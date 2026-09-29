"""One-job NPU process. Called only by coordinator.py after H3 is drained."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

from common import GIB, read_json, write_json


def runtime():
    import torch
    import torch_npu  # Explicit Ascend registration; no CUDA monkeypatching.
    if not torch.npu.is_available() or torch.npu.device_count() != 1:
        raise RuntimeError("Expected exactly one visible NPU. Check device mapping/container isolation")
    torch.npu.set_device(0)
    return torch


def snapshot(torch):
    free, total = torch.npu.mem_get_info(0)
    info = {"free_gib": free / GIB, "total_gib": total / GIB,
            "device_name": torch.npu.get_device_name(0),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__}
    for package in ("torch-npu", "diffusers", "transformers", "accelerate"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = "missing"
    return info


def generate(cfg, request, output):
    torch = runtime()
    from diffusers import QwenImage21Pipeline
    start = time.monotonic()
    initial = snapshot(torch)
    if initial["free_gib"] < cfg["min_free_npu_gib"]:
        raise RuntimeError(f"Insufficient free NPU memory: {initial['free_gib']:.2f} GiB")
    cap = min(cfg["max_qwen_allocator_gib"], initial["free_gib"] - cfg["reserve_npu_gib"])
    # Limits this process's caching allocator, not all CANN/driver allocations.
    torch.npu.set_per_process_memory_fraction(cap / initial["total_gib"], 0)
    torch.npu.reset_peak_memory_stats(0)
    probe = torch.ones((16, 16), dtype=torch.bfloat16, device="npu:0")
    _ = probe @ probe
    torch.npu.synchronize()
    del probe, _
    torch.npu.empty_cache()
    pipe = None
    try:
        # Load on CPU, never pipe.to('npu') before enabling offload.
        pipe = QwenImage21Pipeline.from_pretrained(
            cfg["model_path"], torch_dtype=torch.bfloat16,
            local_files_only=True, low_cpu_mem_usage=True,
        )
        if cfg["offload_mode"] == "model":
            pipe.enable_model_cpu_offload(device="npu", gpu_id=0)
        else:
            pipe.enable_sequential_cpu_offload(device="npu", gpu_id=0)
        loaded = time.monotonic()

        def check_memory(_pipe, step, _timestep, kwargs):
            free, _total = torch.npu.mem_get_info(0)
            print(json.dumps({"step": step + 1, "free_gib": free / GIB}), flush=True)
            if free / GIB < cfg["reserve_npu_gib"]:
                raise RuntimeError("NPU free memory crossed reserve; aborting Qwen, not H3")
            return kwargs

        with torch.inference_mode():
            result = pipe(
                prompt=request["prompt"], width=request["width"], height=request["height"],
                num_inference_steps=request["steps"], num_images_per_prompt=1,
                true_cfg_scale=1.0,
                generator=torch.Generator(device="cpu").manual_seed(request["seed"]),
                callback_on_step_end=check_memory,
            )
        torch.npu.synchronize()
        image = result.images[0]
        if image.size != (request["width"], request["height"]):
            raise RuntimeError(f"Unexpected output image size: {image.size}")
        image.save(output / "image.png")
        write_json(output / "metrics.json", {
            "initial": initial, "allocator_cap_gib": cap,
            "peak_allocated_gib": torch.npu.max_memory_allocated(0) / GIB,
            "peak_reserved_gib": torch.npu.max_memory_reserved(0) / GIB,
            "cpu_load_and_setup_seconds": loaded - start,
            "inference_and_save_seconds": time.monotonic() - loaded,
            "note": "Allocator peaks exclude other processes and some driver workspaces. Hardware validation required.",
        })
    finally:
        if pipe is not None:
            # Process exit is the final cleanup boundary even if offload cleanup fails.
            try:
                pipe.maybe_free_model_hooks()
            finally:
                del pipe
        gc.collect()
        torch.npu.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--job-dir", type=Path)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(snapshot(runtime())), flush=True)
    elif args.job_dir:
        generate(read_json(args.job_dir / "config.resolved.json"),
                 read_json(args.job_dir / "request.json"), args.job_dir)
    else:
        parser.error("--probe or --job-dir is required")


if __name__ == "__main__":
    main()
