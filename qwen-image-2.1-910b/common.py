"""Standard-library configuration, journal and cross-process locking."""
import contextlib
import json
import math
import os
from pathlib import Path

GIB = 1024 ** 3


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_config(path, require_ready=True):
    path = Path(path).resolve()
    cfg = read_json(path)
    for key in ("state_dir", "output_dir"):
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be nonempty")
        cfg[key] = str((path.parent / value).resolve())
    for key in ("h3_hook_timeout_seconds", "h3_drain_timeout_seconds", "poll_seconds",
                "worker_timeout_seconds", "release_timeout_seconds", "release_tolerance_gib",
                "min_free_npu_gib", "reserve_npu_gib", "max_qwen_allocator_gib",
                "min_available_host_gib", "max_pixels"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["reserve_npu_gib"] >= cfg["min_free_npu_gib"]:
        raise ValueError("reserve_npu_gib must be smaller than min_free_npu_gib")
    if cfg.get("offload_mode") not in ("model", "sequential"):
        raise ValueError("offload_mode must be model or sequential")
    if require_ready:
        model = cfg.get("model_path")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model_path is empty: fill in the local Diffusers model directory")
        model = Path(model)
        if not model.is_absolute() or not (model / "model_index.json").is_file():
            raise ValueError("model_path must be an absolute local Diffusers directory containing model_index.json")
        cfg["model_path"] = str(model)
        if read_json(model / "model_index.json").get("_class_name") != "QwenImage21Pipeline":
            raise ValueError("model_index.json must identify QwenImage21Pipeline")
        for component in ("transformer", "text_encoder", "vae", "scheduler", "processor"):
            if not (model / component).is_dir():
                raise ValueError(f"Missing Diffusers component: {component}")
        device = cfg.get("physical_device_id")
        if not isinstance(device, str) or not device.isdecimal() or cfg.get("device_mapping_confirmed") is not True:
            raise ValueError("Confirm the physical card mapping and set physical_device_id + device_mapping_confirmed")
        argv = cfg.get("h3_control_argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x.strip() for x in argv):
            raise ValueError("h3_control_argv is unconfigured; refusing to run without H3 admission control")
    return cfg


def validate_request(request, cfg):
    if not isinstance(request, dict) or set(request) - {"prompt", "width", "height", "steps", "seed"}:
        raise ValueError("Only prompt, width, height, steps, seed are supported (single-image T2I)")
    result = {"width": 1024, "height": 1024, "steps": 40, "seed": 42, **request}
    if not isinstance(result.get("prompt"), str) or not result["prompt"].strip() or len(result["prompt"]) > 4000:
        raise ValueError("prompt must contain 1..4000 characters")
    for key in ("width", "height", "steps", "seed"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if any(result[k] < 256 or result[k] % 32 for k in ("width", "height")):
        raise ValueError("width/height must be >=256 and divisible by 32")
    if result["width"] * result["height"] > cfg["max_pixels"]:
        raise ValueError("Resolution exceeds max_pixels; raise only after capacity testing")
    if not 1 <= result["steps"] <= 100 or not 0 <= result["seed"] < 2 ** 63:
        raise ValueError("steps must be 1..100, seed must be 0..2**63-1")
    return result


def worker_env(cfg):
    env = os.environ.copy()
    # This process is never launched via torchrun; do not inherit H3 distributed state.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(ASCEND_RT_VISIBLE_DEVICES=cfg["physical_device_id"], HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    return env


@contextlib.contextmanager
def exclusive_lock(path):
    """One coordinator per shared state directory; never unlink this lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another coordinator is running; retry after it completes") from exc
        try:
            yield handle
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
