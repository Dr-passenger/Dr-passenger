"""Run on the actual Ascend server before full-video inference."""
import argparse
import json
import platform
from pathlib import Path
import av
import numpy as np
import torch
import torch_npu
from upscale import Upscaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    if not args.device.startswith("npu:"):
        parser.error("This check must run on NPU")
    engine = Upscaler(args.weights, args.device, args.precision, tile=32)
    pattern = np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    result = engine(pattern)
    torch.npu.synchronize()
    assert result.shape == (64, 64, 3)
    print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "av": av.__version__,
        "device": args.device, "device_name": torch.npu.get_device_name(),
        "precision": args.precision, "tiny_inference": "passed",
        "encoder": av.codec.Codec("libx264", "w").name}, ensure_ascii=False, indent=2))
    print("Tiny inference only: full-resolution memory, speed and visual quality still require a real-video test.")


if __name__ == "__main__":
    main()
