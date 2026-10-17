#!/usr/bin/env bash
# Run from an already validated torch/torch_npu/CANN environment on Linux 910B.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
SR_BASE_PYTHON="${SR_BASE_PYTHON:-python3}"
"$SR_BASE_PYTHON" -c 'import torch, torch_npu; assert torch.npu.is_available(), "NPU unavailable"; print("Using existing stack:", torch.__version__, torch_npu.__version__)'
if [[ -e .venv ]]; then
    echo 'Refusing to replace existing .venv; use it or choose a fresh package directory.' >&2
    exit 1
fi
"$SR_BASE_PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import av, torch, torch_npu; print("encoder:", av.codec.Codec("libx264", "w").name); assert torch.npu.is_available()'
echo 'Ready. Download/locate weights, then run check_environment.py before video inference.'
