#!/usr/bin/env bash
# Call only after H3's successful completion and resource-release decision.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'Usage: bash after_h3.sh H3_VIDEO OUTPUT_VIDEO WEIGHTS [2k|4k] [npu:0] [additional upscale.py flags...]' >&2
    exit 2
fi
SR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SR_INPUT="$1"
SR_OUTPUT="$2"
SR_WEIGHTS="$3"
shift 3
SR_TARGET="${1:-4k}"
if [[ $# -gt 0 ]]; then shift; fi
SR_DEVICE="${1:-npu:0}"
if [[ $# -gt 0 ]]; then shift; fi
SR_PYTHON="${SR_PYTHON:-$SR_DIR/.venv/bin/python}"
exec "$SR_PYTHON" "$SR_DIR/upscale.py" \
    --input "$SR_INPUT" --output "$SR_OUTPUT" --weights "$SR_WEIGHTS" \
    --target "$SR_TARGET" --device "$SR_DEVICE" "$@"
