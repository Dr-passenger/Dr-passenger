"""Run on Linux in a cloned/independent, already working torch + torch-npu environment."""
import argparse
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusers-ref", required=True, help="Reviewed full 40-hex upstream Diffusers commit with QwenImage21Pipeline")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Use the Linux Ascend server, not this Windows machine")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.diffusers_ref):
        parser.error("Use a full immutable commit SHA, not main or an unreviewed floating branch")
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "torch-npu")}
    target = ROOT / ".venv"
    if target.exists():
        parser.error(".venv already exists; preserve it. Use a fresh deployment directory for an upgrade")
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(target)], check=True)
    python = str(target / "bin" / "python")
    constraints = ROOT / "torch.constraints.txt"
    constraints.write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n", encoding="utf-8")
    source = f"git+https://github.com/huggingface/diffusers.git@{args.diffusers_ref}"
    subprocess.run([python, "-m", "pip", "install", "-c", str(constraints), "-r", str(ROOT / "requirements.in"), source], check=True)
    subprocess.run([python, "-m", "pip", "check"], check=True)
    subprocess.run([python, "-c", "import torch, torch_npu; from diffusers import QwenImage21Pipeline; print('Imports OK; NPU generation remains untested')"], check=True)
    freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements.resolved.txt").write_text(freeze, encoding="utf-8")
    print("Installed in .venv; base H3 packages were not upgraded. No weights downloaded.")


if __name__ == "__main__":
    main()
