"""Explicit download from the official release; never runs during inference."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/RealESRGAN_x4plus.pth"))
    parser.add_argument("--sha256", help="Optional trusted digest from your model registry")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Destination exists; will not overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix=".download-", dir=args.output.parent) as folder:
        part = Path(folder) / "weights.part"
        with urlopen(URL, timeout=60) as response, part.open("xb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                stream.write(chunk)
        value = digest.hexdigest()
        if args.sha256 and value.lower() != args.sha256.lower():
            raise ValueError("SHA-256 mismatch; download not published")
        os.link(part, args.output)
    print(f"Saved {args.output}\nSHA256 {value}")
    print("A locally calculated digest records this download; without a trusted expected digest it is not independent authenticity verification.")


if __name__ == "__main__":
    main()
