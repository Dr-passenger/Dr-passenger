"""Independent H3 -> Real-ESRGAN x4plus -> QHD/UHD video, with audio remux.

Only NPU inference is used by default. CPU is explicit and intended for tests.
Input contract: zero-origin CFR, square-pixel SDR video (normal H3 output).
"""
import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import av
import numpy as np

TARGETS = {"2k": (2560, 1440), "4k": (3840, 2160)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fitted_size(width, height, target, fit):
    tw, th = target
    scale = (min if fit == "pad" else max)(tw / width, th / height)
    rounding = math.floor if fit == "pad" else math.ceil
    return max(1, rounding(width * scale)), max(1, rounding(height * scale))


def fit_frame(rgb, target, fit):
    h, w = rgb.shape[:2]
    rw, rh = fitted_size(w, h, target, fit)
    scaled = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
        width=rw, height=rh, format="rgb24", interpolation="LANCZOS").to_ndarray()
    tw, th = target
    if fit == "crop":
        x, y = (rw - tw) // 2, (rh - th) // 2
        return np.ascontiguousarray(scaled[y:y + th, x:x + tw])
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x, y = (tw - rw) // 2, (th - rh) // 2
    canvas[y:y + rh, x:x + rw] = scaled
    return canvas


def check_timestamp(frame, number, fps):
    if frame.pts is None or frame.time_base is None:
        raise ValueError("Video frame has no timestamp; refusing to guess A/V timing")
    actual = frame.pts * frame.time_base
    # Allow container quantization, including millisecond MKV timestamps.
    if abs(actual - Fraction(number, 1) / fps) > Fraction(1, 500):
        raise ValueError(f"Requires zero-origin CFR video: frame {number}, timestamp {actual}")


def inspect_video(path):
    with av.open(str(path)) as src:
        if len(src.streams.video) != 1:
            raise ValueError("Expected exactly one video stream")
        video = src.streams.video[0]
        fps = video.average_rate
        if not fps or not 0 < fps <= 120:
            raise ValueError("Missing or unsupported frame rate")
        if video.sample_aspect_ratio not in (None, Fraction(1, 1)):
            raise ValueError("Non-square pixels are not supported")
        if video.start_time is not None and abs(video.start_time * video.time_base) > Fraction(1, 500):
            raise ValueError("Video must start at timestamp zero")
        frame = next(src.decode(video), None)
        if frame is None:
            raise ValueError("Input has no decodable video frames")
        check_timestamp(frame, 0, fps)
        if max(c.bits for c in frame.format.components) > 8:
            raise ValueError("Only 8-bit SDR H3 input is supported; no implicit HDR conversion")
        if video.codec_context.color_trc in (16, 18) or frame.colorspace in (9, 10):
            raise ValueError("HDR/BT.2020 input requires an explicit tone-mapping pipeline")
        if float(getattr(frame, "rotation", 0)) != 0 or video.metadata.get("rotate", "0") != "0":
            raise ValueError("Rotated input is not supported; normalize orientation first")
        return {"width": frame.width, "height": frame.height, "fps": fps,
                "audio_streams": len(src.streams.audio)}


class Upscaler:
    def __init__(self, weights, device="npu:0", precision="fp32", tile=256, overlap=32):
        import torch
        from rrdbnet import RRDBNet
        self.torch = torch
        if re.fullmatch(r"npu:\d+", device):
            import torch_npu  # noqa: F401; registers the NPU device
            if not torch.npu.is_available():
                raise RuntimeError("NPU unavailable; check CANN/driver/torch_npu. No CPU fallback.")
            torch.npu.set_device(device)
        elif device != "cpu":
            raise ValueError("Use npu:<logical_index>, or explicit cpu for verification")
        if device == "cpu" and precision != "fp32":
            raise ValueError("CPU verification uses fp32")
        self.device = torch.device(device)
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16}[precision]
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params"))
        if state is None:
            raise ValueError("Expected official RealESRGAN_x4plus params_ema/params checkpoint")
        self.model = RRDBNet()
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        self.tile, self.overlap = tile, overlap

    def __call__(self, rgb):
        torch = self.torch
        h, w = rgb.shape[:2]
        # Stitch on CPU: the full 4x canvas never occupies NPU memory.
        result = np.empty((h * 4, w * 4, 3), dtype=np.uint8)
        with torch.inference_mode():
            for y in range(0, h, self.tile):
                for x in range(0, w, self.tile):
                    x1, y1 = min(x + self.tile, w), min(y + self.tile, h)
                    left, top = max(0, x - self.overlap), max(0, y - self.overlap)
                    right, bottom = min(w, x1 + self.overlap), min(h, y1 + self.overlap)
                    array = np.ascontiguousarray(rgb[top:bottom, left:right].transpose(2, 0, 1))
                    tensor = torch.from_numpy(array).unsqueeze(0).to(self.device, dtype=self.dtype) / 255.0
                    out = self.model(tensor)
                    if not torch.isfinite(out).all().item():
                        raise RuntimeError("Non-finite model output; retry in fp32, check NPU precision")
                    out = out[0, :, (y - top) * 4:(y1 - top) * 4, (x - left) * 4:(x1 - left) * 4]
                    patch = out.float().clamp(0, 1).mul(255).round().byte().cpu().numpy().transpose(1, 2, 0)
                    result[y * 4:y1 * 4, x * 4:x1 * 4] = patch
                    del tensor, out, patch
        return result


def audio_fingerprints(path):
    """Hash compressed packet payloads, not re-encoded/decoded audio."""
    with av.open(str(path)) as src:
        streams = list(src.streams.audio)
        digests = {s.index: hashlib.sha256() for s in streams}
        counts = {s.index: 0 for s in streams}
        starts, ends = {}, {}
        if streams:
            for packet in src.demux(streams):
                if not packet.size:
                    continue
                key = packet.stream.index
                digests[key].update(bytes(packet))
                counts[key] += 1
                if packet.pts is None:
                    raise ValueError("Audio packet has no timestamp")
                start = packet.pts * packet.time_base
                end = start + (packet.duration or 0) * packet.time_base
                starts[key] = min(starts.get(key, start), start)
                ends[key] = max(ends.get(key, end), end)
        return [{"codec": s.codec_context.name, "sample_rate": s.codec_context.sample_rate,
                 "channels": s.codec_context.channels, "packets": counts[s.index],
                 "sha256": digests[s.index].hexdigest(),
                 "start": float(starts.get(s.index, 0)), "end": float(ends.get(s.index, 0))}
                for s in streams]


def verify_output(source_audio, output, target, fps, count):
    observed = 0
    with av.open(str(output)) as src:
        for frame in src.decode(video=0):
            if (frame.width, frame.height) != target:
                raise ValueError("Output resolution mismatch")
            check_timestamp(frame, observed, fps)
            observed += 1
    if observed != count:
        raise ValueError(f"Output frame count mismatch: {observed} != {count}")
    result_audio = audio_fingerprints(output)
    if len(result_audio) != len(source_audio):
        raise ValueError("Audio stream count changed")
    for before, after in zip(source_audio, result_audio):
        for key in ("codec", "sample_rate", "channels", "packets", "sha256"):
            if before[key] != after[key]:
                raise ValueError(f"Original audio not preserved: {key}")
        for key in ("start", "end"):
            if abs(before[key] - after[key]) > 0.002:
                raise ValueError(f"Audio timeline changed: {key}")
    return result_audio


def process_video(source, output, upscaler, target, fit="pad", crf=18, preset="medium"):
    """Write a private partial file; publish only after frame/audio validation."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError("Input/output must differ and output must not already exist")
    if output.suffix.lower() not in (".mp4", ".mkv"):
        raise ValueError("Output must be .mp4 or .mkv")
    info = inspect_video(source)
    source_audio = audio_fingerprints(source)
    fps = info["fps"]
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    # Temp directory is on the destination filesystem; final hard-link publication
    # is atomic and refuses a concurrently created output (unlike os.replace).
    with tempfile.TemporaryDirectory(prefix=".sr-", dir=output.parent) as temporary:
        partial = Path(temporary) / ("video" + output.suffix.lower())
        options = {"movflags": "+faststart", "avoid_negative_ts": "disabled"} if output.suffix.lower() == ".mp4" else {"avoid_negative_ts": "disabled"}
        with av.open(str(source)) as src, av.open(str(partial), "w", options=options) as dst:
            vin = src.streams.video[0]
            vout = dst.add_stream("libx264", rate=fps)
            vout.width, vout.height = target
            vout.pix_fmt = "yuv420p"
            vout.time_base = Fraction(1, 1) / fps
            vout.codec_context.time_base = vout.time_base
            vout.codec_context.thread_count = 4
            vout.codec_context.color_range = 1  # limited-range output
            vout.codec_context.colorspace = 1  # BT.709
            vout.codec_context.color_primaries = 1
            vout.codec_context.color_trc = 1
            vout.options = {"crf": str(crf), "preset": preset, "bf": "0"}
            audio = {s.index: dst.add_stream_from_template(s) for s in src.streams.audio}
            for s in src.streams.audio:
                audio[s.index].metadata.update(s.metadata)
            for packet in src.demux([vin, *src.streams.audio]):
                if packet.stream.index in audio:
                    if packet.size:
                        packet.stream = audio[packet.stream.index]
                        dst.mux(packet)
                    continue
                for frame in packet.decode():
                    check_timestamp(frame, count, fps)
                    if (frame.width, frame.height) != (info["width"], info["height"]):
                        raise ValueError("Input changes resolution mid-stream")
                    rgb = frame.to_ndarray(format="rgb24")
                    enlarged = upscaler(rgb)
                    if enlarged.dtype != np.uint8 or enlarged.shape != (frame.height * 4, frame.width * 4, 3):
                        raise ValueError("Upscaler did not produce an RGB uint8 4x frame")
                    fitted = fit_frame(enlarged, target, fit)
                    out_frame = av.VideoFrame.from_ndarray(fitted, format="rgb24").reformat(
                        format="yuv420p", dst_colorspace="ITU709", src_color_range=2, dst_color_range=1)
                    out_frame.pts, out_frame.time_base = count, Fraction(1, 1) / fps
                    for encoded in vout.encode(out_frame):
                        dst.mux(encoded)
                    count += 1
                    if count == 1 or count % 24 == 0:
                        print(json.dumps({"frames": count, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            if not count:
                raise ValueError("No video frames processed")
            for encoded in vout.encode(None):
                dst.mux(encoded)
        audio_result = verify_output(source_audio, partial, target, fps, count)
        os.link(partial, output)
    return {"input": str(source), "output": str(output), "width": target[0], "height": target[1],
            "frames": count, "fps": str(fps), "duration_seconds": float(count / fps),
            "audio": audio_result, "audio_copy_verified": True,
            "elapsed_seconds": round(time.monotonic() - started, 3)}


@contextmanager
def device_lock(device):
    """Serializes this program per visible NPU; does not stop or lock H3."""
    if not device.startswith("npu:"):
        yield
        return
    import fcntl  # Ascend target is Linux
    index = int(device.split(":")[1])
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    physical = visible.split(",")[index].strip() if visible else str(index)
    key = hashlib.sha256(physical.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / (f"h3-realesrgan-{os.getuid()}-" + key + ".lock")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another super-resolution job holds this NPU lock") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-sha256", help="Optional trusted checkpoint digest")
    parser.add_argument("--target", choices=TARGETS, default="4k")
    parser.add_argument("--fit", choices=("pad", "crop"), default="pad")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", choices=("ultrafast", "fast", "medium", "slow"), default="medium")
    args = parser.parse_args()
    if args.tile < 32 or args.overlap < 0 or args.overlap >= args.tile or not 0 <= args.crf <= 51:
        parser.error("tile >= 32; 0 <= overlap < tile; 0 <= crf <= 51")
    if not re.fullmatch(r"npu:\d+|cpu", args.device):
        parser.error("Use npu:<logical_index>, or explicit cpu for verification")
    if args.output.suffix.lower() not in (".mp4", ".mkv"):
        parser.error("Output must be .mp4 or .mkv")
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    if report_path.exists():
        parser.error("Output report already exists; choose a new output path")
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error("Output already exists or equals input")
    inspect_video(args.input)  # fail before loading weights
    digest = sha256_file(args.weights)
    if args.weights_sha256 and digest.lower() != args.weights_sha256.lower():
        parser.error("Weights SHA-256 mismatch")
    with device_lock(args.device):
        engine = Upscaler(args.weights, args.device, args.precision, args.tile, args.overlap)
        report = process_video(args.input, args.output, engine, TARGETS[args.target], args.fit, args.crf, args.preset)
        report.update({"weights_sha256": digest, "device": args.device, "precision": args.precision,
                       "tile": args.tile, "overlap": args.overlap, "fit": args.fit})
        # Sidecar is not a completion signal: the final video itself is published last
        # in process_video only after successful validation.
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
