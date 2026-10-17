"""Local CPU/codec tests. No claim of Ascend hardware or pretrained quality validation."""
from fractions import Fraction
import tempfile
from pathlib import Path
import unittest

import av
import numpy as np
import torch
from rrdbnet import RRDBNet
from upscale import (Upscaler, TARGETS, audio_fingerprints, check_timestamp,
                     fit_frame, fitted_size, inspect_video, process_video)


def make_video(path, audio=True, count=6, fps=Fraction(24, 1), vfr=False, audio_offset=0):
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = 16, 12, "yuv420p"
        video.options = {"bf": "0", "preset": "ultrafast"}
        sounds = [out.add_stream("aac", rate=32000) for _ in range(int(audio))]
        for sound in sounds:
            sound.layout = "stereo"
        for index in range(count):
            rgb = np.full((12, 16, 3), (20 + index * 15, 80, 150), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, "rgb24")
            frame.pts = index + (1 if vfr and index >= 2 else 0)
            frame.time_base = 1 / fps
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        for sound in sounds:
            samples = int(count / fps * 32000)
            wave = (0.1 * np.sin(np.arange(samples) * 2 * np.pi * 440 / 32000)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(np.stack([wave, wave]), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts, frame.time_base = 32000, audio_offset, Fraction(1, 32000)
            for packet in sound.encode(frame):
                out.mux(packet)
            for packet in sound.encode(None):
                out.mux(packet)


def fake_x4(rgb):
    return rgb.repeat(4, axis=0).repeat(4, axis=1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_audio_copy_and_frames(self):
        source, output = self.root / "source.mp4", self.root / "out.mp4"
        make_video(source)
        report = process_video(source, output, fake_x4, (64, 48), preset="ultrafast")
        self.assertEqual(report["frames"], 6)
        self.assertTrue(report["audio_copy_verified"])
        self.assertEqual(audio_fingerprints(source), audio_fingerprints(output))
        self.assertEqual(inspect_video(output)["fps"], Fraction(24, 1))

    def test_silent_fractional_fps(self):
        source, output = self.root / "silent.mp4", self.root / "out.mp4"
        make_video(source, audio=False, fps=Fraction(24000, 1001))
        report = process_video(source, output, fake_x4, (64, 48), preset="ultrafast")
        self.assertEqual(report["audio"], [])
        self.assertEqual(report["fps"], "24000/1001")

    def test_multiple_audio_tracks_and_offset(self):
        source, output = self.root / "source.mp4", self.root / "out.mp4"
        make_video(source, audio=2, audio_offset=3200)
        report = process_video(source, output, fake_x4, (64, 48), preset="ultrafast")
        self.assertEqual(len(report["audio"]), 2)
        self.assertEqual(audio_fingerprints(source), audio_fingerprints(output))

    def test_silent_mkv(self):
        source, output = self.root / "source.mp4", self.root / "out.mkv"
        make_video(source, audio=False)
        report = process_video(source, output, fake_x4, (64, 48), preset="ultrafast")
        self.assertEqual(report["frames"], 6)

    def test_exact_2k_and_4k(self):
        source = self.root / "one.mp4"
        make_video(source, audio=False, count=1)
        for name, size in TARGETS.items():
            output = self.root / (name + ".mp4")
            report = process_video(source, output, fake_x4, size, preset="ultrafast")
            self.assertEqual((report["width"], report["height"]), size)

    def test_aspect_ratio(self):
        self.assertEqual(fitted_size(1344, 768, (2560, 1440), "pad"), (2520, 1440))
        self.assertEqual(fitted_size(1344, 768, (3840, 2160), "pad"), (3780, 2160))
        image = np.full((12, 16, 3), 255, dtype=np.uint8)
        padded = fit_frame(image, (40, 20), "pad")
        self.assertTrue((padded[:, 0] == 0).all())
        cropped = fit_frame(image, (40, 20), "crop")
        self.assertEqual(cropped.shape, (20, 40, 3))
        self.assertTrue((cropped == 255).all())

    def test_no_overwrite(self):
        source, output = self.root / "source.mp4", self.root / "out.mp4"
        make_video(source)
        output.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            process_video(source, output, fake_x4, (64, 48))
        self.assertEqual(output.read_bytes(), b"keep me")
        with self.assertRaises(FileExistsError):
            process_video(source, source, fake_x4, (64, 48))

    def test_failure_not_published(self):
        source, output = self.root / "source.mp4", self.root / "out.mp4"
        make_video(source)
        def broken(frame):
            raise RuntimeError("synthetic NPU OOM")
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            process_video(source, output, broken, (64, 48))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".sr-*")), [])

    def test_vfr_rejected(self):
        source, output = self.root / "vfr.mp4", self.root / "out.mp4"
        make_video(source, audio=False, vfr=True)
        with self.assertRaisesRegex(ValueError, "CFR"):
            process_video(source, output, fake_x4, (64, 48))
        self.assertFalse(output.exists())

    def test_real_architecture_cpu(self):
        torch.set_num_threads(1)
        model = RRDBNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 16697987)
        weights = self.root / "random-test-only.pth"
        torch.save({"params_ema": model.state_dict()}, weights)
        engine = Upscaler(weights, device="cpu", tile=32)
        rgb = np.full((2, 3, 3), 128, dtype=np.uint8)
        result = engine(rgb)
        self.assertEqual(result.shape, (8, 12, 3))
        source, output = self.root / "source.mp4", self.root / "actual-net.mp4"
        make_video(source, count=1)
        report = process_video(source, output, engine, (64, 48), preset="ultrafast")
        self.assertEqual(report["frames"], 1)

    def test_tile_stitch(self):
        # An exactly local x4 model exposes indexing mistakes without training weights.
        engine = Upscaler.__new__(Upscaler)
        engine.torch, engine.device, engine.dtype = torch, torch.device("cpu"), torch.float32
        engine.tile, engine.overlap = 4, 2
        engine.model = lambda t: t.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        rgb = np.random.default_rng(42).integers(0, 256, (9, 11, 3), dtype=np.uint8)
        np.testing.assert_array_equal(engine(rgb), fake_x4(rgb))

    def test_missing_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            check_timestamp(av.VideoFrame(16, 12), 0, Fraction(24, 1))


if __name__ == "__main__":
    unittest.main()
