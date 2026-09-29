"""No NPU, torch, H3 server or network required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common
import coordinator as app


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.cfg = common.read_json(ROOT / "config.json")
        self.cfg.update(state_dir=str(self.base / "state"), output_dir=str(self.base / "output"),
                        model_path=str(self.base / "model"), physical_device_id="7",
                        device_mapping_confirmed=True, h3_control_argv=["example-adapter"])
        self.events = []
        self.baseline = {"free_gib": 29, "total_gib": 64, "visible_devices": "7"}
        self.request = common.validate_request({"prompt": "test"}, self.cfg)

    @property
    def state_path(self):
        return Path(self.cfg["state_dir"]) / "active.json"

    def hook(self, cfg, action, lease):
        self.events.append(action)
        return {"lease_id": lease, "paused": True, "active_jobs": 0, "resumed": True}

    def worker(self, cfg, job, journal, state_path):
        self.events.append("worker")
        journal.update(phase="worker_exited", worker_exited=True)
        common.write_json(state_path, journal)
        (job / "image.png").write_bytes(b"test fixture, not a real image")
        common.write_json(job / "metrics.json", {})

    def run_mock(self, worker=None, release=None):
        with patch.object(app, "hook", side_effect=self.hook), \
             patch.object(app, "probe", return_value=self.baseline), \
             patch.object(app, "available_host_gib", return_value=100), \
             patch.object(app, "run_worker", side_effect=worker or self.worker), \
             patch.object(app, "wait_released", side_effect=release):
            return app.run_job(self.cfg, self.request)

    def test_success_order_and_journal_removed(self):
        job = self.run_mock(release=lambda *_: self.baseline)
        self.assertEqual(self.events, ["pause", "status", "worker", "status", "resume"])
        self.assertFalse(self.state_path.exists())
        self.assertEqual(common.read_json(job / "result.json")["phase"], "completed")

    def test_worker_failure_releases_before_resuming(self):
        def broken(cfg, job, journal, path):
            self.worker(cfg, job, journal, path)
            raise RuntimeError("inference failed")
        def release(*_):
            self.events.append("released")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_mock(worker=broken, release=release)
        self.assertEqual(self.events[-3:], ["released", "status", "resume"])
        self.assertFalse(self.state_path.exists())

    def test_uncertain_worker_exit_never_resumes(self):
        def broken(cfg, job, journal, path):
            journal["phase"] = "worker_running"
            common.write_json(path, journal)
            raise RuntimeError("cannot reap worker")
        with self.assertRaisesRegex(RuntimeError, "termination uncertain"):
            self.run_mock(worker=broken)
        self.assertTrue(self.state_path.exists())
        self.assertNotIn("resume", self.events)

    def test_release_failure_preserves_pause(self):
        with self.assertRaisesRegex(RuntimeError, "memory leak"):
            self.run_mock(release=RuntimeError("memory leak"))
        self.assertTrue(self.state_path.exists())
        self.assertNotIn("resume", self.events)

    def test_pause_uncertainty_retains_lease(self):
        with patch.object(app, "hook", side_effect=TimeoutError("uncertain pause")), \
             patch.object(app, "run_worker") as worker:
            with self.assertRaises(TimeoutError):
                app.run_job(self.cfg, self.request)
        worker.assert_not_called()
        self.assertEqual(common.read_json(self.state_path)["phase"], "pause_requested")

    def test_drain_failure_never_launches_worker(self):
        with patch.object(app, "hook", side_effect=self.hook), \
             patch.object(app, "wait_idle", side_effect=TimeoutError("busy")), \
             patch.object(app, "run_worker") as worker:
            with self.assertRaises(TimeoutError):
                app.run_job(self.cfg, self.request)
        worker.assert_not_called()
        self.assertNotIn("resume", self.events)

    def test_existing_lease_blocks_all_actions(self):
        common.write_json(self.state_path, {"phase": "worker_running"})
        with patch.object(app, "hook") as hook:
            with self.assertRaisesRegex(RuntimeError, "Unresolved lease"):
                app.run_job(self.cfg, self.request)
        hook.assert_not_called()

    def test_insufficient_memory_does_not_launch_worker(self):
        self.baseline["free_gib"] = 20
        with self.assertRaisesRegex(RuntimeError, "Need >="):
            self.run_mock(release=lambda *_: self.baseline)
        self.assertNotIn("worker", self.events)
        self.assertEqual(self.events[-1], "resume")

    def test_resume_failure_retains_journal(self):
        normal = self.hook
        def broken(cfg, action, lease):
            if action == "resume":
                raise TimeoutError("resume uncertain")
            return normal(cfg, action, lease)
        self.hook = broken
        with self.assertRaises(TimeoutError):
            self.run_mock(release=lambda *_: self.baseline)
        self.assertEqual(common.read_json(self.state_path)["phase"], "resume_requested")

    def test_hook_rejects_wrong_lease(self):
        reply = subprocess.CompletedProcess([], 0, json.dumps({"lease_id": "wrong", "paused": True}))
        with patch.object(app.subprocess, "run", return_value=reply):
            with self.assertRaisesRegex(RuntimeError, "lease"):
                app.hook(self.cfg, "pause", "correct")

    def test_hook_rejects_boolean_active_count(self):
        reply = subprocess.CompletedProcess([], 0, json.dumps({"lease_id": "x", "paused": True, "active_jobs": False}))
        with patch.object(app.subprocess, "run", return_value=reply):
            with self.assertRaisesRegex(RuntimeError, "integer"):
                app.hook(self.cfg, "status", "x")

    def test_memory_release_timeout(self):
        with patch.object(app, "probe", return_value={"free_gib": 10}), \
             patch.object(app.time, "monotonic", side_effect=[0, 1000]):
            with self.assertRaisesRegex(RuntimeError, "not back to baseline"):
                app.wait_released(self.cfg, self.baseline)

    def test_exclusive_lock(self):
        path = self.base / "test.lock"
        with common.exclusive_lock(path):
            with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                with common.exclusive_lock(path):
                    pass
        with common.exclusive_lock(path):
            pass

    def test_empty_model_path_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "model_path is empty"):
            common.load_config(ROOT / "config.json")

    def test_model_layout_and_unconfigured_h3(self):
        model = Path(self.cfg["model_path"])
        for name in ("transformer", "text_encoder", "vae", "scheduler", "processor"):
            (model / name).mkdir(parents=True)
        common.write_json(model / "model_index.json", {"_class_name": "QwenImage21Pipeline"})
        path = self.base / "config.json"
        common.write_json(path, self.cfg)
        self.assertEqual(common.load_config(path)["physical_device_id"], "7")
        self.cfg["h3_control_argv"] = []
        common.write_json(path, self.cfg)
        with self.assertRaisesRegex(ValueError, "h3_control_argv"):
            common.load_config(path)

    def test_oversize_and_unknown_request_fields(self):
        for request in ({"prompt": "test", "width": 2048, "height": 2048},
                        {"prompt": "test", "batch": 2}, {"prompt": "test", "width": True},
                        {"prompt": "test", "steps": 0}, {"prompt": ""}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                common.validate_request(request, self.cfg)

    def test_environment_is_single_device_and_offline(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "8", "RANK": "0"}):
            env = common.worker_env(self.cfg)
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "7")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("WORLD_SIZE", env)
        self.assertNotIn("RANK", env)

    def test_recovery_requires_confirmation(self):
        with self.assertRaisesRegex(RuntimeError, "requires"):
            app.recover(self.cfg, False)

    def test_recovery_success_after_release_failure(self):
        with self.assertRaises(RuntimeError):
            self.run_mock(release=RuntimeError("temporary pressure"))
        with patch.object(app, "hook", side_effect=self.hook), \
             patch.object(app, "wait_released", return_value=self.baseline):
            app.recover(self.cfg, True)
        self.assertFalse(self.state_path.exists())
        self.assertEqual(self.events[-3:], ["pause", "status", "resume"])

    def test_recovery_rejects_device_change(self):
        with self.assertRaises(RuntimeError):
            self.run_mock(release=RuntimeError("temporary pressure"))
        self.cfg["physical_device_id"] = "6"
        with self.assertRaisesRegex(RuntimeError, "changed"):
            app.recover(self.cfg, True)

    def test_recovery_rejects_live_worker(self):
        common.write_json(self.state_path, {
            "config": self.cfg, "worker_pid": 12345, "worker_exited": False,
        })
        with patch.object(app.os, "kill", return_value=None), \
             self.assertRaisesRegex(RuntimeError, "still exists"):
            app.recover(self.cfg, True)

    def test_status_breach_after_worker_never_resumes(self):
        normal = self.hook
        def broken(cfg, action, lease):
            reply = normal(cfg, action, lease)
            if action == "status" and "worker" in self.events:
                reply["active_jobs"] = 1
            return reply
        self.hook = broken
        with self.assertRaisesRegex(RuntimeError, "ran jobs while"):
            self.run_mock(release=lambda *_: self.baseline)
        self.assertTrue(self.state_path.exists())
        self.assertNotIn("resume", self.events)

    def test_drain_busy_then_idle(self):
        with patch.object(app, "hook", side_effect=[{"active_jobs": 1}, {"active_jobs": 0}]), \
             patch.object(app.time, "sleep") as sleep:
            app.wait_idle(self.cfg, "lease")
        sleep.assert_called_once_with(self.cfg["poll_seconds"])

    def test_hook_default_refuses_to_fake_pause(self):
        result = subprocess.run([sys.executable, str(ROOT / "h3_adapter.py"), "pause", "x"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("not connected", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
