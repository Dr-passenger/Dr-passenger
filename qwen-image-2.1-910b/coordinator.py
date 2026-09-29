"""Fail-closed H3/Qwen scheduling. No torch imports in this parent process."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from common import GIB, exclusive_lock, load_config, read_json, validate_request, worker_env, write_json

ROOT = Path(__file__).resolve().parent


def hook(cfg, action, lease):
    result = subprocess.run(
        cfg["h3_control_argv"] + [action, lease], check=True,
        capture_output=True, text=True, encoding="utf-8",
        timeout=cfg["h3_hook_timeout_seconds"],
    )
    reply = json.loads(result.stdout)
    if not isinstance(reply, dict) or reply.get("lease_id") != lease:
        raise RuntimeError(f"H3 {action}: missing/mismatched lease acknowledgement")
    if action in ("pause", "status") and reply.get("paused") is not True:
        raise RuntimeError("H3 admission is not paused by this lease")
    if action == "status" and (type(reply.get("active_jobs")) is not int or reply["active_jobs"] < 0):
        raise RuntimeError("H3 status must include a nonnegative integer active_jobs")
    if action == "resume" and reply.get("resumed") is not True:
        raise RuntimeError("H3 did not acknowledge resuming this lease")
    return reply


def wait_idle(cfg, lease):
    deadline = time.monotonic() + cfg["h3_drain_timeout_seconds"]
    while True:
        if hook(cfg, "status", lease)["active_jobs"] == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("H3 did not drain before timeout; no Qwen process was started")
        time.sleep(cfg["poll_seconds"])


def probe(cfg):
    result = subprocess.run(
        [sys.executable, str(ROOT / "worker.py"), "--probe"],
        env=worker_env(cfg), check=True, capture_output=True,
        text=True, encoding="utf-8", timeout=120,
    )
    # torch/CANN may print diagnostics to stdout before our final JSON record.
    data = json.loads(result.stdout.strip().splitlines()[-1])
    if not 0 <= data["free_gib"] <= data["total_gib"]:
        raise RuntimeError("Invalid NPU memory report")
    if data["visible_devices"] != cfg["physical_device_id"]:
        raise RuntimeError("Unexpected device mapping in probe")
    return data


def available_host_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GIB
    raise RuntimeError("Cannot determine available host RAM")


def check_capacity(cfg, baseline):
    if baseline["free_gib"] < cfg["min_free_npu_gib"]:
        raise RuntimeError(f"Need >= {cfg['min_free_npu_gib']} GiB free NPU memory; got {baseline['free_gib']:.2f}")
    available = available_host_gib()
    if available < cfg["min_available_host_gib"]:
        raise RuntimeError(f"Need >= {cfg['min_available_host_gib']} GiB available host RAM; got {available:.2f}")


def run_worker(cfg, job, journal, state_path):
    journal["phase"] = "worker_starting"
    write_json(state_path, journal)
    with (job / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "worker.py"), "--job-dir", str(job)],
            env=worker_env(cfg), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            journal.update(phase="worker_running", worker_pid=process.pid)
            write_json(state_path, journal)
            code = process.wait(timeout=cfg["worker_timeout_seconds"])
            if code:
                raise RuntimeError(f"Qwen worker exited {code}; see {job / 'worker.log'}")
            if not (job / "image.png").is_file() or not (job / "metrics.json").is_file():
                raise RuntimeError("Worker exited without image/metrics")
        finally:
            # Only terminate this newly created Qwen process group, never H3/audit.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            journal.update(phase="worker_exited", worker_exited=True)
            write_json(state_path, journal)


def wait_released(cfg, baseline):
    deadline = time.monotonic() + cfg["release_timeout_seconds"]
    target = max(cfg["reserve_npu_gib"], baseline["free_gib"] - cfg["release_tolerance_gib"])
    last = None
    while True:
        last = probe(cfg)
        if last["free_gib"] >= target:
            return last
        if time.monotonic() >= deadline:
            raise RuntimeError(f"NPU memory not back to baseline: {last['free_gib']:.2f} < {target:.2f} GiB; H3 remains paused")
        time.sleep(cfg["poll_seconds"])


def run_job(cfg, request):
    state_dir = Path(cfg["state_dir"])
    state_path = state_dir / "active.json"
    with exclusive_lock(state_dir / "coordinator.lock"):
        if state_path.exists():
            raise RuntimeError(f"Unresolved lease at {state_path}; investigate/recover before another job")
        lease = uuid.uuid4().hex
        job = Path(cfg["output_dir"]) / lease
        job.mkdir(parents=True, exist_ok=False)
        write_json(job / "config.resolved.json", cfg)
        write_json(job / "request.json", request)
        journal = {"lease_id": lease, "phase": "pause_requested", "job_dir": str(job),
                   "config": cfg, "worker_exited": False, "baseline": None}
        write_json(state_path, journal)  # Durable BEFORE a possibly uncertain pause call.
        hook(cfg, "pause", lease)
        journal["phase"] = "draining"
        write_json(state_path, journal)
        # If pause or drain fails, leave the lease for explicit recovery.
        wait_idle(cfg, lease)
        journal["phase"] = "h3_idle"
        write_json(state_path, journal)
        failure = None
        try:
            journal["baseline"] = probe(cfg)
            write_json(state_path, journal)
            check_capacity(cfg, journal["baseline"])
            run_worker(cfg, job, journal, state_path)
        except BaseException as exc:
            failure = exc
        # Never resume H3 if cleanup or process exit is uncertain.
        if journal["phase"] in ("worker_starting", "worker_running"):
            raise RuntimeError("Worker termination uncertain; lease retained for operator recovery") from failure
        journal["phase"] = "checking_release"
        write_json(state_path, journal)
        if journal["baseline"] is not None:
            journal["released"] = wait_released(cfg, journal["baseline"])
        if hook(cfg, "status", lease)["active_jobs"] != 0:
            raise RuntimeError("H3 ran jobs while admission was paused; do not trust this adapter")
        journal["phase"] = "resume_requested"
        write_json(state_path, journal)
        hook(cfg, "resume", lease)
        journal["phase"] = "failed" if failure else "completed"
        journal["error"] = str(failure) if failure else None
        write_json(job / "result.json", journal)
        state_path.unlink()
        if failure:
            raise failure
        print(json.dumps({"image": str(job / "image.png"), "h3_resumed": True}, ensure_ascii=False))
        return job


def recover(cfg, confirmed):
    if not confirmed:
        raise RuntimeError("Recovery requires --confirm-no-qwen-process after checking the server")
    state_dir = Path(cfg["state_dir"])
    with exclusive_lock(state_dir / "coordinator.lock"):
        state_path = state_dir / "active.json"
        journal = read_json(state_path)
        saved = journal["config"]
        if any(saved[k] != cfg[k] for k in ("physical_device_id", "h3_control_argv")):
            raise RuntimeError("Device/H3 adapter changed since failure; restore the original config")
        pid = journal.get("worker_pid")
        if pid and not journal.get("worker_exited"):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"PID {pid} still exists; inspect it manually, do not blindly kill it")
        lease = journal["lease_id"]
        # Reassert the same durable lease (adapter must support idempotent retries).
        hook(saved, "pause", lease)
        wait_idle(saved, lease)
        if journal.get("baseline"):
            wait_released(saved, journal["baseline"])
        hook(saved, "resume", lease)
        journal["phase"] = "operator_recovered"
        write_json(Path(journal["job_dir"]) / "recovery.json", journal)
        state_path.unlink()
        print("Recovered lease; H3 admission restored")


def interrupted(_signum, _frame):
    # Let cleanup finish even if Ctrl-C is pressed repeatedly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt("Interrupted; cleaning up this Qwen job")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate config/model layout without touching H3 or NPU")
    sub.add_parser("status", help="Read the persistent local lease journal")
    generate = sub.add_parser("generate")
    generate.add_argument("--request", required=True)
    recovery = sub.add_parser("recover")
    recovery.add_argument("--confirm-no-qwen-process", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, require_ready=args.command != "status")
    if args.command == "check":
        print("Config/model layout valid. This is NOT an NPU/operator/H3 integration test.")
        return
    if args.command == "status":
        path = Path(cfg["state_dir"]) / "active.json"
        print(json.dumps(read_json(path), ensure_ascii=False, indent=2) if path.exists() else "No local active lease (not proof H3 is idle)")
        return
    if sys.platform != "linux":
        raise RuntimeError("Actual generation/recovery requires the Linux Ascend server")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "recover":
        recover(cfg, args.confirm_no_qwen_process)
    else:
        run_job(cfg, validate_request(read_json(args.request), cfg))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
