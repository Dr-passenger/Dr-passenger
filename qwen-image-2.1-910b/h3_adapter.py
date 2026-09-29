"""Integration point for YOUR H3 scheduler, deliberately fail-closed.

argv: python /absolute/path/h3_adapter.py pause|status|resume LEASE_ID
Replace main() using the documented contract in README.md. Never fake success.
"""
import sys


def main():
    raise RuntimeError(
        "H3 adapter is not connected. Implement durable admission pause, global active-job "
        "count, lease ownership and idempotent resume in your actual H3 scheduler. "
        "Do not implement pause by stopping/killing H3 or disabling its audit model."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
