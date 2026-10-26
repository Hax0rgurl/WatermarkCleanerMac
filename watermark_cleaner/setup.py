"""Prepare both local restoration models before opening the cleaner."""

from __future__ import annotations

import argparse
import json
import sys

from watermark_cleaner.model_store import ensure_models, models_status


def report_progress(fraction: float, message: str) -> None:
    """A line protocol lets the native launcher display real download progress."""
    fraction = min(1.0, max(0.0, float(fraction)))
    message = " ".join(str(message).split())
    print(f"MAC_WATERMARK_MODEL_PROGRESS={fraction:.4f}|{message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the local video restoration models.")
    parser.add_argument("--status", action="store_true", help="Check the verified cache only.")
    arguments = parser.parse_args(argv)
    try:
        if not arguments.status:
            ensure_models(None, progress=report_progress)
        status = models_status()
        print("MAC_WATERMARK_MODEL_STATUS=" + json.dumps(status, sort_keys=True), flush=True)
        if not status or not all(status.values()):
            print(
                "ERROR: Both restoration models must be ready before the app can open.",
                file=sys.stderr,
                flush=True,
            )
            return 1
        if not arguments.status:
            report_progress(1.0, "E2FGVI-HQ and ProPainter are ready")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: Model setup failed: {error}", file=sys.stderr, flush=True)
        print(
            "Check the connection and reopen the app. Verified downloads will be reused.",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
