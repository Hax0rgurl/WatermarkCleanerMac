"""Writable local runtime paths shared by the native launcher and video engine."""

from __future__ import annotations
import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = Path(
    os.environ.get(
        "MAC_WATERMARK_RUNTIME_DIR",
        str(Path.home() / "Library/Application Support/MAC WATERMARK CLEANER"),
    )
).expanduser()
MODELS_DIR = RUNTIME_ROOT / "Models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
CACHE_DIR = RUNTIME_ROOT / "Cache"
LOGS_PATH = RUNTIME_ROOT / "Logs"
WORKING_DIR = RUNTIME_ROOT / "Working"
OUTPUT_DIR = Path(
    os.environ.get("MAC_WATERMARK_OUTPUT_DIR", str(Path.home() / "Movies/MAC WATERMARK CLEANER"))
).expanduser()
for directory in (RUNTIME_ROOT, MODELS_DIR, CACHE_DIR, LOGS_PATH, WORKING_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
