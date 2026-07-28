"""Runtime paths and model configuration for MAC WATERMARK CLEANER.

This file is derived from DeMark-World and was changed for the macOS app. In
particular, no directory below the installed .app bundle is ever used for
mutable state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _default_runtime_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MAC WATERMARK CLEANER"
    return Path.home() / ".mac-watermark-cleaner"


RUNTIME_ROOT = Path(
    os.environ.get("MAC_WATERMARK_RUNTIME_DIR")
    or os.environ.get("SORA_WATERMARK_RUNTIME_DIR")
    or str(_default_runtime_root())
).expanduser()
RUNTIME_ROOT.mkdir(exist_ok=True, parents=True)

STATIC_RESOURCES_DIR = SOURCE_ROOT / "resources"
MODELS_DIR = RUNTIME_ROOT / "Models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
OUTPUT_DIR = Path(
    os.environ.get("MAC_WATERMARK_OUTPUT_DIR")
    or os.environ.get("SORA_WATERMARK_OUTPUT_DIR")
    or str(Path.home() / "Movies" / "MAC WATERMARK CLEANER")
).expanduser()
WORKING_DIR = RUNTIME_ROOT / "Working"
LOGS_PATH = RUNTIME_ROOT / "Logs"
DATA_PATH = RUNTIME_ROOT / "Data"
CACHE_DIR = RUNTIME_ROOT / "Cache"

for directory in (
    MODELS_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    WORKING_DIR,
    LOGS_PATH,
    DATA_PATH,
    CACHE_DIR,
):
    directory.mkdir(exist_ok=True, parents=True)

# Keep libraries that respect these common cache variables out of the app
# bundle and in the same user-controlled runtime area.
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))

WATER_MARK_TEMPLATE_IMAGE_PATH = STATIC_RESOURCES_DIR / "watermark_template.png"
WATER_MARK_DETECT_YOLO_WEIGHTS = CHECKPOINT_DIR / "watermark-best.pt"
WATER_MARK_DETECT_YOLO_WEIGHTS_REMOTE_URL = (
    "https://github.com/linkedlist771/DeMark-World/releases/download/"
    "V0.0.1/watermark-best.pt"
)
WATER_MARK_DETECT_YOLO_WEIGHTS_SHA256 = (
    "6ac71b6ab8db27ec7928b5176e60a359"
    "c65e1579a5c1d58cf2f98df30cf3085e"
)
WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON = MODELS_DIR / "model_version.json"

SPYNET_CHECKPOINT_PATH = CHECKPOINT_DIR / "spynet_20210409-c6c1bd09.pth"
PHY_NET_CHECKPOINT_REMOTE_URL = (
    "https://download.openmmlab.com/mmediting/restorers/basicvsr/"
    "spynet_20210409-c6c1bd09.pth"
)
PHY_NET_CHECKPOINT_PATH = SPYNET_CHECKPOINT_PATH

E2FGVI_HQ_CHECKPOINT_PATH = CHECKPOINT_DIR / "E2FGVI-HQ-CVPR22.pth"
E2FGVI_HQ_CHECKPOINT_REMOTE_URL = (
    "https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/"
    "V0.0.1/E2FGVI-HQ-CVPR22.pth"
)

DEFAULT_WATERMARK_REMOVE_MODEL = "lama"
SQLITE_PATH = DATA_PATH / "db.sqlite3"

# These paths are retained for API compatibility with upstream. They are
# static and are deliberately not created at import time.
FRONTUI_DIR = SOURCE_ROOT / "frontend"
FRONTUI_DIST_DIR = FRONTUI_DIR / "dist"
FRONTUI_DIST_DIR_ASSETS = FRONTUI_DIST_DIR / "assets"
FRONTUI_DIST_DIR_INDEX_HTML = FRONTUI_DIST_DIR / "index.html"

ENABLE_E2FGVI_HQ_TORCH_COMPILE = (
    os.environ.get("MAC_WATERMARK_ENABLE_TORCH_COMPILE")
    or os.environ.get("SORA_ENABLE_TORCH_COMPILE")
    or "0"
).lower() in {"1", "true", "yes"}
TORCH_COMPILE_DIR = CACHE_DIR / "torch_compile"
E2FGVI_HQ_TORCH_COMPILE_DIR = TORCH_COMPILE_DIR / "e2fgvi_hq"
E2FGVI_HQ_TORCH_COMPILE_DIR.mkdir(exist_ok=True, parents=True)
E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS = (
    E2FGVI_HQ_TORCH_COMPILE_DIR / "artifacts.bin"
)
E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS_BF16 = (
    E2FGVI_HQ_TORCH_COMPILE_DIR / "artifacts_bf16.bin"
)

DEFAULT_DETECT_BATCH_SIZE = 4
CHUNK_SIZE_PER_GB_VRAM = 5
