import hashlib
import json
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from demark_world.configs import (
    WATER_MARK_DETECT_YOLO_WEIGHTS,
    WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON,
)


def ensure_model_downloaded(
    model_path: Path,
    url: str,
    force_download: bool = False,
    expected_sha256: str | None = None,
):
    if model_path.exists() and expected_sha256:
        actual_hash = generate_sha256_hash(model_path)
        if actual_hash.lower() != expected_sha256.lower():
            logger.warning(
                f"Cached model failed SHA-256 verification and will be "
                f"re-downloaded: {model_path}"
            )
            model_path.unlink()

    if not model_path.exists() or force_download:
        logger.debug(f"Downloading model from {url}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = model_path.with_suffix(model_path.suffix + ".download")
        temp_file.unlink(missing_ok=True)
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(15, 300),
                headers={"User-Agent": "Sora-Watermark-Cleaner-macOS/1.0.2"},
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with open(temp_file, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            if temp_file.stat().st_size == 0:
                raise RuntimeError("download returned an empty file")
            if expected_sha256:
                actual_hash = generate_sha256_hash(temp_file)
                if actual_hash.lower() != expected_sha256.lower():
                    raise RuntimeError(
                        "downloaded model failed SHA-256 verification "
                        f"(expected {expected_sha256}, got {actual_hash})"
                    )
            temp_file.replace(model_path)
            logger.success(f"✓ Model downloaded: {model_path}")
            return True
        except Exception as e:
            temp_file.unlink(missing_ok=True)
            raise RuntimeError(
                "The universal watermark detector could not be downloaded. "
                "First use requires an internet connection. "
                f"Source: {url}. Destination: {model_path}. Details: {e}"
            ) from e
    else:
        logger.debug(f"Model already exists: {model_path}")


def generate_sha256_hash(file_path: Path) -> str:
    """Generate SHA256 hash for a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_local_hash() -> str | None:
    """Get local model hash from JSON file, generate if not exists."""
    if WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON.exists():
        with WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON.open("r") as f:
            return json.load(f).get("sha256")

    if WATER_MARK_DETECT_YOLO_WEIGHTS.exists():
        logger.info(f"Generating SHA256 hash for {WATER_MARK_DETECT_YOLO_WEIGHTS}")
        return generate_sha256_hash(WATER_MARK_DETECT_YOLO_WEIGHTS)

    return None



def download_detector_weights(force_download: bool = False):
    raise NotImplementedError("Downloading detector weights is not implemented yet.")
