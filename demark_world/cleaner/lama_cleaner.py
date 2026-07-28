"""A small, macOS-friendly LaMa runner.

The implementation keeps DeMark-World's LaMa behavior while avoiding imports
of unrelated diffusion backends.  The model is downloaded only when this
cleaner is first selected and is stored under the app's writable runtime root.

The LaMa model URL and preprocessing behavior are derived from IOPaint
(https://github.com/Sanster/IOPaint), which is included in the vendored
DeMark-World source and covered by the notices shipped with this project.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from loguru import logger

from demark_world.configs import MODELS_DIR
from demark_world.utils.devices_utils import get_device

LAMA_MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/"
    "add_big_lama/big-lama.pt"
)
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
LAMA_MODEL_PATH = MODELS_DIR / "big-lama.pt"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model(path: Path) -> None:
    if path.exists() and _md5(path) == LAMA_MODEL_MD5:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(LAMA_MODEL_URL, stream=True, timeout=(15, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        actual_md5 = _md5(temporary)
        if actual_md5 != LAMA_MODEL_MD5:
            raise RuntimeError(
                f"Downloaded LaMa model failed integrity verification "
                f"(expected {LAMA_MODEL_MD5}, got {actual_md5})."
            )
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "The LaMa model could not be downloaded. First use requires an "
            f"internet connection. Source: {LAMA_MODEL_URL}. Details: {exc}"
        ) from exc


def _pad_to_modulo(array: np.ndarray, modulo: int = 8) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    height, width = array.shape[:2]
    out_height = ((height + modulo - 1) // modulo) * modulo
    out_width = ((width + modulo - 1) // modulo) * modulo
    return np.pad(
        array,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def _normalize(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    return np.transpose(array, (2, 0, 1)).astype("float32") / 255.0


class LamaCleaner:
    """Load the TorchScript LaMa model and inpaint one frame at a time."""

    def __init__(self) -> None:
        _download_model(LAMA_MODEL_PATH)
        self.device = get_device()
        logger.info(f"Loading LaMa on {self.device} from {LAMA_MODEL_PATH}")
        try:
            self.model = torch.jit.load(
                str(LAMA_MODEL_PATH), map_location="cpu"
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"LaMa could not be loaded from {LAMA_MODEL_PATH}: {exc}"
            ) from exc

    @torch.inference_mode()
    def clean(
        self, input_image: np.ndarray, watermark_mask: np.ndarray
    ) -> np.ndarray:
        original_height, original_width = input_image.shape[:2]
        image = _pad_to_modulo(input_image)
        mask = _pad_to_modulo(watermark_mask)

        image_tensor = torch.from_numpy(_normalize(image)).unsqueeze(0).to(
            self.device
        )
        mask_tensor = torch.from_numpy(_normalize(mask)).unsqueeze(0).to(
            self.device
        )
        mask_tensor = (mask_tensor > 0).to(image_tensor.dtype)

        output = self.model(image_tensor, mask_tensor)
        output = output[0].permute(1, 2, 0).detach().cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype("uint8")
        # The input and FFmpeg stream are both BGR, so preserve channel order.
        return output[:original_height, :original_width]
