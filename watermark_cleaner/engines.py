"""Local, frame-preserving video restoration engine interface."""

from __future__ import annotations
from enum import StrEnum
from functools import lru_cache
import os
import threading
from typing import Callable, Protocol, TYPE_CHECKING
import numpy as np

Progress = Callable[[float, str], None]


class EngineName(StrEnum):
    E2FGVI = "e2fgvi"
    PROPAINTER = "propainter"


class TemporalEngine(Protocol):
    def restore(
        self, frames: np.ndarray, masks: np.ndarray, progress: Progress | None = None
    ) -> np.ndarray:
        """Restore BGR uint8 frames, retaining every unmasked source pixel."""
        ...


def get_engine(engine_id: str) -> TemporalEngine:
    """Select an explicit local video model; unknown engine names fail clearly."""
    with _FACTORY_LOCK:
        return _cached_engine(EngineName(engine_id).value)


if TYPE_CHECKING:
    from .backends.e2fgvi.network import InpaintGenerator as E2Generator
    from .backends.propainter.network import InpaintGenerator as PPGenerator
    from .backends.propainter.modules.flow_comp_raft import RAFT_bi
    from .backends.propainter.recurrent_flow_completion import RecurrentFlowCompleteNet

_INFERENCE_LOCK = threading.RLock()
_FACTORY_LOCK = threading.Lock()


def _notify(progress: Progress | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def _inputs(frames: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("Video frames must be a uint8 array")
    if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape[:3]) < 1:
        raise ValueError("Video frames must have shape [frames, height, width, 3]")
    if frames.shape[0] > 80:
        raise ValueError("A temporal engine call supports at most 80 frames")
    if (
        not isinstance(masks, np.ndarray)
        or masks.dtype != np.uint8
        or masks.shape != frames.shape[:3]
    ):
        raise ValueError("Watermark masks must be uint8 and match the video frame dimensions")
    return np.ascontiguousarray(frames), np.ascontiguousarray(masks > 0)


def _padded(
    frames: np.ndarray, masks: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    ph, pw = height - frames.shape[1], width - frames.shape[2]
    # Symmetric extension matches the reflected borders in upstream HQ inference.
    return (
        np.pad(frames, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="symmetric"),
        np.pad(masks, ((0, 0), (0, ph), (0, pw)), mode="symmetric"),
    )


def _finish(frames: np.ndarray, masks: np.ndarray, prediction_rgb: np.ndarray) -> np.ndarray:
    if prediction_rgb.shape != frames.shape or not np.isfinite(prediction_rgb).all():
        raise RuntimeError("The video model returned invalid frames. The original video was kept.")
    result = frames.copy()
    bgr = np.clip(prediction_rgb, 0, 255).astype(np.uint8)[..., ::-1]
    result[masks] = bgr[masks]
    return result


class E2FGVIEngine:
    name = EngineName.E2FGVI
    display_name = "E2FGVI-HQ"

    def __init__(self) -> None:
        self._model: E2Generator | None = None

    def _load(self, progress: Progress | None) -> E2Generator:
        import torch
        from .backends.e2fgvi.network import InpaintGenerator
        from .model_store import ensure_models

        if self._model is None:
            _notify(progress, 0.0, "Preparing E2FGVI-HQ")
            paths = ensure_models(
                self.name, lambda value, message: _notify(progress, value * 0.05, message)
            )
            model = InpaintGenerator(init_weights=False)
            state = torch.load(paths["E2FGVI-HQ-CVPR22.pth"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            self._model = model
        return self._model

    def restore(
        self, frames: np.ndarray, masks: np.ndarray, progress: Progress | None = None
    ) -> np.ndarray:
        import torch

        source, selected = _inputs(frames, masks)
        if not selected.any():
            _notify(progress, 1.0, "No selected watermark pixels in this section")
            return source.copy()
        with _INFERENCE_LOCK:
            if torch.get_num_threads() > 6:
                torch.set_num_threads(min(6, os.cpu_count() or 1))
            model = self._load(progress)
            length, height, width = source.shape[:3]
            # One-frame cuts use duplicate context, then retain the one source frame.
            working = source if length > 1 else np.repeat(source, 2, axis=0)
            mask = selected if length > 1 else np.repeat(selected, 2, axis=0)
            padded_h = max(120, ((height + 59) // 60) * 60)
            padded_w = max(216, ((width + 107) // 108) * 108)
            working, mask = _padded(working, mask, padded_h, padded_w)
            images = (
                torch.from_numpy(working[..., ::-1].transpose(0, 3, 1, 2).copy())
                .unsqueeze(0)
                .float()
                / 127.5
                - 1
            )
            mask_t = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(2).float()
            count = working.shape[0]
            predictions: list[np.ndarray | None] = [None] * count
            windows = list(range(0, count, 5))
            _notify(progress, 0.1, "Restoring video with E2FGVI-HQ")
            with torch.inference_mode():
                for step, center in enumerate(windows):
                    neighbors = list(range(max(0, center - 5), min(count, center + 6)))
                    references = [index for index in range(0, count, 10) if index not in neighbors]
                    indices = neighbors + references
                    predicted, _ = model(
                        images[:, indices] * (1 - mask_t[:, indices]), len(neighbors)
                    )
                    if not torch.isfinite(predicted).all():
                        raise RuntimeError(
                            "E2FGVI-HQ returned invalid pixels. The source was kept."
                        )
                    predicted = (
                        ((predicted[:, :, :height, :width] + 1) * 127.5)
                        .clamp(0, 255)
                        .permute(0, 2, 3, 1)
                        .numpy()
                    )
                    for local, index in enumerate(neighbors):
                        old = predictions[index]
                        predictions[index] = (
                            predicted[local].copy()
                            if old is None
                            else (old + predicted[local]) * 0.5
                        )
                    _notify(
                        progress,
                        0.1 + 0.85 * (step + 1) / len(windows),
                        f"E2FGVI-HQ: restoring section {step + 1} of {len(windows)}",
                    )
            if any(item is None for item in predictions[:length]):
                raise RuntimeError("E2FGVI-HQ did not return every requested frame")
            result = _finish(source, selected, np.stack(predictions[:length]))
            _notify(progress, 1.0, "E2FGVI-HQ section finished")
            return result


class ProPainterEngine:
    name = EngineName.PROPAINTER
    display_name = "ProPainter"

    def __init__(self) -> None:
        self._model: PPGenerator | None = None
        self._raft: RAFT_bi | None = None
        self._flow: RecurrentFlowCompleteNet | None = None

    def _load(
        self, progress: Progress | None
    ) -> tuple[PPGenerator, RAFT_bi, RecurrentFlowCompleteNet]:
        import torch
        from .backends.propainter.network import InpaintGenerator
        from .backends.propainter.modules.flow_comp_raft import RAFT_bi
        from .backends.propainter.recurrent_flow_completion import RecurrentFlowCompleteNet
        from .model_store import ensure_models

        if self._model is None or self._raft is None or self._flow is None:
            _notify(progress, 0.0, "Preparing ProPainter")
            paths = ensure_models(
                self.name, lambda value, message: _notify(progress, value * 0.05, message)
            )
            raft = RAFT_bi(str(paths["raft-things.pth"]), device=torch.device("cpu"))
            flow = RecurrentFlowCompleteNet()
            flow.load_state_dict(
                torch.load(
                    paths["recurrent_flow_completion.pth"], map_location="cpu", weights_only=True
                ),
                strict=True,
            )
            model = InpaintGenerator(init_weights=False)
            model.load_state_dict(
                torch.load(paths["ProPainter.pth"], map_location="cpu", weights_only=True),
                strict=True,
            )
            for network in (raft, flow, model):
                network.eval().requires_grad_(False)
            self._model, self._raft, self._flow = model, raft, flow
        return self._model, self._raft, self._flow

    def restore(
        self, frames: np.ndarray, masks: np.ndarray, progress: Progress | None = None
    ) -> np.ndarray:
        import torch

        source, selected = _inputs(frames, masks)
        if not selected.any():
            _notify(progress, 1.0, "No selected watermark pixels in this section")
            return source.copy()
        with _INFERENCE_LOCK:
            if torch.get_num_threads() > 6:
                torch.set_num_threads(min(6, os.cpu_count() or 1))
            model, raft, flow = self._load(progress)
            length, height, width = source.shape[:3]
            working = source if length > 1 else np.repeat(source, 2, axis=0)
            mask = selected if length > 1 else np.repeat(selected, 2, axis=0)
            working, mask = _padded(
                working, mask, max(128, (height + 7) // 8 * 8), max(128, (width + 7) // 8 * 8)
            )
            count, padded_h, padded_w = working.shape[:3]
            images = (
                torch.from_numpy(working[..., ::-1].transpose(0, 3, 1, 2).copy())
                .unsqueeze(0)
                .float()
                / 127.5
                - 1
            )
            mask_t = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(2).float()
            _notify(progress, 0.1, "ProPainter: estimating motion between frames")
            with torch.inference_mode():
                forward, backward = [], []
                chunks = list(range(0, count, 12))
                for step, start in enumerate(chunks):
                    end = min(count, start + 12)
                    a, b = raft(images[:, max(0, start - 1) : end], iters=20)
                    forward.append(a)
                    backward.append(b)
                    _notify(
                        progress,
                        0.1 + 0.15 * (step + 1) / len(chunks),
                        f"ProPainter: estimating motion {step + 1} of {len(chunks)}",
                    )
                observed_flows = (torch.cat(forward, dim=1), torch.cat(backward, dim=1))
                completed, _ = flow.forward_bidirect_flow(observed_flows, mask_t)
                completed = flow.combine_flow(observed_flows, completed, mask_t)
                _notify(progress, 0.4, "ProPainter: rebuilding motion behind the watermark")
                propagated, remaining = model.img_propagation(
                    images * (1 - mask_t), completed, mask_t, "nearest"
                )
                updated = (
                    images * (1 - mask_t)
                    + propagated.view(1, count, 3, padded_h, padded_w) * mask_t
                )
                remaining = remaining.view(1, count, 1, padded_h, padded_w)
                predictions: list[np.ndarray | None] = [None] * count
                windows = list(range(0, count, 5))
                for step, center in enumerate(windows):
                    neighbors = list(range(max(0, center - 5), min(count, center + 6)))
                    references = [index for index in range(0, count, 10) if index not in neighbors]
                    indices = neighbors + references
                    local_flows = (completed[0][:, neighbors[:-1]], completed[1][:, neighbors[:-1]])
                    predicted = model(
                        updated[:, indices],
                        local_flows,
                        mask_t[:, indices],
                        remaining[:, indices],
                        len(neighbors),
                    )
                    predicted = predicted.view(-1, 3, padded_h, padded_w)
                    if not torch.isfinite(predicted).all():
                        raise RuntimeError(
                            "ProPainter returned invalid pixels. The source was kept."
                        )
                    predicted = (
                        ((predicted + 1) * 127.5)
                        .clamp(0, 255)
                        .permute(0, 2, 3, 1)
                        .numpy()
                        .astype(np.uint8)
                    )
                    for local, index in enumerate(neighbors):
                        candidate = predicted[local, :height, :width]
                        old = predictions[index]
                        predictions[index] = (
                            candidate.copy()
                            if old is None
                            else (
                                old.astype(np.float32) * 0.5 + candidate.astype(np.float32) * 0.5
                            ).astype(np.uint8)
                        )
                    _notify(
                        progress,
                        0.45 + 0.5 * (step + 1) / len(windows),
                        f"ProPainter: restoring section {step + 1} of {len(windows)}",
                    )
            if any(item is None for item in predictions[:length]):
                raise RuntimeError("ProPainter did not return every requested frame")
            result = _finish(source, selected, np.stack(predictions[:length]))
            _notify(progress, 1.0, "ProPainter section finished")
            return result


# Canonical string keys avoid separate cached models for a StrEnum and its value.
@lru_cache(maxsize=2)
def _cached_engine(name: str) -> TemporalEngine:
    return E2FGVIEngine() if name == EngineName.E2FGVI.value else ProPainterEngine()
