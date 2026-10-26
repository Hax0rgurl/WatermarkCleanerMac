"""Contracts and actual pretrained CPU inference for both selectable engines."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from watermark_cleaner import model_store
from watermark_cleaner.engines import EngineName, E2FGVIEngine, ProPainterEngine, get_engine
from watermark_cleaner.backends.e2fgvi.modules.cpu_deform import modulated_deform_conv2d


@pytest.mark.parametrize("engine_name", [EngineName.E2FGVI, EngineName.PROPAINTER])
def test_empty_masks_retain_frames_without_loading_models(engine_name, monkeypatch):
    monkeypatch.setattr(
        model_store,
        "ensure_models",
        lambda *args, **kwargs: pytest.fail("Empty masks loaded models"),
    )
    frames = np.arange(2 * 7 * 9 * 3, dtype=np.uint8).reshape(2, 7, 9, 3)
    masks = np.zeros(frames.shape[:3], np.uint8)
    result = get_engine(engine_name).restore(frames, masks)
    assert np.array_equal(result, frames)
    assert not np.shares_memory(result, frames)


@pytest.mark.parametrize("engine_type", [E2FGVIEngine, ProPainterEngine])
def test_invalid_inputs_fail_before_model_loading(engine_type):
    engine = engine_type()
    frames = np.zeros((2, 7, 9, 3), np.uint8)
    mask = np.zeros(frames.shape[:3], np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        engine.restore(frames.astype(np.float32), mask)
    with pytest.raises(ValueError, match="match"):
        engine.restore(frames, np.zeros((2, 8, 9), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        engine.restore(frames, mask.astype(bool))
    with pytest.raises(ValueError, match="80"):
        engine.restore(np.zeros((81, 7, 9, 3), np.uint8), np.zeros((81, 7, 9), np.uint8))


def test_unknown_engine_is_not_silently_replaced():
    with pytest.raises(ValueError):
        get_engine("lama")
    with pytest.raises(ValueError):
        get_engine("unrecognized")


def test_deformable_cpu_adapter_preserves_grouped_convolution():
    torch.manual_seed(2)
    x = torch.rand(1, 4, 6, 6)
    weight = torch.rand(6, 2, 3, 3)
    bias = torch.rand(6)
    offset = torch.zeros(1, 36, 6, 6)
    modulation = torch.ones(1, 18, 6, 6)
    result = modulated_deform_conv2d(
        x, offset, modulation, weight, bias, padding=1, groups=2, deform_groups=2
    )
    expected = F.conv2d(x, weight, bias, padding=1, groups=2)
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_deformable_cpu_adapter_offset_order_is_vertical_then_horizontal():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    offset = torch.zeros(1, 2, 4, 4)
    offset[:, 0] = 1
    result = modulated_deform_conv2d(
        x, offset, torch.ones(1, 1, 4, 4), torch.ones(1, 1, 1, 1), None
    )
    expected = F.pad(x[:, :, 1:, :], (0, 0, 0, 1))
    assert torch.equal(result, expected)


@pytest.fixture(scope="module")
def pretrained_models():
    # These are actual public checkpoints, not mocked network outputs. The same
    # verified cache can be supplied by CI or an offline developer machine.
    directory = Path(
        os.environ.get(
            "WATERMARK_TEST_MODEL_DIR",
            str(Path(tempfile.gettempdir()) / "watermark-cleaner-pretrained-test-cache"),
        )
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(model_store, "model_directory", lambda: directory)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(3, previous_threads))
    try:
        model_store.ensure_models(None)
        assert model_store.models_status() == {"e2fgvi": True, "propainter": True}
        yield directory
    finally:
        torch.set_num_threads(previous_threads)
        patch.undo()


@pytest.mark.parametrize("name", [EngineName.E2FGVI, EngineName.PROPAINTER])
def test_real_pretrained_cpu_inference_preserves_color_size_and_unmasked_pixels(
    pretrained_models, name
):
    # Odd spatial dimensions exercise native-size padding; the final frame is
    # clear temporal reference footage. BGR blue detects channel-order mistakes.
    frames = np.empty((3, 95, 129, 3), np.uint8)
    frames[:] = [180, 40, 20]
    masks = np.zeros(frames.shape[:3], np.uint8)
    masks[:2, 38:50, 44:62] = 255
    frames[masks > 0] = 255
    original = frames.copy()
    progress = []
    result = get_engine(name).restore(
        frames, masks, lambda value, text: progress.append((value, text))
    )
    assert result.shape == frames.shape
    assert result.dtype == np.uint8
    assert np.array_equal(frames, original)
    assert np.array_equal(result[masks == 0], frames[masks == 0])
    assert np.any(result[masks > 0] != frames[masks > 0])
    mean = result[masks > 0].mean(axis=0)
    assert mean[0] > mean[2] + 80, "Reconstruction swapped RGB and BGR"
    assert progress[-1][0] == 1.0
    assert all(0 <= value <= 1 for value, _ in progress)
    assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))


@pytest.mark.parametrize("name", [EngineName.E2FGVI, EngineName.PROPAINTER])
def test_real_pretrained_single_frame_cut(pretrained_models, name):
    frames = np.full((1, 55, 75, 3), [160, 60, 30], np.uint8)
    masks = np.zeros(frames.shape[:3], np.uint8)
    masks[:, :8, :10] = 255
    frames[masks > 0] = 255
    result = get_engine(name).restore(frames, masks)
    assert result.shape == frames.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[masks == 0], frames[masks == 0])
    assert np.any(result[masks > 0] != frames[masks > 0])


def test_vendor_namespaces_do_not_collide(pretrained_models):
    from watermark_cleaner.backends.e2fgvi.network import InpaintGenerator as E2Generator
    from watermark_cleaner.backends.propainter.network import InpaintGenerator as PPGenerator

    assert E2Generator.__module__ == "watermark_cleaner.backends.e2fgvi.network"
    assert PPGenerator.__module__ == "watermark_cleaner.backends.propainter.network"
    e2 = get_engine("e2fgvi")
    pp = get_engine("propainter")
    e2._load(None)
    pp._load(None)
    assert type(e2._model) is E2Generator
    assert type(pp._model) is PPGenerator
    assert e2 is get_engine(EngineName.E2FGVI)
    assert pp is get_engine(EngineName.PROPAINTER)
