import numpy as np
import pytest
import torch
from demark_world.cleaner.lama_cleaner import LamaCleaner, MAX_INPAINT_SIZE

class RecordingModel:
    def __init__(self):
        self.image = None
        self.mask = None

    def __call__(self, image, mask):
        self.image, self.mask = image.clone(), mask.clone()
        # A deliberately different RGB prediction makes channel order and
        # unintended changes outside the mask observable.
        output = torch.zeros_like(image)
        output[:, 0] = 1
        return output


def cleaner_with_recording_model():
    cleaner = LamaCleaner.__new__(LamaCleaner)
    cleaner.device = torch.device('cpu')
    cleaner.model = RecordingModel()
    return cleaner


def test_restoration_preserves_unmasked_pixels_and_converts_rgb():
    cleaner = cleaner_with_recording_model()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (10, 80, 230)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[35:65, 1800:1900] = 255
    result = cleaner.clean(frame, mask)
    assert result.shape == frame.shape
    assert np.array_equal(result[mask == 0], frame[mask == 0])
    assert np.all(result[mask > 0] == (0, 0, 255))
    assert max(cleaner.model.image.shape[-2:]) <= MAX_INPAINT_SIZE
    assert cleaner.model.image.shape[-2] < 1080
    assert np.allclose(cleaner.model.image[0, :, 0, 0], np.array([230, 80, 10]) / 255)


@pytest.mark.parametrize('height,width', [(17, 19), (1080, 1920), (2160, 3840)])
def test_large_masks_are_bounded_without_resizing_output(height, width):
    cleaner = cleaner_with_recording_model()
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    mask = np.full((height, width), 255, dtype=np.uint8)
    result = cleaner.clean(frame, mask)
    assert result.shape == frame.shape
    assert np.all(result == (0, 0, 255))
    assert max(cleaner.model.image.shape[-2:]) <= MAX_INPAINT_SIZE


def test_empty_mask_skips_model():
    cleaner = cleaner_with_recording_model()
    frame = np.full((40, 80, 3), 123, dtype=np.uint8)
    result = cleaner.clean(frame, np.zeros((40, 80), dtype=np.uint8))
    assert np.array_equal(result, frame)
    assert cleaner.model.image is None


def test_mismatched_mask_is_rejected():
    cleaner = cleaner_with_recording_model()
    with pytest.raises(ValueError, match='dimensions'):
        cleaner.clean(np.zeros((20, 20, 3), np.uint8), np.zeros((5, 5), np.uint8))


def test_manual_regions_scale_to_the_original_video():
    from demark_world.core import region_boxes
    assert region_boxes([[.75, .05, .95, .15]], 1280, 720) == [(960, 36, 1216, 108)]


@pytest.mark.parametrize('regions', [[], [[0, 0, 0, 1]], [[0, 0, 1.1, 1]], [[0, 0, float('nan'), 1]], [[1, 0, 0, 1]]])
def test_invalid_manual_regions_are_rejected(regions):
    from demark_world.core import region_boxes
    with pytest.raises(ValueError):
        region_boxes(regions, 1280, 720)


def test_manual_mode_never_loads_a_watermark_detector(monkeypatch):
    import demark_world.core as core
    def fail():
        raise AssertionError('manual restoration must not depend on a trained logo detector')
    monkeypatch.setattr(core, 'DeMarkWorldDetector', fail)
    engine = core.DeMarkWorld(automatic_detection=False, cleaner=object())
    assert engine.detector is None


def test_manual_regions_restore_both_logos_without_repainting_caption(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace
    import demark_world.core as core
    frame = np.full((120, 200, 3), 50, dtype=np.uint8)
    class RecordingCleaner:
        def clean(self, image, mask):
            result = image.copy()
            result[mask > 0] = 200
            return result
    monkeypatch.setattr(core, 'VideoLoader', lambda _: [frame])
    sink = SimpleNamespace(stdin=io.BytesIO())
    engine = core.DeMarkWorld(automatic_detection=False, cleaner=RecordingCleaner())
    engine._clean_lama(tmp_path/'input.mp4', [None], 200, 120, 1, sink, None, True,
                       manual_boxes=[(20, 20, 40, 40), (160, 20, 180, 40)])
    result = np.frombuffer(sink.stdin.getvalue(), np.uint8).reshape(frame.shape)
    assert np.all(result[25:35, 25:35] == 200)
    assert np.all(result[25:35, 165:175] == 200)
    assert np.all(result[80:110, 60:140] == 50)
