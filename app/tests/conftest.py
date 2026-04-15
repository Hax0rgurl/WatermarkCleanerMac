import tempfile
from pathlib import Path

import numpy as np
import pytest

from sorawm.schemas import CleanerType


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a simple RGB image (100x100)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a mock video file path for testing."""
    return temp_dir / "test_video.mp4"


@pytest.fixture
def sample_mask():
    """Create a sample mask for testing."""
    # Create a binary mask (100x100)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # White square in center
    return mask


@pytest.fixture
def mock_video_frames():
    """Create mock video frames for testing."""
    # Create 10 frames of 100x100 RGB images
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def mock_video_masks():
    """Create mock video masks for testing."""
    # Create 10 masks of 100x100
    masks = []
    for i in range(10):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Add some variation to masks
        mask[10 + i : 50 + i, 10 + i : 50 + i] = 255
        masks.append(mask)
    return masks


@pytest.fixture
def cleaner_types():
    """Provide all available cleaner types."""
    return [CleanerType.LAMA, CleanerType.E2FGVI_HQ]


@pytest.fixture
def detection_result_with_watermark():
    """Mock detection result with watermark found."""
    return {
        "detected": True,
        "bbox": (10, 10, 50, 50),
        "confidence": 0.95,
        "center": (30, 30),
    }


@pytest.fixture
def detection_result_no_watermark():
    """Mock detection result with no watermark."""
    return {
        "detected": False,
        "bbox": None,
        "confidence": None,
        "center": None,
    }
