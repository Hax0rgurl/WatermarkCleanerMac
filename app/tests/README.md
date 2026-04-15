# Sora Watermark Cleaner - Unit Tests

This directory contains unit tests for the Sora Watermark Cleaner project.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and common fixtures
├── test_basic.py           # Basic functionality tests that don't require heavy dependencies
├── unit/                   # Unit tests
│   ├── test_watermark_detector.py    # Tests for SoraWaterMarkDetector
│   ├── test_watermark_cleaner.py     # Tests for WaterMarkCleaner factory
│   ├── test_core.py                  # Tests for SoraWM core class
│   ├── test_imputation_utils.py      # Tests for imputation utilities
│   └── test_video_utils.py          # Tests for video utilities
└── integration/            # Integration tests (future)
```

## Running Tests

### Prerequisites

The project requires several dependencies. To run all tests, install them:

```bash
pip install -e .
```

### Running Basic Tests

The basic tests can run without heavy ML dependencies:

```bash
python tests/test_basic.py
```

### Running Full Test Suite

To run the full test suite with pytest:

```bash
pytest
```

Or run specific test files:

```bash
pytest tests/unit/test_watermark_detector.py -v
```

## Test Coverage

The test suite covers:

1. **SoraWaterMarkDetector**: Watermark detection using YOLO models
   - Initialization with proper model loading
   - Detection with and without watermarks
   - Tensor to Python type conversion
   - Multiple detection handling

2. **WaterMarkCleaner**: Factory pattern for cleaner implementations
   - LAMA cleaner instantiation
   - E2FGVI-HQ cleaner instantiation
   - Invalid type handling

3. **SoraWM Core**: Main processing pipeline
   - Initialization with different cleaner types
   - Batch processing of video directories
   - Single video processing with both cleaner types
   - Audio track merging
   - Progress callback handling

4. **Imputation Utils**: Data preprocessing utilities
   - Breakpoint detection for missing frames
   - Interval-based bbox averaging
   - Index interval mapping
   - Breakpoint refinement by chunk size

5. **Video Utils**: Video processing utilities
   - Frame overlap blending for smooth transitions
   - VideoLoader class for video reading
   - Frame slicing and iteration

## Mocking Strategy

Since this project uses heavy ML dependencies (PyTorch, ultralytics, etc.), the tests extensively use mocking:

- **YOLO models**: Mocked to return predefined detection results
- **FFmpeg processes**: Mocked for video processing
- **ML model inference**: Mocked to avoid GPU requirements
- **File I/O**: Mocked for testing without actual video files

## Fixtures

Common test fixtures are provided in `conftest.py`:

- `temp_dir`: Temporary directory for file operations
- `sample_image`: Sample numpy image for testing
- `sample_mask`: Sample binary mask
- `mock_video_frames`: List of mock video frames
- `cleaner_types`: All available cleaner types
- Detection result fixtures for different scenarios

## Notes

- Tests are designed to run in CI/CD environments without GPU access
- All external dependencies are mocked to ensure fast, reliable tests
- The test suite validates the core logic without requiring actual ML model weights
- Integration tests would require actual video files and model weights (future work)
