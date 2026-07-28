"""Cleaner selection with optional temporal dependencies loaded lazily.

Changed from DeMark-World so selecting LaMa on macOS does not import MMCV or
the E2FGVI-HQ graph. This keeps the default Mac install usable while retaining
the upstream temporal backend for environments that install its extras.
"""

from demark_world.cleaner.lama_cleaner import LamaCleaner
from demark_world.schemas import CleanerType


class WaterMarkCleaner:
    def __new__(
        cls,
        cleaner_type: CleanerType,
        enable_torch_compile: bool = False,
        use_bf16: bool = False,
    ):
        if cleaner_type == CleanerType.LAMA:
            return LamaCleaner()
        if cleaner_type == CleanerType.E2FGVI_HQ:
            try:
                from demark_world.cleaner.e2fgvi_hq_cleaner import (
                    E2FGVIHDConfig,
                    E2FGVIHDCleaner,
                )
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "E2FGVI-HQ is an optional GPU backend and is not included "
                    "in the standard macOS install. Use LaMa on this Mac, or "
                    "install the temporal extras on a supported CUDA system."
                ) from exc
            config = E2FGVIHDConfig(
                enable_torch_compile=enable_torch_compile,
                use_bf16=use_bf16,
            )
            return E2FGVIHDCleaner(config=config)
        raise ValueError(f"Invalid cleaner type: {cleaner_type}")
