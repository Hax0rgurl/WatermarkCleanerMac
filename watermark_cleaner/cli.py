"""Command-line access to the same verified local restoration pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(
        description=(
            "Restore local video watermarks with E2FGVI-HQ or ProPainter. "
            "Originals are kept and existing output files are never overwritten."
        )
    )
    arguments.add_argument("--version", action="version", version=__version__)
    arguments.add_argument("-i", "--input", required=True, type=Path, help="Video file or folder")
    arguments.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="New MP4 file, or output folder for a batch",
    )
    arguments.add_argument("--engine", choices=["e2fgvi", "propainter"], default="e2fgvi")
    arguments.add_argument(
        "--watermark-text",
        default="",
        help="Comma-separated branding names; omit to scan the built-in platform-name list",
    )
    arguments.add_argument(
        "--regions",
        type=Path,
        help="Reviewed timed-region JSON for one video; skips automatic scanning",
    )
    arguments.add_argument("--quiet", action="store_true", help="Hide processing progress")
    return arguments


def _read_regions(path: Path | None) -> list | None:
    if path is None:
        return None
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("regions")
    if not isinstance(data, list) or not data:
        raise ValueError("The regions file must contain a nonempty JSON list of timed boxes.")
    return data


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    options = arguments.parse_args(argv)
    source, destination = options.input.expanduser(), options.output.expanduser()
    try:
        if source.is_dir():
            if options.regions is not None:
                raise ValueError(
                    "Use --regions with one video, since each video has its own timeline."
                )
            if destination.exists() and not destination.is_dir():
                raise ValueError("A folder input needs an output folder.")
            sources = sorted(
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not sources:
                raise ValueError(f"No supported videos were found in {source}.")
            batch = True
        else:
            if not source.is_file():
                raise FileNotFoundError(f"The input video does not exist: {source}")
            if destination.suffix.lower() != ".mp4":
                raise ValueError("A single video needs a new .mp4 output path.")
            if source.resolve() == destination.resolve():
                raise ValueError("The source and cleaned output must be different files.")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
            sources, batch = [source], False
        regions = _read_regions(options.regions)
        from .core import WatermarkCleaner, unique_mp4_path

        cleaner = WatermarkCleaner(options.engine)
        for video in sources:
            output = unique_mp4_path(destination, video.stem) if batch else destination
            previous = -1

            def progress(value: int) -> None:
                nonlocal previous
                if not options.quiet and value != previous:
                    print(f"{video.name}: {value}%", file=sys.stderr, flush=True)
                    previous = value

            cleaner.run(
                video,
                output,
                progress_callback=progress,
                watermark_regions=regions,
                watermark_text=options.watermark_text,
            )
            print(f"Completed: {output}", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Original videos were kept.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
