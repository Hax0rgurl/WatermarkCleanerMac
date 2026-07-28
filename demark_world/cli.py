"""Installed command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from demark_world.configs import OUTPUT_DIR
from demark_world.core import DeMarkWorld, unique_mp4_path
from demark_world.schemas import CleanerType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove AI-video watermarks without altering originals."
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model",
        choices=[item.value for item in CleanerType],
        default=CleanerType.LAMA.value,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = DeMarkWorld(
        cleaner_type=CleanerType(args.model),
        detect_batch_size=args.batch_size,
        enable_torch_compile=False,
    )
    if args.input.is_dir():
        engine.run_batch(args.input, args.output, quiet=args.quiet)
        return

    output = args.output
    if output.suffix == "":
        output.mkdir(parents=True, exist_ok=True)
        output = unique_mp4_path(output, args.input.stem)
    elif output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    engine.run(args.input, output, quiet=args.quiet)
    print(output)


if __name__ == "__main__":
    main()
