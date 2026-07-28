from pathlib import Path

from demark_world import CleanerType, DeMarkWorld

engine = DeMarkWorld(cleaner_type=CleanerType.LAMA)
engine.run(Path("input.mp4"), Path("output-cleaned.mp4"))
