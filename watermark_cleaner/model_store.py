"""Verified, atomic downloads of the two local video model checkpoints.

No footage leaves this computer. Only these published checkpoint files are
fetched. A completed file is accepted only after its pinned SHA-256 matches.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    name: str
    url: str
    sha256: str
    size: int


PROPAINTER_RELEASE = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
MODEL_SPECS = (
    ModelSpec(
        "e2fgvi",
        "E2FGVI-HQ-CVPR22.pth",
        "https://huggingface.co/spaces/VIPLab/Track-Anything/resolve/bb879e543c98747057ec76529bcdb3232c22f7fd/checkpoints/E2FGVI-HQ-CVPR22.pth",
        "afff989d41205598a79ce24630b9c83af4b0a06f45b137979a25937d94c121a5",
        164535938,
    ),
    ModelSpec(
        "propainter",
        "raft-things.pth",
        PROPAINTER_RELEASE + "raft-things.pth",
        "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
        21108000,
    ),
    ModelSpec(
        "propainter",
        "recurrent_flow_completion.pth",
        PROPAINTER_RELEASE + "recurrent_flow_completion.pth",
        "22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283",
        20348681,
    ),
    ModelSpec(
        "propainter",
        "ProPainter.pth",
        PROPAINTER_RELEASE + "ProPainter.pth",
        "12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591",
        157780510,
    ),
)
_THREAD_LOCK = threading.RLock()
_VERIFIED: dict[tuple[str, int, int, int, int, int], bool] = {}


def model_directory() -> Path:
    from .configs import MODELS_DIR

    return Path(MODELS_DIR) / "temporal-v2"


def model_path(spec: ModelSpec) -> Path:
    return model_directory() / spec.engine / spec.name


def _valid(spec: ModelSpec, path: Path) -> bool:
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size != spec.size:
            return False
        key = (
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if key in _VERIFIED:
            return _VERIFIED[key]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        valid = digest.hexdigest() == spec.sha256
        _VERIFIED[key] = valid
        return valid
    except OSError:
        return False


def models_status() -> dict[str, bool]:
    """Report readiness only when every checkpoint for an engine is verified."""
    return {
        name: any(spec.engine == name for spec in MODEL_SPECS)
        and all(_valid(spec, model_path(spec)) for spec in MODEL_SPECS if spec.engine == name)
        for name in ("e2fgvi", "propainter")
    }


@contextmanager
def _download_lock(root: Path, progress: Progress | None) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    # flock releases automatically when a process exits or is interrupted.
    with _THREAD_LOCK, (root / ".download.lock").open("a+b") as lock:
        notified = False
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if progress and not notified:
                    progress(
                        0.0,
                        "Another app window is downloading the models. Waiting for it to finish.",
                    )
                    notified = True
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _download(
    spec: ModelSpec, destination: Path, progress: Progress | None, completed: int, total: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{spec.name}.", suffix=".partial", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            received = 0
            request = Request(spec.url, headers={"User-Agent": "MAC-WATERMARK-CLEANER/2.0"})
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > spec.size:
                        raise ValueError(f"{spec.name} download exceeded its expected size")
                    stream.write(chunk)
                    digest.update(chunk)
                    if progress:
                        progress(
                            min((completed + received) / total, 0.999),
                            f"Downloading {spec.name}: {received / 1048576:.1f} of {spec.size / 1048576:.1f} MB",
                        )
            stream.flush()
            os.fsync(stream.fileno())
        if received != spec.size or digest.hexdigest() != spec.sha256:
            raise ValueError(f"{spec.name} did not match its published checkpoint checksum")
        os.replace(temporary, destination)
        temporary = None
        if not _valid(spec, destination):
            raise ValueError(f"{spec.name} failed verification after saving")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_models(
    engine_id: str | None = None, progress: Progress | None = None
) -> dict[str, Path]:
    """Download both model sets on first launch, or verify one selected engine.

    Interrupted downloads never replace a working checkpoint. Re-running this
    function reuses verified files and retries incomplete downloads.
    """
    if engine_id is not None and engine_id not in {"e2fgvi", "propainter"}:
        raise ValueError(f"Unknown video engine: {engine_id}")
    specs = [spec for spec in MODEL_SPECS if engine_id is None or spec.engine == engine_id]
    total = sum(spec.size for spec in specs)
    completed = 0
    ready: dict[str, Path] = {}
    with _download_lock(model_directory(), progress):
        for spec in specs:
            path = model_path(spec)
            if not _valid(spec, path):
                if progress:
                    progress(completed / total, f"Preparing {spec.name} download")
                path.parent.mkdir(parents=True, exist_ok=True)
                for orphan in path.parent.glob(f".{spec.name}.*.partial"):
                    orphan.unlink(missing_ok=True)
                try:
                    _download(spec, path, progress, completed, total)
                except (OSError, ValueError, HTTPError, URLError, HTTPException) as error:
                    raise RuntimeError(
                        f"Could not prepare {spec.name}. Check your internet connection and available disk space, "
                        f"then choose Retry. Verified model files are kept. Details: {error}"
                    ) from error
            ready[spec.name] = path
            completed += spec.size
            if progress:
                progress(completed / total, f"{spec.name} is ready")
    return ready
