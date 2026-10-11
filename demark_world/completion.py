"""Identity-pinned, recoverable completion handling for cleaned videos."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol

from loguru import logger

from demark_world.configs import LOGS_PATH, RUNTIME_ROOT, WORKING_DIR


FFMPEG_POSTPROCESS_TIMEOUT_SECONDS = 3600


class CompletionError(RuntimeError):
    """Base class for completion safety failures."""


class OutputVerificationError(CompletionError):
    """The cleaned output could not be proven fully decodable."""


class SourceIdentityError(CompletionError):
    """The selected source path no longer names the selected file."""


class TrashMoveError(CompletionError):
    """The Trash helper did not return a confirmed success receipt."""


class TrashLocationUnknown(CompletionError):
    """Foundation committed a Trash operation but returned no usable location."""


class JobStateError(CompletionError):
    """A durable per-job state transition could not be recorded safely."""


class CommandRunner(Protocol):
    def __call__(self, *args, **kwargs): ...


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    device: int
    inode: int
    file_type: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> SourceIdentity:
        normalized = Path(os.path.abspath(os.path.expanduser(str(path))))
        try:
            details = os.lstat(normalized)
        except OSError as exc:
            raise SourceIdentityError(
                f"Selected source is unavailable: {normalized}"
            ) from exc
        if not stat.S_ISREG(details.st_mode):
            kind = "symbolic link" if stat.S_ISLNK(details.st_mode) else "non-file item"
            raise SourceIdentityError(
                f"Selected source is a {kind}, not a regular video: {normalized}"
            )
        return cls(
            path=str(normalized),
            device=int(details.st_dev),
            inode=int(details.st_ino),
            file_type=int(stat.S_IFMT(details.st_mode)),
            size=int(details.st_size),
            mtime_ns=int(details.st_mtime_ns),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SourceIdentity:
        try:
            return cls(
                path=str(value["path"]),
                device=int(value["device"]),
                inode=int(value["inode"]),
                file_type=int(value["file_type"]),
                size=int(value["size"]),
                mtime_ns=int(value["mtime_ns"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceIdentityError("Selection manifest has an invalid file identity.") from exc

    @property
    def source_path(self) -> Path:
        return Path(self.path)


@dataclass(frozen=True)
class TrashReceipt:
    status: str
    source: str
    trashed: str | None
    device: int
    inode: int


@dataclass(frozen=True)
class CompletionResult:
    source: str
    output: str
    trashed: str
    completed_at: str


def validate_source_identity(identity: SourceIdentity) -> None:
    """Require that a path still names the exact selected regular file."""

    current = SourceIdentity.capture(identity.source_path)
    if current != identity:
        raise SourceIdentityError(
            "The selected source changed after it was queued. "
            "Nothing at that path was moved to Trash."
        )


def ensure_safe_destination(source: Path, output: Path) -> None:
    """Refuse collisions before the processing engine touches either path."""

    source = Path(os.path.abspath(os.path.expanduser(str(source))))
    output = Path(os.path.abspath(os.path.expanduser(str(output))))
    if source == output:
        raise FileExistsError("The cleaned output cannot replace the original video.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)


def verify_cleaned_video(
    output: Path,
    *,
    source: Path | None = None,
    runner: CommandRunner = subprocess.run,
    ffmpeg_executable: str | None = None,
) -> None:
    """Decode every output video/audio packet with FFmpeg ``-xerror``."""

    # Keep the lexical output path. Resolving here would follow a symlink that
    # appeared after processing and could accidentally validate another file.
    output = Path(os.path.abspath(os.path.expanduser(str(output))))
    try:
        details = os.lstat(output)
    except OSError as exc:
        raise OutputVerificationError("The cleaned output is missing.") from exc
    if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
        raise OutputVerificationError("The cleaned output is not a non-empty regular file.")

    if source is not None:
        normalized_source = Path(os.path.abspath(os.path.expanduser(str(source))))
        if normalized_source == output:
            raise OutputVerificationError(
                "The cleaned output points to the selected source."
            )

    ffmpeg = (
        ffmpeg_executable
        or os.environ.get("MAC_WATERMARK_FFMPEG")
        or shutil.which("ffmpeg")
    )
    if not ffmpeg:
        raise OutputVerificationError(
            "The bundled full-file video verifier is unavailable."
        )

    try:
        completed = runner(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-i",
                str(output),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=FFMPEG_POSTPROCESS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OutputVerificationError(
            f"Could not decode the complete cleaned output: {exc}"
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 400:
            detail = f"{detail[:400]}…"
        suffix = f" ({detail})" if detail else ""
        raise OutputVerificationError(
            f"The complete cleaned output did not decode successfully{suffix}."
        )

    try:
        current = os.lstat(output)
    except OSError as exc:
        raise OutputVerificationError(
            "The cleaned output disappeared during full-file validation."
        ) from exc
    original_identity = (
        int(details.st_dev),
        int(details.st_ino),
        int(stat.S_IFMT(details.st_mode)),
        int(details.st_size),
        int(details.st_mtime_ns),
    )
    current_identity = (
        int(current.st_dev),
        int(current.st_ino),
        int(stat.S_IFMT(current.st_mode)),
        int(current.st_size),
        int(current.st_mtime_ns),
    )
    if current_identity != original_identity:
        raise OutputVerificationError(
            "The cleaned output changed during full-file validation."
        )


def move_to_macos_trash(
    source: Path,
    identity: SourceIdentity,
    *,
    runner: CommandRunner = subprocess.run,
    helper: str | None = None,
) -> TrashReceipt:
    """Ask the identity-checking Foundation helper to commit a Trash move."""

    source = Path(os.path.abspath(os.path.expanduser(str(source))))
    if source != identity.source_path:
        raise SourceIdentityError("Trash request does not match the queued source path.")
    validate_source_identity(identity)

    helper_path = helper or os.environ.get("MAC_WATERMARK_TRASH_HELPER")
    if not helper_path or not Path(helper_path).is_file():
        raise TrashMoveError("The recoverable macOS Trash helper is unavailable.")

    command = [
        helper_path,
        str(source),
        str(identity.device),
        str(identity.inode),
        str(identity.file_type),
        str(identity.size),
        str(identity.mtime_ns),
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrashMoveError(
            "The Trash operation was not confirmed. Check the source and Trash."
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise TrashMoveError(
            "The Trash helper did not confirm a commit"
            + (f": {detail}" if detail else ".")
        )

    try:
        payload = json.loads(completed.stdout)
        receipt = TrashReceipt(
            status=str(payload["status"]),
            source=str(payload["source"]),
            trashed=(
                str(payload["trashed"])
                if isinstance(payload.get("trashed"), str)
                else None
            ),
            device=int(payload["device"]),
            inode=int(payload["inode"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TrashMoveError(
            "The Trash helper returned no valid commit receipt. "
            "Check the source and Trash."
        ) from exc

    if (
        receipt.source != str(source)
        or receipt.device != identity.device
        or receipt.inode != identity.inode
    ):
        raise TrashMoveError("The Trash receipt does not match the queued source.")
    if receipt.status == "location_unknown":
        return receipt
    if (
        receipt.status != "trashed"
        or not receipt.trashed
        or not Path(receipt.trashed).is_absolute()
    ):
        raise TrashMoveError("The Trash helper returned an unsupported receipt.")
    return receipt


def _append_audit(event: dict, audit_path: Path) -> bool:
    """Append an audit event without ever changing the operation outcome."""

    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return True
    except Exception as exc:
        logger.error("Could not append cleanup audit {}: {}", audit_path, exc)
        return False


def finalize_cleaned_video(
    identity: SourceIdentity,
    output: Path,
    *,
    verifier: Callable[..., None] = verify_cleaned_video,
    trash: Callable[[Path, SourceIdentity], TrashReceipt] = move_to_macos_trash,
    audit_path: Path | None = None,
    workspace: Path | None = None,
) -> CompletionResult:
    """Verify output, revalidate identity, then commit source to Trash."""

    source = identity.source_path
    output = Path(os.path.abspath(os.path.expanduser(str(output))))
    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    started_at = datetime.now(UTC).isoformat()

    try:
        if workspace is not None:
            mark_job_phase(workspace, "verifying")
            validate_recorded_output_identity(workspace, output)
        verifier(output, source=source)
        if workspace is not None:
            validate_recorded_output_identity(workspace, output)
        validate_source_identity(identity)
    except Exception as exc:
        cleanup = _quarantine_failed_output(
            output,
            workspace=workspace,
            audit_path=audit_path,
        )
        _append_audit(
            {
                "event": (
                    "source_identity_changed"
                    if isinstance(exc, SourceIdentityError)
                    else "verification_failed"
                ),
                "source": str(source),
                "source_identity": asdict(identity),
                "output": str(output),
                "at": started_at,
                "error": str(exc),
                "failed_output_cleanup": cleanup,
            },
            audit_path,
        )
        raise

    try:
        if workspace is not None:
            mark_job_phase(workspace, "trash_requested")
        receipt = trash(source, identity)
    except Exception as exc:
        if workspace is not None:
            record_job_phase_best_effort(
                workspace,
                "trash_unconfirmed",
                audit_path=audit_path,
                error=str(exc),
            )
        try:
            validate_source_identity(identity)
        except SourceIdentityError:
            cleanup = "source_state_ambiguous_output_preserved"
        else:
            cleanup = _quarantine_failed_output(
                output,
                workspace=workspace,
                audit_path=audit_path,
            )
        _append_audit(
            {
                "event": "trash_unconfirmed",
                "source": str(source),
                "source_identity": asdict(identity),
                "output": str(output),
                "at": started_at,
                "error": str(exc),
                "failed_output_cleanup": cleanup,
            },
            audit_path,
        )
        raise

    if receipt.status == "location_unknown":
        if workspace is not None:
            record_job_phase_best_effort(
                workspace,
                "trash_location_unknown",
                audit_path=audit_path,
            )
        event = {
            "event": "trash_location_unknown",
            "source": str(source),
            "source_identity": asdict(identity),
            "output": str(output),
            "at": started_at,
        }
        _append_audit(event, audit_path)
        raise TrashLocationUnknown(
            "macOS committed a Trash operation but did not provide its location. "
            "The job was not marked complete; inspect the source path and Trash."
        )
    if (
        receipt.status != "trashed"
        or receipt.source != str(source)
        or receipt.device != identity.device
        or receipt.inode != identity.inode
        or receipt.trashed is None
        or not Path(receipt.trashed).is_absolute()
    ):
        _append_audit(
            {
                "event": "trash_receipt_invalid",
                "source": str(source),
                "source_identity": asdict(identity),
                "output": str(output),
                "at": started_at,
            },
            audit_path,
        )
        raise TrashMoveError(
            "The Trash operation returned an invalid receipt. "
            "Inspect both the source path and Trash."
        )

    # Foundation's valid receipt is the commit point. Everything below is
    # deliberately best-effort and cannot reverse or misreport that outcome.
    result = CompletionResult(
        source=str(source),
        output=str(output),
        trashed=receipt.trashed,
        completed_at=datetime.now(UTC).isoformat(),
    )
    if workspace is not None:
        record_job_phase_best_effort(
            workspace,
            "trash_committed",
            audit_path=audit_path,
            trash_receipt=asdict(result),
        )
    _append_audit(
        {"event": "completed_and_trashed", **asdict(result)},
        audit_path,
    )
    return result


def run_verified_cleaning_job(
    identity: SourceIdentity,
    output: Path,
    cleaner: Callable[[Path, Path], None],
    *,
    verifier: Callable[..., None] = verify_cleaned_video,
    trash: Callable[[Path, SourceIdentity], TrashReceipt] = move_to_macos_trash,
    audit_path: Path | None = None,
    workspace: Path | None = None,
) -> CompletionResult:
    """Run a cleaner while pinning source identity across the whole job."""

    source = identity.source_path
    output = Path(os.path.abspath(os.path.expanduser(str(output))))
    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    validate_source_identity(identity)
    try:
        ensure_safe_destination(source, output)
    except Exception as exc:
        _append_audit(
            {
                "event": "collision_refused",
                "source": str(source),
                "source_identity": asdict(identity),
                "output": str(output),
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        raise

    try:
        cleaner(source, output)
    except Exception as exc:
        cleanup = _quarantine_failed_output(
            output,
            workspace=workspace,
            audit_path=audit_path,
        )
        _append_audit(
            {
                "event": "processing_failed",
                "source": str(source),
                "source_identity": asdict(identity),
                "output": str(output),
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
                "failed_output_cleanup": cleanup,
            },
            audit_path,
        )
        raise

    return finalize_cleaned_video(
        identity,
        output,
        verifier=verifier,
        trash=trash,
        audit_path=audit_path,
        workspace=workspace,
    )


def _unique_quarantine_path(quarantine_root: Path, name: str) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    candidate = quarantine_root / f"{timestamp}-{name}"
    counter = 2
    while candidate.exists():
        candidate = quarantine_root / f"{timestamp}-{counter}-{name}"
        counter += 1
    return candidate


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _job_manifest_path(workspace: Path) -> Path:
    workspace = _normalized_path(workspace)
    try:
        details = os.lstat(workspace)
    except OSError as exc:
        raise JobStateError(f"Job workspace is unavailable: {workspace}") from exc
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise JobStateError(f"Job workspace is not a real directory: {workspace}")
    return workspace / "job.json"


def _read_job_manifest(workspace: Path) -> dict[str, object]:
    manifest_path = _job_manifest_path(workspace)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobStateError(f"Job manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise JobStateError(f"Job manifest has an unsupported format: {manifest_path}")
    return payload


def _write_job_manifest(workspace: Path, payload: Mapping[str, object]) -> None:
    """Durably replace a job manifest before the next destructive phase."""

    manifest_path = _job_manifest_path(workspace)
    temporary = manifest_path.with_name(f".job-{uuid.uuid4().hex}.tmp")
    try:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
        directory_fd = os.open(
            manifest_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise JobStateError(f"Could not persist job phase: {manifest_path}") from exc


def mark_job_phase(
    workspace: Path,
    phase: str,
    **fields: object,
) -> None:
    payload = _read_job_manifest(workspace)
    payload["phase"] = phase
    payload["phase_updated_at"] = datetime.now(UTC).isoformat()
    payload.update(fields)
    _write_job_manifest(workspace, payload)


def record_job_phase_best_effort(
    workspace: Path,
    phase: str,
    *,
    audit_path: Path,
    **fields: object,
) -> bool:
    try:
        mark_job_phase(workspace, phase, **fields)
        return True
    except Exception as exc:
        _append_audit(
            {
                "event": "job_phase_record_failed",
                "workspace": str(workspace),
                "phase": phase,
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        return False


def prepare_job_publication(
    workspace: Path,
    partial_output: Path,
    visible_output: Path,
) -> SourceIdentity:
    """Record the exact inode before core hard-links it into the output folder."""

    partial_identity = SourceIdentity.capture(_normalized_path(partial_output))
    visible_output = _normalized_path(visible_output)
    expected_output = SourceIdentity(
        path=str(visible_output),
        device=partial_identity.device,
        inode=partial_identity.inode,
        file_type=partial_identity.file_type,
        size=partial_identity.size,
        mtime_ns=partial_identity.mtime_ns,
    )
    payload = _read_job_manifest(workspace)
    try:
        manifest_output = _normalized_path(Path(str(payload["output"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise JobStateError("Job manifest has no valid output path.") from exc
    if manifest_output != visible_output:
        raise JobStateError("Core publication does not match the job output path.")
    payload["phase"] = "publish_prepared"
    payload["phase_updated_at"] = datetime.now(UTC).isoformat()
    payload["published_output"] = asdict(expected_output)
    _write_job_manifest(workspace, payload)
    return expected_output


def validate_recorded_output_identity(
    workspace: Path,
    visible_output: Path,
) -> SourceIdentity:
    """Require the visible file to be the inode core prepared to publish."""

    try:
        payload = _read_job_manifest(workspace)
        raw_identity = payload.get("published_output")
        if not isinstance(raw_identity, Mapping):
            raise JobStateError("Job has no published output identity.")
        expected = SourceIdentity.from_mapping(raw_identity)
        visible_output = _normalized_path(visible_output)
        if expected.source_path != visible_output:
            raise JobStateError("Published output path does not match this job.")
        current = SourceIdentity.capture(visible_output)
    except (JobStateError, SourceIdentityError) as exc:
        raise OutputVerificationError(
            "The visible cleaned output is not the file this job published."
        ) from exc
    if current != expected:
        raise OutputVerificationError(
            "The visible cleaned output changed after publication."
        )
    return expected


def quarantine_recorded_output(
    workspace: Path,
    *,
    quarantine_root: Path | None = None,
    audit_path: Path | None = None,
    reason: str = "failed_job",
) -> str:
    """Quarantine only the exact output inode recorded before publication."""

    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    quarantine_root = quarantine_root or (RUNTIME_ROOT / "Failed Jobs")
    try:
        payload = _read_job_manifest(workspace)
        raw_identity = payload.get("published_output")
        if not isinstance(raw_identity, Mapping):
            return "no_published_output_identity"
        expected = SourceIdentity.from_mapping(raw_identity)
        manifest_output = _normalized_path(Path(str(payload["output"])))
        if expected.source_path != manifest_output:
            raise JobStateError(
                "Published output identity does not match the manifest output path."
            )
        current = SourceIdentity.capture(expected.source_path)
        if current != expected:
            _append_audit(
                {
                    "event": "recorded_output_identity_mismatch",
                    "workspace": str(workspace),
                    "output": str(expected.source_path),
                    "reason": reason,
                    "at": datetime.now(UTC).isoformat(),
                },
                audit_path,
            )
            return "identity_mismatch_output_preserved"
        destination = _unique_quarantine_path(
            quarantine_root,
            f"output-{expected.source_path.name}",
        )
        os.replace(expected.source_path, destination)
    except SourceIdentityError:
        return "recorded_output_missing_or_not_regular"
    except Exception as exc:
        _append_audit(
            {
                "event": "failed_output_quarantine_failed",
                "workspace": str(workspace),
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        return "quarantine_failed"
    _append_audit(
        {
            "event": "failed_output_quarantined",
            "workspace": str(workspace),
            "output": str(expected.source_path),
            "quarantined_to": str(destination),
            "reason": reason,
            "at": datetime.now(UTC).isoformat(),
        },
        audit_path,
    )
    return f"quarantined:{destination}"


def _quarantine_failed_output(
    output: Path,
    *,
    workspace: Path | None,
    audit_path: Path,
) -> str:
    """Move an invalid output only when this job durably pinned its identity."""

    if workspace is None:
        return "output_not_identity_pinned"
    try:
        payload = _read_job_manifest(workspace)
        if _normalized_path(Path(str(payload["output"]))) != _normalized_path(output):
            return "output_path_mismatch_preserved"
    except Exception:
        return "job_manifest_unreadable_output_preserved"
    return quarantine_recorded_output(
        workspace,
        audit_path=audit_path,
        reason="live_job_failure",
    )


def create_job_workspace(
    identity: SourceIdentity,
    output: Path,
    *,
    working_root: Path = WORKING_DIR,
) -> Path:
    """Create one private runtime directory for all transient job artifacts."""

    working_root.mkdir(parents=True, exist_ok=True)
    workspace = working_root / f"job-{uuid.uuid4()}"
    workspace.mkdir(mode=0o700)
    manifest = {
        "version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "phase": "created",
        "source": asdict(identity),
        "output": str(Path(os.path.abspath(os.path.expanduser(str(output))))),
        "published_output": None,
    }
    _write_job_manifest(workspace, manifest)
    return workspace


def quarantine_job_workspace(
    workspace: Path,
    *,
    quarantine_root: Path | None = None,
    audit_path: Path | None = None,
) -> Path | None:
    """Move a failed or stale workspace intact into app-owned quarantine."""

    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    quarantine_root = quarantine_root or (RUNTIME_ROOT / "Failed Jobs")
    try:
        if not workspace.exists() and not workspace.is_symlink():
            return None
        destination = _unique_quarantine_path(quarantine_root, workspace.name)
        os.replace(workspace, destination)
    except Exception as exc:
        _append_audit(
            {
                "event": "job_workspace_quarantine_failed",
                "workspace": str(workspace),
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        return None
    _append_audit(
        {
            "event": "job_workspace_quarantined",
            "workspace": str(workspace),
            "quarantined_to": str(destination),
            "at": datetime.now(UTC).isoformat(),
        },
        audit_path,
    )
    return destination


def complete_job_workspace(
    workspace: Path,
    *,
    audit_path: Path | None = None,
) -> bool:
    """Remove an empty successful workspace; quarantine any leftover partials."""

    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    try:
        if not workspace.exists():
            return True
        leftovers = [path for path in workspace.iterdir() if path.name != "job.json"]
        if leftovers:
            return quarantine_job_workspace(
                workspace,
                audit_path=audit_path,
            ) is not None
        (workspace / "job.json").unlink(missing_ok=True)
        workspace.rmdir()
        return True
    except Exception:
        return quarantine_job_workspace(
            workspace,
            audit_path=audit_path,
        ) is not None


def reconcile_stale_workspace(
    workspace: Path,
    *,
    quarantine_root: Path | None = None,
    audit_path: Path | None = None,
) -> str:
    """Resolve a crash window without guessing whether Trash committed."""

    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    quarantine_root = quarantine_root or (RUNTIME_ROOT / "Failed Jobs")
    try:
        payload = _read_job_manifest(workspace)
        phase = str(payload.get("phase", "unknown"))
        if phase in {"trash_committed", "trash_location_unknown"}:
            _append_audit(
                {
                    "event": "stale_output_preserved_after_trash",
                    "workspace": str(workspace),
                    "phase": phase,
                    "at": datetime.now(UTC).isoformat(),
                },
                audit_path,
            )
            return "trash_confirmed_or_location_unknown_output_preserved"
        if not isinstance(payload.get("published_output"), Mapping):
            return "no_published_output"
        raw_source = payload.get("source")
        if not isinstance(raw_source, Mapping):
            raise JobStateError("Stale job has no valid source identity.")
        source_identity = SourceIdentity.from_mapping(raw_source)
        try:
            validate_source_identity(source_identity)
        except SourceIdentityError:
            _append_audit(
                {
                    "event": "stale_output_preserved_ambiguous_trash_state",
                    "workspace": str(workspace),
                    "phase": phase,
                    "source": source_identity.path,
                    "at": datetime.now(UTC).isoformat(),
                },
                audit_path,
            )
            return "source_identity_missing_or_changed_output_preserved"
        return quarantine_recorded_output(
            workspace,
            quarantine_root=quarantine_root,
            audit_path=audit_path,
            reason=f"stale_job_{phase}",
        )
    except Exception as exc:
        _append_audit(
            {
                "event": "stale_job_reconciliation_failed_output_preserved",
                "workspace": str(workspace),
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        return "reconciliation_failed_output_preserved"


def recover_stale_workspaces(
    *,
    working_root: Path = WORKING_DIR,
    quarantine_root: Path | None = None,
    audit_path: Path | None = None,
) -> list[Path]:
    """Quarantine every prior-run workspace before accepting new jobs."""

    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    quarantine_root = quarantine_root or (RUNTIME_ROOT / "Failed Jobs")
    recovered: list[Path] = []
    try:
        working_root.mkdir(parents=True, exist_ok=True)
        workspaces = sorted(working_root.iterdir())
    except Exception as exc:
        _append_audit(
            {
                "event": "stale_workspace_scan_failed",
                "working_root": str(working_root),
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        return recovered
    for workspace in workspaces:
        reconcile_stale_workspace(
            workspace,
            quarantine_root=quarantine_root,
            audit_path=audit_path,
        )
        destination = quarantine_job_workspace(
            workspace,
            quarantine_root=quarantine_root,
            audit_path=audit_path,
        )
        if destination is not None:
            recovered.append(destination)
    return recovered


def consume_selection_manifest(
    manifest_path: Path,
    identity: SourceIdentity,
    *,
    audit_path: Path | None = None,
) -> bool:
    """Atomically remove a successfully committed identity from the queue."""

    audit_path = audit_path or (LOGS_PATH / "cleanup-audit.jsonl")
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    temporary = manifest_path.with_suffix(
        manifest_path.suffix + f".{uuid.uuid4().hex}.tmp"
    )
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            sources = payload.get("sources", [])
            if not isinstance(sources, list):
                raise ValueError("Selection manifest sources must be a list.")
            retained: list[object] = []
            for value in sources:
                try:
                    queued_identity = SourceIdentity.from_mapping(value)
                except SourceIdentityError:
                    # Preserve unrelated malformed records for diagnosis. One
                    # bad entry must not keep a committed source in the queue.
                    retained.append(value)
                    continue
                if queued_identity != identity:
                    retained.append(value)
            payload["sources"] = retained
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, manifest_path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return True
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        _append_audit(
            {
                "event": "queue_consume_failed_after_trash",
                "source_identity": asdict(identity),
                "manifest": str(manifest_path),
                "at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            },
            audit_path,
        )
        return False
