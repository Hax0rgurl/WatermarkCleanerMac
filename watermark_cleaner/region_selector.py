"""Optional visual review of automatically found local watermark ranges."""

from __future__ import annotations

import base64
from contextlib import closing
from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from watermark_cleaner.completion import validate_source_identity
from watermark_cleaner.configs import RUNTIME_ROOT
from watermark_cleaner.timed_regions import normalize_regions
from watermark_cleaner.video_stream import VideoPreviewServer

_select_regions = components.declare_component(
    "watermark_regions", path=str(Path(__file__).parent / "watermark_selector")
)


@st.cache_resource(show_spinner=False)
def _preview_server():
    return VideoPreviewServer()


@st.cache_data(show_spinner=False, max_entries=24)
def video_preview(path: str, mtime_ns: int):
    # The metadata probe decodes exact timestamps, including variable-rate clips.
    from watermark_cleaner.video import iter_frames, probe_video

    info = probe_video(Path(path))
    with closing(iter_frames(Path(path))) as frames:
        try:
            image = Image.fromarray(next(frames)[:, :, ::-1])
        except StopIteration as exc:
            raise ValueError("Could not read the video preview.") from exc
    image.thumbnail((1000, 560), Image.Resampling.LANCZOS)
    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=90)
    return {
        "image": "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode(),
        "duration": float(info.duration),
        "fps": float(info.fps),
        "total_frames": int(info.total_frames),
        "frame_times": list(map(float, info.frame_times)),
    }


def selection_key(identity) -> str:
    return f"region-v2-{identity.device}-{identity.inode}-{identity.mtime_ns}"


def selection_files(identity):
    digest = hashlib.sha256(
        (str(identity.source_path) + selection_key(identity)).encode()
    ).hexdigest()[:24]
    saved = RUNTIME_ROOT / "Selections" / f"v2-areas-{digest}.json"
    return saved, saved.with_suffix(".scan.json")


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _normalized(regions, total_frames: int) -> list[dict]:
    if not regions:
        return []
    normalized = normalize_regions(regions, total_frames)
    # Preserve scanner-specific mask information after validating standard fields.
    return [
        {**raw, **valid} if isinstance(raw, dict) else valid
        for raw, valid in zip(regions, normalized, strict=True)
    ]


def _target(value: str) -> str:
    return " ".join(value.casefold().split())


def _load_state(identity) -> str:
    key = selection_key(identity)
    if f"{key}-areas" not in st.session_state:
        saved, summary_file = selection_files(identity)
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary.get("schema") != 3:
                raise ValueError("Old selection format")
            regions = _normalized(
                json.loads(saved.read_text(encoding="utf-8")), int(summary["total_frames"])
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            regions, summary = [], {}
        st.session_state[f"{key}-areas"] = regions
        st.session_state[f"{key}-scan-result"] = summary
        st.session_state[f"{key}-edited"] = bool(summary.get("edited", False))
        st.session_state[f"{key}-revision"] = 0
    return key


def selected_regions(identity, watermark_text: str = "") -> list[dict] | None:
    """Return current reviewed ranges, or None to request an automatic scan."""
    key = _load_state(identity)
    summary = st.session_state.get(f"{key}-scan-result", {})
    if summary.get("schema") != 3 or summary.get("target") != _target(watermark_text):
        return None
    regions = st.session_state.get(f"{key}-areas", [])
    if st.session_state.get(f"{key}-edited") and not regions:
        raise ValueError(
            "No watermark ranges are selected. Find the times again or add a missed watermark."
        )
    if regions and (summary.get("timing_verified") or st.session_state.get(f"{key}-edited")):
        return regions
    return None


def find_watermark_regions(identity, *, watermark_text: str = "", progress=None) -> list[dict]:
    from watermark_cleaner.scanner import scan_video

    key = _load_state(identity)
    validate_source_identity(identity)
    result = scan_video(identity.source_path, watermark_text=watermark_text, progress=progress)
    validate_source_identity(identity)
    total = int(result["total_frames"])
    regions = _normalized(result.get("regions", []), total)
    summary = {
        "schema": 3,
        "target": _target(watermark_text),
        "total_frames": total,
        "timing_verified": bool(regions) and all(r.get("timing_verified") for r in regions),
        "regions_found": len(regions),
        "edited": False,
    }
    st.session_state[f"{key}-areas"] = regions
    st.session_state[f"{key}-scan-result"] = summary
    st.session_state[f"{key}-revision"] += 1
    st.session_state[f"{key}-edited"] = False
    saved, summary_file = selection_files(identity)
    _save(saved, regions)
    _save(summary_file, summary)
    return regions


def regions_for_processing(identity, *, watermark_text: str = "", progress=None) -> list[dict]:
    regions = selected_regions(identity, watermark_text)
    if regions is None:
        regions = find_watermark_regions(identity, watermark_text=watermark_text, progress=progress)
    if not regions:
        raise ValueError(
            "No matching watermark was confirmed. Enter its name or add an area on the player."
        )
    return regions


def select_watermark_regions(identity, *, watermark_text: str = "") -> list[dict]:
    key = _load_state(identity)
    try:
        validate_source_identity(identity)
        preview = video_preview(str(identity.source_path), identity.mtime_ns)
        video_url = _preview_server().register(identity)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return []
    if st.button("FIND WATERMARK TIMES", key=f"{key}-scan"):
        progress = st.progress(0.0, text="Checking the video for watermark appearances…")
        try:
            find_watermark_regions(
                identity,
                watermark_text=watermark_text,
                progress=lambda value, text: progress.progress(
                    max(0.0, min(1.0, float(value))), text=str(text)
                ),
            )
        except Exception as exc:
            st.error(str(exc))
        finally:
            progress.empty()
    summary = st.session_state.get(f"{key}-scan-result", {})
    target_matches = summary.get("target") == _target(watermark_text)
    if summary.get("schema") == 3 and target_matches:
        count = len(st.session_state[f"{key}-areas"])
        if count:
            st.caption(
                f"{count} watermark ranges found. Play or scrub to check them; correct a miss only if needed."
            )
        else:
            st.info(
                "No matching branding was found. Enter its name in Watermark options or add an area below."
            )
    elif summary:
        st.info("The watermark name changed. Find the times again to update these ranges.")
    previous = st.session_state[f"{key}-areas"] if target_matches else []
    revision = st.session_state[f"{key}-revision"]
    value = _select_regions(
        image=preview["image"],
        video_url=video_url,
        regions=previous,
        source_key=key + "-" + _target(watermark_text),
        revision=revision,
        fps=preview["fps"],
        total_frames=preview["total_frames"],
        frame_times=preview["frame_times"],
        duration=preview["duration"],
        key=key,
        default={"revision": revision, "regions": previous},
    )
    incoming = (
        value.get("regions", previous)
        if isinstance(value, dict) and value.get("revision") == revision
        else previous
    )
    try:
        regions = _normalized(incoming, preview["total_frames"])
    except (ValueError, TypeError):
        st.error("A correction has an invalid area or time range.")
        return previous
    if regions != previous:
        st.session_state[f"{key}-edited"] = True
        summary = {
            "schema": 3,
            "target": _target(watermark_text),
            "total_frames": preview["total_frames"],
            "timing_verified": False,
            "regions_found": len(regions),
            "edited": True,
        }
        st.session_state[f"{key}-scan-result"] = summary
        st.session_state[f"{key}-areas"] = regions
        saved, summary_file = selection_files(identity)
        try:
            _save(saved, regions)
            _save(summary_file, summary)
        except OSError:
            st.warning("These corrections could not be saved for the next session.")
    return regions
