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


def _target(value: str) -> str:
    return " ".join(value.casefold().split())


def selection_key(identity, watermark_text: str = "", watermark_style: str = "auto") -> str:
    key = f"region-v2-{identity.device}-{identity.inode}-{identity.mtime_ns}"
    if watermark_style != "auto" or _target(watermark_text):
        scope = json.dumps([watermark_style, _target(watermark_text)])
        key += "-" + hashlib.sha256(scope.encode()).hexdigest()[:16]
    return key


def selection_files(
    identity, watermark_text: str | None = None, watermark_style: str | None = None
):
    # Keep the prior source-only location readable. New scoped files never
    # overwrite legacy manual work saved under a different watermark name.
    legacy = watermark_text is None and watermark_style is None
    watermark_text, watermark_style = watermark_text or "", watermark_style or "auto"
    digest = hashlib.sha256(
        (
            str(identity.source_path) + selection_key(identity, watermark_text, watermark_style)
        ).encode()
    ).hexdigest()[:24]
    saved = RUNTIME_ROOT / "Selections" / f"{'v2' if legacy else 'v3'}-areas-{digest}.json"
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


def _scope_matches(summary, watermark_text: str, watermark_style: str) -> bool:
    return (
        summary.get("schema") == 3
        and summary.get("target") == _target(watermark_text)
        and summary.get("style", "auto") == watermark_style
    )


def _load_state(identity, watermark_text: str = "", watermark_style: str = "auto") -> str:
    key = selection_key(identity, watermark_text, watermark_style)
    if f"{key}-areas" not in st.session_state:
        paths = [selection_files(identity, watermark_text, watermark_style)]
        legacy = selection_files(identity)
        if legacy != paths[0]:
            paths.append(legacy)
        regions, summary = [], {}
        for saved, summary_file in paths:
            try:
                candidate = json.loads(summary_file.read_text(encoding="utf-8"))
                if not _scope_matches(candidate, watermark_text, watermark_style):
                    continue
                regions = _normalized(
                    json.loads(saved.read_text(encoding="utf-8")), int(candidate["total_frames"])
                )
                summary = candidate
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
        st.session_state[f"{key}-areas"] = regions
        st.session_state[f"{key}-scan-result"] = summary
        st.session_state[f"{key}-edited"] = bool(summary.get("edited", False))
        st.session_state[f"{key}-revision"] = 0
    return key


def _automatic_current(summary, regions) -> bool:
    automatic = [region for region in regions if region.get("source") == "suggested"]
    if not automatic:
        return True
    from watermark_cleaner.scanner import SCAN_VERSION

    return summary.get("scan_version") == SCAN_VERSION and all(
        region.get("timing_verified") is True for region in automatic
    )


def selected_regions(
    identity, watermark_text: str = "", *, watermark_style: str = "auto"
) -> list[dict] | None:
    """Return current reviewed ranges, or None to refresh automatic detection."""
    key = _load_state(identity, watermark_text, watermark_style)
    summary = st.session_state.get(f"{key}-scan-result", {})
    if not _scope_matches(summary, watermark_text, watermark_style):
        return None
    regions = st.session_state.get(f"{key}-areas", [])
    if st.session_state.get(f"{key}-edited") and not regions:
        raise ValueError("No watermark ranges are selected. Add a missed watermark on the player.")
    if regions and _automatic_current(summary, regions):
        return regions
    return None


def _same_occurrence(first, second) -> bool:
    if first["start_frame"] >= second["end_frame"] or second["start_frame"] >= first["end_frame"]:
        return False
    a, b = first["bbox"], second["bbox"]
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and intersection / smaller >= 0.4


def _retain_unedited_intervals(region, overrides) -> list[dict]:
    """Keep refreshed coverage outside the old occurrence that a user edited."""
    intervals = [(region["start_frame"], region["end_frame"])]
    for override in overrides:
        if not _same_occurrence(region, override):
            continue
        start, stop = override["start_frame"], override["end_frame"]
        retained = []
        for first, last in intervals:
            if stop <= first or start >= last:
                retained.append((first, last))
                continue
            if first < start:
                retained.append((first, start))
            if stop < last:
                retained.append((stop, last))
        intervals = retained
    if intervals == [(region["start_frame"], region["end_frame"])]:
        return [region]
    return [
        {
            **region,
            "id": f"{region['id']}-part-{start}-{stop}",
            "start_frame": start,
            "end_frame": stop,
        }
        for start, stop in intervals
    ]


def _overrides(summary, total_frames: int) -> list[dict]:
    try:
        return _normalized(summary.get("overrides", []), total_frames)
    except (ValueError, TypeError):
        return []


def _store_state(identity, watermark_text, watermark_style, key, regions, summary) -> None:
    st.session_state[f"{key}-areas"] = regions
    st.session_state[f"{key}-scan-result"] = summary
    st.session_state[f"{key}-edited"] = bool(summary.get("edited"))
    saved, summary_file = selection_files(identity, watermark_text, watermark_style)
    _save(saved, regions)
    _save(summary_file, summary)


def find_watermark_regions(
    identity, *, watermark_text: str = "", watermark_style: str = "auto", progress=None
) -> list[dict]:
    from watermark_cleaner.scanner import SCAN_VERSION, scan_video

    key = _load_state(identity, watermark_text, watermark_style)
    previous = st.session_state.get(f"{key}-areas", [])
    previous_summary = st.session_state.get(f"{key}-scan-result", {})
    validate_source_identity(identity)
    result = scan_video(
        identity.source_path,
        watermark_text=watermark_text,
        watermark_style=watermark_style,
        progress=progress,
    )
    validate_source_identity(identity)
    total = int(result["total_frames"])
    automatic = _normalized(result.get("regions", []), total)
    manual = _normalized([r for r in previous if r.get("source") != "suggested"], total)
    overrides = (
        _overrides(previous_summary, total) if "overrides" in previous_summary else manual.copy()
    )
    blockers = overrides.copy()
    for correction in manual:
        # Editing an automatic area retains its ID. Apply its corrected times
        # to the recorded old geometry, even if the user moved or tightened it.
        # Newly drawn areas have no recorded association and remain additive.
        blockers.extend(
            {
                **override,
                "start_frame": correction["start_frame"],
                "end_frame": correction["end_frame"],
            }
            for override in overrides
            if override["id"] == correction["id"]
        )
    regions = [
        retained
        for region in automatic
        for retained in _retain_unedited_intervals(region, blockers)
    ] + manual
    regions.sort(key=lambda region: (region["start_frame"], region["end_frame"], region["id"]))
    summary = {
        "schema": 3,
        "target": _target(watermark_text),
        "style": watermark_style,
        "scan_version": SCAN_VERSION,
        "total_frames": total,
        "timing_verified": bool(regions)
        and all(r.get("source") != "suggested" or r.get("timing_verified") for r in regions),
        "regions_found": len(regions),
        "edited": bool(manual or overrides),
        "overrides": overrides,
    }
    st.session_state[f"{key}-revision"] += 1
    _store_state(identity, watermark_text, watermark_style, key, regions, summary)
    return regions


def regions_for_processing(
    identity, *, watermark_text: str = "", watermark_style: str = "auto", progress=None
) -> list[dict]:
    regions = selected_regions(identity, watermark_text, watermark_style=watermark_style)
    if regions is None:
        regions = find_watermark_regions(
            identity,
            watermark_text=watermark_text,
            watermark_style=watermark_style,
            progress=progress,
        )
    if not regions:
        raise ValueError(
            "No matching watermark was confirmed. Choose its style, enter its name, or add an area on the player."
        )
    return regions


def select_watermark_regions(
    identity, *, watermark_text: str = "", watermark_style: str = "auto"
) -> list[dict]:
    key = _load_state(identity, watermark_text, watermark_style)
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
                watermark_style=watermark_style,
                progress=lambda value, text: progress.progress(
                    max(0.0, min(1.0, float(value))), text=str(text)
                ),
            )
        except Exception as exc:
            st.error(str(exc))
        finally:
            progress.empty()
    summary = st.session_state.get(f"{key}-scan-result", {})
    previous = st.session_state[f"{key}-areas"]
    if _scope_matches(summary, watermark_text, watermark_style):
        if previous:
            st.caption(
                f"{len(previous)} watermark ranges. Play or scrub to check them; correct a miss only if needed."
            )
            if not _automatic_current(summary, previous):
                st.caption(
                    "Automatic timing and masks will refresh before processing. Your manual corrections are kept."
                )
        else:
            st.info(
                "No matching branding was found. Choose its style or enter its name in Watermark options, or add an area below."
            )
    revision = st.session_state[f"{key}-revision"]
    value = _select_regions(
        image=preview["image"],
        video_url=video_url,
        regions=previous,
        source_key=key,
        revision=revision,
        fps=preview["fps"],
        total_frames=preview["total_frames"],
        frame_times=preview["frame_times"],
        duration=preview["duration"],
        key=key,
        default={"revision": revision, "source_key": key, "regions": previous},
    )
    incoming = (
        value.get("regions", previous)
        if isinstance(value, dict)
        and value.get("revision") == revision
        and value.get("source_key") == key
        else previous
    )
    try:
        regions = _normalized(incoming, preview["total_frames"])
    except (ValueError, TypeError):
        st.error("A correction has an invalid area or time range.")
        return previous
    if regions != previous:
        overrides = _overrides(summary, preview["total_frames"])
        by_id = {region["id"]: region for region in regions}
        for prior in previous:
            if prior.get("source") == "suggested" and prior != by_id.get(prior["id"]):
                if prior not in overrides:
                    overrides.append(prior)
        summary = {
            **summary,
            "schema": 3,
            "target": _target(watermark_text),
            "style": watermark_style,
            "total_frames": preview["total_frames"],
            "timing_verified": False,
            "regions_found": len(regions),
            "edited": True,
            "overrides": overrides,
        }
        try:
            _store_state(identity, watermark_text, watermark_style, key, regions, summary)
        except OSError:
            st.warning("These corrections could not be saved for the next session.")
    return regions
