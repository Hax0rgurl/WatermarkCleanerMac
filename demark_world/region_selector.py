"""Automatic named-watermark scanning with a playable correction timeline."""

import base64
import hashlib
import json
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from demark_world.timed_regions import normalize_regions
from demark_world.configs import RUNTIME_ROOT
from demark_world.completion import validate_source_identity
from demark_world.video_stream import VideoPreviewServer
from demark_world.vision_runtime import cached_model, scan_video
from demark_world.vision_scan import VISION_MODELS
from demark_world.watermark_guard import target_names

_select_regions = components.declare_component(
    "watermark_regions", path=str(Path(__file__).parent / "watermark_selector")
)


@st.cache_resource(show_spinner=False)
def _preview_server():
    return VideoPreviewServer()


@st.cache_data(show_spinner=False, max_entries=24)
def video_preview(path: str, mtime_ns: int, seconds: float):
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("Could not read the selected video.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError("The selected video has no readable frames.")
        duration = count / fps
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round(seconds * fps)))
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Could not read the video preview.")
        height, width = frame.shape[:2]
        scale = min(1, 1000 / width, 560 / height)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not create the video preview.")
        return ("data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
                duration, fps, count)
    finally:
        capture.release()


def selection_key(identity):
    return f"region-{identity.device}-{identity.inode}-{identity.mtime_ns}"


def selection_files(identity):
    key = selection_key(identity)
    saved = RUNTIME_ROOT / "Selections" / (
        "areas-" + hashlib.sha256((str(identity.source_path) + key).encode()).hexdigest()[:24] + ".json")
    return saved, saved.with_suffix(".scan.json")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def scan_scope_matches(summary, options):
    """Model/server changes apply to a rescan, not to already checked visible ranges."""
    previous = summary.get("options", {})
    try:
        same_names = sorted(target_names(previous.get("watermark_text", ""))) == sorted(
            target_names(options.get("watermark_text", "")))
    except ValueError:
        return False
    return (same_names and previous.get("sampling_seconds", 5) <= options.get("sampling_seconds", 1)
            and (not options.get("use_vision_model") or summary.get("semantic_model_check")))


def find_watermark_regions(identity, *, progress=None):
    """Replace old ranges with a new fully checked scan of this exact source."""
    key = selection_key(identity)
    options = st.session_state[f"{key}-options"]
    validate_source_identity(identity)
    result = scan_video(identity.source_path, **options, refine=True, progress=progress)
    validate_source_identity(identity)
    regions = normalize_regions(result["regions"], result["total_frames"]) if result["regions"] else []
    st.session_state[f"{key}-areas"] = regions
    st.session_state[f"{key}-revision"] += 1
    st.session_state[f"{key}-edited"] = False
    summary = {"schema": 2, "options": options, "model": result["model"],
               "semantic_model_check": result["semantic_model_check"],
               "timing_verified": result.get("timing_verified", False),
               "server_fallback": result.get("server_fallback", False),
               "cadence": result.get("cadence", {}), "regions_found": len(regions)}
    st.session_state[f"{key}-scan-result"] = summary
    saved, scan_file = selection_files(identity)
    _save(saved, regions)
    _save(scan_file, summary)
    return regions


def regions_for_processing(identity, *, progress=None):
    """Auto-scan uncalibrated legacy boxes; use current checked timeline corrections."""
    key = selection_key(identity)
    regions = st.session_state.get(f"{key}-areas", [])
    summary = st.session_state.get(f"{key}-scan-result", {})
    same_options = scan_scope_matches(summary, st.session_state.get(f"{key}-options", {}))
    checked = (summary.get("schema") == 2 and summary.get("timing_verified") and same_options)
    edited = st.session_state.get(f"{key}-edited", False) and (not summary or same_options)
    if not regions or not (checked or edited):
        regions = find_watermark_regions(identity, progress=progress)
    if not regions:
        raise ValueError("No matching watermark was confirmed. The source was kept. "
                         "Use the player to add a missed watermark or change the watermark name.")
    return regions


def select_watermark_regions(identity) -> list[dict]:
    source, key = identity.source_path, selection_key(identity)
    try:
        validate_source_identity(identity)
        poster, duration, fps, total = video_preview(str(source), identity.mtime_ns, 0.0)
        video_url = _preview_server().register(identity)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return []
    area_key, revision_key = f"{key}-areas", f"{key}-revision"
    saved, scan_file = selection_files(identity)
    if area_key not in st.session_state:
        try:
            previous = json.loads(saved.read_text())
            restored = normalize_regions(previous, total) if previous else []
            st.session_state[area_key] = [r for r in restored if r["source"] != "suggested"
                                        or r.get("verification") == "named-ocr-v1"]
        except (OSError, ValueError, TypeError):
            st.session_state[area_key] = []
        try:
            st.session_state[f"{key}-scan-result"] = json.loads(scan_file.read_text())
        except (OSError, ValueError):
            st.session_state[f"{key}-scan-result"] = {}
    st.session_state.setdefault(revision_key, 0)
    st.session_state.setdefault(f"{key}-edited", False)
    remembered = st.session_state.get(f"{key}-scan-result", {}).get("options", {})

    watermark_text = st.text_input("Watermark to remove", value=remembered.get("watermark_text", "InVideo"),
                                   placeholder="InVideo, Storyblocks, Sora, or a creator name",
                                   key=f"{key}-target")
    st.caption("Finds the named branding throughout the video and checks when each appearance starts and stops. Captions stay outside the automatic selection.")
    profile, endpoint, server_model, interval, use_model = "Quick", "", "", 1, True
    with st.expander("Scan settings"):
        use_model = st.checkbox("Confirm branding with a local vision model", value=remembered.get("use_vision_model", True),
                                key=f"{key}-use-model")
        if use_model:
            profiles = list(VISION_MODELS)
            installed = [name for name in profiles if cached_model(VISION_MODELS[name][0])]
            default_profile = "Quick" if "Quick" in installed or not installed else installed[0]
            if remembered.get("profile") in profiles:
                default_profile = remembered["profile"]
            profile = st.selectbox("Vision model", profiles, index=profiles.index(default_profile),
                                   key=f"{key}-profile")
            st.caption(f"{VISION_MODELS[profile][0]}. " + (
                "Uses the downloaded model." if cached_model(VISION_MODELS[profile][0]) else
                f"First use downloads about {'16 GB' if profile == 'Latest quality' else '3 GB'}.") + " Frames stay on your Mac.")
            use_server = st.checkbox("Use a vision server already running on this Mac", value=bool(remembered.get("endpoint")), key=f"{key}-server")
            st.caption("Downloaded models run directly. If the optional server is unavailable, the scan uses the downloaded model.")
            endpoint = st.text_input("Local OpenAI-compatible URL", remembered.get("endpoint") or "http://127.0.0.1:8080/v1",
                                     key=f"{key}-url") if use_server else ""
            server_model = st.text_input("Local server model name", value=remembered.get("server_model", ""), key=f"{key}-model") if use_server else ""
        else:
            st.caption("Uses Apple Vision text recognition with an exact watermark-name check.")
        interval = st.select_slider("Check for appearances every", options=[0.5, 1, 2], value=remembered.get("sampling_seconds", 1),
                                    format_func=lambda v: f"{v:g} second(s)", key=f"{key}-interval")
    options = {"profile": profile, "sampling_seconds": interval, "endpoint": endpoint,
               "server_model": server_model, "watermark_text": watermark_text,
               "use_vision_model": use_model}
    st.session_state[f"{key}-options"] = options
    try:
        names = target_names(watermark_text)
        valid_target = True
        st.caption("Looking for: " + " · ".join(names))
    except ValueError:
        valid_target = False
    if st.button("FIND & CHECK WATERMARK TIMES", key=f"{key}-scan", disabled=not valid_target):
        bar = st.progress(0, text="Preparing the automatic scan…")
        try:
            find_watermark_regions(identity, progress=lambda value, text:
                                   bar.progress(float(value), text=text))
        except Exception as exc:
            st.error(str(exc))
        finally:
            bar.empty()

    summary = st.session_state.get(f"{key}-scan-result", {})
    if summary.get("schema") == 2 and scan_scope_matches(summary, options):
        count = summary["regions_found"]
        model = summary["model"]
        if count:
            cadence = summary.get("cadence", {})
            repeated = f" Repeats about every {cadence['period_seconds']:.1f} seconds." if cadence.get("period_seconds") else ""
            st.success(f"{count} appearances found. Start/stop frames checked.{repeated}")
            st.caption(f"These ranges were checked with {model}. Scrub or play any range below. FIND & CHECK reruns the scan using the current settings.")
            if summary.get("server_fallback"):
                st.caption("The optional server was unavailable, so the downloaded vision model ran directly.")
        else:
            st.info("No matching branding was confirmed. Change the name or use Add missed watermark on the player.")
    elif st.session_state[area_key]:
        st.caption("The next automatic scan replaces the older selections with checked appearance times.")

    previous, revision = st.session_state[area_key], st.session_state[revision_key]
    value = _select_regions(image=poster, video_url=video_url, regions=previous,
                            source_key=key, revision=revision, fps=fps, total_frames=total,
                            key=key, default={"revision": revision, "regions": previous})
    regions = value.get("regions", previous) if isinstance(value, dict) and (
        value.get("revision") == revision) else previous
    try:
        regions = normalize_regions(regions, total) if regions else []
    except (ValueError, TypeError):
        st.error("A timeline correction has an invalid area or range.")
        return []
    if regions != previous:
        st.session_state[f"{key}-edited"] = True
    st.session_state[area_key] = regions
    try:
        _save(saved, regions)
    except OSError:
        st.warning("These selections could not be saved for the next session.")
    return regions
