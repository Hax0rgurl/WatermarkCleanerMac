"""Local desktop interface for E2FGVI-HQ and ProPainter video restoration."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from loguru import logger

from watermark_cleaner.completion import (
    SourceIdentity,
    SourceIdentityError,
    complete_job_workspace,
    create_job_workspace,
    prepare_job_publication,
    quarantine_job_workspace,
    quarantine_recorded_output,
    validate_source_identity,
)
from watermark_cleaner.configs import LOGS_PATH, OUTPUT_DIR, RUNTIME_ROOT
from watermark_cleaner.region_selector import regions_for_processing, select_watermark_regions
from watermark_cleaner.templates import STYLE_OPTIONS

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ENGINE_OPTIONS = {"E2FGVI-HQ": "e2fgvi", "ProPainter": "propainter"}


@st.cache_resource(show_spinner=False)
def _configure_logging() -> int:
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    return logger.add(LOGS_PATH / "app.log", rotation="5 MB", retention=3)


def _model_status() -> dict[str, bool]:
    from watermark_cleaner.model_store import models_status

    status = models_status()
    return {name: status.get(name) is True for name in ENGINE_OPTIONS.values()}


def _download_models(progress=None) -> None:
    from watermark_cleaner.model_store import ensure_models

    ensure_models(None, progress=progress)


def _models_ready() -> bool:
    """Finish first-launch provisioning and expose an explicit retry on failure."""
    try:
        status = _model_status()
    except Exception as exc:
        status = dict.fromkeys(ENGINE_OPTIONS.values(), False)
        st.session_state["setup_error"] = str(exc)
    if all(status.values()):
        st.session_state.pop("setup_error", None)
        return True
    if not st.session_state.get("setup_attempted", False):
        st.session_state["setup_attempted"] = True
        bar = st.progress(0.0, text="Preparing both local restoration models…")
        try:
            _download_models(
                lambda value, text: bar.progress(max(0.0, min(1.0, float(value))), text=str(text))
            )
            status = _model_status()
            if not all(status.values()):
                raise RuntimeError("Model preparation is incomplete. Reopen the app to retry.")
            st.session_state.pop("setup_error", None)
        except Exception as exc:
            logger.exception("First-launch model preparation failed")
            st.session_state["setup_error"] = str(exc)
        finally:
            bar.empty()
    ready = all(status.values())
    if not ready:
        st.error(
            "Setup could not finish: "
            + st.session_state.get("setup_error", "One or both model downloads are incomplete.")
        )
        if st.button("RETRY MODEL SETUP", key="retry-model-setup"):
            st.session_state["setup_attempted"] = False
            st.rerun()
    return ready


def _output_path(source: Path) -> Path:
    return OUTPUT_DIR / f"{source.stem}-cleaned-{datetime.now():%Y%m%d-%H%M%S-%f}.mp4"


def _selection_manifest_path() -> Path | None:
    value = os.environ.get("MAC_WATERMARK_SELECTION_FILE")
    return Path(value).expanduser() if value else None


def _load_native_selection() -> list[SourceIdentity]:
    manifest = _selection_manifest_path()
    if manifest is None or not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload.get("sources", [])
        if not isinstance(records, list):
            raise ValueError("Invalid source queue")
    except (OSError, ValueError, AttributeError):
        logger.warning("Could not read the native video selection")
        return []
    selected, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            identity = SourceIdentity.from_mapping(record)
            validate_source_identity(identity)
        except SourceIdentityError:
            continue
        key = identity.device, identity.inode
        if key not in seen and identity.source_path.suffix.lower() in VIDEO_TYPES:
            selected.append(identity)
            seen.add(key)
    return selected


def _new_cleaner(engine_id: str):
    from watermark_cleaner.core import WatermarkCleaner

    return WatermarkCleaner(engine_id=engine_id)


def _process_source(
    identity: SourceIdentity,
    cleaner,
    file_index: int,
    total: int,
    *,
    final_path: Path | None = None,
    watermark_regions=None,
    watermark_text: str = "",
    watermark_style: str = "auto",
) -> Path:
    """Publish a verified result while retaining the exact selected source."""
    validate_source_identity(identity)
    destination = final_path or _output_path(identity.source_path)
    workspace = create_job_workspace(identity, destination)
    bar = st.progress(0, text=f"Preparing {identity.source_path.name}…")

    def update(percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        overall = int(((file_index + min(percent, 99) / 100) / max(1, total)) * 100)
        if percent < 15:
            text = "Finding and checking watermark appearances…"
        elif percent < 90:
            text = "Restoring video frames locally…"
        else:
            text = "Preserving audio and verifying the complete video…"
        bar.progress(overall, text=f"{file_index + 1}/{total}: {text}")

    def before_publish(partial: Path, visible: Path):
        validate_source_identity(identity)
        return prepare_job_publication(workspace, partial, visible)

    try:
        if watermark_regions is None:
            watermark_regions = regions_for_processing(
                identity,
                watermark_text=watermark_text,
                watermark_style=watermark_style,
                progress=lambda value, text: update(int(max(0.0, min(1.0, float(value))) * 14)),
            )
            validate_source_identity(identity)
        cleaner.run(
            identity.source_path,
            destination,
            progress_callback=update,
            quiet=True,
            working_directory=workspace,
            before_publish=before_publish,
            watermark_regions=watermark_regions,
            watermark_text=watermark_text,
            watermark_style=watermark_style,
        )
        validate_source_identity(identity)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("The cleaner did not produce a verified video.")
    except Exception:
        quarantine_recorded_output(workspace, reason="restoration_failed")
        quarantine_job_workspace(workspace)
        raise
    finally:
        bar.empty()
    complete_job_workspace(workspace)
    return destination


def _inject_vaporwave_style() -> None:
    st.markdown(
        """
        <style>
          :root {
            --void: #070611;
            --ink: #0a0a14;
            --panel: rgba(26, 0, 51, .76);
            --panel-solid: #16052c;
            --pink: #ff006e;
            --hot-pink: #ff10f0;
            --cyan: #00f5ff;
            --violet: #8338ec;
            --acid: #39ff14;
            --text: #f8f4ff;
            --muted: #c8b8d8;
          }

          html, body, [class*="css"] {
            font-family: "SFMono-Regular", Menlo, Monaco, Consolas,
              "Liberation Mono", "Courier New", monospace;
          }

          [data-testid="stAppViewContainer"], .stApp {
            color: var(--text);
            background:
              radial-gradient(circle at 50% 0%, rgba(131, 56, 236, .28), transparent 46%),
              radial-gradient(circle at 94% 16%, rgba(255, 0, 110, .13), transparent 30%),
              linear-gradient(180deg, #0d0a20 0%, var(--void) 100%);
          }

          .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .15;
            z-index: 0;
            background-image:
              linear-gradient(rgba(255, 0, 110, .32) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 245, 255, .25) 1px, transparent 1px);
            background-size: 24px 24px;
            mask-image: linear-gradient(to bottom, transparent, black 35%, black);
          }

          .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .12;
            z-index: 999;
            background: repeating-linear-gradient(
              0deg, transparent, transparent 3px,
              rgba(0, 245, 255, .14) 3px, rgba(0, 245, 255, .14) 4px
            );
          }

          [data-testid="stHeader"] {
            background: transparent;
          }

          [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            padding: 2.1rem 2.4rem 4rem;
          }

          [data-testid="stSidebar"] {
            background: rgba(10, 3, 26, .94);
            border-right: 1px solid rgba(0, 245, 255, .42);
            box-shadow: 10px 0 40px rgba(131, 56, 236, .17);
          }

          [data-testid="stSidebar"] * {
            color: var(--text);
          }

          .vw-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.2rem 1.8rem;
            margin-bottom: 1.4rem;
            border: 1px solid var(--cyan);
            background:
              linear-gradient(135deg, rgba(131, 56, 236, .28), rgba(255, 0, 110, .09)),
              rgba(10, 4, 26, .82);
            box-shadow:
              0 0 0 1px rgba(255, 0, 110, .54) inset,
              0 0 28px rgba(0, 245, 255, .13),
              0 24px 80px rgba(0, 0, 0, .38);
          }

          .vw-kicker {
            color: var(--cyan);
            letter-spacing: .18em;
            font-size: .78rem;
            font-weight: 700;
            margin-bottom: .7rem;
          }

          .vw-hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4.6vw, 3.9rem);
            line-height: 1;
            letter-spacing: -.055em;
            background: linear-gradient(90deg, var(--hot-pink), #b463ff 48%, var(--cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 24px rgba(255, 16, 240, .22);
          }

          .vw-hero p {
            color: #d9d0e8;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 780px;
            margin: .9rem 0 0;
          }

          .vw-safety {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: .8rem;
            align-items: start;
            margin: 1rem 0 1.4rem;
            padding: 1rem 1.1rem;
            border-left: 3px solid var(--acid);
            background: rgba(57, 255, 20, .065);
            color: #dcffd4;
          }

          .vw-safety strong {
            color: var(--acid);
            letter-spacing: .08em;
          }

          .vw-empty {
            padding: 2rem;
            margin: 1rem 0;
            border: 1px dashed rgba(0, 245, 255, .62);
            background: rgba(0, 245, 255, .045);
            text-align: center;
            color: var(--muted);
          }

          .vw-empty strong {
            display: block;
            color: var(--cyan);
            font-size: 1.1rem;
            margin-bottom: .5rem;
          }

          div[data-testid="stButton"] > button[kind="primary"] {
            min-height: 3.25rem;
            border: 1px solid var(--cyan);
            border-radius: 0;
            color: white;
            background: linear-gradient(90deg, var(--violet), var(--pink));
            box-shadow: 0 0 22px rgba(255, 0, 110, .23);
            font-weight: 800;
            letter-spacing: .08em;
          }

          div[data-testid="stButton"] > button[kind="primary"]:hover {
            border-color: white;
            box-shadow: 0 0 30px rgba(0, 245, 255, .3);
            transform: translateY(-1px);
          }

          [data-testid="stAlert"] {
            border-radius: 0;
            border: 1px solid rgba(0, 245, 255, .34);
            background: rgba(26, 0, 51, .68);
          }

          [data-testid="stCodeBlock"] {
            border: 1px solid rgba(131, 56, 236, .48);
            border-radius: 0;
          }

          [data-testid="stExpander"] details,
          [data-testid="stExpander"] summary,
          [data-testid="stCode"] pre {
            background: var(--panel-solid);
            color: var(--text);
          }

          [data-testid="stSlider"] label,
          [data-testid="stAlert"] p,
          [data-testid="stExpander"] summary p {
            color: var(--text);
          }

          hr {
            border-color: rgba(0, 245, 255, .2) !important;
          }

          h1, h2, h3 {
            color: var(--text);
          }

          a {
            color: var(--cyan) !important;
          }

          .small-note {
            color: #9c8ead;
            font-size: .82rem;
            line-height: 1.55;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="MAC WATERMARK CLEANER",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _configure_logging()
    _inject_vaporwave_style()
    st.markdown(
        """<section class="vw-hero"><div class="vw-kicker">LOCAL VIDEO RESTORATION</div>
      <h1>MAC WATERMARK CLEANER</h1><p>Choose your videos, pick an engine, and remove the watermark.
      Your original files stay where they are.</p></section>""",
        unsafe_allow_html=True,
    )
    ready = _models_ready()
    engine_label = st.selectbox(
        "Restoration engine", list(ENGINE_OPTIONS), index=0, key="restoration-engine"
    )
    st.caption(
        "E2FGVI-HQ is the faster default on this Mac. ProPainter offers a second restoration model."
    )
    if ready:
        st.caption("Both models are ready. Video processing stays on this Mac.")
    with st.expander("Watermark options"):
        style_label = st.selectbox(
            "Watermark style", list(STYLE_OPTIONS), index=0, key="watermark-style"
        )
        watermark_style = STYLE_OPTIONS[style_label]
        watermark_text = st.text_input(
            "Watermark name (optional)",
            value="",
            placeholder="Automatic detection, or enter a specific brand",
            key="watermark-name",
        )
        st.caption(
            "Auto finds common platform watermarks. Choose a style for its logo and creator text, or enter a specific name."
        )
    sources = _load_native_selection()
    st.subheader("VIDEOS")
    if sources:
        for identity in sources:
            st.write(f"**{identity.source_path.name}** · {identity.size / 1_048_576:.1f} MB")
        review = st.toggle("Review or adjust watermark times", value=False, key="review-watermarks")
        if review:
            for identity in sources:
                with st.expander(identity.source_path.name, expanded=True):
                    select_watermark_regions(
                        identity, watermark_text=watermark_text, watermark_style=watermark_style
                    )
    else:
        st.info("Use CHOOSE VIDEOS in the app toolbar to select a video.")
    if st.button(
        "REMOVE WATERMARKS",
        type="primary",
        use_container_width=True,
        disabled=not (ready and sources),
        key="remove-watermarks",
    ):
        completed = list(st.session_state.get("completed", []))
        cleaner = None
        try:
            cleaner = _new_cleaner(ENGINE_OPTIONS[engine_label])
        except Exception as exc:
            st.error(f"Could not prepare {engine_label}: {exc}")
            logger.exception("Cleaner could not start")
        if cleaner is not None:
            for index, identity in enumerate(sources):
                try:
                    path = _process_source(
                        identity,
                        cleaner,
                        index,
                        len(sources),
                        watermark_text=watermark_text,
                        watermark_style=watermark_style,
                    )
                except Exception as exc:
                    st.error(f"{identity.source_path.name}: {exc}")
                    logger.exception("Restoration failed for {}", identity.source_path)
                else:
                    completed.append(
                        {
                            "output": str(path),
                            "source": str(identity.source_path),
                            "engine": engine_label,
                        }
                    )
                    st.success(f"Saved and verified {path.name}. Original kept.")
        st.session_state["completed"] = completed
    completed = [
        record
        for record in st.session_state.get("completed", [])
        if Path(record["output"]).is_file()
    ]
    if completed:
        st.subheader("CLEANED VIDEOS")
        st.caption("Files are already saved in the OUTPUT folder. Download saves another copy.")
        for index, record in enumerate(completed):
            path = Path(record["output"])
            st.write(f"**{path.name}** · {record.get('engine', 'Local restoration')}")
            st.download_button(
                "DOWNLOAD",
                data=path.read_bytes(),
                file_name=path.name,
                mime="video/mp4",
                key=f"download-{index}-{path}",
                on_click="ignore",
            )
    with st.sidebar:
        st.caption("OUTPUT FOLDER")
        st.code(str(OUTPUT_DIR), language=None)
        st.caption("APP DATA")
        st.code(str(RUNTIME_ROOT), language=None)
        st.caption(
            "Models download automatically once, then run locally. No cloud processing or credits."
        )
        st.caption(
            "The included research models have noncommercial licenses. See the bundled notices for permissions."
        )


if __name__ == "__main__":
    main()
