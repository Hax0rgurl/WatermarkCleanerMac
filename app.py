"""MAC WATERMARK CLEANER macOS web interface.

The native AppKit shell owns source-file selection so this process receives
real macOS paths. That lets successful jobs move the actual originals to Trash
after the cleaned output has been verified.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path

import streamlit as st
from loguru import logger

from demark_world.completion import (
    CompletionResult,
    OutputVerificationError,
    SourceIdentity,
    SourceIdentityError,
    TrashLocationUnknown,
    TrashMoveError,
    complete_job_workspace,
    consume_selection_manifest,
    create_job_workspace,
    prepare_job_publication,
    quarantine_job_workspace,
    recover_stale_workspaces,
    run_verified_cleaning_job,
    validate_source_identity,
    verify_cleaned_video,
)
from demark_world.configs import LOGS_PATH, OUTPUT_DIR, RUNTIME_ROOT
from demark_world.core import DeMarkWorld
from demark_world.schemas import CleanerType

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@st.cache_resource(show_spinner=False)
def _configure_logging() -> int:
    return logger.add(LOGS_PATH / "app.log", rotation="5 MB", retention=3)


_LOG_SINK_ID = _configure_logging()


@st.cache_resource(show_spinner=False)
def _recover_stale_workspaces_once() -> int:
    recovered = recover_stale_workspaces()
    return len(recovered)


_RECOVERED_STALE_JOBS = _recover_stale_workspaces_once()


@st.cache_resource(show_spinner=False)
def _load_engine() -> DeMarkWorld:
    return DeMarkWorld(
        cleaner_type=CleanerType.LAMA,
        detect_batch_size=4,
        enable_torch_compile=False,
    )


def _output_path(source: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # The engine always produces H.264/AAC. MP4 is the reliable container for
    # every accepted input type and prevents invalid WebM/AVI combinations.
    return OUTPUT_DIR / f"{source.stem}-cleaned-{timestamp}.mp4"


def _selection_manifest_path() -> Path | None:
    value = os.environ.get("MAC_WATERMARK_SELECTION_FILE")
    return Path(value).expanduser() if value else None


def _load_native_selection() -> list[SourceIdentity]:
    """Read and revalidate identity-pinned records from the native picker."""

    manifest = _selection_manifest_path()
    if manifest is None or not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read native selection manifest {}", manifest)
        return []

    selected: list[SourceIdentity] = []
    seen: set[tuple[int, int]] = set()
    for raw_identity in payload.get("sources", []):
        if not isinstance(raw_identity, dict):
            continue
        try:
            identity = SourceIdentity.from_mapping(raw_identity)
            validate_source_identity(identity)
        except SourceIdentityError as exc:
            logger.warning("Skipping stale native selection: {}", exc)
            continue
        key = (identity.device, identity.inode)
        if (
            key not in seen
            and identity.source_path.suffix.lower() in VIDEO_TYPES
        ):
            selected.append(identity)
            seen.add(key)
    return selected


def _process_source(
    identity: SourceIdentity,
    engine: DeMarkWorld,
    file_index: int,
    total: int,
    *,
    final_path: Path | None = None,
) -> tuple[Path, CompletionResult]:
    """Clean one pinned source and commit only after full validation."""

    source = identity.source_path
    progress = st.progress(0)
    status = st.empty()
    final_path = final_path or _output_path(source)
    workspace = create_job_workspace(identity, final_path)

    def update(percent: int) -> None:
        safe_percent = min(percent, 98)
        overall = int(((file_index + safe_percent / 100) / total) * 100)
        progress.progress(max(0, min(overall, 100)))
        if percent < 50:
            status.info(
                f"File {file_index + 1}/{total}: detecting watermark regions…"
            )
        elif percent < 95:
            status.info(
                f"File {file_index + 1}/{total}: restoring video frames…"
            )
        else:
            status.info(
                f"File {file_index + 1}/{total}: preserving audio and verifying…"
            )

    def verify(output_path: Path, *, source: Path) -> None:
        status.info(
            f"File {file_index + 1}/{total}: decoding the complete cleaned file…"
        )
        verify_cleaned_video(output_path, source=source)

    try:
        completion = run_verified_cleaning_job(
            identity,
            final_path,
            lambda input_path, output_path: engine.run(
                input_path,
                output_path,
                progress_callback=update,
                quiet=True,
                working_directory=workspace,
                before_publish=lambda partial, visible: prepare_job_publication(
                    workspace,
                    partial,
                    visible,
                ),
            ),
            verifier=verify,
            workspace=workspace,
        )
    except Exception:
        quarantine_job_workspace(workspace)
        raise

    # The source Trash receipt is already committed. These cleanup operations
    # are best-effort and cannot change or misreport that result.
    complete_job_workspace(workspace)
    manifest = _selection_manifest_path()
    if manifest is not None:
        consume_selection_manifest(manifest, identity)
    return final_path, completion


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
        initial_sidebar_state="expanded",
    )
    _inject_vaporwave_style()

    st.markdown(
        """
        <section class="vw-hero">
          <div class="vw-kicker">PRIVATE VIDEO RESTORATION // macOS</div>
          <h1>MAC WATERMARK CLEANER</h1>
          <p>Universal DeMark-World detection for Sora, Veo, Runway, Pika,
          Kling, and other AI-video watermarks—inside one private Mac app.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("PROCESSING")
        batch_size = st.slider(
            "Detection batch size",
            min_value=1,
            max_value=8,
            value=4,
            help="Lower this if memory pressure is high.",
        )
        st.success("● LaMa engine ready")
        if platform.system() == "Darwin":
            st.caption(
                "E2FGVI-HQ source is included for CUDA systems. This Mac build "
                "uses the optimized LaMa path."
            )
        st.divider()
        st.caption("CLEANED VIDEOS")
        st.code(str(OUTPUT_DIR), language=None)
        st.caption("APP DATA")
        st.code(str(RUNTIME_ROOT), language=None)
        st.caption("CLEANUP RECEIPTS")
        st.code(str(LOGS_PATH / "cleanup-audit.jsonl"), language=None)

    st.info(
        "The first run downloads the universal detector and LaMa model. "
        "They remain in Application Support for later offline use."
    )
    if _RECOVERED_STALE_JOBS:
        st.warning(
            f"Recovered {_RECOVERED_STALE_JOBS} interrupted job workspace(s) "
            "into App Data → Failed Jobs. Any matching unverified visible "
            "output was quarantined; no queued source was moved."
        )
    st.markdown(
        """
        <div class="vw-safety">
          <strong>RECOVERABLE</strong>
          <span>The app decodes the entire cleaned file, then confirms the
          source is still the exact device/inode selected before asking macOS
          to Trash it. Unconfirmed jobs are never marked complete.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sources = _load_native_selection()
    st.subheader("SOURCE QUEUE")
    if not sources:
        st.markdown(
            """
            <div class="vw-empty">
              <strong>NO VIDEOS SELECTED</strong>
              Use CHOOSE VIDEOS in the app toolbar above to select files from
              anywhere on your Mac.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write(f"{len(sources)} video(s) ready:")
        for identity in sources:
            size = identity.size / 1_048_576
            st.code(
                f"{identity.source_path}  ·  {size:.1f} MB",
                language=None,
            )

        if st.button(
            "REMOVE WATERMARKS",
            type="primary",
            use_container_width=True,
        ):
            results: list[tuple[Path, CompletionResult]] = []
            failures: list[tuple[Path, str]] = []
            try:
                with st.spinner(
                    "Loading detector and LaMa model. First use can take several minutes…"
                ):
                    engine = _load_engine()
                    engine.detect_batch_size = batch_size
            except Exception as exc:
                st.error(f"Could not load the cleaner: {exc}")
                logger.exception("Cleaner engine failed to load")
                engine = None

            if engine is not None:
                for index, identity in enumerate(sources):
                    source = identity.source_path
                    try:
                        completed_job = _process_source(
                            identity,
                            engine,
                            index,
                            len(sources),
                        )
                    except SourceIdentityError as exc:
                        failures.append((source, str(exc)))
                        st.error(f"{source.name}: {exc}")
                        st.warning(
                            "The path changed identity, so the app refused to "
                            "move anything at that path."
                        )
                    except OutputVerificationError as exc:
                        failures.append((source, str(exc)))
                        st.error(f"{source.name}: {exc}")
                        st.warning(
                            "Full-file validation failed before any source "
                            "Trash request."
                        )
                    except TrashLocationUnknown as exc:
                        failures.append((source, str(exc)))
                        st.error(f"{source.name}: {exc}")
                        st.warning(
                            "macOS did not provide a recoverable location. "
                            "Inspect both the source path and Trash."
                        )
                    except TrashMoveError as exc:
                        failures.append((source, str(exc)))
                        st.error(f"{source.name}: {exc}")
                        st.warning(
                            "The Trash commit was not confirmed. Inspect both "
                            "the source path and Trash before retrying."
                        )
                    except Exception as exc:
                        failures.append((source, str(exc)))
                        logger.exception("Cleaning failed for {}", source)
                        st.error(f"{source.name}: {exc}")
                        st.warning(
                            "The job stopped before a confirmed Trash commit. "
                            "Review the source path and cleanup audit."
                        )
                    else:
                        results.append(completed_job)
                        st.success(
                            f"Verified {completed_job[0].name}. "
                            "Foundation confirmed the original in macOS Trash."
                        )

            if results:
                st.session_state["completed"] = [
                    {
                        "output": str(path),
                        "source": receipt.source,
                        "trashed": receipt.trashed,
                        "completed_at": receipt.completed_at,
                    }
                    for path, receipt in results
                ]
                st.success(
                    f"Completed {len(results)} video(s). Verified outputs remain; "
                    "their originals are recoverable from macOS Trash."
                )
            if failures:
                st.warning(
                    f"{len(failures)} file(s) were not completed. "
                    "Review their individual status before retrying."
                )

    completed = [
        item
        for item in st.session_state.get("completed", [])
        if Path(item["output"]).is_file()
    ]
    if completed:
        st.subheader("COMPLETED")
        for item in completed:
            path = Path(item["output"])
            left, right = st.columns([3, 1])
            left.code(
                f"{path}\nOriginal → {item['trashed']}",
                language=None,
            )
            right.download_button(
                "DOWNLOAD",
                data=path.read_bytes(),
                file_name=path.name,
                mime="video/mp4",
                key=f"download-{path}",
                use_container_width=True,
            )

    st.divider()
    st.markdown(
        """
        <p class="small-note">
        Universal engine derived from
        <a href="https://github.com/linkedlist771/DeMark-World">DeMark-World</a>
        by LLinkedlist (MIT). Original Sora project distributed under
        Apache-2.0. Full notices are included with the app and source.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
