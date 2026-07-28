"""MAC WATERMARK CLEANER macOS web interface.

The processing engine is the universal DeMark-World implementation, with a
simple workflow for Sora and other supported AI-video sources.
"""

from __future__ import annotations

import platform
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from loguru import logger

from demark_world.configs import LOGS_PATH, OUTPUT_DIR, RUNTIME_ROOT
from demark_world.core import DeMarkWorld
from demark_world.schemas import CleanerType

VIDEO_TYPES = ["mp4", "mov", "m4v", "avi", "mkv", "webm"]


@st.cache_resource(show_spinner=False)
def _configure_logging() -> int:
    return logger.add(LOGS_PATH / "app.log", rotation="5 MB", retention=3)


_LOG_SINK_ID = _configure_logging()


def _safe_name(name: str) -> str:
    candidate = Path(name).name.strip()
    return candidate or "video.mp4"


@st.cache_resource(show_spinner=False)
def _load_engine() -> DeMarkWorld:
    return DeMarkWorld(
        cleaner_type=CleanerType.LAMA,
        detect_batch_size=4,
        enable_torch_compile=False,
    )


def _output_path(original_name: str) -> Path:
    source = Path(_safe_name(original_name))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # The engine always produces H.264/AAC. MP4 is the reliable container for
    # every accepted input type and prevents invalid WebM/AVI combinations.
    return OUTPUT_DIR / f"{source.stem}-cleaned-{timestamp}.mp4"


def _process_upload(uploaded_file, engine: DeMarkWorld, file_index: int, total: int):
    progress = st.progress(0)
    status = st.empty()
    final_path = _output_path(uploaded_file.name)

    with tempfile.TemporaryDirectory(prefix="mac-watermark-cleaner-") as temporary:
        input_path = Path(temporary) / _safe_name(uploaded_file.name)
        input_path.write_bytes(uploaded_file.getvalue())

        def update(percent: int) -> None:
            overall = int(((file_index + percent / 100) / total) * 100)
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
                    f"File {file_index + 1}/{total}: preserving audio and saving…"
                )

        engine.run(input_path, final_path, progress_callback=update, quiet=True)

    progress.progress(int(((file_index + 1) / total) * 100))
    status.success(f"Saved {final_path.name}")
    return final_path


def main() -> None:
    st.set_page_config(
        page_title="MAC WATERMARK CLEANER",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
          .block-container {max-width: 1100px; min-width: 0; padding-top: 2rem;}
          .hero {
            padding: 1.7rem 2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #101827, #1d3055);
            color: white;
            box-shadow: 0 18px 45px rgba(10, 30, 65, .18);
            margin-bottom: 1.5rem;
          }
          .hero h1 {
            margin: 0;
            font-size: clamp(1.65rem, 2.8vw, 2.4rem);
            line-height: 1.12;
            white-space: normal;
            overflow-wrap: break-word;
          }
          .hero p {
            opacity: .84;
            font-size: 1.05rem;
            margin-bottom: 0;
            overflow-wrap: break-word;
          }
          .small-note {font-size: .88rem; color: #667085;}
        </style>
        <div class="hero">
          <h1>🎬 MAC WATERMARK CLEANER</h1>
          <p>Now powered by DeMark-World’s universal detector for Sora, Veo,
          Runway, Pika, Kling, and other AI-video watermarks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Processing")
        batch_size = st.slider(
            "Detection batch size",
            min_value=1,
            max_value=8,
            value=4,
            help="Lower this if memory pressure is high.",
        )
        st.success("LaMa — optimized for this Mac")
        if platform.system() == "Darwin":
            st.caption(
                "The E2FGVI-HQ source is included for CUDA systems, but its "
                "compiled MMCV operations are not part of the standard Mac install."
            )
        st.divider()
        st.caption("Writable data")
        st.code(str(RUNTIME_ROOT), language=None)
        st.caption("Cleaned videos")
        st.code(str(OUTPUT_DIR), language=None)
        st.caption("Logs")
        st.code(str(LOGS_PATH), language=None)

    st.info(
        "First processing run downloads the universal detector and LaMa model. "
        "They stay in Application Support for later offline use."
    )
    st.caption(
        "Use this tool only with media you own or are authorized to modify. "
        "The app never deletes the original video."
    )

    uploads = st.file_uploader(
        "Drop one or more videos here",
        type=VIDEO_TYPES,
        accept_multiple_files=True,
        help="Original files remain untouched. Results are written to Movies.",
    )

    if uploads:
        st.write(f"Ready to process {len(uploads)} video(s).")
        for upload in uploads:
            st.write(f"• {_safe_name(upload.name)} — {upload.size / 1_048_576:.1f} MB")

        if st.button(
            "Remove Watermarks",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Loading the detector and LaMa model. First use can take several minutes…"
                ):
                    engine = _load_engine()
                    engine.detect_batch_size = batch_size

                results = []
                for index, upload in enumerate(uploads):
                    results.append(
                        _process_upload(upload, engine, index, len(uploads))
                    )

                st.session_state["completed_paths"] = [str(path) for path in results]
                st.success(
                    f"Finished {len(results)} video(s). Originals were not changed."
                )
            except Exception as exc:
                st.error(str(exc))
                st.warning(
                    f"Open the log folder for details: {LOGS_PATH}. "
                    "If this is first use, confirm the Mac is online and try again."
                )

    completed = [
        Path(path)
        for path in st.session_state.get("completed_paths", [])
        if Path(path).is_file()
    ]
    if completed:
        st.subheader("Completed")
        for path in completed:
            left, right = st.columns([3, 1])
            left.code(str(path), language=None)
            right.download_button(
                "Download",
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
