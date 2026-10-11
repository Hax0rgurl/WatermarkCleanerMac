"""Prepare the optional vision environment without changing the cleaner runtime."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from urllib.parse import urlparse

from demark_world.configs import RUNTIME_ROOT, SOURCE_ROOT
from demark_world.vision_scan import VISION_MODELS, local_endpoint
from demark_world.watermark_guard import target_names

VISION_REQUIREMENTS = [
    "mlx-vlm==0.7.6", "mlx==0.32.3", "mlx-metal==0.32.3",
    "transformers==5.19.0", "huggingface-hub==1.33.0", "llguidance==1.9.1",
    "opencv-python==5.0.0.93", "numpy==2.5.3", "pillow==12.3.0", "requests==2.34.2",
]


def cached_model(model_id: str) -> str | None:
    roots = [RUNTIME_ROOT/"Models"/"vision"/"hub", Path.home()/".cache/huggingface/hub"]
    try:
        config = json.loads((Path.home()/".pinokio/config.json").read_text())
        if config.get("home"):
            roots.append(Path(config["home"])/"cache/HF_HOME/hub")
    except (OSError, ValueError):
        pass
    name = "models--" + model_id.replace("/", "--")
    for root in roots:
        for snapshot in (root/name/"snapshots").glob("*"):
            # A tokenizer/config alone is a partial download, not an installed model.
            try:
                index = json.loads((snapshot/"model.safetensors.index.json").read_text())
                files = set(index["weight_map"].values())
                if files and all((snapshot/f).is_file() for f in files) and (
                    snapshot/"preprocessor_config.json").is_file():
                    return str(snapshot)
            except (OSError, KeyError, ValueError):
                if (snapshot/"model.safetensors").is_file() and (
                    snapshot/"preprocessor_config.json").is_file():
                    return str(snapshot)
    return None


def scan_video(path, *, profile="Quick", sampling_seconds=1, endpoint="", server_model="",
               refine=True, watermark_text="", use_vision_model=True, progress=None) -> dict:
    target_names(watermark_text)
    if endpoint:
        endpoint = local_endpoint(endpoint)
    server_fallback = False
    if endpoint:
        address = urlparse(endpoint)
        try:
            with socket.create_connection((address.hostname, address.port or 80), timeout=2):
                pass
            if not server_model.strip():
                raise ValueError("No model was selected on the server")
        except (OSError, ValueError):
            # A downloaded MLX model runs directly. It does not require the
            # optional OpenAI-compatible server to have been started separately.
            endpoint, server_model, server_fallback = "", "", True
            if progress:
                progress(0, "The local server is unavailable. Using the downloaded vision model directly…")
    environment = dict(os.environ)
    environment["HF_HOME"] = str(RUNTIME_ROOT/"Models"/"vision")
    environment["UV_CACHE_DIR"] = str(RUNTIME_ROOT/"Cache"/"vision-uv")
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    vision_python = RUNTIME_ROOT/"VisionPython"/"bin/python"
    import sys
    uv_candidates = [SOURCE_ROOT.parent/"tools/uv", Path.home()/"pinokio/bin/miniforge/bin/uv"]
    uv = next((str(p) for p in uv_candidates if p.is_file()), shutil.which("uv"))
    if use_vision_model and not uv:
        raise RuntimeError("The optional vision installer could not find uv. Manual timing is available.")
    if use_vision_model and not vision_python.is_file():
        if progress:
            progress(0, "Preparing the separate local vision runtime…")
        subprocess.run([uv, "venv", "--python", sys.executable, str(vision_python.parent.parent)],
                       env=environment, check=True, capture_output=True, text=True)
    marker = vision_python.parent.parent/"vision-version.txt"
    version = "\n".join(VISION_REQUIREMENTS)
    if use_vision_model and (not marker.exists() or marker.read_text() != version):
        if progress:
            progress(0, "Installing local vision components. First use can take several minutes…")
        installed = subprocess.run([uv, "pip", "install", "--python", str(vision_python),
                                    *VISION_REQUIREMENTS], env=environment, capture_output=True, text=True)
        if installed.returncode:
            raise RuntimeError("Vision installation failed: " + installed.stderr[-1500:])
        marker.write_text(version)
    request = {"path": str(path), "profile": profile, "sampling_seconds": sampling_seconds,
               "endpoint": endpoint, "server_model": server_model, "refine": refine,
               "watermark_text": watermark_text,
               "use_vision_model": use_vision_model,
               "cached_model": cached_model(VISION_MODELS[profile][0])}
    with tempfile.TemporaryDirectory(prefix="vision-scan-", dir=RUNTIME_ROOT/"Working") as temp:
        request_path = Path(temp)/"request.json"
        request_path.write_text(json.dumps(request))
        log_path = Path(temp)/"vision.log"
        result = None
        with log_path.open("w+") as log:
            python = str(vision_python) if use_vision_model else sys.executable
            process = subprocess.Popen([python, "-m", "demark_world.vision_worker",
                                        str(request_path)], env=environment, stdout=subprocess.PIPE,
                                       stderr=log, text=True)
            try:
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    if "result" in event:
                        result = event["result"]
                    if progress and "status" in event:
                        progress(event.get("progress", 0), event["status"])
                returncode = process.wait()
                if returncode or result is None:
                    log.seek(0)
                    raise RuntimeError("Vision scan did not complete: " + log.read()[-1500:])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        result["server_fallback"] = server_fallback
        return result
