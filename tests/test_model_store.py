"""Exercise real HTTP downloads, interruption, verification and locking."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

import pytest
from watermark_cleaner import model_store as store


@contextmanager
def published_model(payload: bytes):
    state = {"body": payload, "requests": 0, "delay": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"] += 1
            time.sleep(state["delay"])
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    spec = store.ModelSpec(
        "e2fgvi",
        "weights.pth",
        f"http://127.0.0.1:{server.server_port}/weights.pth",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )
    try:
        yield spec, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "model_directory", lambda: tmp_path)
    store._VERIFIED.clear()
    return tmp_path


def test_streamed_download_and_offline_reuse(monkeypatch, isolated_store):
    payload = b"verified test checkpoint" * 50000
    with published_model(payload) as (spec, server):
        monkeypatch.setattr(store, "MODEL_SPECS", (spec,))
        events = []
        result = store.ensure_models(None, lambda value, text: events.append((value, text)))
        assert result[spec.name].read_bytes() == payload
        assert store.models_status() == {"e2fgvi": True, "propainter": False}
        assert events[-1][0] == 1.0
        assert all(0 <= value <= 1 for value, _ in events)
        assert not list(isolated_store.rglob("*.partial"))
        monkeypatch.setattr(
            store,
            "urlopen",
            lambda *args, **kwargs: pytest.fail("Verified cache attempted a network request"),
        )
        store.ensure_models("e2fgvi")
        assert server["requests"] == 1


def test_corrupt_download_never_publishes_and_retry_recovers(monkeypatch, isolated_store):
    payload = b"correct checkpoint bytes" * 128
    with published_model(payload) as (spec, server):
        monkeypatch.setattr(store, "MODEL_SPECS", (spec,))
        path = store.model_path(spec)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"old incomplete data")
        server["body"] = b"x" * len(payload)
        with pytest.raises(RuntimeError, match="Retry"):
            store.ensure_models("e2fgvi")
        assert path.read_bytes() == b"old incomplete data"
        assert not store.models_status()["e2fgvi"]
        assert not list(isolated_store.rglob("*.partial"))
        server["body"] = payload
        store.ensure_models("e2fgvi")
        assert path.read_bytes() == payload
        assert store.models_status()["e2fgvi"]


def test_short_download_is_rejected(monkeypatch, isolated_store):
    with published_model(b"full checkpoint payload") as (spec, server):
        monkeypatch.setattr(store, "MODEL_SPECS", (spec,))
        server["body"] = b"short"
        with pytest.raises(RuntimeError, match="checksum"):
            store.ensure_models("e2fgvi")
        assert not store.model_path(spec).exists()
        assert not list(isolated_store.rglob("*.partial"))


def test_cached_hash_is_invalidated_when_file_changes(monkeypatch, isolated_store):
    with published_model(b"valid bytes") as (spec, server):
        monkeypatch.setattr(store, "MODEL_SPECS", (spec,))
        store.ensure_models("e2fgvi")
        assert store.models_status()["e2fgvi"]
        store.model_path(spec).write_bytes(b"wrong bytes")
        assert not store.models_status()["e2fgvi"]


def test_parallel_windows_only_download_once(monkeypatch, isolated_store):
    with published_model(b"thread-safe checkpoint" * 64) as (spec, server):
        monkeypatch.setattr(store, "MODEL_SPECS", (spec,))
        server["delay"] = 0.1
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(store.ensure_models, "e2fgvi") for _ in range(2)]
            results = [future.result(timeout=5) for future in futures]
        assert results[0] == results[1]
        assert server["requests"] == 1


def test_abandoned_partial_is_removed_on_retry(monkeypatch, isolated_store):
    with published_model(b"checkpoint") as (spec, server):
        monkeypatch.setattr(store, "MODEL_SPECS", (spec,))
        path = store.model_path(spec)
        path.parent.mkdir(parents=True)
        orphan = path.parent / f".{spec.name}.abandoned.partial"
        orphan.write_bytes(b"interrupted download")
        store.ensure_models("e2fgvi")
        assert not orphan.exists()
        assert path.read_bytes() == b"checkpoint"


def test_another_process_waits_for_download_lock(monkeypatch, isolated_store):
    with published_model(b"process-safe checkpoint") as (spec, server):
        isolated_store.mkdir(exist_ok=True)
        lock = (isolated_store / ".download.lock").open("a+b")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = json.dumps({"root": str(isolated_store), "spec": spec.__dict__})
        code = """import json,sys
from pathlib import Path
from watermark_cleaner import model_store as s
x=json.loads(sys.argv[1]);s.model_directory=lambda:Path(x["root"]);s.MODEL_SPECS=(s.ModelSpec(**x["spec"]),)
def progress(value,message):
 if "Waiting" in message:print("waiting",flush=True)
s.ensure_models("e2fgvi",progress);print("ready",flush=True)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", code, data],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        try:
            assert select.select([process.stdout], [], [], 10)[0], (
                "Child did not report its lock wait"
            )
            assert process.stdout.readline().strip() == "waiting"
            assert process.poll() is None
            assert server["requests"] == 0
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            output, errors = process.communicate(timeout=10)
            assert process.returncode == 0, errors
            assert "ready" in output
            assert server["requests"] == 1
        finally:
            lock.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def test_unknown_engine_does_not_download():
    with pytest.raises(ValueError, match="Unknown"):
        store.ensure_models("unsupported")
