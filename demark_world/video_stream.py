"""Seekable local previews of identity-pinned videos, without loading them into RAM."""

import atexit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import os
import re
import secrets
import stat
import threading
from urllib.parse import urlparse

from demark_world.completion import SourceIdentityError


def byte_range(header, size):
    """Return an inclusive single HTTP range; reject unsatisfiable requests."""
    if not header:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Range outside file")
    return start, end


class VideoPreviewServer:
    def __init__(self):
        self.sources = {}
        self.tokens = {}
        self.lock = threading.Lock()
        preview = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.serve(head=True)

            def do_GET(self):
                self.serve(head=False)

            def serve(self, head):
                host = urlparse("http://" + self.headers.get("Host", "")).hostname
                origin = self.headers.get("Origin")
                if host not in {"127.0.0.1", "localhost"} or (origin and
                        urlparse(origin).hostname not in {"127.0.0.1", "localhost"}):
                    self.send_error(403)
                    return
                token = self.path.split("?", 1)[0].removeprefix("/video/")
                with preview.lock:
                    identity = preview.sources.get(token)
                if identity is None:
                    self.send_error(404)
                    return
                try:
                    descriptor = os.open(identity.path, os.O_RDONLY | os.O_NOFOLLOW)
                except OSError:
                    self.send_error(410, "The selected video is unavailable")
                    return
                with os.fdopen(descriptor, "rb") as video:
                    details = os.fstat(video.fileno())
                    actual = (details.st_dev, details.st_ino, stat.S_IFMT(details.st_mode),
                              details.st_size, details.st_mtime_ns)
                    expected = (identity.device, identity.inode, identity.file_type,
                                identity.size, identity.mtime_ns)
                    if actual != expected:
                        self.send_error(410, "The selected video changed")
                        return
                    try:
                        start, end = byte_range(self.headers.get("Range"), identity.size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{identity.size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    partial = bool(self.headers.get("Range"))
                    self.send_response(206 if partial else 200)
                    self.send_header("Content-Type", mimetypes.guess_type(identity.path)[0]
                                     or "application/octet-stream")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Cache-Control", "private, no-store")
                    if partial:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{identity.size}")
                    if origin:
                        self.send_header("Access-Control-Allow-Origin", origin)
                        self.send_header("Vary", "Origin")
                    self.end_headers()
                    if head:
                        return
                    video.seek(start)
                    remaining = end - start + 1
                    try:
                        while remaining > 0:
                            chunk = video.read(min(128 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # Browsers cancel previous ranges when the playhead moves.

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def register(self, identity):
        if identity.size <= 0:
            raise SourceIdentityError("The selected video is empty.")
        with self.lock:
            if identity not in self.tokens:
                token = secrets.token_urlsafe(32)
                self.tokens[identity] = token
                self.sources[token] = identity
            token = self.tokens[identity]
        return f"http://127.0.0.1:{self.http.server_port}/video/{token}"

    def close(self):
        if self.thread.is_alive():
            self.http.shutdown()
            self.http.server_close()
            self.thread.join(timeout=2)
