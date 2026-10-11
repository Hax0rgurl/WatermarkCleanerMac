import urllib.error
import urllib.request

import pytest

from demark_world.completion import SourceIdentity
from demark_world.video_stream import VideoPreviewServer


@pytest.fixture
def preview(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(bytes(range(256)) * 64)
    server = VideoPreviewServer()
    yield video, server, server.register(SourceIdentity.capture(video))
    server.close()


def request(url, *, method="GET", **headers):
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers))


def test_browser_can_seek_forward_backward_and_request_the_mp4_tail(preview):
    video, server, url = preview
    data = video.read_bytes()
    for header, start, end in [("bytes=1024-2047", 1024, 2047),
                               ("bytes=0-511", 0, 511),
                               ("bytes=-128", len(data)-128, len(data)-1),
                               ("bytes=16000-", 16000, len(data)-1)]:
        with request(url, Range=header) as response:
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Range"] == f"bytes {start}-{end}/{len(data)}"
            assert response.read() == data[start:end+1]


def test_metadata_request_has_size_and_type_without_video_body(preview):
    video, server, url = preview
    with request(url, method="HEAD", Origin="http://127.0.0.1:8501") as response:
        assert int(response.headers["Content-Length"]) == video.stat().st_size
        assert response.headers["Content-Type"] == "video/mp4"
        assert response.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:8501"
        assert response.read() == b""


@pytest.mark.parametrize("header", ["bytes=99999-", "bytes=5-1", "bytes=0-4,8-12", "bytes=-0"])
def test_invalid_seek_is_rejected(preview, header):
    _, _, url = preview
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(url, Range=header)
    assert exc.value.code == 416


def test_preview_refuses_replaced_file_or_symlink(preview, tmp_path):
    video, _, url = preview
    video.unlink()
    other = tmp_path / "other.mp4"
    other.write_bytes(b"not the selected video")
    video.symlink_to(other)
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(url)
    assert exc.value.code == 410


def test_unknown_urls_and_remote_origins_cannot_read_local_files(preview):
    _, server, url = preview
    for target, headers, code in [(url + "unknown", {}, 404),
                                  (url, {"Origin": "https://example.com"}, 403),
                                  (url, {"Host": "example.com"}, 403)]:
        with pytest.raises(urllib.error.HTTPError) as exc:
            request(target, **headers)
        assert exc.value.code == code
