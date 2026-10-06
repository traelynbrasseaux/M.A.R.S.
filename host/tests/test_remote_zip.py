import io
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

from mhm.datasets import remote_zip


@pytest.fixture
def served_zip(monkeypatch: pytest.MonkeyPatch) -> zipfile.ZipFile:
    """An in-memory zip whose bytes stand in for the remote server's ranged responses."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("a/stored.bin", bytes(range(256)) * 4, compress_type=zipfile.ZIP_STORED)
        z.writestr("a/deflated.csv", "1.0,2.0\n" * 5000, compress_type=zipfile.ZIP_DEFLATED)
    blob = buffer.getvalue()
    monkeypatch.setattr(remote_zip, "fetch_range", lambda url, start, end: blob[start : end + 1])
    return zipfile.ZipFile(io.BytesIO(blob))


def test_fetch_member_matches_zipfile(served_zip: zipfile.ZipFile):
    for info in served_zip.infolist():
        assert remote_zip.fetch_member("url", info) == served_zip.read(info)


def test_extract_skips_complete_files(served_zip: zipfile.ZipFile, tmp_path: Path):
    members = served_zip.infolist()
    remote_zip.extract("url", members, tmp_path)
    assert (tmp_path / "a" / "deflated.csv").read_text() == "1.0,2.0\n" * 5000
    (tmp_path / "a" / "stored.bin").write_bytes(b"truncated")
    remote_zip.extract("url", members, tmp_path)
    assert (tmp_path / "a" / "stored.bin").read_bytes() == bytes(range(256)) * 4


def test_fetch_range_waits_out_rate_limit(monkeypatch: pytest.MonkeyPatch):
    class Response(io.BytesIO):
        status = 206

    calls, sleeps = [], []

    def fake_urlopen(request):
        calls.append(request)
        if len(calls) < 3:
            raise HTTPError(request.full_url, 429, "TOO MANY REQUESTS", {"Retry-After": "7"}, None)
        return Response(b"abc")

    monkeypatch.setattr(remote_zip, "urlopen", fake_urlopen)
    monkeypatch.setattr(remote_zip.time, "sleep", sleeps.append)
    assert remote_zip.fetch_range("http://example.invalid/x.zip", 0, 2) == b"abc"
    assert sleeps == [7, 7]
