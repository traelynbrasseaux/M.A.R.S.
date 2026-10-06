"""Read single members out of a large remote zip using HTTP range requests."""

import io
import itertools
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

LOCAL_HEADER_SIZE = 30
MAX_RETRIES = 6


def fetch_range(url: str, start: int, end: int) -> bytes:
    """Bytes start through end inclusive, waiting out rate limits (Zenodo returns 429)."""
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in itertools.count():
        try:
            with urlopen(request) as response:
                # A 200 here would mean the server ignored the range and is sending the whole file.
                if response.status != 206:
                    raise OSError(f"{url} ignored the range request")
                return response.read()
        except HTTPError as e:
            if e.code not in (429, 503) or attempt == MAX_RETRIES:
                raise
            retry_after = e.headers.get("Retry-After", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else 2**attempt)


class HttpRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file, enough for zipfile to read the index."""

    def __init__(self, url: str):
        self.url = url
        with urlopen(Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        match whence:
            case io.SEEK_SET:
                self.pos = offset
            case io.SEEK_CUR:
                self.pos += offset
            case io.SEEK_END:
                self.pos = self.size + offset
        return self.pos

    def readinto(self, buffer) -> int:
        end = min(self.pos + len(buffer), self.size)
        if end <= self.pos:
            return 0
        data = fetch_range(self.url, self.pos, end - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """A member's contents in two ranged GETs (local header, then data), CRC checked."""
    header = fetch_range(url, info.header_offset, info.header_offset + LOCAL_HEADER_SIZE - 1)
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = info.header_offset + LOCAL_HEADER_SIZE + name_len + extra_len
    data = fetch_range(url, start, start + info.compress_size - 1)
    match info.compress_type:
        case zipfile.ZIP_STORED:
            pass
        case zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, wbits=-15)
        case other:
            raise ValueError(f"{info.filename} uses unsupported compression {other}")
    if zlib.crc32(data) != info.CRC:
        raise OSError(f"{info.filename} failed its CRC check")
    return data


def extract(url: str, members: list[zipfile.ZipInfo], out_dir: Path, workers: int = 4) -> None:
    """Fetch members into out_dir in parallel, skipping files already present at full size."""

    def present(info: zipfile.ZipInfo) -> bool:
        target = out_dir / info.filename
        return target.exists() and target.stat().st_size == info.file_size

    todo = [m for m in members if not present(m)]

    def save(info: zipfile.ZipInfo) -> str:
        target = out_dir / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(fetch_member(url, info))
        partial.replace(target)
        return info.filename

    with ThreadPoolExecutor(workers) as pool:
        for i, name in enumerate(pool.map(save, todo), 1):
            print(f"[{i}/{len(todo)}] {name}", flush=True)


def total_gb(members: list[zipfile.ZipInfo]) -> float:
    return sum(m.compress_size for m in members) / 1e9
