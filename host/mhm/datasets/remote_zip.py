"""Read single members out of a large remote zip using HTTP range requests."""

import io
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


class HttpRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file; each read is one ranged GET."""

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
        request = Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urlopen(request) as response:
            # A 200 here would mean the server ignored the range and is sending the whole file.
            if response.status != 206:
                raise OSError(f"{self.url} ignored the range request")
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 18))


def extract(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], out_dir: Path) -> None:
    """Extract members under out_dir, skipping files already present at full size."""
    for i, info in enumerate(members, 1):
        target = out_dir / info.filename
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with archive.open(info) as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        partial.replace(target)
        print(f"[{i}/{len(members)}] {info.filename}", flush=True)


def total_gb(members: list[zipfile.ZipInfo]) -> float:
    return sum(m.compress_size for m in members) / 1e9
