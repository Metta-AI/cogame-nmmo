"""Read/write Coworld artifact URIs: file:// and http(s)://.

Conventions mirror the Coworld runtime contract (bitworld runtime /
Cookbook "Raw Docker Shape"):

- ``file://`` URIs strip the scheme and keep the absolute path, so the
  platform's mount pattern ``file:///coworld/out/results.json`` resolves
  to ``/coworld/out/results.json``. Parent directories are created on
  write.
- Plain scheme-less strings are treated as local paths.
- ``http(s)://`` reads are GET, writes are PUT with a Content-Type
  header (signed URLs on the hosted platform). Non-2xx raises. Writes
  are bounded (30s request timeout) and retried a few times with a short
  backoff before giving up — artifacts are the episode's whole point.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp

_HTTP_SCHEMES = ("http://", "https://")

WRITE_ATTEMPTS = 3
WRITE_BACKOFF_SECONDS = 0.5
WRITE_TIMEOUT_SECONDS = 30.0


def local_path(uri: str) -> Path | None:
    """The local path for a file:// URI or plain path, else None."""
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    if "://" not in uri:
        return Path(uri)
    return None


async def read_uri(uri: str) -> bytes:
    path = local_path(uri)
    if path is not None:
        return path.read_bytes()
    if uri.startswith(_HTTP_SCHEMES):
        async with aiohttp.ClientSession() as session:
            async with session.get(uri) as resp:
                if not 200 <= resp.status < 300:
                    raise IOError(
                        f"GET {uri} failed with status {resp.status}")
                return await resp.read()
    raise ValueError(f"unsupported URI scheme: {uri}")


async def write_uri(uri: str, data: bytes,
                    content_type: str = "application/octet-stream", *,
                    attempts: int = WRITE_ATTEMPTS,
                    backoff_seconds: float = WRITE_BACKOFF_SECONDS,
                    timeout_seconds: float = WRITE_TIMEOUT_SECONDS) -> None:
    path = local_path(uri)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if uri.startswith(_HTTP_SCHEMES):
        last_error: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(backoff_seconds * attempt)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.put(
                            uri, data=data,
                            headers={"Content-Type": content_type}) as resp:
                        if 200 <= resp.status < 300:
                            return
                        body = (await resp.text())[:200]
                        last_error = IOError(
                            f"PUT {uri} failed with status "
                            f"{resp.status}: {body}")
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_error = IOError(f"PUT {uri} failed: {exc!r}")
        raise last_error
    raise ValueError(f"unsupported URI scheme: {uri}")
