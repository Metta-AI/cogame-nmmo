"""Binary replay format v1: header JSON + packed per-tick actions.

Layout:

    bytes 0-3   magic b"MOBA"
    byte  4     format version, u8 = 1
    bytes 5-8   header_len, u32 little-endian
    ...         header JSON, utf-8, header_len bytes
    ...         body: tick_count * 60 bytes
                (per tick: 10 heroes x 6 uint8 action values, post-clamp,
                 exactly as fed to the sim)

Header JSON keys: format_version, sim_wasm_sha256, config (fully resolved
game config incl. seed and player names — names MUST live in the replay
bytes per the Coworld static-viewer contract; tokens excluded), result
(filled at finalize), tick_count.

A replay plus the pinned sim wasm fully determines the episode: re-run
the sim from ``config.seed`` feeding each tick's actions and every obs/
reward byte reproduces (see tests/test_replay.py re-simulation test).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import defaults
from .config import GameConfig
from .sim import DEFAULT_WASM_PATH

MAGIC = b"MOBA"
FORMAT_VERSION = 1
_PREFIX_LEN = len(MAGIC) + 1 + 4  # magic + version u8 + header_len u32le
BYTES_PER_TICK = defaults.NUM_HEROES * defaults.ACTIONS_PER_HERO  # 60


class ReplayError(ValueError):
    """Malformed or unsupported replay bytes."""


def sim_wasm_sha256(wasm_path: str | Path = DEFAULT_WASM_PATH) -> str:
    """Hex sha256 of the sim wasm binary (recorded in replay headers)."""
    return hashlib.sha256(Path(wasm_path).read_bytes()).hexdigest()


class ReplayWriter:
    """Accumulates per-tick actions; finalize() renders the full file.

    Body is buffered in memory: episodes are at most 40000 ticks x 60 B
    = 2.4 MB, so streaming to disk buys nothing.
    """

    def __init__(self, config: GameConfig, sim_wasm_sha256: str):
        self._config = config
        self._sha = sim_wasm_sha256
        self._body = bytearray()
        self._tick_count = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def append_tick(self, tick: int, actions: np.ndarray) -> None:
        """Record one tick's (10, 6) post-clamp action matrix.

        Signature matches the engine's ``on_tick`` hook; ``tick`` must be
        the next sequential tick (catches skipped/duplicated ticks).
        """
        if tick != self._tick_count:
            raise ValueError(
                f"non-sequential tick {tick}, expected {self._tick_count}")
        actions = np.asarray(actions)
        if actions.shape != (defaults.NUM_HEROES, defaults.ACTIONS_PER_HERO):
            raise ValueError(
                f"actions must be ({defaults.NUM_HEROES}, "
                f"{defaults.ACTIONS_PER_HERO}), got {actions.shape}")
        self._body += actions.astype(np.uint8).tobytes()
        self._tick_count += 1

    def finalize(self, result: dict) -> bytes:
        """Render the complete replay file with the episode result."""
        header = json.dumps({
            "format_version": FORMAT_VERSION,
            "sim_wasm_sha256": self._sha,
            "config": self._config.to_dict(),
            "result": result,
            "tick_count": self._tick_count,
        }, separators=(",", ":")).encode("utf-8")
        return b"".join((
            MAGIC,
            bytes([FORMAT_VERSION]),
            len(header).to_bytes(4, "little"),
            header,
            bytes(self._body),
        ))


class Replay:
    """Parsed replay: validated header + per-tick action access."""

    def __init__(self, header: dict, body: bytes):
        self.header = header
        self._body = body
        self.tick_count = header["tick_count"]

    @classmethod
    def parse(cls, data: bytes) -> "Replay":
        if len(data) < _PREFIX_LEN:
            raise ReplayError(f"replay too short ({len(data)} bytes)")
        if data[:4] != MAGIC:
            raise ReplayError(f"bad magic {data[:4]!r}, expected {MAGIC!r}")
        if data[4] != FORMAT_VERSION:
            raise ReplayError(
                f"unsupported format version {data[4]}, "
                f"expected {FORMAT_VERSION}")
        header_len = int.from_bytes(data[5:9], "little")
        if _PREFIX_LEN + header_len > len(data):
            raise ReplayError("header extends past end of file")
        try:
            header = json.loads(data[_PREFIX_LEN:_PREFIX_LEN + header_len])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReplayError(f"header is not valid JSON: {exc}") from exc
        if not isinstance(header, dict) or \
                not isinstance(header.get("tick_count"), int):
            raise ReplayError("header missing integer tick_count")
        body = data[_PREFIX_LEN + header_len:]
        expected = header["tick_count"] * BYTES_PER_TICK
        if len(body) != expected:
            raise ReplayError(
                f"body is {len(body)} bytes, expected {expected} "
                f"({header['tick_count']} ticks x {BYTES_PER_TICK})")
        return cls(header, body)

    def actions(self, tick: int) -> np.ndarray:
        """The (10, 6) uint8 action matrix for one tick."""
        if not 0 <= tick < self.tick_count:
            raise IndexError(f"tick {tick} out of range 0..{self.tick_count - 1}")
        start = tick * BYTES_PER_TICK
        return np.frombuffer(
            self._body, dtype=np.uint8,
            count=BYTES_PER_TICK, offset=start,
        ).reshape(defaults.NUM_HEROES, defaults.ACTIONS_PER_HERO)

    def __iter__(self):
        for tick in range(self.tick_count):
            yield self.actions(tick)

    def __len__(self) -> int:
        return self.tick_count
