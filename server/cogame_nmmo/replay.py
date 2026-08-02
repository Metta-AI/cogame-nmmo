"""Binary replay format v1: header JSON + packed per-tick actions.

Layout:

    bytes 0-3   magic b"NMMO"
    byte  4     format version, u8 = 1
    bytes 5-8   header_len, u32 little-endian
    ...         header JSON, utf-8, header_len bytes
    ...         body: tick_count * num_agents bytes
                (per tick: one uint8 26-way action per agent, post-clamp,
                 exactly as fed to the sim; num_agents comes from the
                 header config: len(players) x heroes_per_seat)

Header JSON keys: format_version, sim_wasm_sha256, config (fully resolved
game config incl. seed and player names — names MUST live in the replay
bytes per the Coworld static-viewer contract; tokens excluded), result
(filled at finalize), tick_count.

A replay plus the pinned sim wasm fully determines the episode: re-run
the sim from ``config.seed`` feeding each tick's actions and every obs/
reward byte and the state digest reproduce (see tests/test_replay.py
re-simulation test).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import defaults
from .config import GameConfig
from .sim import DEFAULT_WASM_PATH

MAGIC = b"NMMO"
FORMAT_VERSION = 1
_PREFIX_LEN = len(MAGIC) + 1 + 4  # magic + version u8 + header_len u32le
# Highest valid action value (inclusive): ACT_HIGH is exclusive.
_MAX_ACTION = defaults.ACT_HIGH[0] - 1


class ReplayError(ValueError):
    """Malformed or unsupported replay bytes."""


def sim_wasm_sha256(wasm_path: str | Path = DEFAULT_WASM_PATH) -> str:
    """Hex sha256 of the sim wasm binary (recorded in replay headers)."""
    return hashlib.sha256(Path(wasm_path).read_bytes()).hexdigest()


class ReplayWriter:
    """Accumulates per-tick actions; finalize() renders the full file.

    Body is buffered in memory: episodes are at most max_ticks x
    num_agents bytes (5000 x 8 = 40 kB at the defaults), so streaming to
    disk buys nothing.
    """

    def __init__(self, config: GameConfig, sim_wasm_sha256: str):
        self._config = config
        self._sha = sim_wasm_sha256
        self._num_agents = config.num_agents
        self._body = bytearray()
        self._tick_count = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def append_tick(self, tick: int, actions: np.ndarray) -> None:
        """Record one tick's (num_agents, 1) post-clamp action matrix.

        Signature matches the engine's ``on_tick`` hook; ``tick`` must be
        the next sequential tick (catches skipped/duplicated ticks).
        Values are range-checked (0..25): the engine only ever feeds
        post-clamp actions, so an out-of-range value here is a bug, not
        player input.
        """
        if tick != self._tick_count:
            raise ValueError(
                f"non-sequential tick {tick}, expected {self._tick_count}")
        actions = np.asarray(actions)
        if actions.shape != (self._num_agents, defaults.ACTIONS_PER_AGENT):
            raise ValueError(
                f"actions must be ({self._num_agents}, "
                f"{defaults.ACTIONS_PER_AGENT}), got {actions.shape}")
        if actions.min() < 0 or actions.max() > _MAX_ACTION:
            raise ValueError(
                f"action values out of range 0..{_MAX_ACTION}: "
                f"min {actions.min()}, max {actions.max()} "
                f"(replays store post-clamp actions only)")
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

    def __init__(self, header: dict, body: bytes, num_agents: int):
        self.header = header
        self._body = body
        self.num_agents = num_agents
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
        num_agents = cls._header_num_agents(header)
        body = data[_PREFIX_LEN + header_len:]
        expected = header["tick_count"] * num_agents
        if len(body) != expected:
            raise ReplayError(
                f"body is {len(body)} bytes, expected {expected} "
                f"({header['tick_count']} ticks x {num_agents} agents)")
        return cls(header, body, num_agents)

    @staticmethod
    def _header_num_agents(header: dict) -> int:
        """num_agents from the header config (players x heroes_per_seat).

        The body stride depends on it, so a header that can't produce a
        positive agent count is malformed.
        """
        config = header.get("config")
        if not isinstance(config, dict):
            raise ReplayError("header missing config object")
        players = config.get("players")
        heroes = config.get("heroes_per_seat")
        if not isinstance(players, list) or not players or \
                not isinstance(heroes, int) or isinstance(heroes, bool) or \
                heroes < 1:
            raise ReplayError(
                "header config must carry a non-empty players array and a "
                "positive integer heroes_per_seat (they set the body stride)")
        return len(players) * heroes

    def actions(self, tick: int) -> np.ndarray:
        """The (num_agents, 1) uint8 action matrix for one tick."""
        if not 0 <= tick < self.tick_count:
            raise IndexError(f"tick {tick} out of range 0..{self.tick_count - 1}")
        start = tick * self.num_agents
        return np.frombuffer(
            self._body, dtype=np.uint8,
            count=self.num_agents, offset=start,
        ).reshape(self.num_agents, defaults.ACTIONS_PER_AGENT)

    def __iter__(self):
        for tick in range(self.tick_count):
            yield self.actions(tick)

    def __len__(self) -> int:
        return self.tick_count
