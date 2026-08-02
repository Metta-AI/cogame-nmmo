"""Game config model for the Coworld runtime contract.

The config JSON arrives via ``COGAME_CONFIG_URI``. Shape (paintarena/
coworld-ctf conventions: ``players`` and ``tokens`` are parallel arrays in
seat-slot order):

    {
      "seed": 1234,                          // optional; derived if absent
      "max_ticks": 5000,
      "heroes_per_seat": 1,                  // agents per seat (default 1)
      "tick_deadline_ms": 1000,
      "player_connect_timeout_seconds": 180,
      "wall_clock_budget_seconds": 3240,   // optional; derived if absent
      "players": [{"name": "..."}, ...],
      "tokens": ["token-0", ...]
    }

A missing seed is derived once at parse time and recorded on the resolved
config so it always reaches the replay header.
"""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from pathlib import Path

from . import defaults


class ConfigError(ValueError):
    """Invalid or inconsistent game config."""


@dataclass(frozen=True)
class PlayerConfig:
    name: str


@dataclass(frozen=True)
class GameConfig:
    players: tuple[PlayerConfig, ...]
    tokens: tuple[str, ...]
    seed: int
    max_ticks: int
    heroes_per_seat: int
    tick_deadline_ms: int
    player_connect_timeout_seconds: float
    # Engine hard stop (end_reason="wall_clock"): keeps the worst-case
    # episode (max_ticks x tick_deadline can reach hours) under the
    # platform's episode_timeout kill, so artifacts are always written.
    wall_clock_budget_seconds: float

    @property
    def num_seats(self) -> int:
        return len(self.players)

    @property
    def num_agents(self) -> int:
        """Total sim agents: seats x heroes_per_seat (NMMO3's num_agents
        is elastic upstream; the default variant is 8 seats x 1)."""
        return defaults.num_agents(self.num_seats, self.heroes_per_seat)

    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        if not isinstance(data, dict):
            raise ConfigError(f"config must be a JSON object, got {type(data).__name__}")

        players_raw = data.get("players")
        if not isinstance(players_raw, list) or not players_raw:
            raise ConfigError("config requires a non-empty 'players' array")
        players = []
        for i, entry in enumerate(players_raw):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) \
                    or not entry["name"]:
                raise ConfigError(f"players[{i}] must be an object with a non-empty 'name'")
            players.append(PlayerConfig(name=entry["name"]))

        tokens_raw = data.get("tokens")
        if not isinstance(tokens_raw, list) or \
                not all(isinstance(t, str) and t for t in tokens_raw):
            raise ConfigError("config requires a 'tokens' array of non-empty strings")
        if len(tokens_raw) != len(players):
            raise ConfigError(
                f"tokens length {len(tokens_raw)} != players length {len(players)}")

        heroes_per_seat = _int_field(
            data, "heroes_per_seat", defaults.DEFAULT_HEROES_PER_SEAT)
        if heroes_per_seat < 1:
            raise ConfigError(
                f"heroes_per_seat must be a positive integer, "
                f"got {heroes_per_seat}")
        total_agents = defaults.num_agents(len(players), heroes_per_seat)
        if total_agents > defaults.MAX_TOTAL_AGENTS:
            raise ConfigError(
                f"players x heroes_per_seat = {len(players)} x "
                f"{heroes_per_seat} = {total_agents} sim agents exceeds the "
                f"cap of {defaults.MAX_TOTAL_AGENTS} (upstream NMMO3 trains "
                f"num_agents=1024, vendor/upstream/nmmo3.ini; the replay "
                f"viewer rejects larger replays at the same bound)")

        max_ticks = _int_field(data, "max_ticks", defaults.DEFAULT_MAX_TICKS)
        if max_ticks <= 0:
            raise ConfigError(f"max_ticks must be positive, got {max_ticks}")

        tick_deadline_ms = _int_field(
            data, "tick_deadline_ms", defaults.DEFAULT_TICK_DEADLINE_MS)
        if tick_deadline_ms <= 0:
            raise ConfigError(
                f"tick_deadline_ms must be positive, got {tick_deadline_ms}")

        timeout = data.get("player_connect_timeout_seconds",
                           defaults.DEFAULT_PLAYER_CONNECT_TIMEOUT_SECONDS)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) \
                or not math.isfinite(timeout) or timeout < 0:
            raise ConfigError(
                "player_connect_timeout_seconds must be a finite non-negative "
                f"number, got {timeout!r}")

        budget = data.get(
            "wall_clock_budget_seconds",
            defaults.derived_wall_clock_budget_seconds(
                max_ticks, tick_deadline_ms))
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) \
                or not math.isfinite(budget) or budget <= 0:
            raise ConfigError(
                "wall_clock_budget_seconds must be a finite positive number, "
                f"got {budget!r}")

        seed = data.get("seed")
        if seed is None:
            # Derive once and record it: the seed must reach the replay header.
            seed = secrets.randbits(32)
        elif not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError(f"seed must be an integer, got {seed!r}")

        return cls(
            players=tuple(players),
            tokens=tuple(tokens_raw),
            seed=seed,
            max_ticks=max_ticks,
            heroes_per_seat=heroes_per_seat,
            tick_deadline_ms=tick_deadline_ms,
            player_connect_timeout_seconds=float(timeout),
            wall_clock_budget_seconds=float(budget),
        )

    @classmethod
    def from_file_uri(cls, uri: str) -> "GameConfig":
        """Parse a config from a local ``file://`` URI or plain path.

        Local-only convenience (tests, dev). The server reads
        ``COGAME_CONFIG_URI`` through :mod:`cogame_nmmo.uris`, which also
        supports http(s).
        """
        path = uri.removeprefix("file://")
        try:
            raw = Path(path).read_text()
        except OSError as exc:
            raise ConfigError(f"cannot read config from {uri}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config at {uri} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Fully-resolved config for the replay header and results.

        Tokens are deliberately excluded: replays and results are public
        artifacts, tokens are per-episode player credentials.
        """
        return {
            "seed": self.seed,
            "max_ticks": self.max_ticks,
            "heroes_per_seat": self.heroes_per_seat,
            "tick_deadline_ms": self.tick_deadline_ms,
            "player_connect_timeout_seconds": self.player_connect_timeout_seconds,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "players": [{"name": p.name} for p in self.players],
        }


def _int_field(data: dict, key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer, got {value!r}")
    return value
