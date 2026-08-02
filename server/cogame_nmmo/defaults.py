"""Server-level config defaults and seat/hero/team topology helpers.

Env-physics values that mirror upstream training defaults (vision_range,
agent_speed, reward weights) live in sim/shim.c ``moba_init`` — not here.
This module owns the *server* contract: config defaults, the no-op action,
and how player seats map onto the sim's 10 hero slots.

Hero pid -> team mapping (ground truth: vendored moba.h ``init_moba``
spawn loop, ``for (int team = 0; team < 2; team++) ... for (int pid =
team*5; pid < team*5 + 5; pid++) player->team = team``):

- pids 0-4  -> team 0 (radiant)
- pids 5-9  -> team 1 (dire)
- within each team the role order is support, assassin, burst, tank,
  carry (``pid = 5*team + {0,1,2,3,4}`` blocks in init_moba).

Seat -> hero mapping: seat ``i`` controls heroes ``[i*h, (i+1)*h)`` where
``h = heroes_per_seat``. With h=1 (default variant) seat i is hero pid i;
with h=5 (team variant) seat 0 is radiant, seat 1 is dire.
"""

from __future__ import annotations

import numpy as np

NUM_HEROES = 10
TEAM_SIZE = 5
NUM_TEAMS = 2
ACTIONS_PER_HERO = 6

# MultiDiscrete action space highs (exclusive), per column: vel_y, vel_x,
# target-filter, use_q, use_w, use_e. Mirrors cogame_nmmo.sim.ACT_HIGH
# (tested equal); duplicated so transport code needs no wasmtime import.
ACT_HIGH = (7, 7, 3, 2, 2, 2)
# No-op: center velocity (3,3 -> 0,0), scan-all filter, no skills.
NOOP_ACTION = (3, 3, 0, 0, 0, 0)

DEFAULT_MAX_TICKS = 40000
DEFAULT_HEROES_PER_SEAT = 1
DEFAULT_TICK_DEADLINE_MS = 1000
DEFAULT_PLAYER_CONNECT_TIMEOUT_SECONDS = 180
VALID_HEROES_PER_SEAT = (1, 5)

# Mirrors the manifest's top-level episode_timeout_minutes (the platform
# kills the container at that point, losing results and replay). The
# default wall-clock budget is derived from it:
# min(0.9 x this, max_ticks x tick_deadline) — so a slow episode ends
# itself, with artifacts written, before the platform kill. Keep in sync
# with coworld_manifest_template.json.
PLATFORM_EPISODE_TIMEOUT_MINUTES = 60


def derived_wall_clock_budget_seconds(max_ticks: int,
                                      tick_deadline_ms: int) -> float:
    """Default wall-clock budget (see PLATFORM_EPISODE_TIMEOUT_MINUTES)."""
    return min(0.9 * PLATFORM_EPISODE_TIMEOUT_MINUTES * 60,
               max_ticks * tick_deadline_ms / 1000.0)


def seat_count(heroes_per_seat: int) -> int:
    """Number of player seats for a variant (10 or 2)."""
    return NUM_HEROES // heroes_per_seat


def seat_hero_pids(seat: int, heroes_per_seat: int) -> range:
    """Hero pids controlled by ``seat``: [seat*h, (seat+1)*h)."""
    return range(seat * heroes_per_seat, (seat + 1) * heroes_per_seat)


def seat_for_pid(pid: int, heroes_per_seat: int) -> int:
    """The seat controlling hero ``pid`` (inverse of seat_hero_pids)."""
    return pid // heroes_per_seat


def team_for_pid(pid: int) -> int:
    """0 = radiant (pids 0-4), 1 = dire (pids 5-9); see module docstring."""
    return pid // TEAM_SIZE


def team_for_seat(seat: int, heroes_per_seat: int) -> int:
    """The team a seat plays for (a seat's heroes are all on one team)."""
    return team_for_pid(seat_hero_pids(seat, heroes_per_seat).start)


def clamp_actions(actions: np.ndarray) -> np.ndarray:
    """Sanitize finite action values exactly as the sim boundary does.

    Truncate toward zero (the sim's C ``(int)`` cast) and clamp per column
    to ``0 .. ACT_HIGH[col]-1``. Returns uint8 (max action value is 6) —
    the post-clamp form stored in replays. Mirrors MobaSim.set_actions.
    """
    high = np.asarray(ACT_HIGH, dtype=np.float64) - 1.0
    return np.clip(np.trunc(np.asarray(actions, dtype=np.float64)),
                   0.0, high).astype(np.uint8)
