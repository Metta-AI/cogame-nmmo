"""Server-level config defaults and seat/agent topology helpers.

Env-physics values that mirror upstream training defaults (world size,
entity counts, reward weights) live in sim/shim_common.h ``nmmo_configure``
— not here. This module owns the *server* contract: config defaults, the
no-op action, and how player seats map onto the sim's agent slots.

Seat -> agent mapping: seat ``i`` controls agents ``[i*h, (i+1)*h)`` where
``h = heroes_per_seat``. NMMO3's ``num_agents`` is elastic upstream, so
the total agent count is ``num_seats * heroes_per_seat`` (the default
variant is 8 seats x 1 agent, free-for-all). The generalized
heroes-per-seat machinery is inherited from the moba fork and kept: a
future variant may hand one policy several agents.
"""

from __future__ import annotations

import numpy as np

# One 26-way discrete action per agent per tick.
ACTIONS_PER_AGENT = 1

# Discrete action high (exclusive). Mirrors cogame_nmmo.sim.ACT_HIGH
# (tested equal); duplicated so transport code needs no wasmtime import.
ACT_HIGH = (26,)
# No-op: ATN_NOOP (vendored nmmo3.h:50).
NOOP_ACTION = (4,)

# The default VARIANT is 8 seats x 1 agent (FFA), but seat count is not a
# server default: it always comes from the config's players array (the
# manifest's variant definitions own the 8).
DEFAULT_MAX_TICKS = 5000
DEFAULT_HEROES_PER_SEAT = 1
DEFAULT_TICK_DEADLINE_MS = 1000
DEFAULT_PLAYER_CONNECT_TIMEOUT_SECONDS = 180

# Mirrors the manifest's top-level episode_timeout_minutes (the platform
# kills the container at that point, losing results and replay). The
# default wall-clock budget is derived from it:
# min(0.9 x this, max_ticks x tick_deadline) — so a slow episode ends
# itself, with artifacts written, before the platform kill. Keep in sync
# with coworld_manifest_template.json.
PLATFORM_EPISODE_TIMEOUT_MINUTES = 60


def derived_wall_clock_budget_seconds(max_ticks: int,
                                      tick_deadline_ms: int) -> float:
    """Default wall-clock budget (see PLATFORM_EPISODE_TIMEOUT_MINUTES).

    With the 5000-tick default and a 1s deadline the worst case (5000s)
    exceeds the platform timeout, so the 0.9 x 60min = 3240s cap wins.
    """
    return min(0.9 * PLATFORM_EPISODE_TIMEOUT_MINUTES * 60,
               max_ticks * tick_deadline_ms / 1000.0)


def num_agents(num_seats: int, heroes_per_seat: int) -> int:
    """Total sim agents for a variant (seats x agents-per-seat)."""
    return num_seats * heroes_per_seat


def seat_hero_pids(seat: int, heroes_per_seat: int) -> range:
    """Agent pids controlled by ``seat``: [seat*h, (seat+1)*h)."""
    return range(seat * heroes_per_seat, (seat + 1) * heroes_per_seat)


def seat_for_pid(pid: int, heroes_per_seat: int) -> int:
    """The seat controlling agent ``pid`` (inverse of seat_hero_pids)."""
    return pid // heroes_per_seat


def clamp_actions(actions: np.ndarray) -> np.ndarray:
    """Sanitize finite action values exactly as the sim boundary does.

    Truncate toward zero (the sim's C ``(int)`` cast) and clamp to
    ``0 .. 25``. Returns uint8 (max action value is 25) — the post-clamp
    form stored in replays. Mirrors NmmoSim.set_actions.
    """
    high = np.asarray(ACT_HIGH, dtype=np.float64) - 1.0
    return np.clip(np.trunc(np.asarray(actions, dtype=np.float64)),
                   0.0, high).astype(np.uint8)
