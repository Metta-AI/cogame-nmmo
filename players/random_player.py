"""Uniform-random cogame-nmmo player: ``python -m players.random_player``.

Every tick, every hero plays a uniform-random action over the sim's
MultiDiscrete space. ``COGAME_PLAYER_SEED`` (optional) seeds the numpy
Generator for deterministic play in tests.
"""

from __future__ import annotations

import sys

import numpy as np

from .client import run_policy_main, seed_from_env

# MultiDiscrete action highs (exclusive), per column: vel_y, vel_x,
# target-filter, use_q, use_w, use_e. Duplicated from
# cogame_nmmo.defaults.ACT_HIGH (tested equal) so players/ stays
# importable without the server package.
ACT_HIGH = (7, 7, 3, 2, 2, 2)
ACTIONS_PER_HERO = len(ACT_HIGH)


class RandomPolicy:
    """policy(tick, obs_rows) -> uniform-random in-range actions."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, tick: int, obs_rows: list) -> list:
        return self.rng.integers(
            0, ACT_HIGH, size=(len(obs_rows), ACTIONS_PER_HERO)).tolist()


def policy_from_env() -> RandomPolicy:
    return RandomPolicy(seed_from_env(default=None))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
