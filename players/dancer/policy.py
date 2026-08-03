"""Wire adapter: ``python -m players.dancer.policy`` plays the dancer.

policy(tick, obs_rows, resets) protocol identical to the other players.
"""

from __future__ import annotations

import sys

from ..client import run_policy_main, seed_from_env
from .executive import DancerMind


class DancerPolicy:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.minds: dict[int, DancerMind] = {}

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        actions = []
        for i, row in enumerate(obs_rows):
            mind = self.minds.get(i)
            if mind is None:
                mind = self.minds[i] = DancerMind(self.seed, i)
            if resets[i]:
                mind.reset()
            actions.append([mind.act(bytes(row))])
        return actions


def policy_from_env() -> DancerPolicy:
    return DancerPolicy(seed_from_env(default=0))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
