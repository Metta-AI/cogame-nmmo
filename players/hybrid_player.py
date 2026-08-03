"""Hybrid cogame-nmmo player: ``python -m players.hybrid_player``.

The pretrained MMONet baseline (players/baseline_player.py) plus thin
scripted overrides in states where the sim source proves a better move
exists. The recurrent state is fed every obs unchanged (it depends only
on the obs stream, never on which action was ultimately played), so
overrides do not corrupt the net.

Overrides (each measured against pure baseline via tools/local_eval.py):

1. STUCK-STATE RESET: the sim force-resets a life (banking its score
   contribution and wiping levels + gear) when min(comb, prof) fails to
   improve across a 500-tick window (c_step, nmmo3.h:1926-1956) - about
   half the baseline's lost lives are these stagnation resets, and some
   seats spend whole episodes looping at min 1. When our mirrored clock
   shows no improvement for STUCK_TICKS ticks, we zero that agent's
   MinGRU state (brain_reset_state): the net re-enters from a fresh
   behavioral mode - a free re-roll, where the sim's own reset would
   cost the whole life.

2. EMERGENCY HERB: below HERB_HP with a herb in a key slot, eat it
   (use_item works in combat and restores 50 + 10*tier hp). Deaths
   wipe gear and divide the mean score; a held herb un-eaten at death
   is pure waste.

Determinism: same contract as the baseline (COGAME_PLAYER_SEED).
"""

from __future__ import annotations

import sys

from .baseline_player import (DEFAULT_BRAIN_WASM_PATH, DEFAULT_NUM_BRAINS,
                              DEFAULT_SEED, NmmoBrain)
from .client import run_policy_main, seed_from_env
from .scripted_player import I_HERB, NUM_KEY_SLOTS, ATN_ONE, Percept, item_type

STUCK_TICKS = 350        # no min(comb,prof) improvement for this many
                         # ticks -> zero the net's recurrent state
STUCK_REARM = 150        # wait this long before another state reset
HERB_HP = 35             # eat a herb below this hp


class _AgentClock:
    def __init__(self):
        self.reset()

    def reset(self):
        self.best_min = 0
        self.since_improve = 0
        self.since_state_reset = 0


class HybridPolicy:
    """policy(tick, obs_rows, resets): hero i in the seat -> brain i."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_agents: int = DEFAULT_NUM_BRAINS,
                 wasm_path=DEFAULT_BRAIN_WASM_PATH):
        self.brain = NmmoBrain(seed=seed, num_agents=num_agents,
                               wasm_path=wasm_path)
        self.clocks = [_AgentClock() for _ in range(num_agents)]

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        actions = []
        for i, row in enumerate(obs_rows):
            clock = self.clocks[i]
            if resets[i]:
                self.brain.reset_state(i)
                clock.reset()
            obs = bytes(row)
            p = Percept(obs)

            # stuck-state bookkeeping (BEFORE the forward, so a reset
            # affects this tick's inference like the demo's terminals
            # path does)
            m = min(p.comb_lvl, p.prof_lvl)
            if m > clock.best_min:
                clock.best_min = m
                clock.since_improve = 0
            else:
                clock.since_improve += 1
            clock.since_state_reset += 1
            if not resets[i] and clock.since_improve >= STUCK_TICKS and \
                    clock.since_state_reset >= STUCK_REARM:
                self.brain.reset_state(i)
                clock.since_state_reset = 0

            act = self.brain.forward(i, obs)

            # emergency herb: keep the net's state advanced (already
            # done) but play the heal instead of its action
            if p.hp < HERB_HP:
                herb = None
                for idx, item in enumerate(p.inventory[:NUM_KEY_SLOTS]):
                    if item and item_type(item) == I_HERB:
                        herb = idx
                        break
                if herb is not None:
                    act = [ATN_ONE + herb]

            actions.append(act)
        return actions


def policy_from_env() -> HybridPolicy:
    return HybridPolicy(seed_from_env(default=DEFAULT_SEED))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
