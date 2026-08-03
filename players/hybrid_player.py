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
from .scripted_player import (ATN_ATTACK, ATN_DOWN, ATN_LEFT, ATN_NOOP,
                              ATN_ONE, ATN_RIGHT, ATN_UP, CENTER_COL,
                              CENTER_ROW, I_GEM_TYPES, I_HERB, I_HILT, I_ORE,
                              I_TOOL, I_WOOD, MOVE_DELTAS, NUM_KEY_SLOTS,
                              Percept, item_type, tier_level)

STUCK_TICKS = 250        # min(comb,prof) still 1 after this many life
                         # ticks -> zero the net's recurrent state
STUCK_REARM = 120        # wait this long before another state reset
HERB_HP = 35             # eat a herb below this hp
STAG_WINDOW = 500        # the sim's strict-improvement window (c_step)
DEADLINE_TICKS = 130     # take over this close to a predicted stagnation
                         # reset when a rescue harvest is visible


class _AgentClock:
    def __init__(self):
        self.reset()

    def reset(self):
        self.best_min = 0
        self.since_improve = 0
        self.since_state_reset = 0
        self.life_tick = 0
        self.window_start_min = 1


def _deadline_rescue(p: Percept, clock: "_AgentClock"):
    """Scripted takeover near a predicted stagnation reset.

    The sim force-resets a life (banking its score and wiping gear) when
    min(comb, prof) at a 500-life-tick boundary fails to exceed the value
    at the previous boundary. For an agent at min m that is a ~m/2 score
    haircut - so once the deadline is close and this window shows no
    improvement, ANY takeover with positive rescue probability beats
    letting the net idle into the reset.

    Rescue move: if prof is the binding skill and a prof-leveling
    resource tile within our tool's tier is visible, step toward it
    (greedy, ignoring non-adjacent threats - the reset is certain, the
    danger is not). If comb is binding and a weak enemy is adjacent,
    attack it. Returns an action or None.
    """
    m = min(p.comb_lvl, p.prof_lvl)
    if m <= clock.window_start_min:
        remaining = STAG_WINDOW - (clock.life_tick % STAG_WINDOW)
    else:
        return None
    if remaining > DEADLINE_TICKS:
        return None
    held_tier = p.held_tool_tier
    if p.prof_lvl <= p.comb_lvl and held_tier > 0:
        best = None
        for r, c, itype, tier in p.item_tiles():
            if itype not in (I_ORE, I_WOOD, I_HILT, I_HERB) and                     itype not in I_GEM_TYPES:
                continue
            if tier > held_tier or p.prof_lvl >= tier_level(tier):
                continue
            d = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
            if best is None or d < best[0]:
                best = (d, r, c)
        if best is not None and not all(p.inventory):
            _d, r, c = best
            dr, dc = r - CENTER_ROW, c - CENTER_COL
            prefs = []
            if abs(dr) >= abs(dc):
                if dr:
                    prefs.append(ATN_DOWN if dr > 0 else ATN_UP)
                if dc:
                    prefs.append(ATN_RIGHT if dc > 0 else ATN_LEFT)
            else:
                prefs.append(ATN_RIGHT if dc > 0 else ATN_LEFT)
                if dr:
                    prefs.append(ATN_DOWN if dr > 0 else ATN_UP)
            for atn in prefs:
                nr = CENTER_ROW + MOVE_DELTAS[atn][0]
                nc = CENTER_COL + MOVE_DELTAS[atn][1]
                if p.passable(nr, nc):
                    return atn
    if p.comb_lvl <= p.prof_lvl:
        for r, c, d, _hpb in p.enemy_hints():
            if d <= 1 and abs(r - CENTER_ROW) + abs(c - CENTER_COL) == 1:
                return ATN_ATTACK
    return None


class HybridPolicy:
    """policy(tick, obs_rows, resets): hero i in the seat -> brain i."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_agents: int = DEFAULT_NUM_BRAINS,
                 wasm_path=DEFAULT_BRAIN_WASM_PATH,
                 enable_stuck_reset: bool = True,
                 enable_rescue: bool = True,
                 enable_herb: bool = True):
        self.enable_stuck_reset = enable_stuck_reset
        self.enable_rescue = enable_rescue
        self.enable_herb = enable_herb
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
            if clock.life_tick % STAG_WINDOW == 0:
                clock.window_start_min = m
            clock.life_tick += 1
            # Reset only TRULY stuck agents (still min 1 deep into the
            # life): a progressing agent's recurrent memory is an asset
            # - never disturb it, even when its progress stalls late.
            if self.enable_stuck_reset and not resets[i] and \
                    clock.best_min <= 1 and \
                    clock.since_improve >= STUCK_TICKS and \
                    clock.since_state_reset >= STUCK_REARM:
                self.brain.reset_state(i)
                clock.since_state_reset = 0

            act = self.brain.forward(i, obs)

            rescue = _deadline_rescue(p, clock) \
                if self.enable_rescue else None
            if rescue is not None:
                self.counters = getattr(self, "counters", {})
                self.counters["rescue"] = self.counters.get("rescue", 0) + 1
                act = [rescue]

            # emergency herb: keep the net's state advanced (already
            # done) but play the heal instead of its action
            if self.enable_herb and p.hp < HERB_HP:
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
