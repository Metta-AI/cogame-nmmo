"""Netstrap hybrid: the pretrained net plays the bootstrap, the dancer
takes over once geared.

Measured basis (2026-08-03): 80% of dancer deaths happen at comb 1-2 in
the first 300 life-ticks - unarmored escape problems under ambiguous
sensing, the class every scripted generation died to. The net is the
proven bootstrap survivor (deaths ~1-3/3000t, geared by ~t750). The
dancer's proven domain is the geared dance-farm (planner gates: kills
with zero damage taken). Armor also neutralizes exactly the tracking
ambiguity that kills the scripted layers. So: hand off at gear, hand
back on death.

This is a PHASE-level handoff, not a micro-override of the net (the
refuted wrapper class): whoever owns the phase owns every action in it.
The dancer's world model shadows net-controlled ticks so the handoff
starts with hot tracks; the net stays warm during dancer control so a
post-death handback is sane.
"""

from __future__ import annotations

import sys

from ..baseline_player import BaselinePolicy
from ..client import run_policy_main, seed_from_env
from ..scripted_player import Percept
from .executive import DancerMind

SWITCH_DEF = 24         # >= 3 armor pieces tier1 (8 each)
SWITCH_COMB = 3

# burst style: branches the dancer may OWN (its gate-proven combat
# domain); anything else hands control back to the net
COMBAT_BRANCHES = ("engage", "herb-prefight", "herb", "loot")


class NetstrapPolicy:
    """style='sticky': dancer owns everything after the gear switch
    (until death). style='burst': the net owns economy/movement always;
    the dancer takes control only while geared, comb < prof, and its
    executive is actually in a combat branch - each burst is one clean
    dance engagement, then control returns."""

    def __init__(self, seed: int | None = None, style: str = "sticky"):
        self.seed = seed
        self.style = style
        self.nets: dict[int, BaselinePolicy] = {}
        self.minds: dict[int, DancerMind] = {}
        self.mode: dict[int, str] = {}

    def _geared(self, mind: DancerMind, p: Percept) -> bool:
        return (int(p.scalars[46]) >= SWITCH_DEF
                and mind._best_sword_tier(p) >= 1)

    def _switch_ready(self, mind: DancerMind, p: Percept) -> bool:
        return self._geared(mind, p) and p.comb_lvl >= SWITCH_COMB

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        actions = []
        for i, row in enumerate(obs_rows):
            net = self.nets.get(i)
            if net is None:
                net = self.nets[i] = BaselinePolicy(
                    seed=(self.seed or 0) + i, num_agents=1)
            mind = self.minds.get(i)
            if mind is None:
                mind = self.minds[i] = DancerMind(self.seed, i)
                self.mode[i] = "net"
            if resets[i]:
                mind.reset()
                self.mode[i] = "net"
            # the net always sees the stream (recurrent state stays
            # valid for the post-death handback)
            net_act = int(net(tick, [row], [resets[i]])[0][0])
            obs = bytes(row)
            p = Percept(obs)
            if self.style == "burst":
                act = self._burst_act(i, mind, p, obs, net_act)
            else:
                if self.mode[i] == "net" and self._switch_ready(mind, p):
                    self.mode[i] = "dancer"
                if self.mode[i] == "dancer":
                    act = mind.act(obs)
                else:
                    act = net_act
                    mind.shadow(obs, act)
            actions.append([act])
        return actions

    def _burst_act(self, i: int, mind: DancerMind, p: Percept,
                   obs: bytes, net_act: int) -> int:
        if not (self._geared(mind, p) and p.comb_lvl < p.prof_lvl):
            self.mode[i] = "net"
            mind.shadow(obs, net_act)
            return net_act
        # run the dancer executive; keep its action only when it is in
        # a combat branch (with a latch: once engaged, combat branches
        # keep control so a dance is never handed over mid-figure)
        act = mind.act(obs)
        branch = mind.branch.split("[")[0]
        if branch in COMBAT_BRANCHES:
            self.mode[i] = "dancer"
            return act
        self.mode[i] = "net"
        # dancer declined to fight: emit the net's action and repair
        # the mind's own-motion record to what we actually did
        mind.last_action = net_act
        mind.world.note_action(net_act)
        return net_act


def policy_from_env() -> NetstrapPolicy:
    return NetstrapPolicy(seed_from_env(default=0))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
