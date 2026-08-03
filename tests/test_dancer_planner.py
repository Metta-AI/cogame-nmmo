"""L2 DuelPlanner gate vs MiniDuel — a pure-python mirror of the
verified duel rules (chase 33/33, players first, damage formulas,
attack arcs). The planner must rediscover:

- the diagonal sword dance: kill any hittable melee with ZERO damage
  taken in open terrain;
- the safe-stall escape: zero damage from an unhittable melee;
- the bare-hands trade: win the bootstrap fight at minimum cost;
- corner play: no catastrophe when boxed against walls.
"""

from __future__ import annotations

import pytest

from players.dancer.planner import plan_duel
from players.scripted_player import ATN_ATTACK, MOVE_DELTAS, RUN_OFFSET


class MiniDuel:
    """Exact local duel sim. Coordinates unbounded; `walls` blocks."""

    def __init__(self, ppos, epos, e_hp=99, p_hp=99, our_dmg=30,
                 their_dmg=40, sword=True, walls=frozenset()):
        self.p = tuple(ppos)
        self.e = tuple(epos)
        self.e_hp = e_hp
        self.p_hp = p_hp
        self.our_dmg = our_dmg
        self.their_dmg = their_dmg
        self.sword = sword
        self.walls = set(walls)
        self.hits_dealt = 0
        self.dmg_taken = 0

    def passable(self, r, c):
        return (r, c) not in self.walls

    def in_arc(self, pr, pc, r, c):
        ar, ac = abs(r - pr), abs(c - pc)
        if ar + ac == 1:
            return True
        return self.sword and ar == 1 and ac == 1

    def step(self, action):
        # player first
        if action == ATN_ATTACK:
            if self.in_arc(*self.p, *self.e) and self.e_hp > 0:
                self.e_hp -= self.our_dmg
                self.hits_dealt += 1
        else:
            walk = action
            run = False
            if walk - RUN_OFFSET in MOVE_DELTAS:
                walk -= RUN_OFFSET
                run = True
            if walk in MOVE_DELTAS:
                dr, dc = MOVE_DELTAS[walk]
                steps = 2 if run else 1
                nr, nc = self.p[0] + dr * steps, self.p[1] + dc * steps
                mid = (self.p[0] + dr, self.p[1] + dc)
                if self.passable(*mid) and mid != self.e and \
                        (not run or (self.passable(nr, nc)
                                     and (nr, nc) != self.e)):
                    self.p = (nr, nc)
        if self.e_hp <= 0:
            return True
        # enemy response (exact chase rule)
        dr = self.p[0] - self.e[0]
        dc = self.p[1] - self.e[1]
        if abs(dr) + abs(dc) == 1:
            self.p_hp -= self.their_dmg
            self.dmg_taken += self.their_dmg
        else:
            if abs(dr) > abs(dc):
                ne = (self.e[0] + (1 if dr > 0 else -1), self.e[1])
            else:
                ne = (self.e[0], self.e[1] + (1 if dc > 0 else -1))
            if self.passable(*ne) and ne != self.p:
                self.e = ne
        return self.e_hp <= 0


def drive(duel: MiniDuel, intent_kill: bool, ticks: int = 80):
    for _t in range(ticks):
        act = plan_duel(duel.p, duel.e, duel.e_hp, duel.our_dmg,
                        duel.their_dmg, duel.sword, intent_kill,
                        duel.passable)
        done = duel.step(act)
        if done:
            return True
    return duel.e_hp <= 0


def test_dance_kills_without_damage():
    """Sword + hittable melee in the open: kill, zero damage taken."""
    for start in ((3, 0), (0, 3), (2, 2), (4, 1), (-3, -2)):
        duel = MiniDuel((0, 0), start, our_dmg=30, their_dmg=45,
                        sword=True)
        killed = drive(duel, intent_kill=True)
        assert killed, f"no kill from {start} (e_hp={duel.e_hp})"
        assert duel.dmg_taken == 0, \
            f"took {duel.dmg_taken} dmg dancing from {start}"


def test_dance_efficiency():
    """The dance lands a hit at least every 4 ticks on average."""
    duel = MiniDuel((0, 0), (3, 1), our_dmg=25, their_dmg=45, sword=True)
    ticks = 0
    for _t in range(80):
        act = plan_duel(duel.p, duel.e, duel.e_hp, duel.our_dmg,
                        duel.their_dmg, True, True, duel.passable)
        ticks += 1
        if duel.step(act):
            break
    hits_needed = -(-99 // 25)
    assert duel.e_hp <= 0
    assert ticks <= hits_needed * 4 + 6, f"kill took {ticks} ticks"
    assert duel.dmg_taken == 0


def test_escape_unhittable():
    """our_dmg=0 vs a big melee: zero damage over a long pursuit."""
    duel = MiniDuel((0, 0), (2, 1), our_dmg=0, their_dmg=80, sword=False)
    for _t in range(60):
        act = plan_duel(duel.p, duel.e, duel.e_hp, 0, 80, False, False,
                        duel.passable)
        duel.step(act)
    assert duel.dmg_taken == 0, f"took {duel.dmg_taken} while escaping"


def test_bare_hands_trade():
    """Bootstrap: no sword, winnable trade. Kill at sane cost."""
    duel = MiniDuel((0, 0), (1, 0), our_dmg=37, their_dmg=18, sword=False)
    killed = drive(duel, intent_kill=True)
    assert killed
    # 3 hits needed; the enemy can land at most ~4 (first-strike traded)
    assert duel.dmg_taken <= 4 * 18, f"trade cost {duel.dmg_taken}"


def test_corner_no_catastrophe():
    """Boxed against a wall with a strong melee: the planner may not
    find zero damage, but must not stand still and tank forever."""
    walls = {(r, -2) for r in range(-4, 5)} | \
            {(r, 2) for r in range(-4, 5)} | \
            {(-3, c) for c in range(-2, 3)}
    duel = MiniDuel((0, 0), (2, 0), our_dmg=20, their_dmg=50,
                    sword=True, walls=walls)
    for _t in range(50):
        act = plan_duel(duel.p, duel.e, duel.e_hp, 20, 50, True, True,
                        duel.passable, )
        if duel.step(act):
            break
    # corridor: the dance degrades to hit-and-step; cost bounded well
    # under stand-and-tank (which would be ~5 hits x 50 = 250)
    assert duel.e_hp <= 0 or duel.dmg_taken <= 150
