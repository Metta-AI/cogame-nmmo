"""L2 DuelPlanner gate vs MiniDuel/MiniMelee — pure-python mirrors of
the verified duel rules (chase 33/33, players first, damage formulas,
attack arcs, enemy-enemy collision blocking). The planner must
rediscover:

- the diagonal sword dance: kill any hittable melee with ZERO damage
  taken in open terrain;
- the safe-stall escape: zero damage from an unhittable melee;
- the bare-hands trade: win the bootstrap fight at minimum cost;
- corner play: no catastrophe when boxed against walls;

and plan_melee (N enemies) additionally must:

- escape a 2-chaser pincer in the open with at most one hit;
- kill a duel target while a second melee closes in, at most one hit
  from the bystander (drag the dance away from the approach line);
- survive a 2-chaser corner without stand-and-tank;
- reproduce every 1v1 gate when given exactly one enemy (n=1
  regression — the same scenarios run through both planners below).
"""

from __future__ import annotations

import pytest

from players.dancer.planner import plan_duel, plan_melee
from players.scripted_player import ATN_ATTACK, ATN_NOOP, MOVE_DELTAS, \
    RUN_OFFSET


class MiniMelee:
    """Exact N-enemy local sim. Coordinates unbounded; `walls` blocks.

    enemies: [(pos, dmg_to_us, hp)]. Player acts first within a tick;
    then each live melee in list order: attack at Manhattan-1, else one
    step reducing the strictly-largest gap axis (tie -> column),
    stalling when the step is blocked by a wall, the player, or another
    live enemy (a cell vacated earlier in the same tick is free).
    """

    def __init__(self, ppos, enemies, our_dmg=30, sword=True,
                 walls=frozenset()):
        self.p = tuple(ppos)
        self.enemies = [[tuple(pos), dmg, hp] for pos, dmg, hp in enemies]
        self.our_dmg = our_dmg
        self.sword = sword
        self.walls = set(walls)
        self.hits_dealt = 0
        self.dmg_taken = 0
        self.hits_taken = 0
        self.moved_ticks = 0

    def passable(self, r, c):
        return (r, c) not in self.walls

    def _occ(self):
        return {e[0] for e in self.enemies if e[2] > 0}

    def in_arc(self, pr, pc, r, c):
        ar, ac = abs(r - pr), abs(c - pc)
        if ar + ac == 1:
            return True
        return self.sword and ar == 1 and ac == 1

    def step(self, action, target=None):
        """target: an entry of self.enemies that ATN_ATTACK strikes
        (the executive always attacks its selected duel target)."""
        # player first
        occ = self._occ()
        if action == ATN_ATTACK:
            if target is not None and target[2] > 0 and \
                    self.in_arc(*self.p, *target[0]):
                target[2] -= self.our_dmg
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
                mid = (self.p[0] + dr, self.p[1] + dc)
                dest = (self.p[0] + dr * steps, self.p[1] + dc * steps)
                if self.passable(*mid) and mid not in occ and \
                        (not run or (self.passable(*dest)
                                     and dest not in occ)):
                    self.p = dest
                    self.moved_ticks += 1
        # enemies respond
        occ = self._occ()
        for e in self.enemies:
            if e[2] <= 0:
                continue
            r, c = e[0]
            dr, dc = self.p[0] - r, self.p[1] - c
            if abs(dr) + abs(dc) == 1:
                self.dmg_taken += e[1]
                self.hits_taken += 1
                continue
            if abs(dr) > abs(dc):
                ne = (r + (1 if dr > 0 else -1), c)
            else:
                ne = (r, c + (1 if dc > 0 else -1))
            if self.passable(*ne) and ne != self.p and ne not in occ:
                occ.discard(e[0])
                occ.add(ne)
                e[0] = ne
        return all(e[2] <= 0 for e in self.enemies)


class MiniDuel(MiniMelee):
    """1v1 wrapper keeping the original test API."""

    def __init__(self, ppos, epos, e_hp=99, p_hp=99, our_dmg=30,
                 their_dmg=40, sword=True, walls=frozenset()):
        super().__init__(ppos, [(epos, their_dmg, e_hp)], our_dmg,
                         sword, walls)
        self.their_dmg = their_dmg

    @property
    def e(self):
        return self.enemies[0][0]

    @property
    def e_hp(self):
        return self.enemies[0][2]

    def step(self, action):
        return super().step(action, target=self.enemies[0])


# --- 1v1 gates, run through BOTH planners (melee1 = n=1 regression) ---

def _melee_adapter(ppos, epos, e_hp, our_dmg, their_dmg, sword,
                   intent_kill, passable):
    return plan_melee(ppos, [(tuple(epos), their_dmg, e_hp)], 0,
                      our_dmg, sword, intent_kill, passable)


@pytest.fixture(params=["duel", "melee1"])
def plan(request):
    return plan_duel if request.param == "duel" else _melee_adapter


def drive(duel: MiniDuel, plan, intent_kill: bool, ticks: int = 80):
    for _t in range(ticks):
        act = plan(duel.p, duel.e, duel.e_hp, duel.our_dmg,
                   duel.their_dmg, duel.sword, intent_kill,
                   duel.passable)
        done = duel.step(act)
        if done:
            return True
    return duel.e_hp <= 0


def test_dance_kills_without_damage(plan):
    """Sword + hittable melee in the open: kill, zero damage taken."""
    for start in ((3, 0), (0, 3), (2, 2), (4, 1), (-3, -2)):
        duel = MiniDuel((0, 0), start, our_dmg=30, their_dmg=45,
                        sword=True)
        killed = drive(duel, plan, intent_kill=True)
        assert killed, f"no kill from {start} (e_hp={duel.e_hp})"
        assert duel.dmg_taken == 0, \
            f"took {duel.dmg_taken} dmg dancing from {start}"


def test_dance_efficiency(plan):
    """The dance lands a hit at least every 4 ticks on average."""
    duel = MiniDuel((0, 0), (3, 1), our_dmg=25, their_dmg=45, sword=True)
    ticks = 0
    for _t in range(80):
        act = plan(duel.p, duel.e, duel.e_hp, duel.our_dmg,
                   duel.their_dmg, True, True, duel.passable)
        ticks += 1
        if duel.step(act):
            break
    hits_needed = -(-99 // 25)
    assert duel.e_hp <= 0
    assert ticks <= hits_needed * 4 + 6, f"kill took {ticks} ticks"
    assert duel.dmg_taken == 0


def test_escape_unhittable(plan):
    """our_dmg=0 vs a big melee: zero damage over a long pursuit."""
    duel = MiniDuel((0, 0), (2, 1), our_dmg=0, their_dmg=80, sword=False)
    for _t in range(60):
        act = plan(duel.p, duel.e, duel.e_hp, 0, 80, False, False,
                   duel.passable)
        duel.step(act)
    assert duel.dmg_taken == 0, f"took {duel.dmg_taken} while escaping"


def test_bare_hands_trade(plan):
    """Bootstrap: no sword, winnable trade. Kill at sane cost."""
    duel = MiniDuel((0, 0), (1, 0), our_dmg=37, their_dmg=18, sword=False)
    killed = drive(duel, plan, intent_kill=True)
    assert killed
    # 3 hits needed; the enemy can land at most ~4 (first-strike traded)
    assert duel.dmg_taken <= 4 * 18, f"trade cost {duel.dmg_taken}"


def test_corner_no_catastrophe(plan):
    """Boxed against a wall with a strong melee: the planner may not
    find zero damage, but must not stand still and tank forever."""
    walls = {(r, -2) for r in range(-4, 5)} | \
            {(r, 2) for r in range(-4, 5)} | \
            {(-3, c) for c in range(-2, 3)}
    duel = MiniDuel((0, 0), (2, 0), our_dmg=20, their_dmg=50,
                    sword=True, walls=walls)
    for _t in range(50):
        act = plan(duel.p, duel.e, duel.e_hp, 20, 50, True, True,
                   duel.passable)
        if duel.step(act):
            break
    # corridor: the dance degrades to hit-and-step; cost bounded well
    # under stand-and-tank (which would be ~5 hits x 50 = 250)
    assert duel.e_hp <= 0 or duel.dmg_taken <= 150


# --- plan_melee N-enemy gates ---

def drive_melee(sim: MiniMelee, target, intent_kill: bool,
                ticks: int = 80):
    """Replan each tick over the live enemy set, exactly as the
    executive will. target: an entry of sim.enemies or None. Stops at
    target death (the executive replans there)."""
    for _t in range(ticks):
        alive = [e for e in sim.enemies if e[2] > 0]
        if not alive:
            return
        enemies = [(e[0], e[1], e[2]) for e in alive]
        tidx = alive.index(target) if target in alive else None
        act = plan_melee(sim.p, enemies, tidx, sim.our_dmg, sim.sword,
                         intent_kill, sim.passable)
        sim.step(act, target if target in alive else None)
        if target is not None and target[2] <= 0:
            return


def test_two_chaser_open_escape():
    """Pincer by two big melee in the open: at most one hit in 60
    ticks (a clean escape takes zero)."""
    sim = MiniMelee((0, 0), [((2, 1), 60, 99), ((-2, -1), 60, 99)],
                    our_dmg=0, sword=False)
    drive_melee(sim, None, intent_kill=False, ticks=60)
    assert sim.hits_taken <= 1, \
        f"took {sim.hits_taken} hits ({sim.dmg_taken} dmg) in pincer"


def test_duel_with_bystander():
    """Kill the duel target while a second melee approaches from 8
    cells: the dance must drag away from the approach line. At most
    one hit from the bystander."""
    sim = MiniMelee((0, 0), [((3, 0), 45, 99), ((0, 8), 60, 99)],
                    our_dmg=30, sword=True)
    target = sim.enemies[0]
    drive_melee(sim, target, intent_kill=True, ticks=80)
    assert target[2] <= 0, f"target survived (hp={target[2]})"
    assert sim.hits_taken <= 1, \
        f"took {sim.hits_taken} hits ({sim.dmg_taken} dmg) from bystander"


def test_two_chaser_corner():
    """Cornered by two strong melee in a dead-end pocket: bounded
    damage, and never stand-and-tank (tanking = ~100 dmg/tick once
    both are adjacent)."""
    walls = {(r, -2) for r in range(-4, 7)} | \
            {(r, 2) for r in range(-4, 7)} | \
            {(-3, c) for c in range(-2, 3)}
    sim = MiniMelee((0, 0), [((4, 0), 50, 99), ((5, 1), 50, 99)],
                    our_dmg=25, sword=True, walls=walls)
    drive_melee(sim, sim.enemies[0], intent_kill=True, ticks=40)
    both_dead = all(e[2] <= 0 for e in sim.enemies)
    assert both_dead or sim.dmg_taken <= 250, \
        f"took {sim.dmg_taken} dmg cornered (hits={sim.hits_taken})"
    assert sim.moved_ticks >= 5, \
        f"stand-and-tank: moved only {sim.moved_ticks} ticks"
