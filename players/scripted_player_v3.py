"""Scripted cogame-nmmo player v3: ``python -m players.scripted_player_v3``.

v2's observation machinery (liveness + ghost proving) with a combat core
built on two decisive mechanics facts the RL baseline never exploits:

1. THE ELEMENT BYTE IS A PERFECT BOW DISCRIMINATOR. Every enemy of
   level >= 15 is ranged AND gets a non-neutral element at spawn (all
   spawn tiles are seasonal grass, nmmo3.h:1815-1830; spawn() never
   clears element on respawn). element == 0 <=> melee <=> level <= 14.
   So the two threat classes v2 could never separate are separable:
   bows (one-shot on-axis at range 4) get hard geometric avoidance;
   melee get farmed.

2. THE DIAGONAL SWORD DANCE: melee enemies attack only at Manhattan-1
   on their own turn, chase 1 tile/tick along the LONGEST axis toward
   the player (tie -> column move; model verified 33/33 against ground
   truth), and players act before enemies within a tick. A sword's
   attack arc covers the diagonals (ATTACK_SWORD). The cycle
       gap (1,1): ATTACK          (free hit; enemy tie-moves to (1,0))
       gap (1,0): RUN 2 away      (-> (3,0); enemy row-moves -> (2,0))
       gap (2,0): step sideways   (-> (2,1); enemy row-moves -> (1,1))
   lands one hit every ~3 ticks and never takes damage, against ANY
   melee enemy the damage math can hurt (dmg = 40 + 2L + equip_atk -
   5L' > 0). Every kill of an enemy at-or-above our level is +1 comb,
   and level 9-16 kills drop TIER-2 tools -> tier-2 armor (48 defense)
   and prof to 16. Comb ceiling via melee-only farming ~ 15.

Everything else (bootstrap trade-fight for the first tool, harvesting
economy, market dump, overwatch movement, stagnation clock) is
inherited from v2. See scripted_player_v2.py and the campaign learnings
for the measured failures that shaped these doctrines.
"""

from __future__ import annotations

import sys

from .client import run_policy_main, seed_from_env
from .scripted_player import (
    ARMOR_SLOT_BY_TYPE, ATN_ATTACK, ATN_DOWN, ATN_LEFT, ATN_NINE, ATN_NOOP,
    ATN_ONE, ATN_RIGHT, ATN_SELL, ATN_UP, CENTER_COL, CENTER_ROW, I_HERB,
    I_HILT, I_SWORD, I_TOOL, MODE_PLAY, MODE_SELL_PRICE, MODE_SELL_SELECT,
    MOVE_DELTAS, NUM_KEY_SLOTS, Percept, RUN_OFFSET, SLOT_HELD, item_tier,
    item_type,
)
from .scripted_player_v2 import (
    BASE_ATTACK, ENEMY_ATTACK_BASE, ENEMY_HP, LEVEL_MUL, ENEMY_LEVEL_MUL,
    MindV2, NPC_AGGRO, clamp_delta_levels,
)

DUEL_STALL_MAX = 8       # dance ticks without progress before aborting
BOW_AVOID_CHEB = 6       # keep this far from any non-ghost bow imprint
BOOTSTRAP_HP = 90        # hp floor to start the bare-hands bootstrap trade
DANCE_MIN_HP = 45        # abort a dance below this (something is wrong)
RECOVER_UNTIL_HP = 88    # after taking damage, evade-only until back here
RECOVER_FLOOR_HP = 70    # entering-recovery threshold


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


class MindV3(MindV2):
    def reset(self) -> None:
        super().reset()
        self.duel: tuple[int, int] | None = None    # target window cell
        self.duel_stall = 0
        self.mode = "hunt"       # hunt (sword) | gather (tool)

    # -- threat classification -----------------------------------------

    def _classify(self, p: Percept):
        """Non-ghost imprints split by the element byte."""
        bows, melee = [], []
        for r, c, d, hpb, elem in p.enemy_cells():
            if self.ghost[r, c]:
                continue
            (bows if elem != 0 else melee).append((r, c, d, hpb))
        return bows, melee

    def _bow_cells(self, bows):
        return [(r, c) for r, c, _d, _h in bows]

    def _dmg_vs(self, p: Percept, delta: int) -> int:
        """Worst-case per-hit damage vs a MELEE enemy in this delta
        bucket (level additionally capped at 14 by the element proof)."""
        lo, hi = clamp_delta_levels(p.comb_lvl, delta)
        hi = min(hi, 14)
        eq_atk = int(p.scalars[45])
        return BASE_ATTACK + LEVEL_MUL * p.comb_lvl + eq_atk \
            - ENEMY_LEVEL_MUL * hi

    def _evade(self, p: Percept, cells) -> int:
        """Safe-stall escape from melee chasers: any move that opens or
        holds distance is attack-free (a chaser that moved cannot hit
        that tick). Prefer directions with 2-tile clearance (dead ends
        are how single chasers grind agents down); run when clear."""
        moves = self._passable_moves(p)
        if not moves:
            return ATN_NOOP
        occ = getattr(self, "_occupied", set())
        def score(atn):
            mr, mc = MOVE_DELTAS[atn]
            nr, nc = CENTER_ROW + mr, CENTER_COL + mc
            free = (nr, nc) not in occ
            clear2 = p.passable(CENTER_ROW + 2 * mr, CENTER_COL + 2 * mc)
            d = self._dist(nr, nc, cells) if cells else 9
            bow_ok = not self._bow_risk(nr, nc, self._bows)
            return (bow_ok, free, d, clear2)
        best = max(moves, key=score)
        return self._maybe_run(p, best)


    # -- exact local duel/escape planner --------------------------------
    #
    # The melee chase rule is deterministic and verified 33/33 against
    # ground truth (attack at Manhattan-1 on its turn, else step 1 tile
    # reducing the row gap if it is strictly largest, else the col gap;
    # blocked steps stall). Players act first. So the local 1v1 game is
    # PERFECTLY simulable: search our action tree a few plies deep and
    # play the line that maximizes damage dealt minus weighted damage
    # taken. This one search subsumes the dance (it rediscovers the
    # free-hit cycle when profitable), the safe-stall escape, corner
    # play, and trade decisions.

    def _plan_duel(self, p: Percept, er: int, ec: int, e_hp: int,
                   our_dmg: int, their_dmg: int, aggressive: bool,
                   sword: bool, depth: int = 4) -> int | None:
        """Best first action vs the melee at window cell (er, ec)."""
        occupied = getattr(self, "_occupied", set())

        def walkable(r, c):
            return p.passable(r, c)

        def occ_pen(r, c):
            # live-imprint cells are probably-occupied: a move there
            # probably fails (wasting a tick) but phantom live cells
            # must never cage the search into standing still
            return 2.5 if (r, c) in occupied else 0.0

        W = 10.0 if not aggressive else 1.6
        KILL_BONUS = 40.0

        def enemy_step(pr, pc, r, c):
            """(new_r, new_c, attacked) for the enemy at (r, c) vs the
            player at (pr, pc), exact chase rule."""
            dr, dc = pr - r, pc - c
            if abs(dr) + abs(dc) == 1:
                return r, c, True
            if abs(dr) > abs(dc):
                nr, nc = r + (1 if dr > 0 else -1), c
            else:
                nr, nc = r, c + (1 if dc > 0 else -1)
            if walkable(nr, nc) and (nr, nc) != (pr, pc):
                return nr, nc, False
            return r, c, False

        def in_arc(pr, pc, r, c):
            ar, ac = abs(r - pr), abs(c - pc)
            if ar + ac == 1:
                return True
            return sword and ar == 1 and ac == 1

        def search(pr, pc, r, c, ehp, d):
            if ehp <= 0:
                return KILL_BONUS + d * 2.0, ATN_NOOP
            if d == 0:
                return max(abs(r - pr), abs(c - pc)) * 0.3, ATN_NOOP
            best = (-1e9, ATN_NOOP)
            # candidate actions: noop, attack (if in arc), 4 walks,
            # 4 runs
            cands = [(ATN_NOOP, pr, pc, False)]
            if in_arc(pr, pc, r, c):
                cands.append((ATN_ATTACK, pr, pc, True))
            for atn, (mr, mc) in MOVE_DELTAS.items():
                nr, nc = pr + mr, pc + mc
                if walkable(nr, nc) and (nr, nc) != (r, c):
                    cands.append((atn, nr, nc, False))
                    n2r, n2c = pr + 2 * mr, pc + 2 * mc
                    if walkable(n2r, n2c) and (n2r, n2c) != (r, c):
                        cands.append((atn + RUN_OFFSET, n2r, n2c, False))
            for atn, npr, npc, attacked in cands:
                nehp = ehp - our_dmg if attacked else ehp
                if nehp <= 0:
                    val = KILL_BONUS + our_dmg / 10.0
                else:
                    ner, nec, ehit = enemy_step(npr, npc, r, c)
                    sub, _ = search(npr, npc, ner, nec, nehp, d - 1)
                    val = sub + (our_dmg / 10.0 if attacked else 0.0) \
                        - (W * their_dmg / 10.0 if ehit else 0.0)
                if (npr, npc) != (pr, pc):
                    val -= occ_pen(npr, npc)
                if val > best[0]:
                    best = (val, atn)
            return best

        _val, act = search(CENTER_ROW, CENTER_COL, er, ec, e_hp, depth)
        return act

    # -- the dance ------------------------------------------------------

    def _dance_step(self, p: Percept, tr: int, tc: int) -> int | None:
        """One dance action vs the melee target at window cell (tr, tc).
        Returns None when the geometry is unrecoverable this tick (the
        caller aborts the duel)."""
        dr, dc = tr - CENTER_ROW, tc - CENTER_COL   # us -> enemy
        adr, adc = abs(dr), abs(dc)

        occ = getattr(self, "_occupied", set())

        def passable_move(atn):
            mr, mc = MOVE_DELTAS[atn]
            nr, nc = CENTER_ROW + mr, CENTER_COL + mc
            return p.passable(nr, nc) and (nr, nc) not in occ

        def away_row():
            return ATN_UP if dr > 0 else ATN_DOWN
        def away_col():
            return ATN_LEFT if dc > 0 else ATN_RIGHT

        if adr <= 1 and adc <= 1 and adr + adc > 0:
            if adr == 1 and adc == 1:
                self.last_attack_pos = (tr, tc)
                return ATN_ATTACK
            # orthogonal adjacency: open the gap along that axis.
            # run 2 restores the full cycle; walk 1 is the safe stall
            # (enemy re-closes but only ever moves, never attacks).
            atn = away_row() if adr == 1 else away_col()
            if passable_move(atn):
                return self._maybe_run(p, atn)
            # boxed on that axis: sidestep to the diagonal (gap (1,1):
            # enemy tie-moves without attacking)
            for side in ((ATN_LEFT, ATN_RIGHT) if adr == 1
                         else (ATN_UP, ATN_DOWN)):
                if passable_move(side):
                    return side
            # fully cornered: swing (sword arc covers orthogonal too)
            self.last_attack_pos = (tr, tc)
            return ATN_ATTACK
        if (adr, adc) in ((2, 0), (0, 2)):
            # form the (2,1) pre-attack geometry
            for side in ((ATN_LEFT, ATN_RIGHT) if adr == 2
                         else (ATN_UP, ATN_DOWN)):
                if passable_move(side):
                    return side
            return ATN_NOOP     # enemy closes to (1,0): stall-safe
        if (adr, adc) in ((2, 1), (1, 2), (2, 2), (3, 0), (0, 3),
                          (3, 1), (1, 3)):
            return ATN_NOOP     # let it walk into the arc
        if max(adr, adc) <= NPC_AGGRO + 1:
            # approach: reduce the longest gap axis, never entering
            # Manhattan-1 (the enemy must initiate adjacency)
            atn = (ATN_DOWN if dr > 0 else ATN_UP) if adr >= adc \
                else (ATN_RIGHT if dc > 0 else ATN_LEFT)
            if passable_move(atn) and adr + adc > 2:
                return atn
            return ATN_NOOP
        return None

    # -- decide ---------------------------------------------------------

    def _decide(self, p: Percept) -> int:
        bows, melee = self._classify(p)
        bow_cells = self._bow_cells(bows)
        self._bows = bow_cells          # for _maybe_run / _safe_moves
        # entity-occupied cells block move() (dest_check checks pids):
        # stepping into one silently fails and stands us still for a
        # free enemy hit. LIVE imprints only - stale residue covers half
        # the window, and treating it as walls poisons every move choice
        # (measured: evade walked TOWARD real chasers because the away
        # tiles looked occupied).
        occ = set()
        ent = p.tiles[:, :, 4]
        import numpy as _np
        for r, c in zip(*_np.nonzero(ent != 0)):
            if self.live[r, c] > 0:
                occ.add((int(r), int(c)))
        self._occupied = occ
        # obstacle cells for goal pathing: bows AND melee (a passed-by
        # melee chases and lands 40-70/hit at L9-14)
        melee_cells = [(r, c) for r, c, _d, _h in melee]
        obstacle_cells = bow_cells + melee_cells
        urgent = self._urgency(p)

        # 0. market UI (inherited flow)
        if p.ui_mode != MODE_PLAY:
            if self.selling and not p.in_combat:
                if p.ui_mode == MODE_SELL_SELECT:
                    junk = self._junk_slot(p)
                    if junk is not None:
                        return ATN_ONE + junk
                elif p.ui_mode == MODE_SELL_PRICE:
                    self.selling = False
                    return ATN_NINE
            self.selling = False
            return self._wander(p, bow_cells)
        self.selling = False

        # 1. emergency herb
        if p.hp < 40:
            herb = self._herb_slot(p)
            if herb is not None:
                return ATN_ONE + herb

        held = p.equipment[SLOT_HELD]
        sword_held = bool(held) and item_type(held) == I_SWORD

        # kill detection -> loot sweep (tier-2 tools are the economy)
        if p.comb_lvl > self.prev_comb:
            self.loot_sweep_left = 18
            self.duel = None
            self.duel_stall = 0

        # 2. bow safety: hard geometric avoidance of every non-ghost
        # bow imprint (they one-shot low-armor players on-axis at 4)
        eq_def = int(p.scalars[46])
        bow_danger = [
            (r, c) for r, c in bow_cells
            if max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
            <= BOW_AVOID_CHEB]
        # with real armor an arrow stops being lethal; relax to
        # funnel-avoidance only
        armored = eq_def >= 40
        if bow_danger and not armored:
            self.duel = None
            moves = self._passable_moves(p)
            if moves:
                def flee_score(atn):
                    mr, mc = MOVE_DELTAS[atn]
                    nr, nc = CENTER_ROW + mr, CENTER_COL + mc
                    return (not self._bow_risk(nr, nc, bow_cells),
                            self._dist(nr, nc, bow_danger))
                best = max(moves, key=flee_score)
                nr, nc = (CENTER_ROW + MOVE_DELTAS[best][0],
                          CENTER_COL + MOVE_DELTAS[best][1])
                here_risk = self._bow_risk(CENTER_ROW, CENTER_COL,
                                           bow_cells)
                gain = self._dist(nr, nc, bow_danger) > \
                    self._dist(CENTER_ROW, CENTER_COL, bow_danger)
                if gain or (here_risk and not self._bow_risk(
                        nr, nc, bow_cells)):
                    return self._maybe_run(p, best)

        # 2b. recovery bookkeeping (used by the planner weights and
        # the goal gates below)
        self.recovering = getattr(self, "recovering", False)
        if p.hp < RECOVER_FLOOR_HP:
            self.recovering = True
        if p.hp >= RECOVER_UNTIL_HP:
            self.recovering = False

        # 3. melee engagement: one exact-model planner call handles
        # dance, trade, escape, and corner play for the nearest melee.
        def cheb(t):
            return max(abs(t[0] - CENTER_ROW), abs(t[1] - CENTER_COL))
        # engage the planner only for real business: anything at
        # touching range (defense), or a worthwhile target within 4
        # while we are fit to fight (offense). Distant melee are
        # obstacles for pathing, not planner time.
        defense = [t for t in melee if cheb(t) <= 2]
        offense = [t for t in melee
                   if cheb(t) <= 4 and not self.recovering
                   and p.hp >= 70 and self._dmg_vs(p, t[2]) > 0
                   and (t[2] >= 1 or p.comb_lvl <= p.prof_lvl
                        or p.held_tool_tier == 0)]
        near_melee = defense or offense
        if near_melee:
            # nearest first; everything else is a static obstacle via
            # self._occupied (live cells)
            r, c, d, hpb = min(near_melee, key=lambda t: (
                max(abs(t[0] - CENTER_ROW), abs(t[1] - CENTER_COL)), t[2]))
            our_dmg = self._dmg_vs(p, d)
            lo, hi = clamp_delta_levels(p.comb_lvl, d)
            hi = min(hi, 14)
            eq_def = int(p.scalars[46])
            their_dmg = max(ENEMY_ATTACK_BASE + ENEMY_LEVEL_MUL * hi
                            - LEVEL_MUL * p.comb_lvl - eq_def, 0)
            e_hp = hpb * 20 + 10
            second = [t for t in near_melee if (t[0], t[1]) != (r, c)]
            healthy = p.hp >= (55 if their_dmg <= 20 else 80)
            worth = our_dmg > 0 and (d >= 1 or p.comb_lvl <= p.prof_lvl
                                     or p.held_tool_tier == 0)
            aggressive = worth and healthy and not second \
                and not self.recovering
            if p.hp < 45:
                herb = self._herb_slot(p)
                if herb is not None:
                    return ATN_ONE + herb
            act = self._plan_duel(p, r, c, e_hp, max(our_dmg, 0),
                                  their_dmg, aggressive, sword_held)
            if act == ATN_ATTACK:
                self.last_attack_pos = (r, c)
            if act is not None:
                return act

        # 6. mode + equipment
        want_comb = p.comb_lvl <= p.prof_lvl or p.held_tool_tier == 0
        best_sword = self._best_sword_tier(p)
        self.mode = "hunt" if (want_comb and best_sword > 0) else "gather"
        if not p.in_combat:
            act = self._equip_for_mode(p)
            if act is not None:
                return act

        # 7. duel acquisition (hunt mode, sword in hand)
        if self.mode == "hunt" and sword_held and p.hp >= 70:
            best = None
            for r, c, d, hpb in melee:
                if self.live[r, c] <= 0:
                    continue
                if self._dmg_vs(p, d) <= 0:
                    continue
                dist = max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
                if dist > NPC_AGGRO + 1:
                    continue
                # isolation: no other threat near us or the target
                others = [(rr, cc) for rr, cc, _d, _h in melee
                          if (rr, cc) != (r, c)] + bow_cells
                if others and (self._dist(r, c, others) <= 3 or
                               self._dist(CENTER_ROW, CENTER_COL,
                                          others) <= 3):
                    continue
                # prefer comb-leveling kills (d >= 1 guarantees
                # defender >= us), then nearness
                key = (-min(d, 1), dist)
                if best is None or key < best[0]:
                    best = (key, (r, c))
            if best is not None:
                self.duel = best[1]
                self.duel_stall = 0
                step = self._dance_step(p, *best[1])
                if step is not None:
                    return step
                self.duel = None

        # 8. bootstrap hold: no tool yet, a lone weak melee nearby ->
        # let it come (first strike ours), then trade at step 5
        if p.held_tool_tier == 0 and not sword_held and \
                p.hp >= BOOTSTRAP_HP:
            cand = [(max(abs(r - CENTER_ROW), abs(c - CENTER_COL)), r, c)
                    for r, c, d, hpb in melee
                    if d == 0 and self.live[r, c] > 0]
            cand = [t for t in cand if t[0] <= NPC_AGGRO]
            others_near = [b for b in bow_cells
                           if self._dist(CENTER_ROW, CENTER_COL,
                                         [b]) <= 6]
            if cand and not others_near:
                self.hold_ticks += 1
                if self.hold_ticks <= 8:
                    return ATN_NOOP
                _d, r, c = min(cand)
                self.ghost[r, c] = True
                self.hold_ticks = 0

        # 8b. recovery: no goal pursuit while wounded - sit and regen
        # (1 hp/tick); threats are already handled above.
        if self.recovering:
            return ATN_NOOP

        # 9. harvest
        target = self._harvest_target(p)
        if target is not None:
            step = self._step_towards(p, obstacle_cells, *target,
                                      run_ok=False)
            if step is not None:
                return step
        elif all(p.inventory) and not p.in_combat and \
                self._junk_slot(p) is not None:
            self.selling = True
            return ATN_SELL

        # 10. overwatch / roam
        return self._wander(p, obstacle_cells, roam=urgent)

    # -- equipment by mode ----------------------------------------------

    def _equip_for_mode(self, p: Percept) -> int | None:
        equipment = p.equipment
        held = equipment[SLOT_HELD]

        # armor always (empty slot or tier upgrade)
        for atype, slot in ARMOR_SLOT_BY_TYPE.items():
            best = self._best_unequipped(p, atype)
            cur = equipment[slot]
            if best is not None and (cur == 0 or best[1] > item_tier(cur)):
                if cur == 0:
                    return ATN_ONE + best[0]
                s = self._slot_of(p, cur, equipped=True)
                if s is not None:
                    return ATN_ONE + s
        want = I_SWORD if self.mode == "hunt" else I_TOOL
        best = self._best_unequipped(p, want)
        held_type = item_type(held) if held else None
        held_tier = item_tier(held) if held else 0
        if held_type == want:
            if best and best[1] > held_tier:
                s = self._slot_of(p, held, equipped=True)
                if s is not None:
                    return ATN_ONE + s
            return None
        if best:
            if held:
                s = self._slot_of(p, held, equipped=True)
                if s is not None:
                    return ATN_ONE + s
            else:
                return ATN_ONE + best[0]
        return None

    def _harvest_target(self, p: Percept):
        """v2's ranking with one change: hilt (-> sword) outranks
        everything while we lack a sword at our tool tier — the sword
        is the whole combat economy."""
        base = super()._harvest_target(p)
        if self._best_sword_tier(p) >= max(p.held_tool_tier, 1):
            return base
        held_tier = p.held_tool_tier
        best = None
        for r, c, itype, tier in p.item_tiles():
            if itype != I_HILT or tier > held_tier or all(p.inventory):
                continue
            dist = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
            if best is None or dist < best[0]:
                best = (dist, (r, c))
        return best[1] if best is not None else base


class ScriptedPolicyV3:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.minds: dict[int, MindV3] = {}

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        actions = []
        for i, row in enumerate(obs_rows):
            mind = self.minds.get(i)
            if mind is None:
                mind = self.minds[i] = MindV3(self.seed, i)
            if resets[i]:
                mind.reset()
            actions.append([mind.act(bytes(row))])
        return actions


def policy_from_env() -> ScriptedPolicyV3:
    return ScriptedPolicyV3(seed_from_env(default=0))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
