"""L4 Executive: phases + goals on top of WorldModel and DuelPlanner.

All combat decisions go through plan_duel (the only combat authority);
the executive only picks WHETHER and WHAT to engage. All movement goes
through one safe-step filter (bow funnels + occupancy + reroute).
Recovery is a gate on offense/goals, never a control-stealing branch.

Phases: BOOTSTRAP (no tool: trade for the first kill) -> ECONOMY
(tool: hilt -> sword, ore -> armor, prof, herbs) <-> FARM (sword:
dance-kill melee for comb + tier-2 loot). RECOVER overlay below 60 hp
until 90. The 500-tick stagnation clock is mirrored and forces the
binding activity when a bank approaches.
"""

from __future__ import annotations

import numpy as np

from ..scripted_player import (ARMOR_SLOT_BY_TYPE, ATN_ATTACK, ATN_DOWN,
                               ATN_LEFT, ATN_NINE, ATN_NOOP, ATN_ONE,
                               ATN_RIGHT, ATN_SELL, ATN_UP, CENTER_COL,
                               CENTER_ROW, I_GEM_TYPES, I_HERB, I_HILT,
                               I_ORE, I_SWORD, I_TOOL, I_WOOD, MODE_PLAY,
                               MODE_SELL_PRICE, MODE_SELL_SELECT,
                               MOVE_DELTAS, NUM_KEY_SLOTS, Percept,
                               RUN_OFFSET, SLOT_HELD, item_tier, item_type,
                               tier_level)
from .planner import plan_melee
from .world import NPC_AGGRO, WorldModel

STAG_WINDOW = 500
HERB_HP = 45
HERB_HP_COMBAT = 65
RECOVER_FLOOR = 60
RECOVER_UNTIL = 90
BOW_AVOID = 6
ENGAGE_R = 4
HERB_STOCK = 2

# combat model constants (nmmo3.h calc_damage)
BASE_ATTACK = 40
ENEMY_BASE = 15
LEVEL_MUL = 2
ENEMY_LEVEL_MUL = 5


def delta_bounds(my_lvl: int, delta: int) -> tuple[int, int]:
    if delta >= 4:
        return my_lvl + 8, my_lvl + 40
    return max(1, my_lvl + 2 * delta), my_lvl + 2 * delta + 1


class DancerMind:
    def __init__(self, seed, agent_idx: int):
        self._seed = seed
        self._agent_idx = agent_idx
        self._generation = -1
        self.world = WorldModel()
        self.tick = 0
        self.reset()

    def reset(self) -> None:
        self._generation += 1
        self.rng = np.random.default_rng(
            (0 if self._seed is None else self._seed,
             self._agent_idx, self._generation))
        self.world.reset(self.tick)
        self.life_tick = 0
        self.window_start_min = 1
        self.recovering = False
        self.prev_comb = 1
        self.loot_sweep_left = 0
        self.selling = False
        self.last_action = ATN_NOOP
        self.wander_action = ATN_DOWN
        self.wander_left = 0
        self.idle_ticks = 0
        self.branch = ""          # last _decide branch (forensics)

    # ------------------------------------------------------------------

    def shadow(self, obs: bytes, action: int) -> None:
        """Advance world/bookkeeping for a tick whose action was chosen
        by ANOTHER policy (netstrap bootstrap): tracking, life clock and
        stagnation window stay warm so a later handoff starts hot."""
        p = Percept(obs)
        self.world.last_action = self.last_action
        self.world.observe(p, self.tick)
        self.life_tick += 1
        if self.life_tick % STAG_WINDOW == 1:
            self.window_start_min = min(p.comb_lvl, p.prof_lvl)
        self.prev_comb = p.comb_lvl
        self.world.note_action(action)
        self.last_action = action
        self.tick += 1

    def act(self, obs: bytes) -> int:
        p = Percept(obs)
        self.world.last_action = self.last_action
        self.world.observe(p, self.tick)
        self.life_tick += 1
        if self.life_tick % STAG_WINDOW == 1:
            self.window_start_min = min(p.comb_lvl, p.prof_lvl)
        action = self._decide(p)
        self.world.note_action(action)
        self.last_action = action
        self.tick += 1
        return action

    # -- shared movement filter ----------------------------------------

    def _veto(self, p: Percept, r: int, c: int, bows, melee) -> bool:
        """Cell (r, c) is unsafe to step into."""
        if not p.passable(r, c):
            return True
        for br, bc, _t in bows:
            if max(abs(r - br), abs(c - bc)) <= 6 and \
                    min(abs(r - br), abs(c - bc)) <= 1:
                return True     # bow funnel
        for mr, mc, _t in melee:
            if abs(r - mr) + abs(c - mc) <= 1:
                return True     # melee reach
        return False

    def _safe_step(self, p: Percept, tr: int, tc: int, bows, melee,
                   occ) -> int | None:
        """One greedy step toward window cell (tr, tc) through the
        shared safety filter; None when nothing safe advances."""
        dr, dc = tr - CENTER_ROW, tc - CENTER_COL
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
        blocked_last = self._last_move_blocked(p)
        if blocked_last and self.last_action in prefs:
            prefs.remove(self.last_action)
        for atn in prefs:
            mr, mc = MOVE_DELTAS[atn]
            nr, nc = CENTER_ROW + mr, CENTER_COL + mc
            if self._veto(p, nr, nc, bows, melee) or (nr, nc) in occ:
                continue
            # WALKS ONLY: every measured one-shot death involved a run
            # chosen while the killer was missing from the track set; a
            # walk into trouble costs one reactable hit, a run costs 99
            return atn
        return None

    def _flee(self, p: Percept, cells, bows, melee, occ) -> int:
        best, best_key = ATN_NOOP, (-1, -1, -1)
        for atn, (mr, mc) in MOVE_DELTAS.items():
            nr, nc = CENTER_ROW + mr, CENTER_COL + mc
            if not p.passable(nr, nc) or (nr, nc) in occ:
                continue
            bow_ok = not any(
                max(abs(nr - br), abs(nc - bc)) <= 6
                and min(abs(nr - br), abs(nc - bc)) <= 1
                for br, bc, _t in bows)
            melee_ok = not any(abs(nr - r) + abs(nc - c) <= 1
                               for r, c, _t in melee)
            d = min((max(abs(nr - r), abs(nc - c)) for r, c in cells),
                    default=9)
            key = (melee_ok, bow_ok, d)
            if key > best_key:
                best_key, best = key, atn
        return best   # walks only (see _safe_step note)

    def _last_move_blocked(self, p: Percept) -> bool:
        was_move = (self.last_action in MOVE_DELTAS
                    or self.last_action - RUN_OFFSET in MOVE_DELTAS)
        return was_move and p.anim == 0

    # -- inventory helpers (ported, unchanged semantics) ----------------

    def _herb_slot(self, p):
        for idx, item in enumerate(p.inventory[:NUM_KEY_SLOTS]):
            if item and item_type(item) == I_HERB:
                return idx
        return None

    def _herb_count(self, p):
        return sum(1 for i in p.inventory if i and item_type(i) == I_HERB)

    def _slot_of(self, p, raw_id, equipped):
        for idx in range(NUM_KEY_SLOTS):
            if p.inventory[idx] == raw_id and \
                    bool(p.is_equipped[idx]) == equipped:
                return idx
        return None

    def _best_unequipped(self, p, t):
        best = None
        for idx in range(NUM_KEY_SLOTS):
            item = p.inventory[idx]
            if item and item_type(item) == t and not p.is_equipped[idx]:
                if best is None or item_tier(item) > best[1]:
                    best = (idx, item_tier(item))
        return best

    def _max_tool_tier(self, p):
        tiers = [item_tier(i) for i in p.inventory
                 if i and item_type(i) == I_TOOL]
        held = p.equipment[SLOT_HELD]
        if held and item_type(held) == I_TOOL:
            tiers.append(item_tier(held))
        return max(tiers, default=0)

    def _best_sword_tier(self, p):
        tiers = [item_tier(i) for i in p.inventory
                 if i and item_type(i) == I_SWORD]
        held = p.equipment[SLOT_HELD]
        if held and item_type(held) == I_SWORD:
            tiers.append(item_tier(held))
        return max(tiers, default=0)

    def _junk_slot(self, p):
        best_tool = self._max_tool_tier(p)
        best_sword = self._best_sword_tier(p)
        candidates = []
        for idx in range(NUM_KEY_SLOTS):
            item = p.inventory[idx]
            if not item or p.is_equipped[idx]:
                continue
            t = item_type(item)
            tier = item_tier(item)
            if t == I_HERB:
                continue
            if t == I_TOOL and tier >= best_tool:
                continue
            if t == I_SWORD and tier >= best_sword:
                continue
            if t in ARMOR_SLOT_BY_TYPE:
                cur = p.equipment[ARMOR_SLOT_BY_TYPE[t]]
                if cur == 0 or item_tier(cur) < tier:
                    continue
            candidates.append((tier, idx))
        return min(candidates)[1] if candidates else None

    # -- decision -------------------------------------------------------

    def _decide(self, p: Percept) -> int:
        bows = self.world.enemies_extended("bow", margin=4)
        melee = self.world.enemies("melee")
        occ = self.world.occupancy()
        # vertical runs are allowed only when no known bow could align
        # in that direction just beyond the shallow (+-5 row) view
        def vrun_ok(down: bool) -> bool:
            for br, bc, _t in bows:
                if abs(bc - CENTER_COL) <= 1 and \
                        (br > CENTER_ROW if down else br < CENTER_ROW):
                    return False
            return True
        self._vruns = {ATN_DOWN: vrun_ok(True), ATN_UP: vrun_ok(False)}

        # market UI
        self.branch = "ui"
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
            return ATN_NOOP   # exits foreign UI modes without moving
        self.selling = False

        # herb
        limit = HERB_HP_COMBAT if p.in_combat else HERB_HP
        if p.hp < limit:
            herb = self._herb_slot(p)
            if herb is not None:
                self.branch = "herb"
                return ATN_ONE + herb

        # recovery bookkeeping
        if p.hp < RECOVER_FLOOR:
            self.recovering = True
        if p.hp >= RECOVER_UNTIL:
            self.recovering = False

        # kill detection -> loot sweep
        if p.comb_lvl > self.prev_comb:
            self.loot_sweep_left = 16
        self.prev_comb = p.comb_lvl

        held = p.equipment[SLOT_HELD]
        sword_held = bool(held) and item_type(held) == I_SWORD
        eq_atk = int(p.scalars[45])
        eq_def = int(p.scalars[46])
        armored = eq_def >= 40

        # bow safety (hard until armored). The flee is melee-aware:
        # bow-flee stepping into tracked melee reach was a measured
        # 16-death class. (Engage-over-bow-flee precedence was ALSO
        # tried 2026-08-03 and REFUTED: bow deaths tripled 11 -> 31 -
        # phantom melee tracks suppressed the flee.)
        bow_close = [(r, c) for r, c, _t in bows
                     if max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
                     <= BOW_AVOID]
        here_funnel = any(
            max(abs(CENTER_ROW - r), abs(CENTER_COL - c)) <= 6
            and min(abs(CENTER_ROW - r), abs(CENTER_COL - c)) <= 1
            for r, c in bow_close)
        if bow_close and not armored:
            d_now = min(max(abs(CENTER_ROW - r), abs(CENTER_COL - c))
                        for r, c in bow_close)
            if here_funnel or d_now <= 4:
                self.branch = "bow-flee"
                return self._flee(p, bow_close, bows, melee, occ)

        # melee engagement via the planner. Wounded agents engage
        # (with avoid intent) a ring earlier - late evasion was a
        # measured death mode.
        engage_r = ENGAGE_R + (2 if self.recovering else 0)
        near = [(r, c, t) for r, c, t in melee
                if max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
                <= engage_r]
        if near:
            def dmg_vs(track):
                _lo, hi = delta_bounds(p.comb_lvl, track.delta)
                hi = min(hi, 14)
                return (BASE_ATTACK + LEVEL_MUL * p.comb_lvl + eq_atk
                        - ENEMY_LEVEL_MUL * hi)
            # target: the SOFTEST worthwhile enemy nearby, else nearest
            soft = [x for x in near if dmg_vs(x[2]) >= 12]
            pool = soft or near
            r, c, t = min(pool, key=lambda x: (
                max(abs(x[0] - CENTER_ROW), abs(x[1] - CENTER_COL)),
                -dmg_vs(x[2])))
            lo, hi = delta_bounds(p.comb_lvl, t.delta)
            hi = min(hi, 14)
            our_dmg = dmg_vs(t)
            their_dmg = max(ENEMY_BASE + ENEMY_LEVEL_MUL * hi
                            - LEVEL_MUL * p.comb_lvl - eq_def, 0)
            e_hp = t.hp_bucket * 20 + 10
            second = [x for x in near if x[2] is not t]
            want_comb = (p.comb_lvl <= p.prof_lvl
                         or p.held_tool_tier == 0
                         or self._urgent_comb(p))
            # a worthwhile kill finishes fast: slow grinds maximize
            # exposure (measured: hp33 'atk' deaths vs L12 at dmg ~2)
            hits_needed = -(-e_hp // our_dmg) if our_dmg > 0 else 99
            fightable = our_dmg >= 12 and hits_needed <= 8 and want_comb
            hp_ok = p.hp >= (55 if their_dmg * 3 < p.hp else 85)
            if p.held_tool_tier == 0 and not sword_held:
                # bootstrap trades are bare-handed hp-for-hp: take only
                # decisive, isolated, full-bar fights (deaths at min 1
                # divide the whole episode's score)
                far_ok = not any(
                    max(abs(x[0] - CENTER_ROW), abs(x[1] - CENTER_COL)) <= 5
                    for x in second)
                hp_ok = p.hp >= 95 and far_ok
            intent_kill = (fightable and hp_ok and not second
                           and not self.recovering and not bow_close)
            # pre-fight herb: a mid-level melee hits for 40-70; do not
            # enter its reach below one-hit headroom
            if their_dmg > 0 and p.hp < min(95, their_dmg + 25):
                herb = self._herb_slot(p)
                if herb is not None:
                    self.branch = "herb-prefight"
                    return ATN_ONE + herb
            # exact multi-enemy model: EVERY in-window melee is a
            # modeled enemy for the search (gated by test_dancer_planner
            # MiniMelee suite); bows stay hazard lines. Other melee must
            # NOT also appear in occupied/hazards - double-counting the
            # same threat as exact enemy + fear cells was the wiring
            # regression (1.7 -> 1.1, kills collapsed).
            def their_dmg_vs(track):
                _lo2, hi2 = delta_bounds(p.comb_lvl, track.delta)
                hi2 = min(hi2, 14)
                return max(ENEMY_BASE + ENEMY_LEVEL_MUL * hi2
                           - LEVEL_MUL * p.comb_lvl - eq_def, 0)
            # FRESH tracks (recently byte-confirmed) are exact bodies
            # for the search; STALE ones are position guesses - as
            # bodies they cage the dance (v3 lesson, re-measured here:
            # melee-lo deaths 29 -> 43 with phantoms as bodies), so
            # they degrade to soft hazard reach-cells instead.
            enemies = []
            tidx = None
            stale = []
            for mr2, mc2, mt in melee:
                fresh = (self.tick - mt.last_confirmed) <= 3
                if mt is t:
                    tidx = len(enemies)
                elif not fresh:
                    stale.append((mr2, mc2))
                    continue
                enemies.append(((mr2, mc2), their_dmg_vs(mt),
                                mt.hp_bucket * 20 + 10))
            enemy_cells = {e[0] for e in enemies}
            occ_soft = {cell for cell in occ if cell not in enemy_cells}
            hazards = set()
            for br, bc, _bt in bows:
                for rr in range(br - 4, br + 5):
                    hazards.add((rr, bc))
                for cc in range(bc - 4, bc + 5):
                    hazards.add((br, cc))
            for mr2, mc2 in stale:
                for dr2, dc2 in ((0, 0), (1, 0), (-1, 0), (0, 1),
                                 (0, -1)):
                    hazards.add((mr2 + dr2, mc2 + dc2))
            self.branch = (f"engage[n={len(enemies)},kill={intent_kill}"
                           f",dmg={our_dmg},their={their_dmg}]")
            act = plan_melee((CENTER_ROW, CENTER_COL), enemies, tidx,
                             our_dmg, sword_held, intent_kill,
                             lambda rr, cc: p.passable(rr, cc),
                             occ_soft, hazards)
            return act

        # loot sweep
        if self.loot_sweep_left > 0:
            self.loot_sweep_left -= 1
            best = None
            for r, c, itype, tier in p.item_tiles():
                if itype != I_TOOL or tier <= self._max_tool_tier(p):
                    continue
                d = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
                if d <= 6 and not all(p.inventory):
                    if best is None or d < best[0]:
                        best = (d, r, c)
            if best is not None:
                step = self._safe_step(p, best[1], best[2], bows, melee,
                                       occ)
                if step is not None:
                    self.branch = "loot"
                    return step

        # recovery: sit and regen
        if self.recovering:
            self.branch = "recover"
            return ATN_NOOP

        # equipment
        act = self._equip(p, sword_held)
        if act is not None and not p.in_combat:
            self.branch = "equip"
            return act

        # harvest
        target = self._harvest_target(p)
        if target is not None:
            step = self._safe_step(p, target[0], target[1], bows, melee,
                                   occ)
            if step is not None:
                self.branch = "harvest"
                return step
        elif all(p.inventory) and not p.in_combat and \
                self._junk_slot(p) is not None:
            self.selling = True
            self.branch = "sell"
            return ATN_SELL

        # overwatch / roam (seeking-roam during bootstrap was measured
        # NET-NEGATIVE: movement is exposure; fights come to us)
        self.branch = "wander"
        return self._wander(p, bows, melee, occ,
                            roam=self._urgent_any(p))

    # -- stagnation ------------------------------------------------------

    def _urgent_any(self, p) -> bool:
        m = min(p.comb_lvl, p.prof_lvl)
        if m > self.window_start_min:
            return False
        return (STAG_WINDOW - (self.life_tick % STAG_WINDOW)) <= 150

    def _urgent_comb(self, p) -> bool:
        return self._urgent_any(p) and p.comb_lvl <= p.prof_lvl

    # -- equipment / harvest --------------------------------------------

    def _equip(self, p: Percept, sword_held: bool) -> int | None:
        equipment = p.equipment
        held = equipment[SLOT_HELD]
        for atype, slot in ARMOR_SLOT_BY_TYPE.items():
            best = self._best_unequipped(p, atype)
            cur = equipment[slot]
            if best is not None and (cur == 0 or best[1] > item_tier(cur)):
                if cur == 0:
                    return ATN_ONE + best[0]
                s = self._slot_of(p, cur, equipped=True)
                if s is not None:
                    return ATN_ONE + s
        # held: sword when hunting is useful and we have one; else tool
        want = I_SWORD if (self._best_sword_tier(p) > 0
                           and p.comb_lvl <= p.prof_lvl) else I_TOOL
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
            return ATN_ONE + best[0]
        return None

    def _harvest_target(self, p: Percept):
        held_tier = p.held_tool_tier
        herbs = self._herb_count(p)
        if all(p.inventory):
            return None
        need_sword = self._best_sword_tier(p) < max(held_tier, 1)
        armor_needs = set()
        for atype, slot in ARMOR_SLOT_BY_TYPE.items():
            cur = p.equipment[slot]
            if cur == 0 or item_tier(cur) < held_tier:
                armor_needs.add(atype)
        best = None
        for r, c, itype, tier in p.item_tiles():
            if itype == I_TOOL:
                if tier <= self._max_tool_tier(p):
                    continue
                priority = 0.0
            elif itype in (I_ORE, I_WOOD, I_HILT, I_HERB) or \
                    itype in I_GEM_TYPES:
                if tier > held_tier:
                    continue
                if itype == I_HILT and need_sword:
                    priority = 0.5
                elif p.prof_lvl < tier_level(tier):
                    priority = 1.0 - 0.1 * tier
                elif itype == I_HERB and herbs < HERB_STOCK:
                    priority = 2.0
                elif itype == I_ORE and armor_needs:
                    priority = 3.0
                else:
                    continue
            else:
                continue
            dist = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
            key = (priority, dist)
            if best is None or key < best[0]:
                best = (key, (r, c))
        return best[1] if best else None

    def _wander(self, p: Percept, bows, melee, occ, roam=False) -> int:
        if not roam and self.wander_left <= 0:
            self.idle_ticks += 1
            if self.idle_ticks < 30:
                return ATN_NOOP
        blocked = self._last_move_blocked(p)
        options = []
        for atn, (mr, mc) in MOVE_DELTAS.items():
            if blocked and atn == self.wander_action:
                continue
            nr, nc = CENTER_ROW + mr, CENTER_COL + mc
            if not self._veto(p, nr, nc, bows, melee) and \
                    (nr, nc) not in occ:
                options.append(atn)
        if self.wander_left <= 0 or blocked or \
                self.wander_action not in options:
            if not options:
                return ATN_NOOP
            self.wander_action = int(
                options[self.rng.integers(len(options))])
            self.wander_left = int(self.rng.integers(6, 12))
        self.wander_left -= 1
        if self.wander_left <= 0:
            self.idle_ticks = 0
        if self.wander_left % 3 == 0:
            return ATN_NOOP
        return self.wander_action
