"""Scripted cogame-nmmo player v2: ``python -m players.scripted_player_v2``.

A ground-up redesign of the v1 FSM around the actual score dynamics
(measured locally and derived from the vendored sim source):

- Score = mean min(comb, prof) per life. One long life with min rising
  beats everything; every death or 500-tick stagnation reset banks the
  life and divides the mean AND wipes levels + gear.
- v1 died ~10x/1000 ticks: half of all enemies are level >= 9
  (level ~ 1 + 8*Exp2, 1/8 forced to 1) and a melee NPC in contact hits
  EVERY tick (enemy dmg = 15 + 5L' - 2L - equip_defense; an L9 hits a
  fresh spawn for ~58/tick). Survival is proactive avoidance, not
  reactive fleeing.
- Combat levels ONLY on kills of enemies with comb >= ours (+1); enemy
  kills also drop a tool of tier = level_tier(enemy comb) and gold.
  Tools do NOT spawn on the map - the first kill gates all harvesting.
- Harvesting ore/hilt/wood (tool tier >= resource tier) levels prof
  while prof < 8*2^(tier-1) and yields armor/sword/bow. Armor defense
  8*2^(t-1) per piece and sword attack 3*8*2^(t-1) turn equal-level
  fights from near-lethal to nearly free.
- Fights are chosen by exact damage arithmetic (calc_damage,
  nmmo3.h:1289): our dmg/tick = 40 + 2L + equip_atk - 5L', enemy
  dmg/tick = 15 + 5L' - 2L - equip_def, both floored at 0, enemy hp 99,
  and the enemy gets the first hit when we initiate (players act before
  enemies within a tick, but stepping adjacent is a move, not an
  attack). The delta byte gives L' within a 2-level bucket: at delta d,
  L' in [L + 2d, L + 2d + 1] for d < 4 (d = 0 also covers weaker
  enemies, clamped up).
- The 500-tick stagnation clock is mirrored exactly: the sim compares
  min(comb, prof) at each life-tick multiple of 500 against the value
  at the previous multiple; no strict improvement -> forced reset
  (banks the life, wipes gear). When the deadline approaches with no
  improvement, fight standards relax.

Observation handling (liveness map, move-failure detection, stale
imprint discipline) is inherited from v1 - see scripted_player.py's
docstring; the layout constants and Percept decoder are imported from
there and remain tripwired by tests/test_scripted.py.

Behavior (priority order in MindV2._decide):
  1. exit foreign market UI modes; finish an intentional junk sale;
  2. emergency herb (works in combat, restores 50+10*tier);
  3. committed fight: attack the adjacent target while the exchange
     stays survivable; herb mid-fight; abort if a second enemy joins;
  4. loot sweep: after a kill, grab the dropped tool (despawns in 20
     ticks);
  5. threat safety: RUN from any live enemy hint within its aggro
     range + margin unless it is our chosen fight target;
  6. fight start: pick an adjacent-or-approachable enemy whose
     worst-case exchange cost fits our hp budget (+ herb credit),
     with no second live hint nearby, when comb needs leveling
     (comb <= prof or stagnation urgency);
  7. equip upgrades (tool by default; sword when no harvesting is
     gated on the held slot - i.e. armor slots full and prof capped
     for our tier);
  8. harvest: tool upgrades on the ground, then highest-tier
     prof-leveling resources, then hilt (sword), missing armor (ore),
     herbs to stock;
  9. inventory-full junk sale;
 10. exploration wander (persistent direction, threat-biased, runs).

Determinism: pure function of (obs history, per-agent RNG seeded from
COGAME_PLAYER_SEED); no wall clock, no global state.
"""

from __future__ import annotations

import sys

import numpy as np

from .client import run_policy_main, seed_from_env
from .scripted_player import (
    ANIM_DEATH, ARMOR_SLOT_BY_TYPE, ATN_ATTACK, ATN_DOWN, ATN_NINE,
    ATN_NOOP, ATN_ONE, ATN_SELL, ATN_UP, ATN_RIGHT, ATN_LEFT,
    CENTER_COL, CENTER_ROW, ENTITY_ENEMY, I_GEM_TYPES, I_HERB, I_HILT,
    I_ORE, I_SWORD, I_TOOL, I_WOOD, INVENTORY_SLOTS, MODE_PLAY,
    MODE_SELL_PRICE, MODE_SELL_SELECT, MOVE_DELTAS, NUM_KEY_SLOTS,
    Percept, RUN_OFFSET, SLOT_GEM, SLOT_HELD, WINDOW_COLS, WINDOW_ROWS,
    item_tier, item_type, tier_level,
)

# -- combat model constants (nmmo3.h) ---------------------------------------
BASE_ATTACK = 40          # player base damage (calc_damage)
ENEMY_ATTACK_BASE = 15    # enemy: 40 + 2L' + 3L' - 25 = 15 + 5L'
LEVEL_MUL = 2
ENEMY_LEVEL_MUL = 5       # 2L' defense + 3L' enemy buff
ENEMY_HP = 99
HERB_HEAL = 60            # tier-1; higher tiers heal more (50 + 10t)
NPC_AGGRO = 4             # square chase/aggro range (NPC_AGGRO_RANGE)
STAG_WINDOW = 500         # stagnation check period (c_step)

# -- tunables ---------------------------------------------------------------
LIVE_TTL = 3
AVOID_MARGIN = 2          # stay aggro+margin away from non-target enemies
FIGHT_HP_MARGIN = 12      # required hp headroom beyond predicted cost
EMERGENCY_HP = 38         # eat a herb below this, always
MID_FIGHT_HERB_HP = 45    # eat a herb mid-fight below this
RECOVER_HP = 85           # do not start fresh fights below this
HERB_STOCK = 3
TOOL_GRAB_DIST = 6        # dropped tools despawn in 20 ticks
ATTACK_STREAK_MAX = 10
URGENCY_TICKS = 160       # relax fight rules this close to a predicted
                          # stagnation reset with no min improvement yet
SECOND_ENEMY_RADIUS = 6   # no fight start with another live hint this close


def clamp_delta_levels(my_lvl: int, delta: int) -> tuple[int, int]:
    """Worst/best-case enemy level for an observed delta bucket."""
    if delta >= 4:
        return my_lvl + 8, my_lvl + 40
    lo = max(1, my_lvl + 2 * delta)
    hi = my_lvl + 2 * delta + 1
    return lo, hi


class MindV2:
    """Per-agent state machine; rebuilt from scratch on respawn."""

    def __init__(self, seed, agent_idx: int):
        self._seed = seed
        self._agent_idx = agent_idx
        self._generation = -1
        self.reset()

    def reset(self) -> None:
        self._generation += 1
        self.rng = np.random.default_rng(
            (0 if self._seed is None else self._seed,
             self._agent_idx, self._generation))
        self.life_tick = 0                  # ticks since this life began
        self.stag_baseline = 0              # min() at last 500-tick mark
        self.min_seen = 0
        self.wander_action = ATN_DOWN
        self.wander_left = 0
        self.idle_ticks = 0
        self.last_action = ATN_NOOP
        self.selling = False
        self.fight_target: tuple[int, int] | None = None  # window cell
        self.kill_streak_cell: tuple[int, int] | None = None
        self.attack_streak = 0
        self.last_attack_pos: tuple[int, int] | None = None
        self.prev_tiles: np.ndarray | None = None
        self.live = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=np.int8)
        # ghost[r, c]: this window cell's entity imprint is PROVEN stale
        # (it stayed at the same window offset across a tick we moved -
        # real entities shift in the window when we move, ghosts follow
        # it). High-recall threat detection = imprints minus ghosts.
        self.ghost = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=bool)
        self.unchanged = np.zeros((WINDOW_ROWS, WINDOW_COLS),
                                  dtype=np.int8)
        self.prev_comb = 1
        self.hold_ticks = 0                 # consecutive fight-hold NOOPs
        self.loot_sweep_left = 0            # ticks left to look for a drop

    # -- liveness (inherited v1 design) --------------------------------

    def _own_shift(self, p: Percept) -> tuple[int, int]:
        walk = self.last_action
        if walk - RUN_OFFSET in MOVE_DELTAS:
            walk -= RUN_OFFSET
        if walk not in MOVE_DELTAS:
            return (0, 0)
        dr, dc = MOVE_DELTAS[walk]
        if p.anim == 1:      # ANIM_MOVE
            return (dr, dc)
        if p.anim == 6:      # ANIM_RUN
            return (2 * dr, 2 * dc)
        return (0, 0)

    def _update_liveness(self, p: Percept) -> None:
        if self.prev_tiles is None:
            self.live = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=np.int8)
            self.ghost = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=bool)
            self.prev_tiles = p.tiles.copy()
            return
        changed = (p.tiles[:, :, 4:]
                   != self.prev_tiles[:, :, 4:]).any(axis=2)
        dr, dc = self._own_shift(p)
        r0, r1 = max(0, -dr), min(WINDOW_ROWS, WINDOW_ROWS - dr)
        c0, c1 = max(0, -dc), min(WINDOW_COLS, WINDOW_COLS - dc)
        shifted = np.zeros_like(self.live)
        if r0 < r1 and c0 < c1:
            shifted[r0:r1, c0:c1] = self.live[r0 + dr:r1 + dr,
                                              c0 + dc:c1 + dc]
        self.live = np.where(changed, np.int8(LIVE_TTL),
                             np.maximum(shifted - 1, 0).astype(np.int8))
        # ghost proving. Proofs that an imprint is stale residue:
        # (a) we RAN this tick (2 tiles) and the cell's bytes did not
        #     change - no enemy walks 2 tiles/tick, so nothing real can
        #     hold its window cell across a run;
        # (b) we WALKED and an unchanged cell is not behind our motion -
        #     an enemy chasing directly behind us at equal speed keeps
        #     both its window cell and its bytes constant, so behind
        #     cells stay unproven while walking (the chaser-ghosting
        #     trap: ghosting them turns real pursuers invisible);
        # (c) we are stationary, not in combat, and the cell's bytes
        #     have been byte-identical for 3+ consecutive ticks - a real
        #     enemy inside its own aggro range would be approaching or
        #     hitting us (both rewrite bytes).
        # A changed cell was rewritten this tick -> real entity there.
        if (dr, dc) != (0, 0):
            running = max(abs(dr), abs(dc)) >= 2
            if running:
                self.ghost = ~changed
            else:
                rows = np.arange(WINDOW_ROWS)[:, None] - CENTER_ROW
                cols = np.arange(WINDOW_COLS)[None, :] - CENTER_COL
                behind = (rows * dr + cols * dc) < 0
                self.ghost = ~changed & (~behind | self.ghost)
            self.unchanged = np.zeros((WINDOW_ROWS, WINDOW_COLS),
                                      dtype=np.int8)
            # cells that slid in from the border are unknown, not ghost
            if dr > 0:
                self.ghost[WINDOW_ROWS - dr:, :] = False
            elif dr < 0:
                self.ghost[:-dr, :] = False
            if dc > 0:
                self.ghost[:, WINDOW_COLS - dc:] = False
            elif dc < 0:
                self.ghost[:, :-dc] = False
        else:
            self.ghost &= ~changed
            self.unchanged = np.where(
                changed, 0,
                np.minimum(self.unchanged + 1, 100)).astype(np.int8)
            if not p.in_combat:
                self.ghost |= self.unchanged >= 3
        self.prev_tiles = p.tiles.copy()

    def _threat_cells(self, p: Percept):
        """High-recall threat set: every enemy imprint not proven ghost
        (corpse anims already filtered by Percept.enemy_hints)."""
        return [(r, c, d) for r, c, d, _h in p.enemy_hints()
                if not self.ghost[r, c]]

    def _live_hints(self, p: Percept):
        """(r, c, delta, hp_bucket) for trusted enemy cells."""
        out = []
        for r, c, d, hpb in p.enemy_hints():
            if self.live[r, c] <= 0:
                if not p.in_combat:
                    continue
                ar, ac = abs(r - CENTER_ROW), abs(c - CENTER_COL)
                in_attack_position = (max(ar, ac) <= 1
                                      or (min(ar, ac) == 0
                                          and max(ar, ac) <= 4))
                if not in_attack_position:
                    continue
            out.append((r, c, d, hpb))
        return out

    # -- movement helpers (v1 lineage) ---------------------------------

    def _passable_moves(self, p: Percept, avoid=None):
        occ = getattr(self, "_occupied", ())
        moves = []
        for atn, (dr, dc) in MOVE_DELTAS.items():
            if avoid is not None and atn == avoid:
                continue
            nr, nc = CENTER_ROW + dr, CENTER_COL + dc
            if p.passable(nr, nc) and (nr, nc) not in occ:
                moves.append(atn)
        return moves

    @staticmethod
    def _dist(r, c, cells):
        return min(max(abs(r - tr), abs(c - tc)) for tr, tc in cells)

    @staticmethod
    def _bow_risk(r, c, bows) -> bool:
        """Is window position (r, c) inside a possible-bow kill funnel?
        Bows (every enemy >= level 15) one-shot along their row/column
        at range <= 4 and walk 1/tick toward alignment: any cell within
        Chebyshev 6 whose smaller axis offset is <= 1 is at most one
        enemy step from a landed arrow."""
        for br, bc in bows:
            if max(abs(r - br), abs(c - bc)) <= 6 and \
                    min(abs(r - br), abs(c - bc)) <= 1:
                return True
        return False

    def _safe_moves(self, p: Percept, threats, avoid=None, bows=None):
        if bows is None:
            bows = getattr(self, '_bows', ())
        moves = self._passable_moves(p, avoid=avoid)
        if not moves or (not threats and not bows):
            return moves
        here = self._dist(CENTER_ROW, CENTER_COL, threats) \
            if threats else 99

        def keeps_distance(atn):
            dr, dc = MOVE_DELTAS[atn]
            return not threats or self._dist(
                CENTER_ROW + dr, CENTER_COL + dc, threats) >= here

        def bow_ok(atn):
            dr, dc = MOVE_DELTAS[atn]
            return not self._bow_risk(CENTER_ROW + dr, CENTER_COL + dc,
                                      bows)

        for pred in ((lambda a: keeps_distance(a) and bow_ok(a)),
                     bow_ok, keeps_distance):
            good = [a for a in moves if pred(a)]
            if good:
                return good
        return moves

    def _maybe_run(self, p: Percept, atn: int) -> int:
        if atn not in MOVE_DELTAS:
            return atn
        dr, dc = MOVE_DELTAS[atn]
        if p.passable(CENTER_ROW + dr, CENTER_COL + dc) and \
                p.passable(CENTER_ROW + 2 * dr, CENTER_COL + 2 * dc) \
                and not self._bow_risk(CENTER_ROW + 2 * dr,
                                       CENTER_COL + 2 * dc,
                                       getattr(self, '_bows', ())):
            return atn + RUN_OFFSET
        return atn

    def _last_move_blocked(self, p: Percept) -> bool:
        was_move = (self.last_action in MOVE_DELTAS
                    or self.last_action - RUN_OFFSET in MOVE_DELTAS)
        return was_move and p.anim == 0    # ANIM_IDLE

    def _step_towards(self, p: Percept, threats, r: int, c: int,
                      run_ok: bool = True) -> int | None:
        dr = r - CENTER_ROW
        dc = c - CENTER_COL
        prefs = []
        row_atn = ATN_DOWN if dr > 0 else ATN_UP
        col_atn = ATN_RIGHT if dc > 0 else ATN_LEFT
        if abs(dr) >= abs(dc):
            if dr != 0:
                prefs.append(row_atn)
            if dc != 0:
                prefs.append(col_atn)
        else:
            prefs.append(col_atn)
            if dr != 0:
                prefs.append(row_atn)
        if self.last_action in prefs and self._last_move_blocked(p):
            prefs.remove(self.last_action)
            if not prefs:
                return None     # single blocked lane: let the caller
                                # pick another goal instead of pressing
        allowed = self._safe_moves(p, threats)
        for atn in prefs:
            if atn in allowed:
                axis = abs(dr) if atn in (ATN_DOWN, ATN_UP) else abs(dc)
                if run_ok and axis >= 2:
                    return self._maybe_run(p, atn)
                return atn
        return None

    def _wander(self, p: Percept, threats, roam: bool = False) -> int:
        """Overwatch movement: stand still by default. A stationary
        observer has near-perfect threat detection (everything real
        moves relative to it) and attracts fights to finish on its own
        terms; roaming is what walks agents into bow funnels. Roam only
        when there has been nothing to do for a while (or the caller
        forces it), in short legs with scan pauses.
        """
        self.idle_ticks = getattr(self, "idle_ticks", 0)
        if not roam and self.wander_left <= 0:
            self.idle_ticks += 1
            if self.idle_ticks < 30:
                return ATN_NOOP
        # take (or continue) a roam leg
        blocked = self._last_move_blocked(p)
        options = self._safe_moves(
            p, threats, avoid=self.wander_action if blocked else None)
        if self.wander_left <= 0 or blocked or \
                self.wander_action not in options:
            if not options:
                return ATN_NOOP
            self.wander_action = int(options[self.rng.integers(len(options))])
            self.wander_left = int(self.rng.integers(6, 12))
        self.wander_left -= 1
        if self.wander_left <= 0:
            self.idle_ticks = 0     # leg over: overwatch pause restarts
        # pause every third step to re-scan (diff detection needs
        # stationary ticks to prove ghosts and spot movers)
        if self.wander_left % 3 == 0:
            return ATN_NOOP
        return self._maybe_run(p, self.wander_action)

    # -- inventory helpers ---------------------------------------------

    def _herb_slot(self, p: Percept) -> int | None:
        for idx, item in enumerate(p.inventory[:NUM_KEY_SLOTS]):
            if item and item_type(item) == I_HERB:
                return idx
        return None

    def _herb_count(self, p: Percept) -> int:
        return sum(1 for i in p.inventory if i and item_type(i) == I_HERB)

    def _slot_of(self, p: Percept, raw_id: int, equipped: bool):
        inv, eq = p.inventory, p.is_equipped
        for idx in range(NUM_KEY_SLOTS):
            if inv[idx] == raw_id and bool(eq[idx]) == equipped:
                return idx
        return None

    def _best_unequipped(self, p: Percept, t: int):
        """(slot, tier) of the best unequipped item of type t."""
        best = None
        for idx in range(NUM_KEY_SLOTS):
            item = p.inventory[idx]
            if item and item_type(item) == t and not p.is_equipped[idx]:
                if best is None or item_tier(item) > best[1]:
                    best = (idx, item_tier(item))
        return best

    def _best_sword_tier(self, p: Percept) -> int:
        tiers = [item_tier(i) for i in p.inventory
                 if i and item_type(i) == I_SWORD]
        held = p.equipment[SLOT_HELD]
        if held and item_type(held) == I_SWORD:
            tiers.append(item_tier(held))
        return max(tiers, default=0)

    def _junk_slot(self, p: Percept) -> int | None:
        """Expendable slot for a dump: not equipped, not herb, not our
        best tool/sword, not armor awaiting an empty slot, not a
        higher-tier armor upgrade."""
        inv, eq = p.inventory, p.is_equipped
        equipment = p.equipment
        best_tool = max([item_tier(i) for i in inv
                         if i and item_type(i) == I_TOOL], default=0)
        best_sword = max([item_tier(i) for i in inv
                          if i and item_type(i) == I_SWORD], default=0)
        candidates = []
        for idx in range(NUM_KEY_SLOTS):
            item = inv[idx]
            if not item or eq[idx]:
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
                cur = equipment[ARMOR_SLOT_BY_TYPE[t]]
                if cur == 0 or item_tier(cur) < tier:
                    continue
            candidates.append((tier, idx))
        if not candidates:
            return None
        return min(candidates)[1]

    # -- combat model ---------------------------------------------------

    def _exchange_cost(self, p: Percept, delta: int,
                       initiate: bool) -> tuple[int, int]:
        """(predicted hp cost, ticks to kill) vs the WORST-case level in
        the delta bucket; 10**6 cost when we cannot damage it."""
        my = p.comb_lvl
        eq_atk = int(p.scalars[45])   # S_EQUIP_ATTACK
        eq_def = int(p.scalars[46])   # S_EQUIP_DEFENSE
        lo, hi = clamp_delta_levels(my, delta)
        our_dmg = BASE_ATTACK + LEVEL_MUL * my + eq_atk - ENEMY_LEVEL_MUL * hi
        their_dmg = max(ENEMY_ATTACK_BASE + ENEMY_LEVEL_MUL * hi
                        - LEVEL_MUL * my - eq_def, 0)
        if our_dmg <= 0:
            return 10 ** 6, 10 ** 6
        ticks = -(-ENEMY_HP // our_dmg)       # ceil
        hits_taken = ticks - 1 + (1 if initiate else 0)
        return hits_taken * their_dmg, ticks

    def _fight_budget(self, p: Percept, urgent: bool) -> int:
        budget = p.hp - FIGHT_HP_MARGIN
        herbs = self._herb_count(p)
        if herbs:
            budget += HERB_HEAL * min(herbs, 2 if urgent else 1)
        if urgent or (p.comb_lvl == 1 and p.held_tool_tier == 0):
            # bootstrap: the first kill gates the whole economy (tools
            # only drop from kills); accept a near-full-bar exchange
            budget = max(budget, p.hp - 4)
        return budget

    def _fight_unsafe(self, p: Percept, threats, target) -> bool:
        """A non-target threat makes engaging here reckless: anything
        within 3 of us or the target, any strong (delta >= 2) imprint
        within 5, or anything on our row/column within bow range + 1
        (level >= 15 enemies one-shot from range 4 on-axis)."""
        tr, tc = target
        for r, c, d in threats:
            if max(abs(r - tr), abs(c - tc)) <= 1:
                continue    # the target itself
            dist = max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
            if dist <= 3:
                return True
            if max(abs(r - tr), abs(c - tc)) <= 3:
                return True
            if d >= 2 and dist <= 5:
                return True
            if (r == CENTER_ROW or c == CENTER_COL) and dist <= 5:
                return True
        return False

    # -- stagnation clock ----------------------------------------------

    def _update_stagnation(self, p: Percept) -> None:
        m = min(p.comb_lvl, p.prof_lvl)
        self.min_seen = max(self.min_seen, m)
        if self.life_tick > 0 and self.life_tick % STAG_WINDOW == 0:
            self.stag_baseline = m

    def _urgency(self, p: Percept) -> bool:
        """True when the next 500-tick check would reset us and it is
        close (act early: the sim's life-tick counter may lead ours by
        the 2-tick death animation)."""
        m = min(p.comb_lvl, p.prof_lvl)
        if m > self.stag_baseline:
            return False
        remaining = STAG_WINDOW - (self.life_tick % STAG_WINDOW)
        return remaining <= URGENCY_TICKS

    # -- the FSM --------------------------------------------------------

    def act(self, obs: bytes) -> int:
        p = Percept(obs)
        self.life_tick += 1
        self._update_liveness(p)
        self._update_stagnation(p)

        # kill detection: comb level rose -> our target died; sweep for
        # the dropped tool (despawns in 20 ticks)
        if p.comb_lvl > self.prev_comb:
            self.loot_sweep_left = 18
            self.fight_target = None
            self.attack_streak = 0
        self.prev_comb = p.comb_lvl

        # missed-attack ghost feedback (v1 design)
        if (self.last_action == ATN_ATTACK and not p.in_combat
                and self.last_attack_pos is not None):
            self.live[self.last_attack_pos] = 0
            self.fight_target = None

        action = self._decide(p)

        if action == ATN_ATTACK:
            self.attack_streak += 1
        else:
            self.attack_streak = 0
            self.last_attack_pos = None
        self.last_action = action
        return action

    def _decide(self, p: Percept) -> int:
        hints = self._live_hints(p)
        all_cells = self._threat_cells(p)
        # STRONG imprints (delta >= 2) are the only things we flee:
        # they include every possible bow (level >= 15 one-shots
        # on-axis at range 4) and melee we cannot trade with. Weak
        # enemies (delta 0-1) are food - something is inside aggro
        # range most of the time in this world, so universal avoidance
        # just thrashes; the durable defense is armor, not distance.
        strong = [(r, c) for r, c, d in all_cells if d >= 2]
        self._bows = strong
        urgent = self._urgency(p)

        # 1. market UI
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
            return self._wander(p, strong)
        self.selling = False

        # 2. emergency herb
        if p.hp < EMERGENCY_HP:
            herb = self._herb_slot(p)
            if herb is not None:
                return ATN_ONE + herb

        held = p.equipment[SLOT_HELD]
        sword_held = bool(held) and item_type(held) == I_SWORD

        def attackable(r, c):
            ar, ac = abs(r - CENTER_ROW), abs(c - CENTER_COL)
            if ar + ac == 1:
                return True
            return sword_held and max(ar, ac) == 1

        # weak imprints adjacent to us right now (attack resolves
        # ground truth: find_target scans all four directions)
        adjacent_weak = [(r, c) for r, c, d in all_cells
                         if d <= 1 and attackable(r, c)]
        adjacent_strong = [(r, c) for r, c in strong if attackable(r, c)]

        # 3. committed fight: keep hitting while an adjacent enemy is
        # there. Aborting refunds nothing (the enemy regens; kills are
        # the only comb source), so fight it out - unless the exchange
        # has turned hopeless: swarmed, or low hp with no heal and the
        # target far from dead. Disengage RUNS (2 tiles vs their 1).
        if self.attack_streak > 0 and p.in_combat:
            if p.hp < MID_FIGHT_HERB_HP:
                herb = self._herb_slot(p)
                if herb is not None:
                    return ATN_ONE + herb
            adj_all = [(r, c, d, hpb)
                       for r, c, d, hpb in
                       [(r, c, d, h) for r, c, d, h in p.enemy_hints()
                        if not self.ghost[r, c]]
                       if attackable(r, c)]
            target = None
            if self.last_attack_pos is not None:
                for cell in adj_all:
                    if max(abs(cell[0] - self.last_attack_pos[0]),
                           abs(cell[1] - self.last_attack_pos[1])) <= 1:
                        target = cell
                        break
            if target is None and adj_all:
                target = min(adj_all, key=lambda t: (t[2], t[3]))
            swarmed = len(adj_all) >= 2
            hopeless = (target is not None and p.hp < 30
                        and self._herb_slot(p) is None
                        and target[3] >= 2)
            if target is not None and not swarmed and not hopeless \
                    and self.attack_streak < ATTACK_STREAK_MAX:
                r, c = target[0], target[1]
                self.last_attack_pos = (r, c)
                return ATN_ATTACK
            if (swarmed or hopeless) and adj_all:
                cells = [(t[0], t[1]) for t in adj_all]
                moves = self._passable_moves(p)
                if moves:
                    away = max(moves, key=lambda a: self._dist(
                        CENTER_ROW + MOVE_DELTAS[a][0],
                        CENTER_COL + MOVE_DELTAS[a][1], cells))
                    return self._maybe_run(p, away)

        # 4. loot sweep: grab the tool our kill just dropped (despawns
        # in 20 ticks)
        if self.loot_sweep_left > 0:
            self.loot_sweep_left -= 1
            best = None
            for r, c, itype, tier in p.item_tiles():
                if itype != I_TOOL or tier <= self._max_tool_tier(p):
                    continue
                d = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
                if d <= TOOL_GRAB_DIST and not all(p.inventory):
                    if best is None or d < best[0]:
                        best = (d, r, c)
            if best is not None:
                step = self._step_towards(p, strong, best[1], best[2],
                                          run_ok=False)
                if step is not None:
                    return step

        # 5. strong-threat safety. Flee only PROVEN MOVERS (live
        # cells): a stationary strong imprint is either stale residue
        # (ghost-proving needs stationary ticks - fleeing it prevents
        # the proof and causes flee-forever loops) or a dormant enemy
        # that cannot hurt us until it moves, which lights it up.
        live_strong = [(r, c) for r, c in strong if self.live[r, c] > 0]
        danger = [(r, c) for r, c in live_strong
                  if max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
                  <= NPC_AGGRO + AVOID_MARGIN]
        bow_here = self._bow_risk(CENTER_ROW, CENTER_COL, live_strong)
        if danger or bow_here:
            moves = self._passable_moves(p)
            if moves:
                def flee_score(atn):
                    dr, dc = MOVE_DELTAS[atn]
                    nr, nc = CENTER_ROW + dr, CENTER_COL + dc
                    dist = self._dist(nr, nc, danger) if danger else 99
                    return (not self._bow_risk(nr, nc, live_strong), dist)
                best = max(moves, key=flee_score)
                here = self._dist(CENTER_ROW, CENTER_COL, danger) \
                    if danger else 99
                nr, nc = (CENTER_ROW + MOVE_DELTAS[best][0],
                          CENTER_COL + MOVE_DELTAS[best][1])
                gain = danger and self._dist(nr, nc, danger) > here
                dodges = bow_here and not self._bow_risk(nr, nc,
                                                         live_strong)
                if gain or dodges or (danger and here <= 2):
                    return self._maybe_run(p, best)
            # boxed in or safe-ish: fall through

        # 6. fight: attack adjacent weak enemies when healthy enough,
        # or hold for an approaching live one (players act before
        # enemies each tick, so the adjacency tick is OUR first strike)
        need_comb = p.comb_lvl <= p.prof_lvl or urgent \
            or p.held_tool_tier == 0
        budget = self._fight_budget(p, urgent)
        cost0, _ = self._exchange_cost(p, 0, initiate=False)
        adjacent_all = [(d, r, c) for r, c, d in all_cells
                        if attackable(r, c)]
        adjacent_ok = []
        for d, r, c in adjacent_all:
            if d > 1:
                continue
            cost_d, _t = self._exchange_cost(p, d, initiate=False)
            if cost_d <= budget or (p.in_combat and d == 0):
                adjacent_ok.append((d, r, c))
        if adjacent_all:
            # an adjacent enemy MUST resolve to attack or flee - melee
            # NPCs hit every tick at contact; standing there undecided
            # is how single low-level enemies grind agents down.
            if need_comb and len(adjacent_all) == 1 and adjacent_ok:
                _d, r, c = min(adjacent_ok)
                self.last_attack_pos = (r, c)
                return ATN_ATTACK
            cells = [(r, c) for _d, r, c in adjacent_all]
            moves = self._passable_moves(p)
            if moves:
                away = max(moves, key=lambda a: self._dist(
                    CENTER_ROW + MOVE_DELTAS[a][0],
                    CENTER_COL + MOVE_DELTAS[a][1], cells))
                return self._maybe_run(p, away)
        if need_comb and not strong and cost0 <= budget:
            # hold for the nearest live weak enemy already chasing us
            cand = [(max(abs(r - CENTER_ROW), abs(c - CENTER_COL)),
                     r, c)
                    for r, c, d, _h in hints if d <= 1]
            cand = [t for t in cand if t[0] <= NPC_AGGRO]
            if cand:
                dist, r, c = min(cand)
                self.hold_ticks += 1
                if self.hold_ticks > 8:
                    self.ghost[r, c] = True
                    self.live[r, c] = 0
                    self.hold_ticks = 0
                else:
                    return ATN_NOOP
            else:
                self.hold_ticks = 0

        # 7. equipment management (out of combat only)
        if not p.in_combat:
            act = self._equip_action(p)
            if act is not None:
                return act

        # 8. harvest (paths keep distance from strong imprints only)
        target = self._harvest_target(p)
        if target is not None:
            step = self._step_towards(p, strong, *target, run_ok=False)
            if step is not None:
                return step
        elif all(p.inventory) and not p.in_combat and \
                self._junk_slot(p) is not None:
            self.selling = True
            return ATN_SELL

        # 9. overwatch / roam (forced roam under stagnation urgency:
        # progress requires finding something to kill or harvest)
        return self._wander(p, strong, roam=urgent)

    def _equip_action(self, p: Percept) -> int | None:
        """Keep: best tool held while harvesting still pays; best sword
        held otherwise; armor in every slot; never gems."""
        equipment = p.equipment
        held = equipment[SLOT_HELD]

        # armor first: an empty slot with a piece waiting, or an upgrade
        for atype, slot in ARMOR_SLOT_BY_TYPE.items():
            best = self._best_unequipped(p, atype)
            cur = equipment[slot]
            if best is not None and (cur == 0 or
                                     best[1] > item_tier(cur)):
                if cur == 0:
                    return ATN_ONE + best[0]
                cur_slot = self._slot_of(p, cur, equipped=True)
                if cur_slot is not None:
                    return ATN_ONE + cur_slot   # unequip old first
        # never hold a gem
        if held and item_type(held) in I_GEM_TYPES:
            s = self._slot_of(p, held, equipped=True)
            if s is not None:
                return ATN_ONE + s

        best_tool = self._best_unequipped(p, I_TOOL)
        best_sword = self._best_unequipped(p, I_SWORD)
        held_type = item_type(held) if held else None
        held_tier = item_tier(held) if held else 0

        want = I_TOOL
        if not self._harvest_pays(p):
            want = I_SWORD if (best_sword or held_type == I_SWORD) \
                else I_TOOL

        if want == I_TOOL:
            if held_type == I_TOOL:
                if best_tool and best_tool[1] > held_tier:
                    s = self._slot_of(p, held, equipped=True)
                    if s is not None:
                        return ATN_ONE + s
                return None
            if best_tool:
                if held:
                    s = self._slot_of(p, held, equipped=True)
                    if s is not None:
                        return ATN_ONE + s
                else:
                    return ATN_ONE + best_tool[0]
            return None
        # want sword
        if held_type == I_SWORD:
            if best_sword and best_sword[1] > held_tier:
                s = self._slot_of(p, held, equipped=True)
                if s is not None:
                    return ATN_ONE + s
            return None
        if best_sword:
            if held:
                s = self._slot_of(p, held, equipped=True)
                if s is not None:
                    return ATN_ONE + s
            else:
                return ATN_ONE + best_sword[0]
        return None

    def _harvest_pays(self, p: Percept) -> bool:
        """Is there still value in holding the tool? True while prof can
        level at our tool tier, an armor slot is missing/upgradeable at
        our tool tier, we lack a sword, or herbs are short."""
        t = self._max_tool_tier(p)
        if t == 0:
            return True     # nothing else to hold anyway
        if p.prof_lvl < tier_level(t):
            return True
        if self._herb_count(p) < HERB_STOCK:
            return True
        if self._best_sword_tier(p) < t:
            return True
        for atype, slot in ARMOR_SLOT_BY_TYPE.items():
            cur = p.equipment[slot]
            if cur == 0 or item_tier(cur) < t:
                return True
        return False

    def _max_tool_tier(self, p: Percept) -> int:
        tiers = [item_tier(i) for i in p.inventory
                 if i and item_type(i) == I_TOOL]
        held = p.equipment[SLOT_HELD]
        if held and item_type(held) == I_TOOL:
            tiers.append(item_tier(held))
        return max(tiers, default=0)

    def _harvest_target(self, p: Percept):
        """Best item tile to walk onto: (priority, dist) minimised.
        Priorities: 0 tool upgrade, 1 prof-leveling resource (highest
        tier first via tier bonus), 2 hilt for a missing sword, 3 ore
        for missing/upgradeable armor, 4 herb stock."""
        held_tier = p.held_tool_tier
        herbs = self._herb_count(p)
        inventory_full = all(p.inventory)
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
                levels_prof = p.prof_lvl < tier_level(tier)
                if levels_prof:
                    priority = 1.0 - 0.1 * tier
                elif itype == I_HILT and need_sword:
                    priority = 2.0
                elif itype == I_ORE and armor_needs:
                    priority = 3.0
                elif itype == I_HERB and herbs < HERB_STOCK:
                    priority = 4.0
                else:
                    continue
            else:
                continue
            if inventory_full:
                continue
            dist = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
            key = (priority, dist)
            if best is None or key < best[0]:
                best = (key, (r, c))
        return None if best is None else best[1]


class ScriptedPolicyV2:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.minds: dict[int, MindV2] = {}

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        actions = []
        for i, row in enumerate(obs_rows):
            mind = self.minds.get(i)
            if mind is None:
                mind = self.minds[i] = MindV2(self.seed, i)
            if resets[i]:
                mind.reset()
            actions.append([mind.act(bytes(row))])
        return actions


def policy_from_env() -> ScriptedPolicyV2:
    return ScriptedPolicyV2(seed_from_env(default=0))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
