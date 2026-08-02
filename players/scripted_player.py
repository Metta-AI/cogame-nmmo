"""Scripted cogame-nmmo player (daveey's league entry):
``python -m players.scripted_player``.

A survival finite-state machine driven ONLY by the egocentric 11x15 tile
window and the 47 self scalars — worlds are procedural per seed, so
there is no global map to embed (unlike the moba scripted bot). Every
layout constant below is derived from the vendored upstream sources and
tripwired against them in tests/test_scripted.py (the factors table,
window dims, scalar offsets, action/item/tile ids).

Observation trust model (from compute_all_obs, vendored nmmo3.h:926-1020):

- Tile bytes 0-3 (terrain season/type, item type/tier) are RELIABLE:
  rewritten for every window tile every tick.
- Tile bytes 4-9 (entity type/element/delta-comb/hp/anim/dir) are
  STALE-PRONE, and the staleness is WINDOW-RELATIVE: the obs buffer is
  indexed by window cell (obs_adr just increments across the 11x15 scan
  each tick), and a cell's entity bytes are written only when an entity
  currently occupies the world tile under it. A cell therefore keeps
  the imprint of the last entity ever observed at that RELATIVE offset
  — the imprint follows the agent around (verified against the sim: an
  enemy imprint stays at the same window cell while the terrain bytes
  under it change as we walk). Consequences: an imprint can never be
  escaped by moving, and reacting to raw entity bytes locks into
  attack-forever / flee-forever loops.

  The bot therefore maintains a LIVENESS map over the window from
  consecutive-obs diffs (the design doc's freshness-tracking option).
  The detector diffs each cell against ITS OWN bytes last tick — same
  window offset, deliberately NOT motion-corrected: entity bytes at an
  offset are only ever written when an entity currently occupies the
  world tile under it, so a changed cell was necessarily REWRITTEN this
  tick, i.e. an entity is there RIGHT NOW (near-perfect precision, no
  matter how we moved). Changed cells are trusted for LIVE_TTL ticks;
  the decaying trust map IS shifted by our own movement (fresh anim
  scalar says whether the last move/run succeeded; the action says
  which way), because the trusted entities are world-anchored. Recall
  is imperfect — an entity whose visible state is static writes
  identical bytes and goes dark — so while our fresh in_combat scalar
  says someone is hitting us, adjacent enemy imprints are trusted
  regardless. Additional cross-checks:
    * corpse filter: a killed enemy's imprint freezes on
      anim==ANIM_DEATH — never a target;
    * hit feedback: attack() sets OUR in_combat to 5 on any connected
      hit (nmmo3.h:1398) and the in_combat scalar is always fresh, so
      "I attacked last tick and in_combat is still 0" proves the target
      cell was a ghost — its liveness is zeroed; an attack-streak cap
      backstops the corner cases.
  Only LIVE hints drive fighting, fleeing, and threat avoidance; a
  stale hint acted on once costs one wasted tick (find_target,
  nmmo3.h:1310, finds nothing), and the machinery above keeps one
  wasted tick from becoming a loop. Harvest pathing ignores entity
  bytes entirely (moving into an occupied tile just fails, and the
  move-failure detector reroutes). The one entity cell that is NEVER
  stale is the window center: we occupy it, so it is rewritten every
  tick.
- The 47 scalars are always fresh (rewritten every tick from our own
  Entity). They are decoded in full below (offsets nmmo3.h:980-1006).
  Move-failure detection uses the anim scalar: c_step sets anim=IDLE
  each tick and move() sets MOVE only on success, so anim==IDLE after a
  move action means the step was blocked.

Behavior (priority order in ScriptedMind._decide):
  1. exit market UI modes we did not initiate (a non-noop, non-number
     action exits without side effects);
  2. finish an intentional inventory-dump sale (sell mode is the only
     way to free slots; driven by the fresh ui_mode scalar);
  3. eat a herb when hp is low (works in combat, +50 hp minimum);
  4. fight: ATN_ATTACK an adjacent LIVE enemy hint whose delta-comb
     byte is 0 — delta = clamp((seen_comb - my_comb)/2, 0, 4), so 0
     means at most one level above us: beatable, and killing a foe at
     our level or above is the ONLY source of comb levels
     (attack(), nmmo3.h:1431) and of tool drops (drop_loot). Fights
     start only rested (or cheaply, on a wounded enemy) and are then
     fought out — aborting refunds nothing;
  5. tool grab: step onto a nearby dropped tool before it despawns
     (20 ticks), even next to enemies — a held tool gates ALL
     harvesting (pickup_item nmmo3.h:1125-1192 requires held tool tier
     >= resource tier; harvesting is move-onto-tile, pickup runs inside
     move());
  6. disengage: RUN from any live enemy hint in touching range that we
     are not actively fighting (every enemy chases every player in
     aggro range; walking cannot break a chase — equal speed);
  7. equip upgrades when out of combat: best tool in the held slot,
     then armor into empty slots. Gems are deliberately never equipped:
     they set our element, and EFFECT_MATRIX rows contain 0x matchups,
     while the neutral attacker row is all 1x;
  8. harvest: walk/run onto the best visible item tile — tools of a
     higher tier than held, then prof-leveling resources (prof levels
     when prof < tier_level(tier), nmmo3.h:1173), then herbs for the
     heal stock; greedy per-axis pathing over terrain passability
     (grass/dirt pass, stone/water block) with perpendicular rerouting
     on blocked moves;
  9. anti-stagnation wander: persistent random direction (seeded RNG),
     run when clear, re-rolled on blockage — the 500-tick
     no-improvement reset (c_step nmmo3.h:1926-1956) is the enemy, so
     keep moving and keep min(comb, prof) rising.

On a wire ``resets[j]`` flag (death or stagnation reset — the new life
starts at comb=prof=1 somewhere else) the per-agent mind state is
cleared and behavior restarts from scratch.

Determinism: a pure function of (obs history, per-agent RNG seeded from
COGAME_PLAYER_SEED) — no wall clock, no global state.
"""

from __future__ import annotations

import sys

import numpy as np

from .client import run_policy_main, seed_from_env

# ---------------------------------------------------------------------------
# Obs layout constants — tripwired against vendor/upstream/{nmmo3.c,nmmo3.h}
# in tests/test_scripted.py. Do not edit without reading compute_all_obs.
# ---------------------------------------------------------------------------

WINDOW_ROWS = 11           # 2*y_window + 1, y_window=5 (nmmo3.c demo struct)
WINDOW_COLS = 15           # 2*x_window + 1, x_window=7
TILE_BYTES = 10
CENTER_ROW = 5             # our own tile within the window
CENTER_COL = 7
NUM_SCALARS = 47
NUM_REWARD_BYTES = 10
OBS_SIZE = WINDOW_ROWS * WINDOW_COLS * TILE_BYTES + NUM_SCALARS + NUM_REWARD_BYTES
SCALARS_OFF = WINDOW_ROWS * WINDOW_COLS * TILE_BYTES   # 1650
REWARD_OFF = SCALARS_OFF + NUM_SCALARS                 # 1697

# One-hot factor table (nmmo3.c:87) — the per-byte value ranges of a tile
# cell: [terrain season, terrain type, item type, item tier, entity type,
# element, delta comb, hp bucket, anim, dir].
FACTORS = (4, 4, 17, 5, 3, 5, 5, 5, 7, 4)

# Tile-cell byte indices (compute_all_obs, nmmo3.h:948-973)
TB_SEASON = 0        # terrain % 4
TB_TERRAIN = 1       # terrain / 4: 0 grass, 1 dirt, 2 stone, 3 water
TB_ITEM_TYPE = 2     # item id % 17 (I_* type, 0 = empty)
TB_ITEM_TIER = 3     # item id / 17 (tier - 1)
TB_ENT_TYPE = 4      # stale-prone from here down
TB_ENT_ELEMENT = 5
TB_ENT_DELTA = 6     # clamp((seen comb - my comb) / 2, 0, 4)
TB_ENT_HP = 7        # hp / 20
TB_ENT_ANIM = 8
TB_ENT_DIR = 9

TERRAIN_GRASS = 0
TERRAIN_DIRT = 1
TERRAIN_STONE = 2
TERRAIN_WATER = 3

# Self-scalar offsets within the 47 (compute_all_obs, nmmo3.h:980-1006)
S_TYPE = 0
S_COMB_LVL = 1
S_ELEMENT = 2
S_DIR = 3
S_ANIM = 4
S_HP = 5
S_HP_MAX = 6
S_PROF_LVL = 7
S_UI_MODE = 8
S_MARKET_TIER = 9
S_SELL_IDX = 10
S_GOLD = 11
S_IN_COMBAT = 12
S_EQUIPMENT = 13         # 5 slots: helm, chest, legs, held, gem
S_INVENTORY = 18         # 12 slots of raw item ids
S_IS_EQUIPPED = 30       # 12 parallel bools
S_WANDER_RANGE = 42
S_RANGED = 43
S_GOAL = 44
S_EQUIP_ATTACK = 45
S_EQUIP_DEFENSE = 46

INVENTORY_SLOTS = 12
NUM_KEY_SLOTS = 9        # only ATN_ONE..ATN_NINE exist: slots 0-8 usable

# Equipment slot indices (nmmo3.h:152-156)
SLOT_HELM = 0
SLOT_CHEST = 1
SLOT_LEGS = 2
SLOT_HELD = 3
SLOT_GEM = 4

# Actions (nmmo3.h:46-70; id 6 is unassigned, ZERO/MINUS/EQUALS are
# semantic no-ops in play mode)
ATN_DOWN = 0
ATN_UP = 1
ATN_RIGHT = 2
ATN_LEFT = 3
ATN_NOOP = 4
ATN_ATTACK = 5
ATN_ONE = 8              # use/select inventory slot 0; ..NINE = slot 8
ATN_NINE = 16
ATN_BUY = 20
ATN_SELL = 21
ATN_DOWN_SHIFT = 22      # run: move 2 tiles (both must be clear)

# Run variant of a move action (nmmo3.h:2113: run ids are the move ids
# shifted by ATN_DOWN_SHIFT - ATN_DOWN)
RUN_OFFSET = ATN_DOWN_SHIFT - ATN_DOWN

MOVE_DELTAS = {          # DELTAS (nmmo3.h:856): action -> (dr, dc)
    ATN_DOWN: (1, 0),
    ATN_UP: (-1, 0),
    ATN_RIGHT: (0, 1),
    ATN_LEFT: (0, -1),
}

# Animations (nmmo3.h:37-43) — used for freshness cross-checks
ANIM_IDLE = 0
ANIM_MOVE = 1
ANIM_DEATH = 5
ANIM_RUN = 6

# UI modes (nmmo3.h:30-34)
MODE_PLAY = 0
MODE_BUY_TIER = 1
MODE_BUY_ITEM = 2
MODE_SELL_SELECT = 3
MODE_SELL_PRICE = 4

# Entity types (nmmo3.h:73-75)
ENTITY_PLAYER = 1
ENTITY_ENEMY = 2

# Item types (nmmo3.h:130-147). On the GROUND, ore/hilt/wood are the
# harvest forms (they become armor/sword/bow in inventory, pickup_item
# nmmo3.h:1180-1189).
I_HELM = 1
I_CHEST = 2
I_LEGS = 3
I_SWORD = 4
I_BOW = 5
I_TOOL = 6
I_GEM_TYPES = (7, 8, 9, 10)   # earth, fire, air, water
I_HERB = 11
I_ORE = 12
I_WOOD = 13
I_HILT = 14
I_N = 17

ARMOR_SLOT_BY_TYPE = {I_HELM: SLOT_HELM, I_CHEST: SLOT_CHEST,
                      I_LEGS: SLOT_LEGS}

TIER_EXP_BASE = 8


def tier_level(tier: int) -> float:
    """Level equivalent of an item tier (nmmo3.h:884): 8 * 2^(tier-1).
    A harvest levels prof while prof < tier_level(resource tier)."""
    return TIER_EXP_BASE * (2.0 ** (tier - 1))


def item_type(item_id: int) -> int:
    """I_* type of a raw inventory/ground item id (0 = empty slot)."""
    return item_id % I_N


def item_tier(item_id: int) -> int:
    """Tier of a raw item id (meaningless for id 0)."""
    return item_id // I_N + 1


# Behavior thresholds (this bot's own tuning, not upstream contract).
# Combat math they encode (calc_damage, nmmo3.h:1289-1308): at comb L we
# deal 40+2L+equip vs enemy defense 5L', an enemy deals 15+5L'-(2L+equip)
# — a delta-0 fight (L' <= L+1) is winnable but costs most of the hp
# bar at low levels, and melee chases at EQUAL walking speed land a hit
# every other tick, so escapes must RUN (2 tiles/tick).
HEAL_HP = 45         # eat a herb below this (herb restores >= 60)
FIGHT_START_HP = 95  # only START a full-hp-enemy fight when rested: a
                     # comb-1 fight vs a delta-0 enemy can cost ~90 hp
FIGHT_WOUNDED_HP = 60   # cheap kills: start on a wounded enemy
                        # (hp bucket <= 2, i.e. <= 59 hp) at 60+
FIGHT_KEEP_HP = 25   # once committed, fight it out — aborting refunds
                     # nothing (the enemy regens) and the kill is the
                     # only source of comb levels and tool drops
TOOL_GRAB_DIST = 3   # a dropped tool this close is worth ignoring the
                     # disengage rule for (tools despawn in 20 ticks,
                     # drop_respawn_buffer nmmo3.h:783)
HERB_STOCK = 2       # keep harvesting herbs until this many in inventory
AVOID_RADIUS = 5     # keep stronger-enemy hints outside this Chebyshev
                     # ring (NPC aggro triggers at 4, nmmo3.h:169)
ATTACK_STREAK_MAX = 8   # a real melee kill needs <= 5 connected hits;
                        # more consecutive attacks than this = ghost
LIVE_TTL = 3         # ticks an entity cell stays trusted after its
                     # bytes last changed (see the liveness docstring)


class Percept:
    """Decoded view of one 1707-byte observation (zero-copy reshape)."""

    def __init__(self, obs: bytes):
        if len(obs) != OBS_SIZE:
            raise ValueError(f"obs must be {OBS_SIZE} bytes, got {len(obs)}")
        arr = np.frombuffer(bytearray(obs), dtype=np.uint8)
        self.tiles = arr[:SCALARS_OFF].reshape(
            WINDOW_ROWS, WINDOW_COLS, TILE_BYTES)
        self.scalars = arr[SCALARS_OFF:REWARD_OFF]
        self.reward = arr[REWARD_OFF:]

    # -- scalars ----------------------------------------------------------
    @property
    def hp(self) -> int:
        return int(self.scalars[S_HP])

    @property
    def comb_lvl(self) -> int:
        return int(self.scalars[S_COMB_LVL])

    @property
    def prof_lvl(self) -> int:
        return int(self.scalars[S_PROF_LVL])

    @property
    def ui_mode(self) -> int:
        return int(self.scalars[S_UI_MODE])

    @property
    def anim(self) -> int:
        return int(self.scalars[S_ANIM])

    @property
    def in_combat(self) -> bool:
        return int(self.scalars[S_IN_COMBAT]) > 0

    @property
    def inventory(self) -> list[int]:
        return [int(v) for v in
                self.scalars[S_INVENTORY:S_INVENTORY + INVENTORY_SLOTS]]

    @property
    def is_equipped(self) -> list[int]:
        return [int(v) for v in
                self.scalars[S_IS_EQUIPPED:S_IS_EQUIPPED + INVENTORY_SLOTS]]

    @property
    def equipment(self) -> list[int]:
        return [int(v) for v in self.scalars[S_EQUIPMENT:S_EQUIPMENT + 5]]

    @property
    def held_tool_tier(self) -> int:
        """Tier of the held tool, 0 when the held slot is empty or holds
        a non-tool (nothing can be harvested then except loose tools)."""
        held = self.equipment[SLOT_HELD]
        if held and item_type(held) == I_TOOL:
            return item_tier(held)
        return 0

    # -- window -----------------------------------------------------------
    def passable(self, r: int, c: int) -> bool:
        """Terrain passability (PASSABLE, nmmo3.h:895: grass/dirt yes,
        stone/water no). Window coords; out-of-window counts blocked."""
        if not (0 <= r < WINDOW_ROWS and 0 <= c < WINDOW_COLS):
            return False
        return int(self.tiles[r, c, TB_TERRAIN]) in (TERRAIN_GRASS,
                                                     TERRAIN_DIRT)

    def enemy_hints(self):
        """(r, c, delta_comb, hp_bucket) for window tiles whose
        STALE-PRONE entity bytes claim an enemy. Hints only — see the
        module docstring. Corpse imprints (anim frozen on ANIM_DEATH — a
        killed enemy's last written frame) are filtered here; departed
        ghosts need the mind's liveness map."""
        ent = self.tiles[:, :, TB_ENT_TYPE]
        out = []
        for r, c in zip(*np.nonzero(ent == ENTITY_ENEMY)):
            if int(self.tiles[r, c, TB_ENT_ANIM]) == ANIM_DEATH:
                continue
            out.append((int(r), int(c), int(self.tiles[r, c, TB_ENT_DELTA]),
                        int(self.tiles[r, c, TB_ENT_HP])))
        return out

    def item_tiles(self):
        """(r, c, item_type, tier) for every visible ground item (tile
        bytes 2-3 — the reliable half of the cell), excluding our own
        tile (pickup already ran when we stepped onto it)."""
        types = self.tiles[:, :, TB_ITEM_TYPE]
        out = []
        for r, c in zip(*np.nonzero(types != 0)):
            if (r, c) == (CENTER_ROW, CENTER_COL):
                continue
            out.append((int(r), int(c), int(types[r, c]),
                        int(self.tiles[r, c, TB_ITEM_TIER]) + 1))
        return out


class ScriptedMind:
    """Per-agent FSM state. Everything here is rebuilt from scratch on a
    respawn reset (the new life has fresh levels and a fresh location)."""

    def __init__(self, seed, agent_idx: int):
        self._seed = seed
        self._agent_idx = agent_idx
        self.reset()

    def reset(self) -> None:
        # Deterministic under COGAME_PLAYER_SEED: same seed + same reset
        # count -> same stream (the reset counter keeps a respawned life
        # from replaying the previous life's wander choices verbatim).
        generation = getattr(self, "_generation", -1) + 1
        self._generation = generation
        self.rng = np.random.default_rng(
            (0 if self._seed is None else self._seed,
             self._agent_idx, generation))
        self.wander_action = ATN_DOWN
        self.wander_left = 0
        self.last_action = ATN_NOOP
        self.selling = False
        # staleness bookkeeping (module docstring): consecutive-window
        # liveness map + attack hit feedback
        self.prev_tiles: np.ndarray | None = None
        self.live = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=np.int8)
        self.last_attack_pos: tuple[int, int] | None = None
        self.attack_streak = 0

    # -- helpers ----------------------------------------------------------

    def _own_shift(self, p: Percept) -> tuple[int, int]:
        """How far the window slid since last tick, from fresh signals:
        the anim scalar proves whether the last move/run succeeded (and
        at which speed), the remembered action gives the direction.
        Teleportitis (0.1%/tick) breaks this for one tick — the diff
        then over-marks cells live briefly, which is safe."""
        walk = self.last_action
        if walk - RUN_OFFSET in MOVE_DELTAS:
            walk -= RUN_OFFSET
        if walk not in MOVE_DELTAS:
            return (0, 0)
        dr, dc = MOVE_DELTAS[walk]
        if p.anim == ANIM_MOVE:
            return (dr, dc)
        if p.anim == ANIM_RUN:
            return (2 * dr, 2 * dc)
        return (0, 0)

    def _update_liveness(self, p: Percept) -> None:
        """Advance the liveness map (module docstring): decay-and-shift
        the trust map by our own movement, then re-arm every cell whose
        entity bytes changed at the SAME window offset — such a cell was
        necessarily rewritten this tick, so an entity is there now."""
        if self.prev_tiles is None:
            self.live = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=np.int8)
            self.prev_tiles = p.tiles.copy()
            return
        # A changed offset was rewritten this tick (writes only happen
        # for currently-present entities). NOT motion-corrected — see
        # the module docstring.
        changed = (p.tiles[:, :, TB_ENT_TYPE:]
                   != self.prev_tiles[:, :, TB_ENT_TYPE:]).any(axis=2)
        # The decaying trust map tracks world-anchored entities: shift
        # it opposite our own movement. Current window cell (r, c)
        # covers the world tile that was at cell (r + dr, c + dc).
        dr, dc = self._own_shift(p)
        r0, r1 = max(0, -dr), min(WINDOW_ROWS, WINDOW_ROWS - dr)
        c0, c1 = max(0, -dc), min(WINDOW_COLS, WINDOW_COLS - dc)
        shifted = np.zeros_like(self.live)
        if r0 < r1 and c0 < c1:
            shifted[r0:r1, c0:c1] = self.live[r0 + dr:r1 + dr,
                                              c0 + dc:c1 + dc]
        self.live = np.where(changed, np.int8(LIVE_TTL),
                             np.maximum(shifted - 1, 0).astype(np.int8))
        self.prev_tiles = p.tiles.copy()

    def _live_hints(self, p: Percept):
        """Enemy hints on LIVE cells only (corpse imprints are already
        filtered in Percept.enemy_hints). While our fresh in_combat
        scalar says someone is actively hitting us, imprints in attack
        position (adjacent, or on-axis within bow range 4) are trusted
        even if their bytes went static — the diff detector cannot see
        a state-frozen attacker (module docstring recall note)."""
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

    def _threats(self, p: Percept):
        """Window cells hinting a stronger enemy (delta >= 1: two or more
        comb levels above us) close enough to matter for avoidance."""
        return [(r, c) for r, c, d, _hp in self._live_hints(p)
                if d >= 1
                and max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
                <= AVOID_RADIUS]

    @staticmethod
    def _threat_distance(r, c, threats):
        return min(max(abs(r - tr), abs(c - tc)) for tr, tc in threats)

    def _passable_moves(self, p: Percept, avoid=None):
        moves = []
        for atn, (dr, dc) in MOVE_DELTAS.items():
            if avoid is not None and atn == avoid:
                continue
            if p.passable(CENTER_ROW + dr, CENTER_COL + dc):
                moves.append(atn)
        return moves

    def _safe_moves(self, p: Percept, threats, avoid=None):
        """Passable moves that do not step toward any nearby
        stronger-enemy hint (never decrease the min Chebyshev distance).
        Falls back to all passable moves when nothing qualifies —
        walking beats standing still next to a chaser."""
        moves = self._passable_moves(p, avoid=avoid)
        if not threats or not moves:
            return moves
        here = self._threat_distance(CENTER_ROW, CENTER_COL, threats)
        safe = [atn for atn in moves
                if self._threat_distance(
                    CENTER_ROW + MOVE_DELTAS[atn][0],
                    CENTER_COL + MOVE_DELTAS[atn][1], threats) >= here]
        return safe or moves

    def _maybe_run(self, p: Percept, atn: int) -> int:
        """Upgrade a walk to a run (2 tiles) when both tiles are clear
        terrain — runs move at double speed (the only way to break a
        melee chase: walkers and chasers move at the same speed) and
        double wander coverage. Harvest approaches must NOT run past
        their target; callers decide when to upgrade."""
        if atn not in MOVE_DELTAS:
            return atn
        dr, dc = MOVE_DELTAS[atn]
        if p.passable(CENTER_ROW + dr, CENTER_COL + dc) and \
                p.passable(CENTER_ROW + 2 * dr, CENTER_COL + 2 * dc):
            return atn + RUN_OFFSET
        return atn

    def _last_move_blocked(self, p: Percept) -> bool:
        """Did last tick's move/run fail? c_step sets anim=IDLE every
        tick and move() sets MOVE/RUN only on success — always fresh."""
        was_move = (self.last_action in MOVE_DELTAS
                    or self.last_action - RUN_OFFSET in MOVE_DELTAS)
        return was_move and p.anim == ANIM_IDLE

    def _wander(self, p: Percept, threats) -> int:
        """Persistent-direction random walk-or-run (anti-stagnation
        coverage), biased away from stronger-enemy hints."""
        blocked_last = self._last_move_blocked(p)
        options = self._safe_moves(
            p, threats, avoid=self.wander_action if blocked_last else None)
        if self.wander_left <= 0 or blocked_last or \
                self.wander_action not in options:
            if not options:
                return ATN_NOOP  # fully boxed in; wait for teleportitis
            self.wander_action = int(options[self.rng.integers(len(options))])
            self.wander_left = int(self.rng.integers(6, 15))
        self.wander_left -= 1
        return self._maybe_run(p, self.wander_action)

    def _step_towards(self, p: Percept, threats, r: int, c: int) -> int | None:
        """One greedy step (run when 2+ tiles remain on the axis) toward
        window tile (r, c): larger-|delta| axis first, other axis when
        blocked/unsafe, None when neither works."""
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
        # last tick's exact move failed (occupied tile, most likely a
        # stale-invisible entity): demote it so we route around
        if (self.last_action in prefs and self._last_move_blocked(p)
                and len(prefs) > 1):
            prefs.remove(self.last_action)
        allowed = self._safe_moves(p, threats)
        for atn in prefs:
            if atn in allowed:
                axis_dist = abs(dr) if atn in (ATN_DOWN, ATN_UP) else abs(dc)
                if axis_dist >= 2:
                    return self._maybe_run(p, atn)
                return atn
        return None

    def _herb_slot(self, p: Percept) -> int | None:
        for idx, item in enumerate(p.inventory[:NUM_KEY_SLOTS]):
            if item and item_type(item) == I_HERB:
                return idx
        return None

    def _equip_action(self, p: Percept) -> int | None:
        """One equip/unequip keypress toward the target loadout: best
        tool held, best armor in every armor slot. Multi-step upgrades
        (unequip old, equip new) emerge one keypress per tick because
        this is re-derived from fresh scalars every tick."""
        inv = p.inventory
        equipped = p.is_equipped
        equipment = p.equipment

        def best_slot_of_type(t):
            best = None
            for idx in range(NUM_KEY_SLOTS):
                item = inv[idx]
                if item and item_type(item) == t and not equipped[idx]:
                    if best is None or item_tier(item) > item_tier(inv[best]):
                        best = idx
            return best

        # Held slot: tools only (module docstring: harvest gating beats
        # weapon damage; bare/tool melee still attacks adjacent tiles).
        best_tool = best_slot_of_type(I_TOOL)
        held = equipment[SLOT_HELD]
        if held == 0:
            if best_tool is not None:
                return ATN_ONE + best_tool
        elif item_type(held) == I_TOOL:
            if best_tool is not None and \
                    item_tier(inv[best_tool]) > item_tier(held):
                # unequip the current tool first (slot must be empty to
                # equip, use_item nmmo3.h:1544); find its inventory key
                for idx in range(NUM_KEY_SLOTS):
                    if equipped[idx] and inv[idx] == held:
                        return ATN_ONE + idx
        else:
            # a non-tool ended up held (e.g. picked-up sword auto-kept):
            # unequip it to make room for a tool
            if best_tool is not None:
                for idx in range(NUM_KEY_SLOTS):
                    if equipped[idx] and inv[idx] == held:
                        return ATN_ONE + idx

        # Armor into empty slots
        for atype, slot in ARMOR_SLOT_BY_TYPE.items():
            if equipment[slot] == 0:
                idx = best_slot_of_type(atype)
                if idx is not None:
                    return ATN_ONE + idx
        return None

    def _junk_slot(self, p: Percept) -> int | None:
        """Lowest-tier expendable slot for an inventory dump: not
        equipped, not a herb (heal stock), not our best tool, not armor
        an empty slot is waiting for. Key-addressable slots only."""
        inv = p.inventory
        equipped = p.is_equipped
        equipment = p.equipment
        best_tool_tier = max(
            [item_tier(i) for i in inv if i and item_type(i) == I_TOOL],
            default=0)
        candidates = []
        for idx in range(NUM_KEY_SLOTS):
            item = inv[idx]
            if not item or equipped[idx]:
                continue
            t = item_type(item)
            if t == I_HERB:
                continue
            if t == I_TOOL and item_tier(item) >= best_tool_tier:
                continue
            if t in ARMOR_SLOT_BY_TYPE and \
                    equipment[ARMOR_SLOT_BY_TYPE[t]] == 0:
                continue
            candidates.append((item_tier(item), idx))
        if not candidates:
            return None
        return min(candidates)[1]

    def _harvest_target(self, p: Percept):
        """Best visible item tile to walk onto, or None. Ranked by
        (priority, distance): tool upgrades, then prof-leveling
        resources, then herb stock."""
        held_tier = p.held_tool_tier
        herbs_held = sum(1 for i in p.inventory
                         if i and item_type(i) == I_HERB)
        inventory_full = all(p.inventory)
        best = None
        for r, c, itype, tier in p.item_tiles():
            if itype == I_TOOL:
                # tools are picked up bare-handed (nmmo3.h:1148)
                if tier <= held_tier:
                    continue
                priority = 0
            elif itype in (I_ORE, I_WOOD, I_HILT, I_HERB) or \
                    itype in I_GEM_TYPES:
                if tier > held_tier:
                    continue  # our tool cannot harvest it
                if p.prof_lvl < tier_level(tier):
                    priority = 1  # levels prof — the main objective
                elif itype == I_HERB and herbs_held < HERB_STOCK:
                    priority = 2
                else:
                    continue  # no value; do not fill inventory with it
            else:
                continue  # not a ground-harvestable type
            if inventory_full:
                continue  # pickup would fail; dump handles this
            dist = abs(r - CENTER_ROW) + abs(c - CENTER_COL)
            key = (priority, dist)
            if best is None or key < best[0]:
                best = (key, (r, c))
        return None if best is None else best[1]

    # -- the FSM ----------------------------------------------------------

    def act(self, obs: bytes) -> int:
        p = Percept(obs)

        self._update_liveness(p)
        # hit feedback (module docstring): a missed attack proves the
        # targeted cell was a stale imprint — kill its liveness now
        if (self.last_action == ATN_ATTACK and not p.in_combat
                and self.last_attack_pos is not None):
            self.live[self.last_attack_pos] = 0

        action = self._decide(p)

        if action == ATN_ATTACK:
            self.attack_streak += 1
        else:
            self.attack_streak = 0
            self.last_attack_pos = None
        self.last_action = action
        return action

    def _decide(self, p: Percept) -> int:
        threats = self._threats(p)

        # 1/2. Market UI handling. Sell mode we initiated: finish the
        # dump (select the junk slot, then price it at the max key).
        # Any other mode (teleportitis mid-sale confusion, stray BUY):
        # exit with a movement action — it leaves the mode without
        # buying/selling and without moving (c_step nmmo3.h:1989-2105).
        if p.ui_mode != MODE_PLAY:
            if self.selling and not p.in_combat:
                if p.ui_mode == MODE_SELL_SELECT:
                    junk = self._junk_slot(p)
                    if junk is not None:
                        return ATN_ONE + junk
                elif p.ui_mode == MODE_SELL_PRICE:
                    self.selling = False
                    return ATN_NINE  # price 9, top of the key range
            self.selling = False
            return self._wander(p, threats)
        self.selling = False

        # 3. Heal: herbs work in combat and restore >= 60 hp.
        if p.hp < HEAL_HP:
            herb = self._herb_slot(p)
            if herb is not None:
                return ATN_ONE + herb

        # 4. Fight: ATN_ATTACK auto-targets any enemy on the 4-adjacent
        # tiles regardless of facing (find_target scans all four
        # directions; bare hands/tool use the ATTACK_BASIC pattern).
        # delta==0 <=> the enemy is at most one comb level above us —
        # beatable, and the only kind whose kill levels comb (and drops
        # the tool that unlocks all harvesting). Fights are STARTED only
        # when rested (or cheaply, on a wounded enemy) but then fought
        # OUT down to FIGHT_KEEP_HP — aborting refunds nothing. The
        # streak cap turns a suspiciously long "fight" into a dead cell
        # so a stale imprint can never pin us to a tile.
        adjacent_fightable = [
            (r, c) for r, c, delta, hpb in self._live_hints(p)
            if delta == 0
            and abs(r - CENTER_ROW) + abs(c - CENTER_COL) == 1
            and (p.hp >= FIGHT_START_HP
                 or (hpb <= 2 and p.hp >= FIGHT_WOUNDED_HP)
                 or (self.attack_streak > 0 and p.hp >= FIGHT_KEEP_HP))]
        for r, c in adjacent_fightable:
            if self.attack_streak >= ATTACK_STREAK_MAX:
                self.live[r, c] = 0
                continue
            self.last_attack_pos = (r, c)
            return ATN_ATTACK

        # 5. Tool grab: dropped tools despawn after 20 ticks, and a held
        # tool gates ALL harvesting — a nearby upgrade outranks even the
        # disengage rule (worth eating a hit or two for).
        held_tier = p.held_tool_tier
        tool_tiles = [(abs(r - CENTER_ROW) + abs(c - CENTER_COL), r, c)
                      for r, c, itype, tier in p.item_tiles()
                      if itype == I_TOOL and tier > held_tier]
        if tool_tiles and not all(p.inventory):
            dist, r, c = min(tool_tiles)
            if dist <= TOOL_GRAB_DIST:
                step = self._step_towards(p, [], r, c)
                if step is not None:
                    return step

        # 6. Disengage: EVERY enemy chases any player inside its aggro
        # range (enemy_ai, nmmo3.h:1572-1637 — level plays no part), so
        # lingering near a live hint we are not actively fighting means
        # eating hits. RUN away (2 tiles/tick; chasers walk) from any
        # live hint in touching range — stronger enemies (delta >= 1,
        # possibly ranged: bows reach 4 on-axis) get the wider berth.
        close = [(r, c) for r, c, d, _hp in self._live_hints(p)
                 if max(abs(r - CENTER_ROW), abs(c - CENTER_COL))
                 <= (4 if d >= 1 else 2)]
        if close:
            moves = self._passable_moves(p)
            if moves:
                def flee_score(atn):
                    dr, dc = MOVE_DELTAS[atn]
                    nr, nc = CENTER_ROW + dr, CENTER_COL + dc
                    dist = self._threat_distance(nr, nc, close)
                    # bows shoot only along axes (ATTACK_BOW patterns):
                    # breaking row/col alignment dodges ranged fire
                    aligned = sum(1 for tr, tc in close
                                  if nr == tr or nc == tc)
                    return (dist, -aligned)
                return self._maybe_run(p, max(moves, key=flee_score))

        # 7. Equip upgrades (blocked while in combat, use_item
        # nmmo3.h:1474; herbs above are exempt from that check).
        if not p.in_combat:
            equip = self._equip_action(p)
            if equip is not None:
                return equip

        # 8. Harvest / gear collection (move-onto-tile).
        target = self._harvest_target(p)
        if target is not None:
            step = self._step_towards(p, threats, *target)
            if step is not None:
                return step
        elif all(p.inventory) and not p.in_combat and \
                self._junk_slot(p) is not None:
            # inventory full (harvesting is pointless: pickup_item drops
            # out on a full inventory, nmmo3.h:1139-1142) and we hold
            # expendable junk: start a market dump to free a slot
            # (3-tick sell flow handled in step 2 above)
            self.selling = True
            return ATN_SELL

        # 9. Anti-stagnation wander (threat-biased).
        return self._wander(p, threats)


class ScriptedPolicy:
    """policy(tick, obs_rows, resets): one ScriptedMind per hero index."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.minds: dict[int, ScriptedMind] = {}

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        actions = []
        for i, row in enumerate(obs_rows):
            mind = self.minds.get(i)
            if mind is None:
                mind = self.minds[i] = ScriptedMind(self.seed, i)
            if resets[i]:
                mind.reset()  # new life: fresh levels, fresh location
            actions.append([mind.act(bytes(row))])
        return actions


def policy_from_env() -> ScriptedPolicy:
    return ScriptedPolicy(seed_from_env(default=0))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
