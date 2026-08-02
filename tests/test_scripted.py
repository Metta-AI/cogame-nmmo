"""Tests for the scripted player (Phase N3, daveey's league entry).

Layers: (1) obs-layout tripwires — every constant the bot derives from
the vendored upstream sources is re-parsed from those sources here, so
an upstream pin bump that shifts the byte layout fails loudly; (2)
decode helpers against hand-built obs AND against real sim obs (ground
truth); (3) FSM transition unit tests on synthetic obs; (4) determinism
+ action-validity fuzz; (5) slow behavioral episodes vs random (assert)
and vs baseline (report).
"""

import re
from pathlib import Path

import numpy as np
import pytest

from cogame_nmmo.sim import NmmoSim
from players import scripted_player as sp
from players.baseline_player import NmmoBrain
from players.scripted_player import Percept, ScriptedMind, ScriptedPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
NMMO3_C = (REPO_ROOT / "vendor" / "upstream" / "nmmo3.c").read_text()
NMMO3_H = (REPO_ROOT / "vendor" / "upstream" / "nmmo3.h").read_text()


# -- (1) obs-layout tripwires against the vendored sources -------------------

def test_factors_table_matches_upstream():
    m = re.search(r"int factors\[10\] = \{([0-9, ]+)\};", NMMO3_C)
    assert m, "factors table not found in vendored nmmo3.c"
    upstream = tuple(int(v) for v in m.group(1).split(","))
    assert sp.FACTORS == upstream
    # the one-hot channel count the net was built for
    assert sum(sp.FACTORS) == 59


def test_window_dims_and_obs_size_match_upstream():
    # windows from the demo env struct (nmmo3.c demo(): trained values)
    assert re.search(r"\.x_window = 7,", NMMO3_C)
    assert re.search(r"\.y_window = 5,", NMMO3_C)
    assert sp.WINDOW_COLS == 2 * 7 + 1
    assert sp.WINDOW_ROWS == 2 * 5 + 1
    # the obs stride expression as upstream writes it (both files)
    assert re.search(r"11\*15\*10\s*\+\s*47\s*\+\s*10", NMMO3_C)
    assert re.search(r"11\*15\*10\s*\+\s*47\s*\+\s*10", NMMO3_H)
    assert sp.OBS_SIZE == 11 * 15 * 10 + 47 + 10 == 1707
    assert sp.SCALARS_OFF == 1650 and sp.REWARD_OFF == 1697


def test_scalar_offsets_match_compute_all_obs():
    """Parse the scalar-writing block of compute_all_obs (nmmo3.h) and
    assert every named offset the bot uses."""
    block = NMMO3_H[NMMO3_H.index("// Player observation"):
                    NMMO3_H.index("// Reward observation")]
    offsets = {}
    for m in re.finditer(
            r"env->observations\[obs_adr(?:\+(\d+))?\] = player->(\w+);",
            block):
        offsets[m.group(2)] = int(m.group(1) or 0)
    assert offsets == {
        "type": sp.S_TYPE, "comb_lvl": sp.S_COMB_LVL,
        "element": sp.S_ELEMENT, "dir": sp.S_DIR, "anim": sp.S_ANIM,
        "hp": sp.S_HP, "hp_max": sp.S_HP_MAX, "prof_lvl": sp.S_PROF_LVL,
        "ui_mode": sp.S_UI_MODE, "market_tier": sp.S_MARKET_TIER,
        "sell_idx": sp.S_SELL_IDX, "gold": sp.S_GOLD,
        "in_combat": sp.S_IN_COMBAT, "wander_range": sp.S_WANDER_RANGE,
        "ranged": sp.S_RANGED, "goal": sp.S_GOAL,
        "equipment_attack": sp.S_EQUIP_ATTACK,
        "equipment_defense": sp.S_EQUIP_DEFENSE,
    }
    # the three array blocks (loops over j)
    assert re.search(r"obs_adr\+13\+j\] = player->equipment\[j\]", block)
    assert re.search(r"obs_adr\+18\+j\] = player->inventory\[j\]", block)
    assert re.search(r"obs_adr\+30\+j\] = player->is_equipped\[j\]", block)
    assert (sp.S_EQUIPMENT, sp.S_INVENTORY, sp.S_IS_EQUIPPED) == (13, 18, 30)
    # reward block starts at scalar offset 47 and writes 9 of 10 bytes
    reward_block = NMMO3_H[NMMO3_H.index("// Reward observation"):
                           NMMO3_H.index("int safe_tile")]
    reward_offsets = sorted(
        int(m.group(1)) for m in
        re.finditer(r"env->observations\[obs_adr\+(\d+)\]\s*=",
                    reward_block))
    assert reward_offsets == list(range(47, 56))  # byte 1706 never written


def _upstream_defines(src, prefix):
    return {m.group(1): int(m.group(2)) for m in
            re.finditer(rf"#define {prefix}(\w+) (\d+)", src)}


def test_action_ids_match_upstream():
    atn = _upstream_defines(NMMO3_H, "ATN_")
    assert sp.ATN_DOWN == atn["DOWN"]
    assert sp.ATN_UP == atn["UP"]
    assert sp.ATN_RIGHT == atn["RIGHT"]
    assert sp.ATN_LEFT == atn["LEFT"]
    assert sp.ATN_NOOP == atn["NOOP"]
    assert sp.ATN_ATTACK == atn["ATTACK"]
    assert sp.ATN_ONE == atn["ONE"]
    assert sp.ATN_NINE == atn["NINE"]
    assert sp.ATN_BUY == atn["BUY"]
    assert sp.ATN_SELL == atn["SELL"]
    assert sp.ATN_DOWN_SHIFT == atn["DOWN_SHIFT"]
    # run ids are the move ids shifted by a fixed offset (c_step decodes
    # them with `action - ATN_DOWN_SHIFT`, nmmo3.h:2114)
    assert sp.RUN_OFFSET == atn["DOWN_SHIFT"] - atn["DOWN"]
    assert atn["LEFT_SHIFT"] == atn["LEFT"] + sp.RUN_OFFSET
    # movement deltas from the DELTAS table (nmmo3.h:856)
    m = re.search(
        r"int DELTAS\[4\]\[2\] = \{\s*\{1, 0\},\s*\{-1, 0\},"
        r"\s*\{0, 1\},\s*\{0, -1\},\s*\};", NMMO3_H)
    assert m, "DELTAS table changed shape in vendored nmmo3.h"
    assert sp.MOVE_DELTAS == {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}


def test_item_tile_mode_and_slot_ids_match_upstream():
    items = _upstream_defines(NMMO3_H, "I_")
    assert sp.I_N == items["N"]
    assert sp.I_HELM == items["HELM"] and sp.I_CHEST == items["CHEST"]
    assert sp.I_LEGS == items["LEGS"] and sp.I_SWORD == items["SWORD"]
    assert sp.I_BOW == items["BOW"] and sp.I_TOOL == items["TOOL"]
    assert sp.I_HERB == items["HERB"] and sp.I_ORE == items["ORE"]
    assert sp.I_WOOD == items["WOOD"] and sp.I_HILT == items["HILT"]
    assert sp.I_GEM_TYPES == (items["EARTH"], items["FIRE"],
                              items["AIR"], items["WATER"])

    tiles = _upstream_defines(NMMO3_H, "TILE_")
    # terrain byte = tile / 4 (compute_all_obs writes terrain/4)
    assert sp.TERRAIN_GRASS == tiles["SPRING_GRASS"] // 4
    assert sp.TERRAIN_DIRT == tiles["SPRING_DIRT"] // 4
    assert sp.TERRAIN_STONE == tiles["SPRING_STONE"] // 4
    assert sp.TERRAIN_WATER == tiles["SPRING_WATER"] // 4
    # PASSABLE: grass+dirt true, stone+water false (nmmo3.h:895)
    assert re.search(
        r"bool PASSABLE\[16\] = \{\s*true, true, true, true,.*?"
        r"true, true, true, true,.*?false, false, false, false,.*?"
        r"false, false, false, false,", NMMO3_H, re.S)

    modes = _upstream_defines(NMMO3_H, "MODE_")
    assert sp.MODE_PLAY == modes["PLAY"]
    assert sp.MODE_SELL_SELECT == modes["SELL_SELECT"]
    assert sp.MODE_SELL_PRICE == modes["SELL_PRICE"]

    slots = _upstream_defines(NMMO3_H, "SLOT_")
    assert sp.SLOT_HELM == slots["HELM"] and sp.SLOT_CHEST == slots["CHEST"]
    assert sp.SLOT_LEGS == slots["LEGS"] and sp.SLOT_HELD == slots["HELD"]
    assert sp.SLOT_GEM == slots["GEM"]

    ent = _upstream_defines(NMMO3_H, "ENTITY_")
    assert sp.ENTITY_PLAYER == ent["PLAYER"]
    assert sp.ENTITY_ENEMY == ent["ENEMY"]

    assert sp.TIER_EXP_BASE == _upstream_defines(
        NMMO3_H, "TIER_EXP_")["BASE"]
    assert sp.INVENTORY_SLOTS == _upstream_defines(
        NMMO3_H, "INVENTORY_")["SIZE"]


def test_tier_level_matches_upstream_formula():
    # tier_level(tier) = 8 * 2^(tier-1) (nmmo3.h:884)
    assert [sp.tier_level(t) for t in (1, 2, 3, 4, 5)] == \
        [8, 16, 32, 64, 128]


# -- (2) decode helpers ------------------------------------------------------

def make_obs(*, hp=99, hp_max=99, comb=1, prof=1, ui_mode=0, in_combat=0,
             anim=0, direction=0, inventory=(), is_equipped=(),
             equipment=(), terrain=sp.TERRAIN_GRASS, items=(), enemies=()):
    """Synthetic 1707-byte obs: uniform terrain, our player at center,
    plus explicit item tiles (r, c, type, tier) and enemy imprints
    (r, c, delta, hp_bucket)."""
    obs = np.zeros(sp.OBS_SIZE, dtype=np.uint8)
    tiles = obs[:sp.SCALARS_OFF].reshape(
        sp.WINDOW_ROWS, sp.WINDOW_COLS, sp.TILE_BYTES)
    tiles[:, :, sp.TB_TERRAIN] = terrain
    for r, c, itype, tier in items:
        tiles[r, c, sp.TB_ITEM_TYPE] = itype
        tiles[r, c, sp.TB_ITEM_TIER] = tier - 1
    for r, c, delta, hp_bucket in enemies:
        tiles[r, c, sp.TB_ENT_TYPE] = sp.ENTITY_ENEMY
        tiles[r, c, sp.TB_ENT_DELTA] = delta
        tiles[r, c, sp.TB_ENT_HP] = hp_bucket
    tiles[sp.CENTER_ROW, sp.CENTER_COL, sp.TB_ENT_TYPE] = sp.ENTITY_PLAYER
    tiles[sp.CENTER_ROW, sp.CENTER_COL, sp.TB_ENT_HP] = hp // 20

    s = obs[sp.SCALARS_OFF:sp.REWARD_OFF]
    s[sp.S_TYPE] = sp.ENTITY_PLAYER
    s[sp.S_COMB_LVL] = comb
    s[sp.S_DIR] = direction
    s[sp.S_ANIM] = anim
    s[sp.S_HP] = hp
    s[sp.S_HP_MAX] = hp_max
    s[sp.S_PROF_LVL] = prof
    s[sp.S_UI_MODE] = ui_mode
    s[sp.S_IN_COMBAT] = in_combat
    for i, item in enumerate(inventory):
        s[sp.S_INVENTORY + i] = item
    for i, flag in enumerate(is_equipped):
        s[sp.S_IS_EQUIPPED + i] = flag
    for i, item in enumerate(equipment):
        s[sp.S_EQUIPMENT + i] = item
    return obs.tobytes()


def test_percept_decodes_hand_built_obs():
    tool_t2 = (2 - 1) * sp.I_N + sp.I_TOOL  # tier-2 tool
    obs = make_obs(hp=55, comb=3, prof=4, ui_mode=2, in_combat=2,
                   inventory=[tool_t2, sp.I_HERB],
                   is_equipped=[1, 0],
                   equipment=[0, 0, 0, tool_t2, 0],
                   items=[(2, 3, sp.I_ORE, 2)],
                   enemies=[(5, 8, 1, 3)])
    p = Percept(obs)
    assert (p.hp, p.comb_lvl, p.prof_lvl) == (55, 3, 4)
    assert p.ui_mode == 2 and p.in_combat
    assert p.inventory[:2] == [tool_t2, sp.I_HERB]
    assert p.is_equipped[:2] == [1, 0]
    assert p.held_tool_tier == 2
    assert (2, 3, sp.I_ORE, 2) in p.item_tiles()
    assert (5, 8, 1, 3) in p.enemy_hints()
    assert p.passable(0, 0)  # grass everywhere in this synthetic map

    with pytest.raises(ValueError, match="1707"):
        Percept(bytes(10))


def test_percept_item_tier_and_held_tool_edge_cases():
    p = Percept(make_obs())
    assert p.held_tool_tier == 0  # empty held slot
    sword = sp.I_SWORD  # tier-1 sword held: not a tool
    p2 = Percept(make_obs(equipment=[0, 0, 0, sword, 0]))
    assert p2.held_tool_tier == 0
    assert sp.item_type((3 - 1) * sp.I_N + sp.I_HERB) == sp.I_HERB
    assert sp.item_tier((3 - 1) * sp.I_N + sp.I_HERB) == 3


def test_percept_decodes_real_sim_obs():
    """Ground truth: fresh-world obs from the actual sim. The window
    center is our own tile (never stale — we occupy it), so its entity
    bytes must mirror the fresh scalars."""
    sim = NmmoSim(seed=5)
    obs = sim.observations()
    for pid in range(8):
        p = Percept(obs[pid].tobytes())
        assert int(p.scalars[sp.S_TYPE]) == sp.ENTITY_PLAYER
        assert (p.comb_lvl, p.prof_lvl) == (1, 1)
        assert p.hp == 99 and int(p.scalars[sp.S_HP_MAX]) == 99
        assert p.ui_mode == sp.MODE_PLAY and not p.in_combat
        assert p.inventory == [0] * 12
        center = p.tiles[sp.CENTER_ROW, sp.CENTER_COL]
        assert int(center[sp.TB_ENT_TYPE]) == sp.ENTITY_PLAYER
        assert int(center[sp.TB_ENT_HP]) == 99 // 20
        assert int(center[sp.TB_ENT_ANIM]) == int(p.scalars[sp.S_ANIM])
        assert int(center[sp.TB_ENT_DIR]) == int(p.scalars[sp.S_DIR])
        # players spawn on grass (safe_tile requires is_grass)
        assert int(center[sp.TB_TERRAIN]) == sp.TERRAIN_GRASS
        # terrain/season bytes within their one-hot factors everywhere
        assert (p.tiles[:, :, sp.TB_SEASON] < 4).all()
        assert (p.tiles[:, :, sp.TB_TERRAIN] < 4).all()


# -- (3) FSM transitions on synthetic obs ------------------------------------

def mind():
    return ScriptedMind(seed=1, agent_idx=0)


def primed_mind(enemy_obs):
    """Mind whose liveness map trusts the enemies in ``enemy_obs``:
    entity bytes only count once they CHANGE between consecutive
    windows (window-relative staleness), so feed an empty window first.
    The mind's first action is a wander move; the second obs reports
    anim=IDLE, i.e. that move was blocked — shift (0,0), diff arms the
    enemy cells."""
    m = mind()
    m.act(make_obs())
    return m


def test_fsm_eats_herb_at_low_hp():
    obs = make_obs(hp=30, inventory=[0, sp.I_HERB])
    assert mind().act(obs) == sp.ATN_ONE + 1


def test_fsm_runs_from_adjacent_strong_enemy():
    # strong live enemy hint directly above (delta 2): disengage by
    # RUNNING down (the uniquely most-distancing move, at run speed)
    obs = make_obs(hp=30, enemies=[(sp.CENTER_ROW - 1, sp.CENTER_COL, 2, 4)])
    m = primed_mind(obs)
    assert m.act(obs) == sp.ATN_DOWN + sp.RUN_OFFSET


def test_fsm_attacks_adjacent_weak_enemy_when_healthy():
    obs = make_obs(hp=99, enemies=[(sp.CENTER_ROW, sp.CENTER_COL + 1, 0, 4)])
    m = primed_mind(obs)
    assert m.act(obs) == sp.ATN_ATTACK


def test_fsm_ignores_stale_enemy_imprints():
    """The same enemy bytes WITHOUT liveness priming (fresh mind, first
    obs) are treated as an imprint: no attack, no flee — the bot goes
    about its business."""
    obs = make_obs(hp=99, enemies=[(sp.CENTER_ROW, sp.CENTER_COL + 1, 0, 4)])
    m = mind()
    first = m.act(obs)
    assert first != sp.ATN_ATTACK
    # and an UNCHANGING imprint never becomes live on later ticks either
    obs_idle = make_obs(hp=99, anim=sp.ANIM_IDLE,
                        enemies=[(sp.CENTER_ROW, sp.CENTER_COL + 1, 0, 4)])
    for _ in range(5):
        assert m.act(obs_idle) != sp.ATN_ATTACK


def test_fsm_does_not_pick_fights_with_stronger_enemies():
    obs = make_obs(hp=99, enemies=[(sp.CENTER_ROW, sp.CENTER_COL + 1, 3, 4)])
    m = primed_mind(obs)
    assert m.act(obs) != sp.ATN_ATTACK


def test_fsm_disengages_from_weak_enemy_at_mid_hp():
    """Below FIGHT_START_HP the bot must not start a fight, and every
    enemy chases regardless of level (enemy_ai has no level check), so
    it must RUN from a live adjacent hint rather than linger."""
    obs = make_obs(hp=70, enemies=[(sp.CENTER_ROW, sp.CENTER_COL + 1, 0, 4)])
    m = primed_mind(obs)
    action = m.act(obs)
    assert action != sp.ATN_ATTACK
    assert action - sp.RUN_OFFSET in sp.MOVE_DELTAS  # runs, not walks


def test_fsm_equips_tool_from_inventory():
    tool = sp.I_TOOL  # tier 1
    obs = make_obs(inventory=[0, 0, tool])
    assert mind().act(obs) == sp.ATN_ONE + 2


def test_fsm_upgrades_tool_by_unequipping_first():
    t1 = sp.I_TOOL
    t3 = (3 - 1) * sp.I_N + sp.I_TOOL
    obs = make_obs(inventory=[t1, t3], is_equipped=[1, 0],
                   equipment=[0, 0, 0, t1, 0])
    # first keypress unequips the held tier-1 tool (slot must be empty
    # before the tier-3 tool can go in)
    assert mind().act(obs) == sp.ATN_ONE + 0


def test_fsm_moves_towards_higher_tier_tool():
    # tier-1 tool held; a tier-2 tool three tiles right -> RUN right
    # (2+ tiles remain on the axis and the terrain is clear)
    t1 = sp.I_TOOL
    obs = make_obs(inventory=[t1], is_equipped=[1],
                   equipment=[0, 0, 0, t1, 0],
                   items=[(sp.CENTER_ROW, sp.CENTER_COL + 3, sp.I_TOOL, 2)])
    assert mind().act(obs) == sp.ATN_RIGHT + sp.RUN_OFFSET
    # one tile out it must WALK (a run would overshoot the pickup)
    obs = make_obs(inventory=[t1], is_equipped=[1],
                   equipment=[0, 0, 0, t1, 0],
                   items=[(sp.CENTER_ROW, sp.CENTER_COL + 1, sp.I_TOOL, 2)])
    assert mind().act(obs) == sp.ATN_RIGHT


def test_fsm_harvests_only_within_tool_tier():
    t1 = sp.I_TOOL
    base = dict(inventory=[t1], is_equipped=[1],
                equipment=[0, 0, 0, t1, 0])
    # tier-1 ore two tiles up: harvestable, levels prof (run: 2 tiles)
    obs = make_obs(items=[(sp.CENTER_ROW - 2, sp.CENTER_COL, sp.I_ORE, 1)],
                   **base)
    assert mind().act(obs) == sp.ATN_UP + sp.RUN_OFFSET
    # tier-3 ore: our tier-1 tool cannot harvest it -> not a target
    obs = make_obs(items=[(sp.CENTER_ROW - 2, sp.CENTER_COL, sp.I_ORE, 3)],
                   **base)
    m = mind()
    assert m._harvest_target(Percept(obs)) is None
    # ...so the bot wanders; pin the wander lock to make that observable
    # (open grass: the wander step is upgraded to a run)
    m.wander_action = sp.ATN_DOWN
    m.wander_left = 5
    assert m.act(obs) == sp.ATN_DOWN + sp.RUN_OFFSET


def test_fsm_ignores_leveled_out_resources():
    """A resource that can no longer level prof (prof >= tier_level) and
    is not herb stock gets ignored — the bot wanders instead of walking
    a pointless pickup."""
    t1 = sp.I_TOOL
    obs = make_obs(prof=8,  # tier_level(1) == 8: tier-1 ore is spent
                   inventory=[t1], is_equipped=[1],
                   equipment=[0, 0, 0, t1, 0],
                   items=[(sp.CENTER_ROW - 2, sp.CENTER_COL, sp.I_ORE, 1)])
    m = mind()
    assert m._harvest_target(Percept(obs)) is None
    # force the wander choice away from UP so the distinction is visible
    m.wander_action = sp.ATN_DOWN
    m.wander_left = 5
    assert m.act(obs) == sp.ATN_DOWN + sp.RUN_OFFSET


def test_fsm_exits_foreign_ui_mode_with_a_move():
    obs = make_obs(ui_mode=sp.MODE_BUY_TIER)
    action = mind().act(obs)
    # any move/run exits the market UI without side effects
    assert action in sp.MOVE_DELTAS or \
        action - sp.RUN_OFFSET in sp.MOVE_DELTAS


def test_fsm_dumps_junk_when_inventory_full():
    t1 = sp.I_TOOL
    junk = sp.I_SWORD  # a sword we never equip (tools gate harvesting)
    inv = [t1] + [junk] * 11
    m = mind()
    obs = make_obs(inventory=inv, is_equipped=[1] + [0] * 11,
                   equipment=[0, 0, 0, t1, 0])
    assert m.act(obs) == sp.ATN_SELL and m.selling
    obs = make_obs(ui_mode=sp.MODE_SELL_SELECT, inventory=inv,
                   is_equipped=[1] + [0] * 11,
                   equipment=[0, 0, 0, t1, 0])
    assert m.act(obs) == sp.ATN_ONE + 1  # sells the first junk slot
    obs = make_obs(ui_mode=sp.MODE_SELL_PRICE, inventory=inv,
                   is_equipped=[1] + [0] * 11,
                   equipment=[0, 0, 0, t1, 0])
    assert m.act(obs) == sp.ATN_NINE
    assert not m.selling


def test_fsm_reset_reorients():
    """The wire resets flag clears all per-life state (wander lock, sell
    flow) and reseeds the RNG for the new generation."""
    policy = ScriptedPolicy(seed=1)
    obs = make_obs()
    policy(0, [obs], [False])
    m = policy.minds[0]
    m.wander_left = 99
    m.wander_action = sp.ATN_LEFT
    m.selling = True
    gen_before = m._generation
    policy(1, [obs], [True])
    assert m._generation == gen_before + 1
    assert not m.selling
    # wander state was re-rolled, not carried across lives
    assert m.wander_left != 99


# -- (4) determinism + fuzz --------------------------------------------------

def test_scripted_policy_is_deterministic():
    sim = NmmoSim(seed=9)
    obs_history = []
    resets_history = []
    a = ScriptedPolicy(seed=7)
    resets = [False] * 8
    for t in range(60):
        obs = sim.observations()
        rows = [obs[p].tobytes() for p in range(8)]
        obs_history.append(rows)
        resets_history.append(list(resets))
        acts = a(t, rows, resets)
        sim.set_actions(np.asarray(acts, dtype=np.float32))
        sim.step()
        resets = sim.dones()

    b = ScriptedPolicy(seed=7)
    b2 = ScriptedPolicy(seed=7)
    c = ScriptedPolicy(seed=8)
    replay_b = [b(t, rows, r) for t, (rows, r)
                in enumerate(zip(obs_history, resets_history))]
    replay_b2 = [b2(t, rows, r) for t, (rows, r)
                 in enumerate(zip(obs_history, resets_history))]
    assert replay_b == replay_b2  # same seed -> identical
    replay_c = [c(t, rows, r) for t, (rows, r)
                in enumerate(zip(obs_history, resets_history))]
    assert replay_c != replay_b  # different seed -> different wander


def test_action_validity_fuzz():
    """Arbitrary obs bytes (valid or garbage) never crash the FSM and
    always yield one in-range action int per hero."""
    rng = np.random.default_rng(123)
    policy = ScriptedPolicy(seed=1)
    for t in range(200):
        rows = [rng.integers(0, 256, sp.OBS_SIZE, dtype=np.uint8).tobytes()
                for _ in range(2)]
        resets = [bool(rng.integers(2)), bool(rng.integers(2))]
        actions = policy(t, rows, resets)
        assert len(actions) == 2
        for row in actions:
            assert len(row) == 1
            assert isinstance(row[0], int)
            assert 0 <= row[0] < 26


# -- (5) behavioral ----------------------------------------------------------

BEHAVIORAL_TICKS = 2000
_episode_cache: dict = {}


def _scripted_vs_random():
    """One cached 2000-tick episode serves both scripted-vs-random
    assertions (raw + per-life)."""
    if "svr" not in _episode_cache:
        _episode_cache["svr"] = _run_groups(
            17, _scripted_group(range(4)), _random_group(range(4, 8)))
    return _episode_cache["svr"]


def _run_groups(seed, group_a, group_b, ticks=BEHAVIORAL_TICKS):
    """Sim episode with two 4-agent groups; each group entry is a
    callable(pids, obs, resets) -> list of action ints."""
    sim = NmmoSim(seed=seed)
    resets = [False] * 8
    for _ in range(ticks):
        obs = sim.observations()
        acts = np.zeros((8, 1), dtype=np.float32)
        for policy, pids in (group_a, group_b):
            vals = policy(pids, obs, resets)
            for pid, v in zip(pids, vals):
                acts[pid, 0] = v
        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()

    def stats(pids):
        scores = [sim.score(p) for p in pids]
        deaths = [sim.agent_stat(p, 1) for p in pids]
        per_life = [s / (d + 1) for s, d in zip(scores, deaths)]
        return {"scores": scores, "deaths": deaths, "per_life": per_life}

    return stats(group_a[1]), stats(group_b[1])


def _scripted_group(pids, seed=3):
    policy = ScriptedPolicy(seed=seed)

    def run(active_pids, obs, resets):
        rows = [obs[p].tobytes() for p in active_pids]
        r = [resets[p] for p in active_pids]
        return [a[0] for a in policy(0, rows, r)]
    return (run, pids)


def _random_group(pids, seed=99):
    rng = np.random.default_rng(seed)

    def run(active_pids, obs, resets):
        return rng.integers(0, 26, size=len(active_pids)).tolist()
    return (run, pids)


def _baseline_group(pids, seed=1):
    brain = NmmoBrain(seed=seed, num_agents=len(pids))

    def run(active_pids, obs, resets):
        out = []
        for i, p in enumerate(active_pids):
            if resets[p]:
                brain.reset_state(i)
            out.append(brain.forward(i, obs[p].tobytes())[0])
        return out
    return (run, pids)


@pytest.mark.slow
def test_scripted_beats_random_on_ranking_score():
    """The N3 gate on the RANKING metric: mean score per life,
    sim.score/(deaths+1) — exactly engine._seat_score since the
    mean-per-life scoring change (commit 8410305; under the original
    cumulative sum this ordering was distorted by death-count farming —
    see test_baseline's history note). The FSM's kill-grab-harvest loop
    banks min(comb,prof) >= 2 in a meaningful fraction of lives, so its
    per-life mean sits measurably above random's ~1.0 floor (measured
    seeds 17/23/31: scripted mean ~1.22-1.33, random 1.00)."""
    scripted, random_ = _scripted_vs_random()
    print(f"\nscripted-vs-random: scripted={scripted} "
          f"random={random_}")
    assert np.mean(scripted["per_life"]) > np.mean(random_["per_life"]), (
        scripted, random_)


@pytest.mark.slow
def test_scripted_vs_baseline_report_only():
    """Per the plan: report scripted vs the pretrained baseline, no bar."""
    scripted, baseline = _run_groups(
        23, _scripted_group(range(4)), _baseline_group(range(4, 8)))
    print(f"\nscripted-vs-baseline: scripted={scripted} "
          f"baseline={baseline} "
          f"(report only — no assertion by design)")
