import numpy as np
import pytest

from cogame_nmmo.sim import (ACT_HIGH, NOOP_ACTION, NUM_AGENTS, OBS_SIZE,
                             STAT_COMB_LVL, STAT_CUM_MIN_COMB_PROF,
                             STAT_DEATHS, STAT_HP, STAT_LIFE_MIN_COMB_PROF,
                             STAT_PROF_LVL, NmmoSim)


def test_step_shapes():
    sim = NmmoSim(seed=7)
    obs = sim.observations()
    assert obs.shape == (NUM_AGENTS, OBS_SIZE)
    assert obs.shape == (8, 1707)
    assert obs.dtype == np.uint8
    # obs is an independent copy: mutating it must not affect later reads
    original_byte = obs[0, 0]
    obs[0, 0] ^= 0xFF
    assert sim.observations()[0, 0] == original_byte

    rew = sim.rewards()
    assert rew.shape == (NUM_AGENTS,)
    assert rew.dtype == np.float32

    term = sim.terminals()
    assert term.shape == (NUM_AGENTS,)
    assert term.dtype == np.float32
    assert not term.any()          # nobody done at tick 0
    assert sim.dones() == [False] * NUM_AGENTS

    assert sim.tick() == 0
    assert sim.fault() == 0

    sim.set_actions(np.array([NOOP_ACTION] * NUM_AGENTS, dtype=np.float32))
    sim.step()
    assert sim.tick() == 1

    # per-agent stats: everyone starts alive at comb=prof=1
    for pid in range(NUM_AGENTS):
        assert sim.agent_stat(pid, STAT_COMB_LVL) == 1
        assert sim.agent_stat(pid, STAT_PROF_LVL) == 1
        assert sim.score(pid) >= 1


def test_set_actions_rejects_non_finite():
    sim = NmmoSim(seed=7)
    acts = np.array([NOOP_ACTION] * NUM_AGENTS, dtype=np.float32)
    acts[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        sim.set_actions(acts)
    acts[3, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        sim.set_actions(acts)
    with pytest.raises(ValueError, match="actions must be"):
        sim.set_actions(np.zeros((NUM_AGENTS, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="actions must be"):
        sim.set_actions(np.zeros((NUM_AGENTS - 1, 1), dtype=np.float32))


def test_set_actions_clamps_out_of_range():
    # out-of-range values are clamped to 0..25: a sim fed wild values must
    # behave identically to one fed the pre-clamped equivalents
    wild = NmmoSim(seed=7)
    tame = NmmoSim(seed=7)

    wild_acts = np.array(
        [[100.0], [-50.0], [25.9], [-0.4], [26.0], [7.5], [0.0], [4.0]],
        dtype=np.float32)
    tame_acts = np.array(
        [[25.0], [0.0], [25.0], [0.0], [25.0], [7.0], [0.0], [4.0]],
        dtype=np.float32)
    for _ in range(10):
        wild.set_actions(wild_acts)
        tame.set_actions(tame_acts)
        wild.step()
        tame.step()
    assert wild.observations().tobytes() == tame.observations().tobytes()
    assert wild.state_digest() == tame.state_digest()


# -- terminals / respawn surface --------------------------------------------

def test_terminals_fire_and_clear_per_tick():
    """Per-agent done flags: the env only SETS terminals (add_player_log);
    the shim zeroes the buffer before each c_step, so each read reflects
    exactly that tick. Random play against 2048 enemies reliably produces
    deaths within a few hundred ticks."""
    sim = NmmoSim(seed=7)
    rng = np.random.default_rng(0)
    death_events = 0
    saw_flag_clear_after_death = False
    prev_dones = [False] * NUM_AGENTS
    for _ in range(600):
        acts = rng.integers(0, ACT_HIGH, size=(NUM_AGENTS, 1)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
        dones = sim.dones()
        death_events += sum(dones)
        # observing any prev-done agent now not-done proves per-tick clearing
        for pid in range(NUM_AGENTS):
            if prev_dones[pid] and not dones[pid]:
                saw_flag_clear_after_death = True
        prev_dones = dones
    assert death_events > 0, "no deaths in 600 random ticks - seed regression?"
    assert saw_flag_clear_after_death, \
        "terminal flags never observed clearing between ticks"
    # the shim's deaths counters must agree with the flags we accumulated
    assert sum(sim.agent_stat(p, STAT_DEATHS) for p in range(NUM_AGENTS)) \
        == death_events


def test_stagnation_reset_attribution():
    """Directed test for the shim's stagnation branch (shim.c: the
    last_alive_min credit). A stagnation reset respawns the agent in the
    SAME tick, wiping its death-time levels before the host can read
    them, so the shim credits the value it cached on the last tick the
    agent was alive. Stagnation is the only terminal that leaves hp > 0
    (in-tick respawn at full health; an attack death leaves hp == 0
    until a later-tick respawn), so hp at the done tick identifies the
    branch.

    Note: an all-NOOP drive cannot reach the 500-tick stagnation
    threshold — idle agents are killed by roaming enemies within
    ~20-100 ticks (verified empirically on several seeds), and spawn()
    zeroes the stagnation ring buffer. Random play on seed 5 was
    scanned instead: it deterministically produces a stagnation event
    (tick 808, pid 2), and the tracked shadow value proves the credit
    is exactly the last-alive min(comb, prof).

    DISCRIMINATION LIMIT (honest accounting): in every pinned/observed
    event the agent's last-alive min(comb, prof) is 1 — and a fresh
    post-respawn life's min is also 1 — so this test CANNOT tell the
    correct credit (the cached last-alive value) from an inverted one
    that reads the just-respawned entity's levels: both equal 1 here.
    What it DOES pin: the stagnation branch fires (done with hp > 0),
    adds exactly one death, and credits exactly the shadow-tracked
    value (ruling out zero, double-counting, and cross-agent mixups).
    A discriminating event needs an agent that levels BOTH stats
    (last-alive min >= 2) and then goes flat for 500 alive ticks. A
    bounded offline scan (2026-08: 4 action mixes incl. a
    level-then-wander two-phase drive, 120 runs x 4000-5000 ticks,
    ~510k sim ticks, ~75 stagnation events observed) found none:
    agents at min >= 2 under random-ish play always die to attack
    before 500 flat ticks. If such a scenario is ever found, pin it
    here as `credit == last-alive min >= 2 != fresh-life min`."""
    sim = NmmoSim(seed=5)
    rng = np.random.default_rng(0)
    # min(comb,prof) at the end of the last tick each agent was alive —
    # the same shadow the shim maintains (everyone starts alive at 1)
    prev_life_min = [1] * NUM_AGENTS
    stagnations = 0
    for _ in range(850):
        prev_cum = [sim.agent_stat(p, STAT_CUM_MIN_COMB_PROF)
                    for p in range(NUM_AGENTS)]
        prev_deaths = [sim.agent_stat(p, STAT_DEATHS)
                       for p in range(NUM_AGENTS)]
        acts = rng.integers(0, ACT_HIGH,
                            size=(NUM_AGENTS, 1)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
        for pid, done in enumerate(sim.dones()):
            if done and sim.agent_stat(pid, STAT_HP) > 0:
                # stagnation reset: credited from the cached last-alive
                # value, exactly one death added
                stagnations += 1
                assert sim.agent_stat(pid, STAT_DEATHS) == \
                    prev_deaths[pid] + 1, (pid, sim.tick())
                assert sim.agent_stat(pid, STAT_CUM_MIN_COMB_PROF) == \
                    prev_cum[pid] + prev_life_min[pid], (pid, sim.tick())
        # refresh the shadow AFTER the assertions (mirrors the shim's
        # post-credit update; a just-respawned agent re-enters at its
        # fresh life's min)
        for pid in range(NUM_AGENTS):
            if sim.agent_stat(pid, STAT_HP) > 0:
                prev_life_min[pid] = min(
                    sim.agent_stat(pid, STAT_COMB_LVL),
                    sim.agent_stat(pid, STAT_PROF_LVL))
    assert stagnations >= 1, \
        "seed 5 no longer produces a stagnation event - seed regression?"


def test_score_identity_and_death_accounting():
    """score(pid) == cum_min_comb_prof + current-life min; the current-life
    part is 0 exactly while the agent is dead-awaiting-respawn (hp == 0)."""
    sim = NmmoSim(seed=11)
    rng = np.random.default_rng(1)
    for _ in range(400):
        acts = rng.integers(0, ACT_HIGH, size=(NUM_AGENTS, 1)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
        for pid in range(NUM_AGENTS):
            cum = sim.agent_stat(pid, STAT_CUM_MIN_COMB_PROF)
            life = sim.agent_stat(pid, STAT_LIFE_MIN_COMB_PROF)
            assert sim.score(pid) == cum + life
            if sim.agent_stat(pid, STAT_HP) == 0:
                assert life == 0
            else:
                assert life == min(sim.agent_stat(pid, STAT_COMB_LVL),
                                   sim.agent_stat(pid, STAT_PROF_LVL))
    assert sum(sim.agent_stat(p, STAT_DEATHS) for p in range(NUM_AGENTS)) > 0, \
        "no deaths in 400 random ticks - accounting untested"


# -- determinism -------------------------------------------------------------

def _run(seed, ticks=800):
    """Full stream fingerprint: obs, rewards, terminals, digest. 800 ticks
    of random play sees dozens of deaths (each death/respawn consumes
    env.rng draws), so equality here proves determinism THROUGH deaths and
    respawns, not just in the peaceful prefix."""
    sim = NmmoSim(seed=seed)
    rng = np.random.default_rng(0)
    chunks = []
    deaths = 0
    for _ in range(ticks):
        acts = rng.integers(0, ACT_HIGH, size=(NUM_AGENTS, 1)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
        chunks.append(sim.observations().tobytes())
        chunks.append(sim.rewards().tobytes())
        chunks.append(sim.terminals().tobytes())
        deaths += sum(sim.dones())
    chunks.append(sim.state_digest().to_bytes(4, "little"))
    return b"".join(chunks), deaths


def test_determinism_same_seed_through_deaths():
    stream_a, deaths_a = _run(7)
    stream_b, deaths_b = _run(7)
    assert deaths_a > 0, "no deaths - the post-respawn RNG path is untested"
    assert deaths_a == deaths_b
    assert stream_a == stream_b


def test_seed_changes_stream():
    # proves env.rng seeding (shim writes env.rng = seed) actually varies runs
    stream_a, _ = _run(7, ticks=50)
    stream_b, _ = _run(8, ticks=50)
    assert stream_a != stream_b


def test_initial_obs_differs_across_seeds():
    # world gen consumes env.rng in c_reset: different seed, different world
    assert NmmoSim(seed=1).observations().tobytes() \
        != NmmoSim(seed=2).observations().tobytes()


def test_obs_size_export_matches_host_constant():
    # obs_size() (sim/shim.c) is the wasm's own statement of the per-agent
    # obs stride; __init__ hard-fails on mismatch, so reaching this line
    # after an instantiation already proves agreement — assert the export
    # value directly anyway for a loud, named failure.
    sim = NmmoSim(seed=7)
    assert sim._exports["obs_size"](sim._store) == OBS_SIZE == 1707
