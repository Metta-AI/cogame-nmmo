"""Tests for the baseline player (Phase N3): wasm MMONet brain.

Unit tests exercise build/nmmo3_brain.wasm directly (load, param count,
valid actions, determinism, reset-on-done semantics). The slow behavioral
test drives the real sim with baseline seats vs random seats — mirroring
the demo loop (vendored nmmo3.c demo(): forward with terminals-driven
state zeroing, then c_step) — and asserts the pretrained policy scores
at least as well as random. A short websocket episode covers
BaselinePolicy end-to-end against the real server.
"""

import asyncio
import json

import numpy as np
import pytest

from cogame_nmmo import defaults
from cogame_nmmo.sim import NmmoSim
from players.baseline_player import (DEFAULT_NUM_BRAINS, EXPECTED_PARAMS,
                                     NUM_ATNS, OBS_SIZE, BaselinePolicy,
                                     NmmoBrain)
from players.client import play_episode
from players.random_player import RandomPolicy

from tests.test_server import ServerHarness, make_config


# -- unit: wasm brain --------------------------------------------------------

def test_brain_loads_with_expected_param_count():
    # brain_init returns the embedded weight count; the constructor
    # raises if it isn't the 4,430,976 params of nmmo3_weights.bin
    # (17,723,904 bytes / 4 — see the sim/brain_shim.c header for the
    # per-tensor breakdown that sums to exactly this)
    NmmoBrain(seed=1)
    assert EXPECTED_PARAMS == 4_430_976


def test_forward_returns_in_range_actions():
    brain = NmmoBrain(seed=1)
    obs = NmmoSim(seed=3).observations()  # real first-tick obs
    for agent in range(DEFAULT_NUM_BRAINS):
        acts = brain.forward(agent, obs[agent].tobytes())
        assert len(acts) == NUM_ATNS
        assert all(isinstance(a, int) for a in acts)
        assert all(0 <= a < high
                   for a, high in zip(acts, defaults.ACT_HIGH))


def test_brain_init_rejects_second_call():
    # re-init would leak nets and reset recurrent state; the shim
    # returns -1 (see brain_shim.c)
    brain = NmmoBrain(seed=1)
    assert brain._exports["brain_init"](brain._store, 1, 8) == -1


def test_brain_init_rejects_bad_num_agents():
    # the shim rejects out-of-range net counts; the host constructor
    # surfaces that as its param-count RuntimeError
    for bad in (0, -1, 33):
        with pytest.raises(RuntimeError, match="weight params"):
            NmmoBrain(seed=1, num_agents=bad)


def test_forward_and_reset_input_validation():
    brain = NmmoBrain(seed=1, num_agents=4)
    with pytest.raises(ValueError, match="agent_idx"):
        brain.forward(4, bytes(OBS_SIZE))
    with pytest.raises(ValueError, match="agent_idx"):
        brain.forward(-1, bytes(OBS_SIZE))
    with pytest.raises(ValueError, match="agent_idx"):
        brain.reset_state(4)
    with pytest.raises(ValueError, match="1707 bytes"):
        brain.forward(0, bytes(12))


def _roll_obs_history(seed, ticks, brain):
    """Drive a real sim with `brain` on all 8 agents (demo-loop shape:
    zero state on the previous step's terminal before consuming the
    obs), recording each tick's (obs, pre-step resets)."""
    sim = NmmoSim(seed=seed)
    resets = [False] * 8
    history = []
    for _ in range(ticks):
        obs = sim.observations()
        history.append((obs.copy(), list(resets)))
        acts = np.zeros((8, 1), dtype=np.float32)
        for p in range(8):
            if resets[p]:
                brain.reset_state(p)
            acts[p, 0] = brain.forward(p, obs[p].tobytes())[0]
        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()
    return history


def _replay_history(history, brain):
    out = []
    for obs, resets in history:
        tick_actions = []
        for p in range(8):
            if resets[p]:
                brain.reset_state(p)
            tick_actions.append(brain.forward(p, obs[p].tobytes())[0])
        out.append(tick_actions)
    return out


def test_two_fresh_instances_are_identical_including_resets():
    """Same seed + same obs sequence + same reset schedule in the same
    call order -> identical actions (recurrent state and the sampling
    RNG both live in-wasm). The obs history is rolled from a live sim so
    it can contain real respawn resets; the replay applies the same
    reset_state calls, so determinism is asserted across them."""
    brain_a = NmmoBrain(seed=1)
    history = _roll_obs_history(seed=5, ticks=20, brain=brain_a)

    brain_a2 = NmmoBrain(seed=1)  # re-derive A's actions from the history
    brain_b = NmmoBrain(seed=1)
    assert _replay_history(history, brain_a2) == \
        _replay_history(history, brain_b)


def test_different_seed_diverges():
    """Sampling is stochastic (softmax_multidiscrete samples via the
    module's rand(), not argmax): different RNG seeds produce different
    action sequences on the same obs."""
    obs = NmmoSim(seed=3).observations()
    a = NmmoBrain(seed=1)
    b = NmmoBrain(seed=1234)
    seq_a = [a.forward(p, obs[p].tobytes()) for _ in range(5)
             for p in range(8)]
    seq_b = [b.forward(p, obs[p].tobytes()) for _ in range(5)
             for p in range(8)]
    assert seq_a != seq_b


def test_reset_state_changes_behavior():
    """The same obs stream with vs without a mid-stream reset_state call
    diverges afterwards — proving reset actually zeroes live recurrent
    state (a no-op reset_state would keep the sequences identical,
    because both instances share seed, obs, and rand-stream position:
    softmax_multidiscrete draws exactly one rand() per forward, so the
    streams stay aligned action-for-action). Fully deterministic: every
    input is pinned, so this cannot flake."""
    roller = NmmoBrain(seed=1, num_agents=1)
    sim = NmmoSim(seed=7)
    obs_stream = []
    for _ in range(40):
        obs = sim.observations()
        obs_stream.append(obs[0].tobytes())
        acts = np.zeros((8, 1), dtype=np.float32)
        acts[0, 0] = roller.forward(0, obs_stream[-1])[0]
        sim.set_actions(acts)
        sim.step()

    reset_at = 20
    plain = NmmoBrain(seed=1, num_agents=1)
    reset = NmmoBrain(seed=1, num_agents=1)
    seq_plain, seq_reset = [], []
    for t, ob in enumerate(obs_stream):
        if t == reset_at:
            reset.reset_state(0)
        seq_plain.append(plain.forward(0, ob)[0])
        seq_reset.append(reset.forward(0, ob)[0])

    # identical up to the reset...
    assert seq_plain[:reset_at] == seq_reset[:reset_at]
    # ...then the zeroed state changes the sampled trajectory
    assert seq_plain[reset_at:] != seq_reset[reset_at:]


# -- end-to-end: BaselinePolicy over the real websocket stack ---------------

async def test_baseline_policy_plays_ws_episode(tmp_path):
    """Short real episode, 2 seats x 4 heroes: BaselinePolicy answers
    every tick (obs + resets in, one action int per hero out)."""
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=8)
    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(
            play_episode(BaselinePolicy(seed=1, num_agents=4),
                         h.ws_url(0, "token-0")),
            play_episode(RandomPolicy(seed=99), h.ws_url(1, "token-1")))
        engine_result = await h.episode_task

    assert engine_result.final_tick == 8
    results = json.loads(h.results_path.read_text())
    assert results["noop_ticks"] == [0, 0]
    assert results["dead_seats"] == [False, False]
    assert all(m["final_tick"] == 8 for m in done_msgs)


# -- behavioral: baseline vs random on score ---------------------------------

BEHAVIORAL_TICKS = 2000
_behavioral_cache: dict[str, dict] = {}


def run_baseline_vs_random_episode():
    """Demo-loop sim drive, cached (one 2000-tick episode serves every
    behavioral assertion): agents 0-3 run the pretrained MMONet
    (per-agent state, reset-on-terminal exactly like the demo forward),
    agents 4-7 play uniform random. Returns per-group raw cumulative
    scores, deaths, and per-life means (score/(deaths+1) — lives = ended
    lives + the current one)."""
    if _behavioral_cache:
        return _behavioral_cache["stats"]
    sim = NmmoSim(seed=11)
    brain = NmmoBrain(seed=1, num_agents=4)
    rng = np.random.default_rng(99)
    resets = [False] * 8
    for _ in range(BEHAVIORAL_TICKS):
        obs = sim.observations()
        acts = np.zeros((8, 1), dtype=np.float32)
        for i in range(4):          # baseline seats: pids 0-3, brain i=pid
            if resets[i]:
                brain.reset_state(i)
            acts[i, 0] = brain.forward(i, obs[i].tobytes())[0]
        acts[4:8, 0] = rng.integers(0, 26, size=4)
        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()

    def group(pids):
        scores = [sim.score(p) for p in pids]
        deaths = [sim.agent_stat(p, 1) for p in pids]
        per_life = [s / (d + 1) for s, d in zip(scores, deaths)]
        return {"scores": scores, "deaths": deaths, "per_life": per_life}

    stats = {"baseline": group(range(4)), "random": group(range(4, 8))}
    print(f"\nbaseline-vs-random after {BEHAVIORAL_TICKS} ticks: {stats}")
    _behavioral_cache["stats"] = stats
    return stats


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason="N2 scoring-design finding (2026-08-02): the CURRENT ranking "
    "score is CUMULATIVE min(comb,prof) over ended lives, so every death "
    "banks at least 1 — and random dies to enemies ~every 50 ticks "
    "(measured seed 11 / 2000 ticks: random scores 44/38/33/30 with "
    "43/37/32/29 deaths; baseline 28/11/25/12 with 3/1/2/1 deaths, banking "
    "~9-14 per life). Death-count farming (~0.02/tick) outpaces coherent "
    "leveling, so this direction assertion fails for METRIC reasons, not "
    "port reasons. Owner decision (2026-08-02): rank by mean score per "
    "life, score/(deaths+1), engine-side — being implemented in the N2 "
    "server lane. FLIP THIS TO A HARD ASSERT (and drop the xfail) once "
    "that scoring commit lands; the per-life companion test below already "
    "asserts the post-fix ordering.")
def test_baseline_scores_at_least_random_raw_cumulative():
    """Intended N3 gate on the CURRENT (cumulative) ranking score; see
    the xfail reason for why the metric inverts it today and what
    replaces it."""
    stats = run_baseline_vs_random_episode()
    assert np.mean(stats["baseline"]["scores"]) >= \
        np.mean(stats["random"]["scores"]), stats


@pytest.mark.slow
def test_baseline_outlevels_random_per_life():
    """Metric-independent behavioral gate (and the ordering under the
    decided mean-per-life ranking score): per life, the pretrained net
    banks several levels of min(comb,prof) — it equips, harvests, and
    fights coherently — while random stays pinned at ~1 (it can level
    essentially only by accident). Measured seed 11: baseline per-life
    ~7-14, random ~1.

    Toolchain-drift note: if these numbers shift right after an
    emcc/emsdk bump, suspect a musl rand()/libm stream change shifting
    the sampled-action trajectory — recalibrate before suspecting the
    port."""
    stats = run_baseline_vs_random_episode()
    baseline_pl = stats["baseline"]["per_life"]
    random_pl = stats["random"]["per_life"]
    assert np.mean(baseline_pl) > np.mean(random_pl), stats
    # not a squeaker: the trained net's per-life banking is a multiple of
    # random's floor (measured ~7-14 vs ~1; assert a conservative 2x)
    assert np.mean(baseline_pl) >= 2 * np.mean(random_pl), stats
