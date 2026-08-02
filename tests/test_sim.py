import numpy as np
import pytest

from cogame_nmmo.sim import ACT_HIGH, NOOP_ACTION, MobaSim


def test_step_shapes():
    sim = MobaSim(seed=7)
    obs = sim.observations()
    assert obs.shape == (10, 510)
    assert obs.dtype == np.uint8
    # obs is an independent copy: mutating it must not affect later reads
    original_byte = obs[0, 0]
    obs[0, 0] ^= 0xFF
    assert sim.observations()[0, 0] == original_byte

    rew = sim.rewards()
    assert rew.shape == (10,)
    assert rew.dtype == np.float32

    assert sim.tick() == 0
    assert sim.done() == 0

    sim.set_actions(np.array([NOOP_ACTION] * 10, dtype=np.float32))
    sim.step()
    assert sim.tick() == 1

    # per-agent stats: everyone starts at level 1, 0 kills/deaths
    assert sim.agent_stat(0, 0) == 1
    assert sim.agent_stat(0, 1) == 0
    # both ancients start at full health (4500, TOWER_HEALTH[22..23])
    assert sim.ancient_health(0) == 4500.0
    assert sim.ancient_health(1) == 4500.0


def test_set_actions_rejects_non_finite():
    sim = MobaSim(seed=7)
    acts = np.array([NOOP_ACTION] * 10, dtype=np.float32)
    acts[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        sim.set_actions(acts)
    acts[3, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        sim.set_actions(acts)
    with pytest.raises(ValueError, match="actions must be"):
        sim.set_actions(np.zeros((10, 5), dtype=np.float32))


def test_set_actions_clamps_out_of_range():
    # out-of-range values are clamped to 0..high-1: a sim fed wild values must
    # behave identically to one fed the pre-clamped equivalents
    wild = MobaSim(seed=7)
    tame = MobaSim(seed=7)

    wild_acts = np.array([[100.0, -50.0, 9.0, -1.0, 2.5, 1.0]] * 10,
                         dtype=np.float32)
    tame_acts = np.array([[6.0, 0.0, 2.0, 0.0, 1.0, 1.0]] * 10,
                         dtype=np.float32)
    for _ in range(10):
        wild.set_actions(wild_acts)
        tame.set_actions(tame_acts)
        wild.step()
        tame.step()
    assert wild.observations().tobytes() == tame.observations().tobytes()


def _run(seed, ticks=300):
    sim = MobaSim(seed=seed)
    rng = np.random.default_rng(0)
    chunks = []
    for _ in range(ticks):
        acts = rng.integers(0, ACT_HIGH, size=(10, 6)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
        chunks.append(sim.observations().tobytes())
        chunks.append(sim.rewards().tobytes())
    return b"".join(chunks)


def test_determinism_same_seed():
    assert _run(7) == _run(7)


def test_seed_changes_stream():
    # proves patch 0002 (srand seeding) actually does something
    assert _run(7) != _run(8)
