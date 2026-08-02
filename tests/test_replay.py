# Inherited cogame-moba suite: exercises the moba-shaped modules this fork
# has not adapted yet. Skipped (not deleted) pending Phase N2 (server adaptation),
# which replaces it — see docs/plans/2026-08-02-cogame-nmmo-implementation.md.
import pytest

pytest.skip("moba-specific suite pending Phase N2 (server adaptation) rewrite",
            allow_module_level=True)

"""Tests for the binary replay format v1: round-trip and re-simulation."""

import json

import numpy as np
import pytest

from cogame_nmmo import defaults, replay
from cogame_nmmo.config import GameConfig
from cogame_nmmo.engine import LockstepEngine
from cogame_nmmo.replay import Replay, ReplayError, ReplayWriter

NOOP = list(defaults.NOOP_ACTION)


def make_config(**overrides):
    d = {
        "players": [{"name": f"hero-{i}"} for i in range(10)],
        "tokens": [f"tok{i}" for i in range(10)],
        "seed": 77,
        "max_ticks": 120,
        "tick_deadline_ms": 2000,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


def random_actions(rng, n_ticks):
    return [rng.integers(0, defaults.ACT_HIGH,
                         size=(10, 6)).astype(np.uint8)
            for _ in range(n_ticks)]


# -- round trip --------------------------------------------------------------

def test_round_trip():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    rng = np.random.default_rng(0)
    ticks = random_actions(rng, 25)
    for t, acts in enumerate(ticks):
        writer.append_tick(t, acts)
    result = {"winner": 1, "end_reason": "ancient", "final_tick": 25}
    data = writer.finalize(result)

    rp = Replay.parse(data)
    assert rp.header["format_version"] == 1
    assert rp.header["sim_wasm_sha256"] == "ab" * 32
    assert rp.header["result"] == result
    assert rp.header["tick_count"] == 25
    assert rp.tick_count == 25
    # config round-trips fully resolved, names included, tokens excluded
    assert rp.header["config"] == cfg.to_dict()
    assert [p["name"] for p in rp.header["config"]["players"]] == \
        [f"hero-{i}" for i in range(10)]
    assert "tokens" not in rp.header["config"]
    assert rp.header["config"]["seed"] == 77
    for t, acts in enumerate(rp):
        assert acts.dtype == np.uint8 and acts.shape == (10, 6)
        np.testing.assert_array_equal(acts, ticks[t])
    np.testing.assert_array_equal(rp.actions(10), ticks[10])


def test_binary_layout():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="cd" * 32)
    writer.append_tick(0, np.zeros((10, 6), dtype=np.uint8))
    data = writer.finalize({})
    assert data[:4] == b"MOBA"
    assert data[4] == 1  # version
    header_len = int.from_bytes(data[5:9], "little")
    header = json.loads(data[9:9 + header_len])
    assert header["tick_count"] == 1
    body = data[9 + header_len:]
    assert len(body) == 60  # 10 heroes x 6 uint8 per tick
    assert body == b"\x00" * 60


# -- validation --------------------------------------------------------------

def test_bad_magic_rejected():
    with pytest.raises(ReplayError):
        Replay.parse(b"NOPE" + b"\x01" + b"\x00" * 20)


def test_bad_version_rejected():
    cfg = make_config()
    data = bytearray(ReplayWriter(cfg, "ee" * 32).finalize({}))
    data[4] = 9
    with pytest.raises(ReplayError):
        Replay.parse(bytes(data))


def test_truncated_body_rejected():
    cfg = make_config()
    writer = ReplayWriter(cfg, "ee" * 32)
    writer.append_tick(0, np.ones((10, 6), dtype=np.uint8))
    data = writer.finalize({})
    with pytest.raises(ReplayError):
        Replay.parse(data[:-10])


def test_truncated_header_rejected():
    with pytest.raises(ReplayError):
        Replay.parse(b"MOBA\x01\xff\xff\xff\x00rest")


def test_writer_rejects_non_sequential_tick():
    writer = ReplayWriter(make_config(), "ee" * 32)
    writer.append_tick(0, np.zeros((10, 6), dtype=np.uint8))
    with pytest.raises(ValueError):
        writer.append_tick(2, np.zeros((10, 6), dtype=np.uint8))


def test_writer_rejects_bad_tick_shape():
    writer = ReplayWriter(make_config(), "ee" * 32)
    with pytest.raises(ValueError):
        writer.append_tick(0, np.zeros((5, 6), dtype=np.uint8))


def test_sim_wasm_sha256_matches_file(tmp_path):
    import hashlib
    p = tmp_path / "x.wasm"
    p.write_bytes(b"wasm bytes here")
    assert replay.sim_wasm_sha256(p) == \
        hashlib.sha256(b"wasm bytes here").hexdigest()


# -- re-simulation -----------------------------------------------------------

async def test_recorded_episode_resimulates_identically():
    """Record a real wasm episode via the engine hook, then re-run a fresh
    sim from the replay's seed feeding the replay's actions: same final
    tick, winner, and final obs bytes."""
    from cogame_nmmo.sim import MobaSim

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(1000 + seat)

        async def get_actions(self, tick, obs):
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(1, 6)).tolist()

    cfg = make_config(max_ticks=120)
    sim = MobaSim(seed=cfg.seed)
    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    engine = LockstepEngine(
        sim, cfg, [RngSource(s) for s in range(10)],
        on_tick=writer.append_tick)
    result = await engine.run()
    data = writer.finalize({
        "winner": result.winner,
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
    })
    recorded_obs = sim.observations().tobytes()

    rp = Replay.parse(data)
    assert rp.tick_count == result.final_tick
    resim = MobaSim(seed=rp.header["config"]["seed"])
    for acts in rp:
        resim.set_actions(acts.astype(np.float32))
        resim.step()
    assert resim.tick() == result.final_tick
    assert resim.done() == sim.done()
    assert resim.winner() == sim.winner()
    assert resim.observations().tobytes() == recorded_obs
