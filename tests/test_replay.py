"""Tests for the binary replay format v1: round-trip and re-simulation."""

import json

import numpy as np
import pytest

from cogame_nmmo import defaults, replay
from cogame_nmmo.config import GameConfig
from cogame_nmmo.engine import LockstepEngine
from cogame_nmmo.replay import Replay, ReplayError, ReplayWriter

NOOP = list(defaults.NOOP_ACTION)


def make_config(num_seats=8, **overrides):
    d = {
        "players": [{"name": f"agent-{i}"} for i in range(num_seats)],
        "tokens": [f"tok{i}" for i in range(num_seats)],
        "seed": 77,
        "max_ticks": 120,
        "tick_deadline_ms": 2000,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


def random_actions(rng, n_ticks, num_agents=8):
    return [rng.integers(0, defaults.ACT_HIGH,
                         size=(num_agents, 1)).astype(np.uint8)
            for _ in range(n_ticks)]


# -- round trip --------------------------------------------------------------

def test_round_trip():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    rng = np.random.default_rng(0)
    ticks = random_actions(rng, 25)
    for t, acts in enumerate(ticks):
        writer.append_tick(t, acts)
    result = {"scores": [3.0] * 8, "end_reason": "tick_cap", "final_tick": 25}
    data = writer.finalize(result)

    rp = Replay.parse(data)
    assert rp.header["format_version"] == 1
    assert rp.header["sim_wasm_sha256"] == "ab" * 32
    assert rp.header["result"] == result
    assert rp.header["tick_count"] == 25
    assert rp.tick_count == 25
    assert rp.num_agents == 8
    # config round-trips fully resolved, names included, tokens excluded
    assert rp.header["config"] == cfg.to_dict()
    assert [p["name"] for p in rp.header["config"]["players"]] == \
        [f"agent-{i}" for i in range(8)]
    assert "tokens" not in rp.header["config"]
    assert rp.header["config"]["seed"] == 77
    for t, acts in enumerate(rp):
        assert acts.dtype == np.uint8 and acts.shape == (8, 1)
        np.testing.assert_array_equal(acts, ticks[t])
    np.testing.assert_array_equal(rp.actions(10), ticks[10])


def test_binary_layout():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="cd" * 32)
    writer.append_tick(0, np.zeros((8, 1), dtype=np.uint8))
    data = writer.finalize({})
    assert data[:4] == b"NMMO"
    assert data[4] == 1  # version
    header_len = int.from_bytes(data[5:9], "little")
    header = json.loads(data[9:9 + header_len])
    assert header["tick_count"] == 1
    body = data[9 + header_len:]
    assert len(body) == 8  # 8 agents x 1 uint8 action per tick
    assert body == b"\x00" * 8


def test_multi_agent_seat_stride():
    # 2 seats x 4 heroes: body stride is still num_agents = 8 bytes/tick
    cfg = make_config(num_seats=2, heroes_per_seat=4)
    writer = ReplayWriter(cfg, "aa" * 32)
    writer.append_tick(0, np.arange(8, dtype=np.uint8).reshape(8, 1))
    rp = Replay.parse(writer.finalize({}))
    assert rp.num_agents == 8
    assert rp.actions(0).flatten().tolist() == list(range(8))


# -- validation --------------------------------------------------------------

def test_bad_magic_rejected():
    with pytest.raises(ReplayError):
        Replay.parse(b"MOBA" + b"\x01" + b"\x00" * 20)


def test_bad_version_rejected():
    cfg = make_config()
    data = bytearray(ReplayWriter(cfg, "ee" * 32).finalize({}))
    data[4] = 9
    with pytest.raises(ReplayError):
        Replay.parse(bytes(data))


def test_truncated_body_rejected():
    cfg = make_config()
    writer = ReplayWriter(cfg, "ee" * 32)
    writer.append_tick(0, np.ones((8, 1), dtype=np.uint8))
    data = writer.finalize({})
    with pytest.raises(ReplayError):
        Replay.parse(data[:-3])


def test_truncated_header_rejected():
    with pytest.raises(ReplayError):
        Replay.parse(b"NMMO\x01\xff\xff\xff\x00rest")


def test_bool_tick_count_rejected():
    # bool is an int subclass: tick_count true must not parse as 1
    header = json.dumps({
        "tick_count": True,
        "config": {"players": [{"name": "a"}], "heroes_per_seat": 1},
    }).encode()
    data = b"NMMO\x01" + len(header).to_bytes(4, "little") + header
    with pytest.raises(ReplayError, match="integer tick_count"):
        Replay.parse(data)


def test_header_without_agent_topology_rejected():
    # the body stride comes from the header config; a header without it
    # is unusable even if tick_count parses
    header = json.dumps({"tick_count": 0, "config": {}}).encode()
    data = b"NMMO\x01" + len(header).to_bytes(4, "little") + header
    with pytest.raises(ReplayError):
        Replay.parse(data)


def test_writer_rejects_non_sequential_tick():
    writer = ReplayWriter(make_config(), "ee" * 32)
    writer.append_tick(0, np.zeros((8, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        writer.append_tick(2, np.zeros((8, 1), dtype=np.uint8))


def test_writer_rejects_bad_tick_shape():
    writer = ReplayWriter(make_config(), "ee" * 32)
    with pytest.raises(ValueError):
        writer.append_tick(0, np.zeros((5, 1), dtype=np.uint8))


def test_writer_rejects_out_of_range_actions():
    # replays store post-clamp actions only: 26+ or negative is a bug
    writer = ReplayWriter(make_config(), "ee" * 32)
    bad = np.zeros((8, 1), dtype=np.int16)
    bad[3, 0] = 26
    with pytest.raises(ValueError, match="out of range"):
        writer.append_tick(0, bad)
    bad[3, 0] = -1
    with pytest.raises(ValueError, match="out of range"):
        writer.append_tick(0, bad)
    writer.append_tick(0, np.full((8, 1), 25, dtype=np.uint8))  # max is fine


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
    tick, final obs bytes, and state digest."""
    from cogame_nmmo.sim import NmmoSim

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(1000 + seat)

        async def get_actions(self, tick, obs, resets):
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(1, 1)).tolist()

    cfg = make_config(max_ticks=120)
    sim = NmmoSim(seed=cfg.seed, num_agents=cfg.num_agents)
    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    engine = LockstepEngine(
        sim, cfg, [RngSource(s) for s in range(8)],
        on_tick=writer.append_tick)
    result = await engine.run()
    data = writer.finalize({
        "scores": list(result.seat_scores),
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
    })
    recorded_obs = sim.observations().tobytes()
    recorded_digest = sim.state_digest()
    assert result.state_digest == recorded_digest

    rp = Replay.parse(data)
    assert rp.tick_count == result.final_tick
    resim = NmmoSim(seed=rp.header["config"]["seed"],
                    num_agents=rp.num_agents)
    for acts in rp:
        resim.set_actions(acts.astype(np.float32))
        resim.step()
    assert resim.tick() == result.final_tick
    assert resim.observations().tobytes() == recorded_obs
    assert resim.state_digest() == recorded_digest
