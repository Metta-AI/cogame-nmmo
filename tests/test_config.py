"""Tests for game config parsing/validation and seat/agent mapping helpers."""

import json

import pytest

from cogame_nmmo import defaults
from cogame_nmmo.config import ConfigError, GameConfig


def base_dict(num_seats=8, **overrides):
    d = {
        "players": [{"name": f"player{i}"} for i in range(num_seats)],
        "tokens": [f"token-{i}" for i in range(num_seats)],
    }
    d.update(overrides)
    return d


# -- defaults + parsing ------------------------------------------------------

def test_defaults_applied():
    cfg = GameConfig.from_dict(base_dict())
    assert cfg.max_ticks == 5000
    assert cfg.heroes_per_seat == 1
    assert cfg.tick_deadline_ms == 1000
    assert cfg.player_connect_timeout_seconds == 180
    assert cfg.num_seats == 8
    assert cfg.num_agents == 8
    assert [p.name for p in cfg.players] == [f"player{i}" for i in range(8)]


def test_seed_derived_and_recorded_when_missing():
    cfg = GameConfig.from_dict(base_dict())
    assert isinstance(cfg.seed, int)
    assert 0 <= cfg.seed <= 0xFFFFFFFF
    # the derived seed must be recorded in the resolved config (replay header)
    assert cfg.to_dict()["seed"] == cfg.seed


def test_explicit_seed_preserved():
    cfg = GameConfig.from_dict(base_dict(seed=1234))
    assert cfg.seed == 1234
    assert cfg.to_dict()["seed"] == 1234


def test_explicit_values_override_defaults():
    cfg = GameConfig.from_dict(base_dict(
        max_ticks=500, tick_deadline_ms=50,
        player_connect_timeout_seconds=2))
    assert cfg.max_ticks == 500
    assert cfg.tick_deadline_ms == 50
    assert cfg.player_connect_timeout_seconds == 2


def test_num_agents_is_seats_times_heroes():
    # num_agents is elastic upstream: seats x heroes_per_seat, any product
    cfg = GameConfig.from_dict(base_dict(num_seats=2, heroes_per_seat=4))
    assert cfg.num_seats == 2
    assert cfg.heroes_per_seat == 4
    assert cfg.num_agents == 8
    cfg = GameConfig.from_dict(base_dict(num_seats=3))
    assert cfg.num_agents == 3


# -- validation --------------------------------------------------------------

@pytest.mark.parametrize("heroes_per_seat", [0, -1, "2", 1.5, True])
def test_invalid_heroes_per_seat_rejected(heroes_per_seat):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(heroes_per_seat=heroes_per_seat))


def test_token_length_mismatch_rejected():
    d = base_dict()
    d["tokens"] = d["tokens"][:5]
    with pytest.raises(ConfigError):
        GameConfig.from_dict(d)


def test_missing_players_rejected():
    with pytest.raises(ConfigError):
        GameConfig.from_dict({"tokens": []})


def test_bad_max_ticks_rejected():
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(max_ticks=0))


def test_bad_tick_deadline_rejected():
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(tick_deadline_ms=-5))


def test_empty_player_name_rejected():
    d = base_dict()
    d["players"][3] = {"name": ""}
    with pytest.raises(ConfigError):
        GameConfig.from_dict(d)


# -- wall-clock budget -------------------------------------------------------

def test_wall_clock_budget_default_derived():
    # 5000-tick default x 1s deadline = 5000s worst case, so the platform
    # cap min(0.9 x episode timeout) = 3240s wins
    cfg = GameConfig.from_dict(base_dict())
    assert cfg.wall_clock_budget_seconds == pytest.approx(
        0.9 * defaults.PLATFORM_EPISODE_TIMEOUT_MINUTES * 60)
    # a short episode is capped by its own worst case instead
    cfg = GameConfig.from_dict(base_dict(max_ticks=100, tick_deadline_ms=500))
    assert cfg.wall_clock_budget_seconds == pytest.approx(50.0)


def test_wall_clock_budget_explicit_override():
    cfg = GameConfig.from_dict(base_dict(wall_clock_budget_seconds=123.5))
    assert cfg.wall_clock_budget_seconds == 123.5
    assert cfg.to_dict()["wall_clock_budget_seconds"] == 123.5


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"),
                                 "60", True, None])
def test_bad_wall_clock_budget_rejected(bad):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(wall_clock_budget_seconds=bad))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_connect_timeout_rejected(bad):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(player_connect_timeout_seconds=bad))


# -- serialization -----------------------------------------------------------

def test_to_dict_excludes_tokens_by_default():
    cfg = GameConfig.from_dict(base_dict(seed=7))
    d = cfg.to_dict()
    assert "tokens" not in d
    assert d["players"] == [{"name": f"player{i}"} for i in range(8)]
    # round-trips through from_dict (tokens re-supplied)
    d2 = dict(d, tokens=list(cfg.tokens))
    cfg2 = GameConfig.from_dict(d2)
    assert cfg2 == cfg


def test_from_file_uri(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base_dict(seed=42)))
    for uri in (f"file://{path}", str(path)):
        cfg = GameConfig.from_file_uri(uri)
        assert cfg.seed == 42


# -- seat/agent mapping helpers ----------------------------------------------

def test_noop_matches_sim_contract():
    from cogame_nmmo import sim
    assert list(defaults.NOOP_ACTION) == list(sim.NOOP_ACTION)
    assert tuple(defaults.ACT_HIGH) == tuple(sim.ACT_HIGH)


def test_seat_hero_pids_solo():
    for seat in range(8):
        assert list(defaults.seat_hero_pids(seat, 1)) == [seat]


def test_seat_hero_pids_multi():
    assert list(defaults.seat_hero_pids(0, 4)) == [0, 1, 2, 3]
    assert list(defaults.seat_hero_pids(1, 4)) == [4, 5, 6, 7]


def test_seat_for_pid_inverts_mapping():
    for h in (1, 4):
        for seat in range(4):
            for pid in defaults.seat_hero_pids(seat, h):
                assert defaults.seat_for_pid(pid, h) == seat


def test_clamp_actions():
    import numpy as np
    raw = np.array([[25.9], [-1.0], [100.0], [4.0], [0.0], [7.5], [26.0],
                    [-0.4]], dtype=np.float64)
    out = defaults.clamp_actions(raw)
    assert out.dtype == np.uint8
    assert out.tolist() == [[25], [0], [25], [4], [0], [7], [25], [0]]
