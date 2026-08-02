"""Tests for the player client library and random player (Phase N3).

Real episodes against the in-process GameServer (fixtures reused from
tests.test_server) plus targeted reconnect/fatal-error/resets-plumbing
tests against a minimal protocol-speaking fake server.
"""

import asyncio
import base64
import json

import numpy as np
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from cogame_nmmo import defaults
from cogame_nmmo.replay import Replay
from players import client, random_player
from players.client import PlayerError, play_episode

from tests.test_server import ServerHarness, make_config

OBS_SIZE = 1707


# -- full episodes driven by the client library ------------------------------

async def test_random_players_complete_8_seat_episode(tmp_path):
    cfg = make_config(max_ticks=12)
    async with ServerHarness(cfg, tmp_path) as h:
        results = await asyncio.gather(*(
            play_episode(random_player.RandomPolicy(seed=s),
                         h.ws_url(s, f"token-{s}"))
            for s in range(8)))
        engine_result = await h.episode_task

    # every client got the same result doc from the done message
    assert all(isinstance(r, dict) for r in results)
    assert all(r["final_tick"] == engine_result.final_tick for r in results)
    # score floor: every agent is alive at min(comb,prof) >= 1 (an ended
    # life banks its min, so a death never drops a seat below 1)
    assert len(results[0]["scores"]) == 8
    assert all(s >= 1 for s in results[0]["scores"])

    # artifacts written: results + parseable replay
    written = json.loads(h.results_path.read_text())
    assert written["scores"] == results[0]["scores"]
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == engine_result.final_tick
    # nobody degraded to NOOP: the clients kept up every tick
    assert written["noop_ticks"] == [0] * 8


async def test_random_players_complete_multi_hero_variant(tmp_path):
    """Same client code drives a 4-hero seat: 4 obs + 4 resets in,
    4 one-int action rows out."""
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=10)
    seen_obs_counts = set()
    seen_resets = []

    class CountingRandomPolicy(random_player.RandomPolicy):
        def __call__(self, tick, obs_rows, resets):
            seen_obs_counts.add(len(obs_rows))
            assert all(len(o) == OBS_SIZE for o in obs_rows)
            seen_resets.append((tick, resets))
            return super().__call__(tick, obs_rows, resets)

    async with ServerHarness(cfg, tmp_path) as h:
        results = await asyncio.gather(
            play_episode(CountingRandomPolicy(seed=0), h.ws_url(0, "token-0")),
            play_episode(CountingRandomPolicy(seed=1), h.ws_url(1, "token-1")))
        engine_result = await h.episode_task

    assert seen_obs_counts == {4}
    # protocol v2: resets reach the policy every tick, all False at tick 0
    assert all(len(r) == 4 and all(isinstance(b, bool) for b in r)
               for _, r in seen_resets)
    tick0_resets = [r for t, r in seen_resets if t == 0]
    assert tick0_resets and all(r == [False] * 4 for r in tick0_resets)
    assert len(results[0]["scores"]) == 2
    written = json.loads(h.results_path.read_text())
    assert written["noop_ticks"] == [0, 0]
    assert engine_result.final_tick == 10


# -- fatal handshake errors --------------------------------------------------

async def test_bad_token_is_fatal_403(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        with pytest.raises(PlayerError, match="403"):
            await play_episode(random_player.RandomPolicy(seed=0),
                               h.ws_url(0, "wrong-token"))


async def test_duplicate_seat_409_is_retried_then_succeeds():
    """PROTOCOL.md: a seat may reconnect. A 409 usually means the seat's
    previous (stale) connection has not been reaped yet — the server
    heartbeats and strike-closes it — so the client must retry, not die."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise web.HTTPConflict(text="slot already connected")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(_obs_msg(0))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                break
        await ws.send_str(json.dumps(
            {"done": True, "result": {"scores": [1.0]}}))
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/player", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        result = await play_episode(
            random_player.RandomPolicy(seed=0),
            str(server.make_url("/player")),
            reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    assert result == {"scores": [1.0]}
    assert calls == 3


async def test_duplicate_seat_409_retry_budget_is_bounded():
    """A slot that stays occupied exhausts the reconnect budget (bounded
    retry, not an infinite loop and not an instant fatal)."""
    calls = 0

    async def always_conflict(request):
        nonlocal calls
        calls += 1
        raise web.HTTPConflict(text="slot already connected")

    app = web.Application()
    app.router.add_get("/player", always_conflict)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="giving up"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")),
                max_connect_attempts=3,
                reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    assert calls == 3


# -- reconnect behavior ------------------------------------------------------

def _obs_msg(tick, heroes=1, resets=None):
    return json.dumps({
        "tick": tick,
        "obs": [base64.b64encode(bytes(OBS_SIZE)).decode("ascii")] * heroes,
        "resets": [False] * heroes if resets is None else resets,
    })


class FlakyProtocolServer:
    """Fake seat endpoint: drops the first connection mid-episode, then
    finishes the episode on the second connection."""

    def __init__(self):
        self.connections = 0
        self.received = []  # decoded action replies across connections

    async def handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1

        async def collect(n):
            got = 0
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                self.received.append(json.loads(msg.data))
                got += 1
                if got >= n:
                    return
            raise AssertionError("client closed early")

        if self.connections == 1:
            await ws.send_str(_obs_msg(0))
            await collect(1)
            await ws.send_str(_obs_msg(1))
            await collect(1)
            await ws.close()  # drop mid-episode
        else:
            await ws.send_str(_obs_msg(2))
            await collect(1)
            await ws.send_str(json.dumps(
                {"done": True, "result": {"scores": [1.0]}}))
            await ws.close()
        return ws


async def test_client_reconnects_after_midgame_drop():
    fake = FlakyProtocolServer()
    app = web.Application()
    app.router.add_get("/player", fake.handler)
    server = TestServer(app)
    await server.start_server()
    try:
        result = await play_episode(
            random_player.RandomPolicy(seed=3),
            str(server.make_url("/player")),
            reconnect_delay_seconds=0.01)
    finally:
        await server.close()

    assert result == {"scores": [1.0]}
    assert fake.connections == 2
    # all three ticks answered, echoing the server's tick numbers
    assert [m["tick"] for m in fake.received] == [0, 1, 2]
    assert all(len(m["actions"]) == 1 and len(m["actions"][0]) == 1
               for m in fake.received)


async def test_client_gives_up_after_bounded_reconnects():
    """Connections that never make progress exhaust the attempt budget."""
    drops = 0

    async def drop_immediately(request):
        nonlocal drops
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        drops += 1
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/player", drop_immediately)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="giving up"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")),
                max_connect_attempts=3,
                reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    assert drops == 3


async def test_unreachable_server_gives_up():
    with pytest.raises(PlayerError, match="giving up"):
        await play_episode(
            random_player.RandomPolicy(seed=0),
            "http://127.0.0.1:9/player",  # port 9 (discard): refused
            max_connect_attempts=2,
            reconnect_delay_seconds=0.01)


async def test_reconnect_attempts_are_logged(capsys):
    fake = FlakyProtocolServer()
    app = web.Application()
    app.router.add_get("/player", fake.handler)
    server = TestServer(app)
    await server.start_server()
    try:
        await play_episode(
            random_player.RandomPolicy(seed=3),
            str(server.make_url("/player")),
            reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    err = capsys.readouterr().err
    assert "connection attempt failed" in err
    assert "2 ticks answered so far" in err


# -- malformed messages ------------------------------------------------------

async def test_malformed_obs_is_clean_player_error():
    """A garbage obs payload raises PlayerError (clean exit-1 path in
    run_policy_main), not a raw TypeError/binascii traceback."""

    async def bad_obs_server(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps({"tick": 0, "obs": "not-a-list!!"}))
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/player", bad_obs_server)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="malformed obs"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")))
    finally:
        await server.close()


@pytest.mark.parametrize("resets_field", [
    None,                # missing entirely
    [True],              # wrong length for 2 heroes
    [1, 0],              # right length, not bools
    "nope",              # not a list
])
async def test_missing_or_malformed_resets_is_clean_player_error(resets_field):
    """Protocol v2 requires resets every tick; a missing/mis-shaped field
    must fail loudly — silently defaulting would corrupt recurrent
    policies' state across respawns."""

    async def bad_resets_server(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        msg = {"tick": 0, "obs": [
            base64.b64encode(bytes(OBS_SIZE)).decode("ascii")] * 2}
        if resets_field is not None:
            msg["resets"] = resets_field
        await ws.send_str(json.dumps(msg))
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/player", bad_resets_server)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="malformed resets"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")))
    finally:
        await server.close()


async def test_resets_reach_the_policy():
    """The client passes each tick's resets list through to the policy
    verbatim (the recurrent-state-zeroing hook)."""
    sent = [[False, False], [True, False], [False, True]]
    seen = []

    async def resets_server(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        for t, resets in enumerate(sent):
            await ws.send_str(_obs_msg(t, heroes=2, resets=resets))
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    break
        await ws.send_str(json.dumps({"done": True, "result": {}}))
        await ws.close()
        return ws

    def recording_policy(tick, obs_rows, resets):
        seen.append((tick, list(resets)))
        return [[4]] * len(obs_rows)

    app = web.Application()
    app.router.add_get("/player", resets_server)
    server = TestServer(app)
    await server.start_server()
    try:
        await play_episode(recording_policy, str(server.make_url("/player")))
    finally:
        await server.close()
    assert seen == [(0, sent[0]), (1, sent[1]), (2, sent[2])]


async def test_policy_row_count_mismatch_is_fatal():
    """A policy returning the wrong number of action rows fails fast
    locally instead of degrading to silent server-side strikes."""

    async def one_hero_server(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(_obs_msg(0, heroes=1))
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/player", one_hero_server)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="2 action rows for 1 heroes"):
            await play_episode(
                lambda tick, obs_rows, resets: [[4]] * 2,
                str(server.make_url("/player")))
    finally:
        await server.close()


# -- env plumbing + policy properties ----------------------------------------

def test_bad_seed_env_is_fatal(monkeypatch):
    monkeypatch.setenv("COGAME_PLAYER_SEED", "not-a-number")
    with pytest.raises(PlayerError, match="COGAME_PLAYER_SEED"):
        client.seed_from_env()
    with pytest.raises(PlayerError, match="COGAME_PLAYER_SEED"):
        random_player.policy_from_env()
    monkeypatch.delenv("COGAME_PLAYER_SEED")
    assert client.seed_from_env(default=7) == 7


def test_ws_url_env_precedence(monkeypatch):
    monkeypatch.delenv("COWORLD_PLAYER_WS_URL", raising=False)
    monkeypatch.delenv("COGAMES_ENGINE_WS_URL", raising=False)
    with pytest.raises(PlayerError, match="COWORLD_PLAYER_WS_URL"):
        client.ws_url_from_env()
    monkeypatch.setenv("COGAMES_ENGINE_WS_URL", "ws://b/player")
    assert client.ws_url_from_env() == "ws://b/player"
    monkeypatch.setenv("COWORLD_PLAYER_WS_URL", "ws://a/player")
    assert client.ws_url_from_env() == "ws://a/player"


def test_random_policy_seeded_and_in_range(monkeypatch):
    # players/ deliberately duplicates ACT_HIGH; keep it in sync
    assert random_player.ACT_HIGH == defaults.ACT_HIGH

    a = random_player.RandomPolicy(seed=42)
    b = random_player.RandomPolicy(seed=42)
    obs_rows = [bytes(OBS_SIZE)] * 5
    resets = [False] * 5
    acts_a = [a(t, obs_rows, resets) for t in range(20)]
    acts_b = [b(t, obs_rows, resets) for t in range(20)]
    assert acts_a == acts_b  # deterministic under COGAME_PLAYER_SEED
    arr = np.asarray(acts_a)
    assert arr.shape == (20, 5, 1)
    assert (arr >= 0).all()
    assert (arr < np.asarray(defaults.ACT_HIGH)).all()

    monkeypatch.setenv("COGAME_PLAYER_SEED", "7")
    p1 = random_player.policy_from_env()
    p2 = random_player.policy_from_env()
    assert p1(0, obs_rows, resets) == p2(0, obs_rows, resets)
