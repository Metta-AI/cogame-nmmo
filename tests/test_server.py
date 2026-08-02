"""End-to-end tests for the Coworld-contract websocket game server.

In-process aiohttp test server + real websocket clients.
"""

import asyncio
import base64
import json

import aiohttp
import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestServer

from cogame_nmmo import defaults, uris
from cogame_nmmo.config import GameConfig
from cogame_nmmo.engine import NOOP_CAUSES
from cogame_nmmo.replay import Replay
from cogame_nmmo.server import GameServer

# The closed results.json key set (triple-sync rule): must match
# GameServer._results_doc, the manifest results_schema, and the
# docker_smoke.sh assertions.
RESULT_KEYS = {
    "names", "scores", "reward_sums", "end_reason", "final_tick", "seed",
    "state_digest", "agent_stats", "noop_ticks", "dead_seats", "noop_causes",
}
AGENT_STAT_KEYS = {
    "cum_min_comb_prof", "deaths", "comb_lvl", "prof_lvl", "gold",
    "time_alive",
}


def make_config(num_seats=8, heroes_per_seat=1, **overrides):
    d = {
        "players": [{"name": f"bot-{i}"} for i in range(num_seats)],
        "tokens": [f"token-{i}" for i in range(num_seats)],
        "heroes_per_seat": heroes_per_seat,
        "seed": 21,
        "max_ticks": 20,
        "tick_deadline_ms": 1000,
        "player_connect_timeout_seconds": 10,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class ServerHarness:
    def __init__(self, cfg, tmp_path, **server_kwargs):
        self.results_path = tmp_path / "results.json"
        self.replay_path = tmp_path / "replay.bin"
        self.failure_path = tmp_path / "player_failure.json"
        self.server = GameServer(
            cfg,
            results_uri=f"file://{self.results_path}",
            save_replay_uri=f"file://{self.replay_path}",
            player_failure_uri=f"file://{self.failure_path}",
            **server_kwargs,
        )
        self.test_server = TestServer(self.server.make_app())
        self.episode_task = None

    async def __aenter__(self):
        await self.test_server.start_server()
        self.episode_task = asyncio.create_task(self.server.run_episode())
        return self

    async def __aexit__(self, *exc):
        if not self.episode_task.done():
            self.episode_task.cancel()
        try:
            await self.episode_task
        except asyncio.CancelledError:
            pass
        await self.test_server.close()

    def ws_url(self, slot, token):
        return str(self.test_server.make_url(
            f"/player?slot={slot}&token={token}"))


async def play_random_client(harness, slot, token, heroes,
                             seen_resets=None):
    """A well-behaved player: random in-range actions until done.

    ``seen_resets`` (optional list) collects each tick's (tick, resets)
    pair for reset-plumbing assertions.
    """
    rng = np.random.default_rng(slot)
    result = None
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(harness.ws_url(slot, token)) as ws:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                if data.get("done"):
                    result = data["result"]
                    break
                obs = [base64.b64decode(o) for o in data["obs"]]
                assert len(obs) == heroes
                assert all(len(o) == 1707 for o in obs)
                # protocol v2: every tick message carries per-agent resets
                resets = data["resets"]
                assert isinstance(resets, list) and len(resets) == heroes
                assert all(isinstance(r, bool) for r in resets)
                if seen_resets is not None:
                    seen_resets.append((data["tick"], resets))
                acts = rng.integers(
                    0, defaults.ACT_HIGH, size=(heroes, 1)).tolist()
                await ws.send_str(json.dumps(
                    {"tick": data["tick"], "actions": acts}))
    return result


# -- full episodes -----------------------------------------------------------

async def test_full_episode_8_seats(tmp_path):
    cfg = make_config(max_ticks=15)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}", 1)
                   for s in range(8)]
        done_msgs = await asyncio.gather(*clients)
        result = await h.episode_task

    # every client got the done message with the result
    assert all(m is not None for m in done_msgs)
    assert done_msgs[0]["final_tick"] == result.final_tick

    results = json.loads(h.results_path.read_text())
    assert set(results) == RESULT_KEYS
    assert results["names"] == [f"bot-{i}" for i in range(8)]
    assert results["final_tick"] == result.final_tick
    assert results["end_reason"] == "tick_cap"
    assert results["seed"] == 21
    # scores: raw per-seat score values (structure, not magnitude — with
    # random actions most agents just survive at min(comb,prof)=1).
    # Genuinely score-increasing play needs obs decoding, which arrives
    # with the Phase-N3 scripted player; its behavioral test covers
    # score growth.
    assert len(results["scores"]) == 8
    assert all(isinstance(s, float) and s >= 0 for s in results["scores"])
    assert sum(results["scores"]) >= 1  # someone is alive at comb=prof>=1
    assert len(results["agent_stats"]) == 8
    for stats in results["agent_stats"]:
        assert set(stats) == AGENT_STAT_KEYS
        assert stats["comb_lvl"] >= 1 and stats["prof_lvl"] >= 1
    assert isinstance(results["state_digest"], int)
    assert results["state_digest"] == result.state_digest

    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick
    assert replay.num_agents == 8
    assert replay.header["config"]["seed"] == 21
    assert [p["name"] for p in replay.header["config"]["players"]] == \
        results["names"]
    assert replay.header["result"]["scores"] == results["scores"]
    assert replay.header["result"]["state_digest"] == results["state_digest"]
    # no failures reported
    assert not h.failure_path.exists()


async def test_full_episode_multi_agent_seats(tmp_path):
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=12)
    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(
            play_random_client(h, 0, "token-0", 4),
            play_random_client(h, 1, "token-1", 4))
        result = await h.episode_task

    assert all(m is not None for m in done_msgs)
    results = json.loads(h.results_path.read_text())
    assert set(results) == RESULT_KEYS
    assert len(results["scores"]) == 2
    assert len(results["agent_stats"]) == 8  # per agent, not per seat
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick
    assert replay.num_agents == 8


# -- resets over the wire (protocol v2) ---------------------------------------

async def test_resets_forwarded_over_websocket(tmp_path):
    """An agent death on tick T's step arrives as resets=[true] with tick
    T+1's obs, on that agent's seat only. Driven by a scripted fake sim
    so the death tick is deterministic (real-sim death forwarding is
    covered in tests/test_engine.py)."""
    from tests.test_engine import FakeSim

    cfg = make_config(max_ticks=6, tick_deadline_ms=500)
    fake = FakeSim(num_agents=8, dones_at={2: [5], 4: [0]})
    async with ServerHarness(
            cfg, tmp_path,
            sim_factory=lambda seed, num_agents: fake) as h:
        seen = [[] for _ in range(8)]
        clients = [play_random_client(h, s, f"token-{s}", 1,
                                      seen_resets=seen[s])
                   for s in range(8)]
        await asyncio.gather(*clients)
        await h.episode_task

    for slot in range(8):
        for tick, resets in seen[slot]:
            expect = (slot == 5 and tick == 3) or (slot == 0 and tick == 5)
            assert resets == [expect], (slot, tick, resets)
    # the flagged ticks were actually observed (not an empty loop)
    assert any(t == 3 for t, _ in seen[5])
    assert any(t == 5 for t, _ in seen[0])


# -- degraded players --------------------------------------------------------

async def test_missing_player_noop_and_failure_report(tmp_path):
    cfg = make_config(max_ticks=6, tick_deadline_ms=200,
                      player_connect_timeout_seconds=0.4)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}", 1)
                   for s in range(7)]  # slot 7 never connects
        await asyncio.gather(*clients)
        result = await h.episode_task

    assert result.final_tick > 0
    failure = json.loads(h.failure_path.read_text())
    assert failure["failed_policy_index"] == 7
    assert "bot-7" in failure["message"]
    assert set(failure) == {"failed_policy_index", "message"}
    assert h.results_path.exists()
    assert h.replay_path.exists()


async def test_malformed_messages_never_crash_episode(tmp_path):
    cfg = make_config(max_ticks=6, tick_deadline_ms=150)

    async def malformed_client(h, slot, token):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                garbage = iter([
                    "not json at all",
                    json.dumps({"tick": -99, "actions": [[0]]}),
                    json.dumps({"nonsense": True}),
                    json.dumps({"tick": None, "actions": "x"}),
                ])
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    try:
                        await ws.send_str(next(garbage))
                    except StopIteration:
                        # then wrong-shaped actions on the right tick
                        await ws.send_str(json.dumps(
                            {"tick": data["tick"], "actions": [[1, 2]]}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        good = [play_random_client(h, s, f"token-{s}", 1) for s in range(7)]
        results = await asyncio.gather(*good, malformed_client(h, 7, "token-7"))
        result = await h.episode_task

    assert result.final_tick == 6
    # the malformed client stayed connected and still got the done message
    assert results[-1] is not None
    assert h.results_path.exists()


async def test_results_report_noop_causes(tmp_path):
    """results.json noop_causes attributes every degrade: a seat that
    keeps answering the wrong tick shows wrong_tick message counts and
    per-tick timeouts; clean seats show all zeros."""
    cfg = make_config(max_ticks=4, tick_deadline_ms=150)

    async def wrong_tick_client(h, slot, token):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    await ws.send_str(json.dumps({
                        "tick": data["tick"] + 1000,
                        "actions": [list(defaults.NOOP_ACTION)]}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        good = [play_random_client(h, s, f"token-{s}", 1) for s in range(7)]
        results_msgs = await asyncio.gather(
            *good, wrong_tick_client(h, 7, "token-7"))
        await h.episode_task

    assert results_msgs[-1] is not None
    results = json.loads(h.results_path.read_text())
    causes = results["noop_causes"]
    assert len(causes) == 8
    assert set(causes[0]) == set(NOOP_CAUSES)
    for seat in range(7):
        assert all(v == 0 for v in causes[seat].values()), (seat, causes)
    assert causes[7]["timeout"] == 4
    assert causes[7]["wrong_tick"] >= 1
    assert results["noop_ticks"][7] == 4


async def test_dead_seat_disconnect_during_probe_then_reconnect_revives(
        tmp_path):
    """A seat that goes strike-dead while connected has a revival probe
    parked on its websocket. If that socket then drops, the probe's
    waiter must be failed (fail_waiter) so the engine can re-probe —
    otherwise a reconnecting player can never revive the seat."""
    cfg = make_config(max_ticks=80, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2)

    async def paced_client(h, slot, token):
        """Well-behaved but slow (~25ms/tick): keeps the episode running
        long enough for the flaky seat's disconnect + reconnect."""
        rng = np.random.default_rng(slot)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    await asyncio.sleep(0.025)
                    acts = rng.integers(0, defaults.ACT_HIGH,
                                        size=(1, 1)).tolist()
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"], "actions": acts}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        gather = asyncio.gather(*(
            paced_client(h, s, f"token-{s}") for s in range(7)))

        async def flaky(h):
            # Phase 1: connect but never reply. The seat racks up strikes,
            # goes dead, and a revival probe parks on this socket (the obs
            # stream stops once the single outstanding probe is parked).
            async with aiohttp.ClientSession() as session:
                ws = await session.ws_connect(h.ws_url(7, "token-7"))
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), 0.5)
                    except (asyncio.TimeoutError, TimeoutError):
                        break  # probe parked: nothing more will arrive
                    if msg.type != WSMsgType.TEXT:
                        break
                await ws.close()  # drop with the probe still parked
            # Phase 2: reconnect and play properly; must revive the seat.
            return await play_random_client(h, 7, "token-7", 1)

        flaky_result = await asyncio.wait_for(flaky(h), 30)
        await gather
        result = await asyncio.wait_for(h.episode_task, 30)

    assert flaky_result is not None
    assert result.seat_dead[7] is False, \
        "reconnected seat never revived (stuck probe waiter?)"
    assert 0 < result.seat_noop_ticks[7] < 80


async def test_strike_death_force_closes_stale_socket_then_revive(tmp_path):
    """When a seat goes strike-dead the server force-closes its (possibly
    half-open) websocket: the client sees the close, reconnects, and the
    seat revives. Without the close a client whose socket went stale
    server-side would keep feeding a black hole forever."""
    cfg = make_config(max_ticks=80, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2)

    async def paced_client(h, slot, token):
        rng = np.random.default_rng(slot)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    await asyncio.sleep(0.025)
                    acts = rng.integers(0, defaults.ACT_HIGH,
                                        size=(1, 1)).tolist()
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"], "actions": acts}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        gather = asyncio.gather(*(
            paced_client(h, s, f"token-{s}") for s in range(7)))

        async def flaky(h):
            # Never reply; the server must close this socket when the
            # seat strikes out.
            async with aiohttp.ClientSession() as session:
                ws = await session.ws_connect(h.ws_url(7, "token-7"))
                while True:
                    msg = await ws.receive()  # no timeout: server closes
                    if msg.type != WSMsgType.TEXT:
                        break
            # Reconnect and play properly; must revive the seat.
            return await play_random_client(h, 7, "token-7", 1)

        flaky_result = await asyncio.wait_for(flaky(h), 30)
        await gather
        result = await asyncio.wait_for(h.episode_task, 30)

    assert flaky_result is not None
    assert result.seat_dead[7] is False
    assert 0 < result.seat_noop_ticks[7] < 80


async def test_wall_clock_budget_writes_artifacts(tmp_path):
    """Wall-clock expiry ends the episode normally: results.json says
    end_reason="wall_clock" and the partial replay is written."""
    cfg = make_config(max_ticks=5000, tick_deadline_ms=1000,
                      wall_clock_budget_seconds=0.5)

    async def paced_client(h, slot, token):
        rng = np.random.default_rng(slot)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    await asyncio.sleep(0.02)
                    acts = rng.integers(0, defaults.ACT_HIGH,
                                        size=(1, 1)).tolist()
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"], "actions": acts}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(*(
            paced_client(h, s, f"token-{s}") for s in range(8)))
        result = await asyncio.wait_for(h.episode_task, 30)

    assert all(m is not None for m in done_msgs)
    results = json.loads(h.results_path.read_text())
    assert results["end_reason"] == "wall_clock"
    assert 0 < results["final_tick"] < 5000
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick


# -- auth + connection management --------------------------------------------

async def test_bad_token_rejected(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await session.ws_connect(h.ws_url(3, "wrong-token"))
            assert exc.value.status == 403


@pytest.mark.parametrize("slot", ["17", "-1", "abc", ""])
async def test_bad_slot_rejected(tmp_path, slot):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await session.ws_connect(
                    str(h.test_server.make_url(
                        f"/player?slot={slot}&token=token-0")))
            assert exc.value.status == 403


async def test_duplicate_slot_rejected_while_alive(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            ws1 = await session.ws_connect(h.ws_url(0, "token-0"))
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await session.ws_connect(h.ws_url(0, "token-0"))
            await ws1.close()
            # dead connection may be replaced; the server's handler may
            # not have observed the close yet, so retry briefly
            for _ in range(40):
                try:
                    ws2 = await session.ws_connect(h.ws_url(0, "token-0"))
                    break
                except aiohttp.WSServerHandshakeError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("reconnect to a dead slot was never accepted")
            await ws2.close()


async def test_seat_lifecycle_logged(tmp_path, capsys):
    """One stderr line each for connect, disconnect, and 409-reject
    (strike-death and revival lines are covered by the engine tests)."""
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            ws1 = await session.ws_connect(h.ws_url(0, "token-0"))
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await session.ws_connect(h.ws_url(0, "token-0"))
            await ws1.close()
            await asyncio.sleep(0.05)  # let the handler's finally run
    err = capsys.readouterr().err
    assert "seat 0 (bot-0) connected at tick 0" in err
    assert "rejected duplicate connection (409)" in err
    assert "seat 0 (bot-0) disconnected at tick 0" in err


async def test_healthz(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(h.test_server.make_url("/healthz")) as resp:
                assert resp.status == 200
                assert (await resp.json())["status"] == "ok"


# -- platform browser/viewer contract (coworld GAME.md) ----------------------
# The local certifier probes GET /client/player?slot&token, GET
# /client/global, and requires the /global websocket to produce a first
# message (coworld.runner.runner.run_episode_containers).

async def test_client_global_page(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    h.test_server.make_url("/client/global")) as resp:
                assert resp.status == 200
                assert "text/html" in resp.headers["Content-Type"]
                assert "/global" in await resp.text()


async def test_client_player_page_token_checked(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(h.test_server.make_url(
                    "/client/player?slot=0&token=token-0")) as resp:
                assert resp.status == 200
                assert "text/html" in resp.headers["Content-Type"]
            async with session.get(h.test_server.make_url(
                    "/client/player?slot=0&token=wrong")) as resp:
                assert resp.status == 403
            async with session.get(h.test_server.make_url(
                    "/client/player?slot=99&token=token-0")) as resp:
                assert resp.status == 403


async def test_global_ws_first_message_and_done(tmp_path):
    """Viewer gets a snapshot immediately, then progress standings, then
    the final done message with the score-ranked result."""
    cfg = make_config(max_ticks=10)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            global_ws = await session.ws_connect(
                str(h.test_server.make_url("/global")))
            first = json.loads((await asyncio.wait_for(
                global_ws.receive(), 5)).data)
            assert first["type"] == "status"
            assert first["players"] == [f"bot-{i}" for i in range(8)]
            assert first["heroes_per_seat"] == 1
            assert first["done"] is False

            clients = [play_random_client(h, s, f"token-{s}", 1)
                       for s in range(8)]
            gather = asyncio.gather(*clients)
            done_msg = None
            progress_scores = []
            while True:
                msg = await asyncio.wait_for(global_ws.receive(), 30)
                if msg.type != WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                if data.get("done"):
                    done_msg = data
                    break
                if "scores" in data:
                    progress_scores.append(data["scores"])
            await gather
            assert done_msg is not None
            assert len(done_msg["result"]["scores"]) == 8
            # the tick-0 broadcast carries live standings (tick 0 % 50
            # == 0; later broadcasts may be dropped under send pressure)
            for scores in progress_scores:
                assert len(scores) == 8


async def test_global_ws_late_viewer_snapshot_is_self_contained(tmp_path):
    """A viewer connecting after the episode ended gets done + result in
    the connect snapshot (no later message to wait for)."""
    cfg = make_config(max_ticks=10)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}", 1)
                   for s in range(8)]
        await asyncio.gather(*clients)
        await h.episode_task  # episode fully finished
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(
                str(h.test_server.make_url("/global")))
            first = json.loads((await asyncio.wait_for(
                ws.receive(), 5)).data)
            assert first["type"] == "status"
            assert first["done"] is True
            assert len(first["result"]["scores"]) == 8
            await ws.close()


async def test_global_ws_sender_never_crashes_episode(tmp_path):
    """A viewer that sends garbage and disconnects mid-episode is harmless."""
    cfg = make_config(max_ticks=60)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            global_ws = await session.ws_connect(
                str(h.test_server.make_url("/global")))
            await asyncio.wait_for(global_ws.receive(), 5)  # snapshot
            await global_ws.send_str("not json at all")
            clients = [play_random_client(h, s, f"token-{s}", 1)
                       for s in range(8)]
            gather = asyncio.gather(*clients)
            await asyncio.sleep(0.1)
            await global_ws.close()  # disconnect while episode is running
            results = await gather
        assert results[0] is not None
        assert h.results_path.exists()


# -- uris --------------------------------------------------------------------

async def test_file_uri_round_trip(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.bin"
    await uris.write_uri(f"file://{target}", b"\x00\x01payload")
    assert await uris.read_uri(f"file://{target}") == b"\x00\x01payload"
    # plain paths (no scheme) also work, matching the runtime convention
    plain = tmp_path / "plain.txt"
    await uris.write_uri(str(plain), b"hello")
    assert await uris.read_uri(str(plain)) == b"hello"


async def test_coworld_mount_style_uri_path():
    # file:///coworld/out/results.json must resolve to /coworld/out/...
    assert uris.local_path("file:///coworld/out/results.json") is not None
    assert str(uris.local_path("file:///coworld/out/results.json")) == \
        "/coworld/out/results.json"


async def test_http_uri_read_write():
    from aiohttp import web

    stored = {}

    async def handle_get(request):
        return web.Response(body=stored.get("blob", b""))

    async def handle_put(request):
        stored["blob"] = await request.read()
        stored["content_type"] = request.content_type
        return web.Response(status=201)

    app = web.Application()
    app.router.add_get("/artifact", handle_get)
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/artifact"))
        await uris.write_uri(url, b"http-bytes", "application/json")
        assert stored["blob"] == b"http-bytes"
        assert stored["content_type"] == "application/json"
        assert await uris.read_uri(url) == b"http-bytes"
    finally:
        await server.close()


async def test_unsupported_scheme_rejected():
    with pytest.raises(ValueError):
        await uris.read_uri("s3://bucket/key")
    with pytest.raises(ValueError):
        await uris.write_uri("ftp://host/file", b"x")


# -- replay mode --------------------------------------------------------------

def _write_replay_bytes():
    from cogame_nmmo.replay import ReplayWriter

    cfg = make_config()
    writer = ReplayWriter(cfg, "aa" * 32)
    rng = np.random.default_rng(3)
    for t in range(8):
        writer.append_tick(
            t, rng.integers(0, defaults.ACT_HIGH,
                            size=(8, 1)).astype(np.uint8))
    return writer.finalize({"scores": [2.0] * 8, "end_reason": "tick_cap",
                            "final_tick": 8})


async def test_replay_mode_serves_bytes_and_viewer():
    from cogame_nmmo.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/replay-data")) as resp:
                assert resp.status == 200
                assert resp.content_type == "application/octet-stream"
                assert await resp.read() == data
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "/replay-data" in html
                # score standings, not moba winner fields
                assert "NMMO" in html
                assert "standings" in html
                assert "result.winner" not in html
            async with session.get(server.make_url("/healthz")) as resp:
                assert resp.status == 200
    finally:
        await server.close()


async def test_replay_mode_legacy_replay_ws_first_message():
    """The certifier's replay-loadable probe (coworld<=0.1.34 runs it
    even with a static bundle declared) needs one non-empty message
    from the /replay websocket."""
    from cogame_nmmo.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(str(server.make_url("/replay")))
            msg = json.loads((await asyncio.wait_for(ws.receive(), 5)).data)
            assert msg["type"] == "replay_header"
            assert msg["header"]["tick_count"] == 8
            assert msg["header"]["result"]["scores"] == [2.0] * 8
            await ws.close()
    finally:
        await server.close()


async def test_replay_mode_serves_viewer_bundle_when_built(tmp_path):
    """With a viewer/dist bundle present, /client/replay serves the real
    viewer index and its static assets (Phase N4 builds the bundle)."""
    from cogame_nmmo.server import make_replay_app

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!DOCTYPE html><title>viewer</title>fetches /replay-data")
    (dist / "nmmo_viewer.js").write_text("// glue")
    (dist / "nmmo_viewer.wasm").write_bytes(b"\x00asm fake")

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data, viewer_dist=dist))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                # pins the relative-asset regression: the slashless URL
                # must 302 to /client/replay/ so the viewer js resolves
                # under /client/replay/, not /client/
                assert [r.status for r in resp.history] == [302]
                assert resp.url.path == "/client/replay/"
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "viewer" in html
                assert "Placeholder" not in html
            async with session.get(
                    server.make_url("/client/replay/nmmo_viewer.js")) as resp:
                assert resp.status == 200
                assert await resp.text() == "// glue"
            async with session.get(
                    server.make_url("/client/replay/nmmo_viewer.wasm")) as resp:
                assert resp.status == 200
                assert await resp.read() == b"\x00asm fake"
            # bundle mode keeps /replay-data intact
            async with session.get(server.make_url("/replay-data")) as resp:
                assert resp.status == 200
                assert await resp.read() == data
    finally:
        await server.close()


async def test_replay_mode_falls_back_to_placeholder_without_bundle(tmp_path):
    """No viewer/dist (emscripten build not run): the placeholder page
    keeps server tests and dev flows working."""
    from cogame_nmmo.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(
        make_replay_app(data, viewer_dist=tmp_path / "no-such-dist"))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "Placeholder" in html
                assert "/replay-data" in html
    finally:
        await server.close()


async def test_replay_mode_rejects_corrupt_replay():
    from cogame_nmmo.replay import ReplayError
    from cogame_nmmo.server import make_replay_app

    with pytest.raises(ReplayError):
        make_replay_app(b"not a replay")


# -- sim fault containment (patch 0002) --------------------------------------

async def test_sim_fault_writes_results_and_partial_replay(tmp_path):
    """A sim fault (patch-0002 flag) ends the episode with results.json
    (end_reason sim_fault, closed key set intact) and a parseable
    partial replay — instead of the pre-patch exit() losing both."""
    from tests.test_engine import FaultingSim

    cfg = make_config(max_ticks=50, tick_deadline_ms=50,
                      player_connect_timeout_seconds=0.1)
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri=f"file://{results_path}",
        save_replay_uri=f"file://{replay_path}",
        sim_factory=lambda seed, num_agents: FaultingSim(
            fault_at=2, num_agents=num_agents),
    )
    result = await asyncio.wait_for(server.run_episode(), 30)
    assert result.end_reason == "sim_fault"

    results = json.loads(results_path.read_text())
    assert results["end_reason"] == "sim_fault"
    # equal scores: an infra fault is nobody's loss
    assert results["scores"] == [0.0] * 8
    # same closed key set as a normal episode
    assert set(results) == RESULT_KEYS
    assert set(results) == set(server._results_doc(result))
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 2


async def test_engine_exception_still_writes_fault_artifacts(tmp_path):
    """Even an unexpected host failure (here: the sim factory raising)
    writes fault results + the (empty) replay before re-raising."""

    def exploding_factory(seed, num_agents):
        raise RuntimeError("host exploded")

    cfg = make_config(max_ticks=10, player_connect_timeout_seconds=0.1)
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri=f"file://{results_path}",
        save_replay_uri=f"file://{replay_path}",
        sim_factory=exploding_factory,
    )
    with pytest.raises(RuntimeError, match="host exploded"):
        await asyncio.wait_for(server.run_episode(), 30)

    results = json.loads(results_path.read_text())
    assert results["end_reason"] == "sim_fault"
    assert results["final_tick"] == 0
    assert set(results) == RESULT_KEYS
    assert results["names"] == [f"bot-{i}" for i in range(8)]
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 0


# -- shutdown robustness ------------------------------------------------------

async def test_unresponsive_client_never_blocks_episode_exit(tmp_path):
    """A connected client that never reads or replies must not prevent
    run_episode from returning (bounded done-broadcast, strike rule)."""
    cfg = make_config(max_ticks=5, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            silent_ws = await session.ws_connect(h.ws_url(7, "token-7"))
            good = [play_random_client(h, s, f"token-{s}", 1)
                    for s in range(7)]
            await asyncio.gather(*good)
            result = await asyncio.wait_for(h.episode_task, timeout=20)
            await silent_ws.close()
    assert result.final_tick == 5
    results = json.loads(h.results_path.read_text())
    assert results["noop_ticks"][7] == 5
    assert results["noop_ticks"][:7] == [0] * 7


async def test_failing_results_uri_does_not_block_replay_write(tmp_path):
    """Artifact writes are independent: a failing results URI must not
    prevent the replay write; the aggregate error is raised after."""
    cfg = make_config(max_ticks=3, tick_deadline_ms=50,
                      player_connect_timeout_seconds=0.1)
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri="badscheme://results",
        save_replay_uri=f"file://{replay_path}",
        player_failure_uri=f"file://{tmp_path / 'failure.json'}",
    )
    with pytest.raises(IOError):
        await server.run_episode()
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 3


async def test_http_write_retries_then_succeeds():
    from aiohttp import web

    attempts = 0
    stored = {}

    async def handle_put(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return web.Response(status=500)
        stored["blob"] = await request.read()
        return web.Response(status=200)

    app = web.Application()
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/artifact"))
        await uris.write_uri(url, b"retried", "application/json",
                             backoff_seconds=0.01)
        assert attempts == 3
        assert stored["blob"] == b"retried"
    finally:
        await server.close()


async def test_http_write_raises_after_exhausted_retries():
    from aiohttp import web

    attempts = 0

    async def handle_put(request):
        nonlocal attempts
        attempts += 1
        return web.Response(status=503)

    app = web.Application()
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(IOError):
            await uris.write_uri(str(server.make_url("/artifact")),
                                 b"x", backoff_seconds=0.01)
        assert attempts == 3
    finally:
        await server.close()
