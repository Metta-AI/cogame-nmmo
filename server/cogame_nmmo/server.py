"""aiohttp game server implementing the Coworld runtime contract.

Episode mode (default): read the game config from ``COGAME_CONFIG_URI``,
serve ``GET /player?slot=N&token=T`` websockets, run one lockstep episode
(starting when all seats are connected or the connect timeout elapses),
then write ``results.json`` to ``COGAME_RESULTS_URI`` and the replay
bytes to ``COGAME_SAVE_REPLAY_URI`` and exit 0. Seats that never connect
are declared to ``COGAME_PLAYER_FAILURE_URI`` and played as NOOP.

Wire protocol v2, one JSON text message per tick each way:

    server -> player  {"tick": t, "obs": ["<base64 1707B>" x heroes],
                       "resets": [bool x heroes]}
    player -> server  {"tick": t, "actions": [[1 int] x heroes]}
    server -> player  {"done": true, "result": {...}}   (episode end)

``resets[i]`` is True when that agent's terminal fired (death or
stagnation respawn) on a sim step this seat has not yet acknowledged
with a valid reply: recurrent policies must zero that agent's state
before consuming this tick's obs. Delivery is at-least-once (the
engine's pending mask, see engine.py and docs/PROTOCOL.md); for a seat
answering every tick it is exactly the previous step's done flags.

Wrong-tick or malformed replies are treated as missing (NOOP for that
tick); the connection stays up and the episode never crashes.

Global viewer (platform contract, coworld GAME.md): ``GET /global`` is
a broadcast-only websocket — an initial status snapshot on connect,
throttled ``{"tick": t}`` progress while the episode runs, and the
final ``{"done": true, "result": ...}``. Browser pages are served at
``GET /client/global`` (live viewer) and ``GET /client/player?slot=N&
token=T`` (token-checked seat page; observing only, play is via the
websocket protocol above).

Replay mode: when ``COGAME_LOAD_REPLAY_URI`` is set, no episode runs;
the recorded replay is served at ``GET /replay-data`` (raw bytes) and
``GET /client/replay`` (viewer page) and the process stays up.

Entry point: ``python -m cogame_nmmo.server``. Binds ``COGAME_HOST``/
``COGAME_PORT`` (default 0.0.0.0:8080).
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import sys
from pathlib import Path

import numpy as np
from aiohttp import WSCloseCode, WSMsgType, web

from . import defaults, uris
from .config import GameConfig
from .engine import (NOOP_CAUSES, STAT_CODES, EpisodeResult,
                     LockstepEngine)
from .replay import Replay, ReplayWriter, sim_wasm_sha256
from .sim import DEFAULT_WASM_PATH, NmmoSim

# After artifacts are written, keep serving briefly so clients can finish
# reading the done message and close their websockets.
SHUTDOWN_GRACE_SECONDS = 1.0

# Per-seat bound on sending the final done message + close: a connected
# client that stopped reading must never stall process exit.
DONE_SEND_TIMEOUT_SECONDS = 3.0

# Global-viewer tick messages are throttled to every N sim ticks: enough
# for a live progress feed, negligible episode overhead.
GLOBAL_TICK_EVERY = 50

# aiohttp ping/pong heartbeat on /player sockets: a half-open connection
# (network blip, silently dropped peer) is reaped within ~this interval
# instead of lingering as a "connected" seat that 409s real reconnects.
PLAYER_WS_HEARTBEAT_SECONDS = 20.0

# Minimal live global-viewer page: connects to /global and renders the
# JSON feed. Static HTML, built with textContent (names are player data).
GLOBAL_CLIENT_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>cogame-nmmo</title>
<style>
  body { font-family: ui-monospace, monospace; margin: 2rem; }
  #log { white-space: pre-wrap; }
</style>
</head>
<body>
<h1>cogame-nmmo live viewer</h1>
<div id="log">connecting to /global ...</div>
<script>
const log = document.getElementById("log");
const proto = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(proto + "//" + location.host + "/global");
ws.onmessage = (ev) => { log.textContent += "\\n" + ev.data; };
ws.onopen = () => { log.textContent = "connected"; };
ws.onclose = () => { log.textContent += "\\n[closed]"; };
</script>
</body>
</html>
"""

PLAYER_CLIENT_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>cogame-nmmo seat</title></head>
<body style="font-family: ui-monospace, monospace; margin: 2rem;">
<h1>cogame-nmmo</h1>
<p>Seat <span id="slot"></span> is played over the websocket protocol
(<code>GET /player?slot=N&amp;token=T</code>, see docs/PROTOCOL.md);
this page only confirms the seat credential is valid.</p>
<script>
document.getElementById("slot").textContent =
  new URLSearchParams(location.search).get("slot");
</script>
</body>
</html>
"""


class WsSeat:
    """One player seat: websocket connection state + engine ActionSource.

    ``get_actions`` sends the tick's obs to the connected player and waits
    for a matching-tick reply; the engine's deadline (asyncio.wait_for)
    cancels the wait when the player is late. No connection -> None
    immediately (the engine plays NOOP without burning the deadline).
    """

    def __init__(self, slot: int, name: str):
        self.slot = slot
        self.name = name
        self.ws: web.WebSocketResponse | None = None
        self.ever_connected = False
        self._waiter: tuple[int, asyncio.Future] | None = None
        # Messages answering a different tick than the pending one; read
        # by the engine into results noop_causes["wrong_tick"].
        self.wrong_tick_count = 0

    @property
    def connected(self) -> bool:
        return self.ws is not None and not self.ws.closed

    async def get_actions(self, tick: int, obs: np.ndarray, resets):
        ws = self.ws
        if ws is None or ws.closed:
            return None
        payload = json.dumps({
            "tick": tick,
            "obs": [base64.b64encode(row.tobytes()).decode("ascii")
                    for row in obs],
            # protocol v2: per-agent not-yet-acknowledged done flags
            # (recurrent policies zero that agent's state; the engine
            # owns the pending mask and its at-least-once semantics)
            "resets": [bool(r) for r in resets],
        })
        fut = asyncio.get_running_loop().create_future()
        self._waiter = (tick, fut)
        try:
            await ws.send_str(payload)
            return await fut
        except Exception:
            return None
        finally:
            self._waiter = None

    def fail_waiter(self) -> None:
        """Resolve any pending tick waiter with None (connection died).

        Called when this seat's websocket goes away. Without it an
        un-deadlined revival probe parked in ``get_actions`` waits on the
        dead socket forever, and the engine (which keeps at most one
        probe outstanding) can never re-probe — a reconnecting player
        could then never revive the seat.
        """
        waiter = self._waiter
        if waiter is None:
            return
        _tick, fut = waiter
        if not fut.done():
            fut.set_result(None)

    def deliver(self, data) -> None:
        """Route one decoded client message to the pending tick waiter.

        Wrong-tick or shapeless messages are dropped (the engine NOOPs on
        deadline); action-payload validation happens in the engine.
        """
        if not isinstance(data, dict):
            return
        waiter = self._waiter
        if waiter is None:
            return
        tick, fut = waiter
        if data.get("tick") != tick:
            self.wrong_tick_count += 1
            if self.wrong_tick_count == 1:
                print(f"seat {self.slot} ({self.name}): first wrong-tick "
                      f"reply (got {data.get('tick')!r}, pending {tick})",
                      file=sys.stderr)
            return
        if not fut.done():
            fut.set_result(data.get("actions"))


class GameServer:
    def __init__(self, config: GameConfig, *,
                 results_uri: str | None = None,
                 save_replay_uri: str | None = None,
                 player_failure_uri: str | None = None,
                 sim_factory=NmmoSim,
                 wasm_path=DEFAULT_WASM_PATH):
        self.config = config
        self.results_uri = results_uri
        self.save_replay_uri = save_replay_uri
        self.player_failure_uri = player_failure_uri
        self.sim_factory = sim_factory
        self.wasm_path = wasm_path
        self.seats = [
            WsSeat(slot, player.name)
            for slot, player in enumerate(config.players)]
        self._all_connected = asyncio.Event()
        self.result: EpisodeResult | None = None
        # The final results.json payload, kept for late global viewers'
        # connect snapshots (None until the episode ends).
        self.results_doc: dict | None = None
        # Broadcast-only global viewer sockets (see GET /global) and, per
        # socket, the in-flight broadcast send task. The dict both keeps
        # a strong reference to every pending task (create_task results
        # are otherwise collectable mid-flight) and serializes sends per
        # socket: a new broadcast is dropped for a socket whose previous
        # send has not finished, so send_str calls never interleave.
        self._global_wss: set[web.WebSocketResponse] = set()
        self._global_send_tasks: dict[
            web.WebSocketResponse, asyncio.Task] = {}
        # Strong refs to in-flight stale-socket close tasks (create_task
        # results are collectable mid-flight otherwise).
        self._stale_close_tasks: set[asyncio.Task] = set()
        # Last engine tick observed via on_tick, for lifecycle log lines
        # (connect/disconnect/409 happen outside the engine loop).
        self._last_tick = 0

    # -- routes --------------------------------------------------------------

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/healthz", self._handle_healthz)
        app.router.add_get("/player", self._handle_player)
        app.router.add_get("/global", self._handle_global)
        app.router.add_get("/client/global", self._handle_global_client)
        app.router.add_get("/client/player", self._handle_player_client)
        return app

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    def _authorized_slot(self, request: web.Request) -> int:
        """Validate ?slot=N&token=T against the config; 403 on any mismatch."""
        try:
            slot = int(request.query.get("slot", ""))
        except ValueError:
            raise web.HTTPForbidden(text="bad slot")
        if not 0 <= slot < len(self.seats):
            raise web.HTTPForbidden(text="bad slot")
        token = request.query.get("token", "")
        # encode before comparing: compare_digest raises on non-ASCII str
        if not hmac.compare_digest(
                token.encode("utf-8"),
                self.config.tokens[slot].encode("utf-8")):
            raise web.HTTPForbidden(text="bad token")
        return slot

    async def _handle_global_client(
            self, request: web.Request) -> web.Response:
        return web.Response(
            text=GLOBAL_CLIENT_HTML, content_type="text/html")

    async def _handle_player_client(
            self, request: web.Request) -> web.Response:
        self._authorized_slot(request)
        return web.Response(
            text=PLAYER_CLIENT_HTML, content_type="text/html")

    async def _handle_global(self, request: web.Request):
        """Broadcast-only global viewer feed (platform contract).

        Sends a status snapshot immediately on connect (the platform
        runner requires a first message), then throttled tick progress
        and the final done message while the socket stays open. Client
        messages are ignored.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        snapshot = {
            "type": "status",
            "players": [s.name for s in self.seats],
            "heroes_per_seat": self.config.heroes_per_seat,
            "max_ticks": self.config.max_ticks,
            "done": self.results_doc is not None,
        }
        if self.results_doc is not None:
            # Late viewer: make the snapshot self-contained.
            snapshot["result"] = self.results_doc
        await ws.send_str(json.dumps(snapshot))
        self._global_wss.add(ws)
        try:
            async for _msg in ws:
                pass  # broadcast-only: ignore anything the viewer sends
        finally:
            self._global_wss.discard(ws)
            self._global_send_tasks.pop(ws, None)
        return ws

    def _broadcast_global(self, payload: dict) -> None:
        """Fire-and-forget send to every connected global viewer.

        Per-socket serialization: if a socket's previous broadcast send
        is still in flight, this update is dropped for that socket (it is
        a throttled progress feed — losing an update beats interleaving
        concurrent send_str calls on one websocket).
        """
        if not self._global_wss:
            return
        message = json.dumps(payload)
        loop = asyncio.get_running_loop()
        for ws in tuple(self._global_wss):
            if ws.closed:
                continue
            prev = self._global_send_tasks.get(ws)
            if prev is not None and not prev.done():
                continue
            task = loop.create_task(self._global_send(ws, message))
            self._global_send_tasks[ws] = task
            task.add_done_callback(
                lambda t, ws=ws: self._discard_global_send(ws, t))

    def _discard_global_send(self, ws: web.WebSocketResponse,
                             task: asyncio.Task) -> None:
        if self._global_send_tasks.get(ws) is task:
            del self._global_send_tasks[ws]

    @staticmethod
    async def _global_send(ws: web.WebSocketResponse, message: str) -> None:
        try:
            await ws.send_str(message)
        except Exception:
            pass  # viewer sockets can never affect the episode

    async def _handle_player(self, request: web.Request):
        slot = self._authorized_slot(request)
        seat = self.seats[slot]
        if seat.connected:
            # one connection per slot; replace only a dead connection
            print(f"seat {slot} ({seat.name}): rejected duplicate "
                  f"connection (409) at tick {self._last_tick}",
                  file=sys.stderr)
            raise web.HTTPConflict(text="slot already connected")

        ws = web.WebSocketResponse(heartbeat=PLAYER_WS_HEARTBEAT_SECONDS)
        await ws.prepare(request)
        if seat.connected:
            # TOCTOU: a concurrent connect for this slot won during prepare
            await ws.close(
                code=WSCloseCode.POLICY_VIOLATION,
                message=b"slot already connected")
            return ws
        seat.ws = ws
        seat.ever_connected = True
        print(f"seat {slot} ({seat.name}) connected at tick "
              f"{self._last_tick}", file=sys.stderr)
        if all(s.connected for s in self.seats):
            self._all_connected.set()

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue  # malformed: never crash the episode
                seat.deliver(data)
        finally:
            if seat.ws is ws:
                seat.ws = None
                seat.fail_waiter()
                print(f"seat {slot} ({seat.name}) disconnected at tick "
                      f"{self._last_tick}", file=sys.stderr)
        return ws

    # -- episode orchestration -----------------------------------------------

    async def run_episode(self) -> EpisodeResult:
        cfg = self.config
        try:
            await asyncio.wait_for(
                self._all_connected.wait(),
                cfg.player_connect_timeout_seconds)
        except (asyncio.TimeoutError, TimeoutError):
            pass
        for seat in self.seats:
            if not seat.ever_connected:
                await self._report_player_failure(seat)

        writer = ReplayWriter(cfg, self._wasm_sha256())
        sim = None

        def live_scores() -> list[float] | None:
            """Current per-seat score standings for the /global feed.

            Same metric as the final results: per-agent mean score per
            life, sim.score / (deaths + 1), summed over the seat's
            agents (see engine._seat_score). A death mid-episode can
            LOWER a seat's standing — expected mean behavior, not a
            bug. Reads the sim directly (cheap wasm calls);
            best-effort — a faulting sim must never break the
            broadcast path.
            """
            if sim is None:
                return None
            deaths_code = STAT_CODES["deaths"]
            try:
                return [
                    float(sum(
                        sim.score(pid)
                        / (sim.agent_stat(pid, deaths_code) + 1)
                        for pid in defaults.seat_hero_pids(
                            seat, cfg.heroes_per_seat)))
                    for seat in range(cfg.num_seats)]
            except Exception:
                return None

        def on_tick(tick, actions):
            self._last_tick = tick
            writer.append_tick(tick, actions)
            if tick % GLOBAL_TICK_EVERY == 0:
                update = {"tick": tick}
                scores = live_scores()
                if scores is not None:
                    update["scores"] = scores
                self._broadcast_global(update)

        try:
            sim = self.sim_factory(seed=cfg.seed, num_agents=cfg.num_agents)
            engine = LockstepEngine(sim, cfg, self.seats, on_tick=on_tick,
                                    on_seat_dead=self._on_seat_dead)
            result = await engine.run()
        except Exception as exc:
            # The engine contains sim faults itself (end_reason
            # "sim_fault"); reaching here is an unexpected host failure.
            # Artifacts are the episode's whole point: best-effort write
            # fault results + the partial replay before re-raising.
            print(f"unexpected engine failure: {type(exc).__name__}: {exc}; "
                  f"writing fault artifacts", file=sys.stderr)
            await self._write_fault_artifacts(writer)
            raise
        self.result = result
        self._log_seat_degrades(result)

        results_doc = self._results_doc(result)
        self.results_doc = results_doc  # late /global viewers snapshot this
        write_errors: list[str] = []

        async def attempt(label: str, uri: str | None,
                          data: bytes, content_type: str) -> None:
            if not uri:
                return
            try:
                await uris.write_uri(uri, data, content_type)
            except Exception as exc:
                write_errors.append(f"{label} -> {uri}: {exc}")

        try:
            # independent writes: attempt all, aggregate errors after
            await attempt(
                "results", self.results_uri,
                (json.dumps(results_doc, indent=2) + "\n").encode("utf-8"),
                "application/json")
            await attempt(
                "replay", self.save_replay_uri,
                writer.finalize(results_doc), "application/octet-stream")
        finally:
            # players get the done message even if artifact writes fail
            await self._broadcast_done(results_doc)

        if write_errors:
            raise IOError(
                "artifact writes failed: " + "; ".join(write_errors))
        return result

    def _on_seat_dead(self, slot: int) -> None:
        """Strike-death hook: force-close the seat's stale websocket.

        Ten consecutive missed ticks with the socket still "connected"
        means the connection is effectively dead (half-open, or the peer
        stopped reading). Closing it makes the client see the drop and
        reconnect (PROTOCOL.md: a seat may reconnect and revive); leaving
        it open would let the stale socket 409 the reconnect instead.
        """
        seat = self.seats[slot]
        print(f"seat {slot} ({seat.name}) marked dead (strike rule) at "
              f"tick {self._last_tick}", file=sys.stderr)
        ws = seat.ws
        if ws is None or ws.closed:
            return

        async def close_stale() -> None:
            try:
                await ws.close(
                    code=WSCloseCode.GOING_AWAY,
                    message=b"seat marked dead (strike rule); "
                            b"reconnect to resume")
            except Exception:
                pass  # the handler's finally still clears seat.ws

        task = asyncio.get_running_loop().create_task(close_stale())
        self._stale_close_tasks.add(task)
        task.add_done_callback(self._stale_close_tasks.discard)

    def _log_seat_degrades(self, result: EpisodeResult) -> None:
        for seat, noops in enumerate(result.seat_noop_ticks):
            if noops:
                dead = " (dead at episode end)" if result.seat_dead[seat] \
                    else ""
                print(
                    f"seat {seat} ({self.seats[seat].name}): "
                    f"{noops}/{result.final_tick} NOOP ticks{dead}",
                    file=sys.stderr)

    def _wasm_sha256(self) -> str:
        try:
            return sim_wasm_sha256(self.wasm_path)
        except OSError:
            return "unknown"  # non-wasm sim_factory (tests with fakes)

    def _results_doc(self, result: EpisodeResult) -> dict:
        """results.json payload.

        Columnar arrays in config player/seat order (names, scores)
        follow the coworld-ctf convention; `scores` is the field
        cross-game Coworld consumers require (final score per seat,
        higher = better — NOT win/loss 1/0; episodes have no winner).
        The score is the seat's summed per-agent MEAN score per life
        (see engine._seat_score: sim.score / (deaths + 1)) — raw
        cumulative values are suicide-farmable; the raw ingredients
        stay available in `agent_stats` (cum_min_comb_prof, deaths).
        `agent_stats` is the score breakdown per agent pid (seat i owns
        pids [i*h, (i+1)*h); with the default heroes_per_seat=1 it IS the
        per-seat breakdown). Closed key set: triple-synced with the
        manifest results_schema and tools/ci/docker_smoke.sh.
        """
        cfg = self.config
        return {
            "names": [p.name for p in cfg.players],
            "scores": list(result.seat_scores),
            "reward_sums": list(result.seat_reward_sums),
            "end_reason": result.end_reason,
            "final_tick": result.final_tick,
            "seed": cfg.seed,
            # sim state digest at episode end: must match a re-sim of the
            # replay (see replay.py) — cheap corruption tripwire
            "state_digest": result.state_digest,
            "agent_stats": [dict(stats) for stats in result.agent_stats],
            # strike-rule observability: per-seat NOOP-fallback tick counts
            # and whether the seat was dead (see engine strike rule)
            "noop_ticks": list(result.seat_noop_ticks),
            "dead_seats": list(result.seat_dead),
            # per-cause degrade counters, keyed by engine.NOOP_CAUSES
            "noop_causes": [dict(c) for c in result.seat_noop_causes],
        }

    def _fault_results_doc(self, final_tick: int) -> dict:
        """A schema-complete results doc for an episode the engine lost.

        Same CLOSED key set as _results_doc (manifest results_schema):
        end_reason "sim_fault", all scores 0.0 (equal = drawn), zeroed
        stats.
        """
        cfg = self.config
        zero_stats = {name: 0 for name in STAT_CODES}
        return {
            "names": [p.name for p in cfg.players],
            "scores": [0.0] * cfg.num_seats,
            "reward_sums": [0.0] * cfg.num_seats,
            "end_reason": "sim_fault",
            "final_tick": final_tick,
            "seed": cfg.seed,
            "state_digest": 0,
            "agent_stats": [dict(zero_stats)
                            for _ in range(cfg.num_agents)],
            "noop_ticks": [0] * cfg.num_seats,
            "dead_seats": [False] * cfg.num_seats,
            "noop_causes": [dict.fromkeys(NOOP_CAUSES, 0)
                            for _ in range(cfg.num_seats)],
        }

    async def _write_fault_artifacts(self, writer: ReplayWriter) -> None:
        """Best-effort fault results + partial replay (never raises)."""
        results_doc = self._fault_results_doc(writer.tick_count)
        self.results_doc = results_doc
        for label, uri, data, ctype in (
                ("results", self.results_uri,
                 (json.dumps(results_doc, indent=2) + "\n").encode("utf-8"),
                 "application/json"),
                ("replay", self.save_replay_uri,
                 writer.finalize(results_doc), "application/octet-stream")):
            if not uri:
                continue
            try:
                await uris.write_uri(uri, data, ctype)
            except Exception as exc:
                print(f"fault-artifact write failed: {label} -> {uri}: "
                      f"{exc}", file=sys.stderr)
        try:
            await self._broadcast_done(results_doc)
        except Exception as exc:
            print(f"fault done-broadcast failed: {exc}", file=sys.stderr)

    async def _report_player_failure(self, seat: WsSeat) -> None:
        """Declare a no-show seat to COGAME_PLAYER_FAILURE_URI.

        Payload shape is exactly what the platform runner parses
        (coworld.runner.io.GamePlayerFailure, extra="forbid"): message +
        failed_policy_index only; the seat name and reason ride in the
        message text.
        """
        print(f"seat {seat.slot} ({seat.name}) never connected; playing NOOP",
              file=sys.stderr)
        if not self.player_failure_uri:
            return
        payload = {
            "message": (
                f"player '{seat.name}' in slot {seat.slot} did not connect "
                f"within {self.config.player_connect_timeout_seconds:g}s "
                f"(reason: connect_timeout); seat plays NOOP unless it "
                f"connects later"),
            "failed_policy_index": seat.slot,
        }
        try:
            await uris.write_uri(
                self.player_failure_uri,
                json.dumps(payload).encode("utf-8"),
                "application/json")
        except Exception as exc:  # best-effort: never blocks the episode
            print(f"player-failure report failed: {exc}", file=sys.stderr)

    async def _broadcast_done(self, results_doc: dict) -> None:
        message = json.dumps({"done": True, "result": results_doc})

        async def send_and_close(seat: WsSeat) -> None:
            ws = seat.ws
            if ws is None or ws.closed:
                return
            try:
                # bounded per seat: a client that stopped reading must
                # never stall process exit
                await asyncio.wait_for(
                    _send(ws), DONE_SEND_TIMEOUT_SECONDS)
            except Exception:
                pass

        async def _send(ws: web.WebSocketResponse) -> None:
            await ws.send_str(message)
            await ws.close()

        async def send_and_close_global(ws: web.WebSocketResponse) -> None:
            # Serialize with any in-flight tick broadcast for this socket
            # before sending done (send_str calls must never interleave).
            prev = self._global_send_tasks.get(ws)
            if prev is not None and not prev.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(prev), DONE_SEND_TIMEOUT_SECONDS)
                except Exception:
                    pass
            try:
                await asyncio.wait_for(
                    _send(ws), DONE_SEND_TIMEOUT_SECONDS)
            except Exception:
                pass

        # gathered independently: one stuck client cannot stall the others
        await asyncio.gather(
            *(send_and_close(s) for s in self.seats),
            *(send_and_close_global(ws) for ws in tuple(self._global_wss)
              if not ws.closed),
            return_exceptions=True)


# -- replay mode -------------------------------------------------------------

# The wasm re-sim viewer bundle built by sim/build_viewer.sh. When absent
# (emscripten build not run — e.g. server-only test environments), replay
# mode falls back to the placeholder page below.
DEFAULT_VIEWER_DIST = \
    Path(__file__).resolve().parents[2] / "viewer" / "dist"

# Placeholder viewer page served when the wasm re-sim viewer bundle is
# absent (emscripten build not run): fetches /replay-data, parses the
# binary header client-side, and renders the header info.
REPLAY_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>cogame-nmmo replay</title>
<style>
  body { font-family: ui-monospace, monospace; margin: 2rem; }
  dt { font-weight: bold; margin-top: .6rem; }
  .note { margin-top: 2rem; color: #666; }
</style>
</head>
<body>
<h1>cogame-nmmo replay</h1>
<dl id="info">loading /replay-data ...</dl>
<p class="note">Placeholder viewer: the full wasm re-simulation viewer
was not bundled (run sim/build_viewer.sh).</p>
<script>
async function load() {
  const resp = await fetch("/replay-data");
  const buf = new Uint8Array(await resp.arrayBuffer());
  const magic = String.fromCharCode(...buf.slice(0, 4));
  if (magic !== "NMMO") throw new Error("bad magic: " + magic);
  const version = buf[4];
  if (version !== 1) throw new Error("unsupported version: " + version);
  const headerLen = new DataView(buf.buffer, 5, 4).getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(buf.slice(9, 9 + headerLen)));
  const info = document.getElementById("info");
  info.textContent = "";  // build with textContent: names are player data
  const add = (label, value) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    info.appendChild(dt);
    info.appendChild(dd);
  };
  const names = header.config.players.map(p => p.name);
  const scores = header.result.scores || [];
  // score standings, best first (no winner: seats rank by score)
  const standings = names
    .map((name, i) => ({name, score: scores[i]}))
    .sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
    .map(e => e.score === undefined ? e.name : `${e.name}: ${e.score}`);
  add("standings", standings.join(", "));
  add("seed", header.config.seed);
  add("ticks", header.tick_count);
}
load().catch(e => {
  document.getElementById("info").textContent = "failed: " + e.message;
});
</script>
</body>
</html>
"""


def make_replay_app(replay_bytes: bytes,
                    viewer_dist: Path | None = None) -> web.Application:
    """Replay-mode app: raw bytes at /replay-data, viewer at /client/replay.

    ``viewer_dist`` (default: the repo's ``viewer/dist``) is the wasm
    re-sim viewer bundle; its index is served at /client/replay and its
    static assets under /client/replay/*. When the bundle is absent the
    placeholder page is served instead (with a warning), so server tests
    and dev flows never require the emscripten build.

    Also serves a legacy ``GET /replay`` websocket that emits the parsed
    replay header as its first message: the certifier's replay-loadable
    probe (coworld<=0.1.34 runs it even when a static viewer bundle is
    declared) requires one non-empty message from that route.

    Raises ReplayError on corrupt bytes (fail at startup, not on request).
    """
    replay = Replay.parse(replay_bytes)
    dist = DEFAULT_VIEWER_DIST if viewer_dist is None else Path(viewer_dist)
    index = dist / "index.html"
    have_bundle = index.is_file()
    if not have_bundle:
        print(f"viewer bundle not found at {dist}; serving placeholder "
              f"page (run sim/build_viewer.sh for the full viewer)",
              file=sys.stderr)

    async def handle_replay_data(request: web.Request) -> web.Response:
        return web.Response(
            body=replay_bytes, content_type="application/octet-stream")

    async def handle_replay_client(request: web.Request):
        if have_bundle:
            # index references its assets relatively (viewer js/wasm next
            # to it); redirect to the slash form so they resolve under
            # /client/replay/ instead of /client/.
            raise web.HTTPFound("/client/replay/")
        return web.Response(
            text=REPLAY_PLACEHOLDER_HTML, content_type="text/html")

    async def handle_replay_index(request: web.Request):
        if have_bundle:
            return web.FileResponse(index)
        return web.Response(
            text=REPLAY_PLACEHOLDER_HTML, content_type="text/html")

    async def handle_healthz(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_replay_ws(request: web.Request):
        """Legacy replay websocket: header JSON first, then stays open."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(
            {"type": "replay_header", "header": replay.header}))
        async for _msg in ws:
            pass  # broadcast-only; the static bundle is the real viewer
        return ws

    app = web.Application()
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/replay", handle_replay_ws)
    app.router.add_get("/replay-data", handle_replay_data)
    app.router.add_get("/client/replay", handle_replay_client)
    app.router.add_get("/client/replay/", handle_replay_index)
    if have_bundle:
        # bundle assets (viewer js/wasm/data) live next to index
        app.router.add_static("/client/replay/", dist)
    return app


# -- process entry point -----------------------------------------------------

async def async_main() -> int:
    host = os.environ.get("COGAME_HOST", "0.0.0.0")
    port = int(os.environ.get("COGAME_PORT", "8080"))

    load_replay_uri = os.environ.get("COGAME_LOAD_REPLAY_URI", "")
    if load_replay_uri:
        # Replay mode: no episode, serve the recorded replay indefinitely.
        replay_bytes = await uris.read_uri(load_replay_uri)
        runner = web.AppRunner(make_replay_app(replay_bytes))
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        print(f"cogame-nmmo replay mode on {host}:{port} "
              f"({len(replay_bytes)} replay bytes)", file=sys.stderr)
        await asyncio.Event().wait()  # serve until the process is stopped
        return 0

    config_uri = os.environ.get("COGAME_CONFIG_URI", "")
    if not config_uri:
        print("COGAME_CONFIG_URI is required", file=sys.stderr)
        return 2
    config = GameConfig.from_dict(
        json.loads(await uris.read_uri(config_uri)))
    server = GameServer(
        config,
        results_uri=os.environ.get("COGAME_RESULTS_URI"),
        save_replay_uri=os.environ.get("COGAME_SAVE_REPLAY_URI"),
        player_failure_uri=os.environ.get("COGAME_PLAYER_FAILURE_URI"),
    )
    runner = web.AppRunner(server.make_app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"cogame-nmmo serving on {host}:{port} "
          f"({config.num_seats} seats x {config.heroes_per_seat} heroes)",
          file=sys.stderr)
    result = await server.run_episode()
    top = max(result.seat_scores) if result.seat_scores else 0.0
    print(f"episode over: end_reason={result.end_reason} "
          f"tick={result.final_tick} top_score={top:g} "
          f"scores={list(result.seat_scores)}",
          file=sys.stderr)
    await asyncio.sleep(SHUTDOWN_GRACE_SECONDS)
    await runner.cleanup()
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
