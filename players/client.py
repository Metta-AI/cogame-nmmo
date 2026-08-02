"""Reusable async player harness for cogame-nmmo websocket seats.

Speaks the server's lockstep wire protocol v2 (see docs/PROTOCOL.md and
``cogame_nmmo.server``), one JSON text message per tick each way:

    server -> player  {"tick": t, "obs": ["<base64 1707B>", ... per hero],
                       "resets": [bool, ... per hero]}
    player -> server  {"tick": t, "actions": [[1 int], ... per hero]}
    server -> player  {"done": true, "result": {...}}    (episode end)

The websocket URL comes from an explicit argument or, failing that, the
``COWORLD_PLAYER_WS_URL`` / ``COGAMES_ENGINE_WS_URL`` environment
variables (both appear in the Coworld cookbook's docker examples).

A policy is a callable ``policy(tick, obs_rows, resets) -> actions``
where ``obs_rows`` is a list of raw 1707-byte observation blobs (one per
hero this seat controls), ``resets`` a parallel list of bools (protocol
v2: ``resets[j]`` true means hero j's life ended on a sim step this
seat has not yet acknowledged — a recurrent policy must zero that
hero's state BEFORE consuming this tick's obs; stateless policies may
ignore it. Delivery is at-least-once, see docs/PROTOCOL.md: for a seat
answering every tick it is exactly the previous step's done flags, and
after a lag/reconnect gap the flag repeats until a valid reply
acknowledges it — a duplicate zeroes one extra tick of state, bounded
and harmless), and the return value is a matching-length nested
sequence of one action int per hero.

Reconnects: the server allows a dead seat to reconnect, so transient
connection drops are retried with a bounded number of consecutive
attempts (a connection that made progress resets the budget). The harness
keeps no tick state across reconnects — it resumes answering whatever
tick the server sends next. A 403 (bad slot/token) can never succeed on
retry and raises ``PlayerError`` immediately; a 409 (slot occupied)
usually means our own previous connection has not been reaped yet, so it
is retried within the same bounded budget — the server heartbeats player
sockets and force-closes a strike-dead seat's socket, so the stale
connection clears.

Only aiohttp is required (stdlib otherwise).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from typing import Callable, Sequence

import aiohttp
from aiohttp import WSMsgType

WS_URL_ENV_VARS = ("COWORLD_PLAYER_WS_URL", "COGAMES_ENGINE_WS_URL")

DEFAULT_MAX_CONNECT_ATTEMPTS = 5
DEFAULT_RECONNECT_DELAY_SECONDS = 0.5

# Bound on establishing one websocket connection (TCP + handshake): a
# black-holed connect must fail fast instead of eating minutes of the
# reconnect budget. Applied via the session ClientTimeout (total=None so
# the long-lived episode websocket itself is never killed).
CONNECT_TIMEOUT_SECONDS = 20.0

# Handshake statuses that can never succeed on retry. 409 (slot already
# connected) is deliberately NOT here: it is usually this seat's own
# stale previous connection, which the server reaps (ws heartbeat +
# strike-death force-close), so 409s are retried on the normal bounded
# budget.
_FATAL_HTTP_STATUSES = {
    403: "connection rejected (403): bad slot or token",
}

Policy = Callable[[int, list, list], Sequence[Sequence[int]]]


class PlayerError(Exception):
    """Fatal player-side failure (bad auth, duplicate seat, server gone)."""


def seed_from_env(default: int | None = None) -> int | None:
    """``COGAME_PLAYER_SEED`` as an int, or ``default`` when unset/empty.

    A non-integer value raises PlayerError (clear failure instead of a
    traceback deep in a policy constructor).
    """
    raw = os.environ.get("COGAME_PLAYER_SEED")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise PlayerError(
            f"COGAME_PLAYER_SEED must be an integer, got {raw!r}") from exc


def ws_url_from_env() -> str:
    """The seat websocket URL from the environment (first env var wins)."""
    for name in WS_URL_ENV_VARS:
        url = os.environ.get(name)
        if url:
            return url
    raise PlayerError(
        "no websocket URL: set " + " or ".join(WS_URL_ENV_VARS))


async def play_episode(
        policy: Policy,
        url: str | None = None,
        *,
        max_connect_attempts: int = DEFAULT_MAX_CONNECT_ATTEMPTS,
        reconnect_delay_seconds: float = DEFAULT_RECONNECT_DELAY_SECONDS,
) -> dict:
    """Play one episode; returns the ``result`` from the done message.

    ``max_connect_attempts`` bounds *consecutive* failed connection
    attempts (or connections dropped before answering any tick); a
    connection that answered at least one tick resets the budget.
    """
    if url is None:
        url = ws_url_from_env()

    failures = 0
    total_answered = 0

    def _fail(reason: str, exc: Exception | None = None):
        nonlocal failures
        failures += 1
        print(f"player: connection attempt failed "
              f"({failures}/{max_connect_attempts} consecutive): {reason}; "
              f"{total_answered} ticks answered so far", file=sys.stderr)
        if failures >= max_connect_attempts:
            raise PlayerError(
                f"giving up after {failures} consecutive failed "
                f"connection attempts: {reason}") from exc

    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=CONNECT_TIMEOUT_SECONDS,
        sock_connect=CONNECT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                ws = await session.ws_connect(url)
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in _FATAL_HTTP_STATUSES:
                    raise PlayerError(
                        _FATAL_HTTP_STATUSES[exc.status]) from exc
                _fail(f"handshake failed with status {exc.status}", exc)
                await asyncio.sleep(reconnect_delay_seconds)
                continue
            except (aiohttp.ClientError, OSError) as exc:
                _fail(str(exc), exc)
                await asyncio.sleep(reconnect_delay_seconds)
                continue

            try:
                result, answered = await _play_connection(ws, policy)
            finally:
                try:
                    await ws.close()
                except Exception:
                    # A close failure after the done message must never
                    # turn a completed episode into a player failure.
                    pass
            total_answered += answered
            if result is not None:
                return result
            # connection dropped without a done message
            if answered > 0:
                failures = 0  # made progress: fresh reconnect budget
            _fail("connection closed before the done message")
            await asyncio.sleep(reconnect_delay_seconds)


async def _play_connection(
        ws: aiohttp.ClientWebSocketResponse, policy: Policy,
) -> tuple[dict | None, int]:
    """Answer ticks on one connection until done or disconnect.

    Returns ``(result, ticks_answered)``; result is None on disconnect.
    """
    answered = 0
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("done"):
                return data.get("result", {}), answered
            if "tick" not in data or "obs" not in data:
                continue
            try:
                obs_rows = [base64.b64decode(o) for o in data["obs"]]
            except (TypeError, ValueError) as exc:
                # fail cleanly (PlayerError -> exit 1), not a raw traceback
                raise PlayerError(
                    f"malformed obs message at tick {data['tick']!r}: "
                    f"{exc}") from exc
            # Protocol v2: every tick message carries per-hero resets.
            # A missing/mis-shaped field is a protocol violation and must
            # fail loudly — silently defaulting to all-False would corrupt
            # a recurrent policy's state across respawns.
            resets = data.get("resets")
            if (not isinstance(resets, list) or len(resets) != len(obs_rows)
                    or not all(isinstance(r, bool) for r in resets)):
                raise PlayerError(
                    f"malformed resets field at tick {data['tick']!r}: "
                    f"expected {len(obs_rows)} bools, got {resets!r}")
            actions = policy(data["tick"], obs_rows, resets)
            if len(actions) != len(obs_rows):
                # fail fast locally: a row-count bug would otherwise show
                # up only as silent server-side strikes/NOOPs
                raise PlayerError(
                    f"policy returned {len(actions)} action rows for "
                    f"{len(obs_rows)} heroes")
            await ws.send_str(json.dumps({
                "tick": data["tick"],
                "actions": [[int(v) for v in row] for row in actions],
            }))
            answered += 1
    except (aiohttp.ClientError, ConnectionError):
        pass  # dropped mid-episode: caller decides whether to reconnect
    return None, answered


def run_policy_main(policy_factory: Callable[[], Policy]) -> int:
    """Entry-point helper: build the policy and play one episode.

    Takes a zero-arg factory (not a policy) so env-parsing errors during
    policy construction (e.g. a bad COGAME_PLAYER_SEED) also surface as
    clean exit codes. Returns a process exit code: 0 on a clean done
    message, 1 on fatal player errors (bad env config, bad auth,
    duplicate seat, reconnect budget exhausted).
    """
    try:
        policy = policy_factory()
        result = asyncio.run(play_episode(policy))
    except PlayerError as exc:
        print(f"player failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(f"episode done: result={json.dumps(result)}", file=sys.stderr)
    return 0
