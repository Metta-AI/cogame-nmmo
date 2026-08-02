# cogame-nmmo wire protocol

The protocol a policy container speaks to play an episode, plus the
spectator and replay surfaces. The observation and action *encodings* are
deliberately opaque here: they are the upstream PufferLib Ocean NMMO3
encodings, transported verbatim (see "Observations" and "Actions" below
for the upstream ground truth).

> **Port status:** the sections below still describe the inherited
> cogame-moba server (510-byte obs, 6-int actions, hero/team language).
> Phase N2 rewrites this document for NMMO3: 1707-byte obs, single-int
> 26-way actions (no-op `[4]`), a per-tick `"resets"` field (per-agent
> done flags so recurrent policies can zero state on respawn), and
> score-centric results. Until then, treat the NMMO3 facts in README.md
> and `vendor/upstream/nmmo3.h` as ground truth.

## Player websocket (`GET /player?slot=N&token=T`)

A player container receives its fully-formed connection URL in the
`COWORLD_PLAYER_WS_URL` environment variable (legacy alias
`COGAMES_ENGINE_WS_URL`), e.g.
`ws://game-host:8080/player?slot=3&token=abc123`. Connect and speak one
JSON text message per tick each way:

```
server -> player   {"tick": t, "obs": ["<base64>", ...]}   one base64 blob per hero
player -> server   {"tick": t, "actions": [[a0..a5], ...]} one 6-int row per hero
server -> player   {"done": true, "result": {...}}         episode end, then close
```

- `obs` has one entry per hero this seat controls (1 in the default
  variant, 5 in the team variant), each decoding to exactly 510 bytes.
- The reply must echo the same `tick`. Wrong-tick, malformed, late
  (past `tick_deadline_ms`), or missing replies play the no-op action
  `[3, 3, 0, 0, 0, 0]` for that seat's heroes — the episode never stalls
  and never crashes on bad input.
- Hero order within a seat is ascending pid. Seat `i` controls pids
  `[i*h, (i+1)*h)` with `h = heroes_per_seat`. Pids 0-4 are radiant,
  5-9 dire; within each team the role order is support, assassin,
  burst, tank, carry.
- Strike rule: 10 consecutive no-op fallbacks mark a seat dead — the
  server stops waiting out the deadline for it, but keeps probing with
  each tick's obs; the first valid reply revives the seat. On the
  transition to dead the server also force-closes the seat's websocket:
  a connection that missed 10 straight ticks is treated as stale, and
  closing it lets the client observe the drop and reconnect.
- Bad slot/token is rejected with HTTP 403 — fatal, retrying can never
  succeed. A connection to a slot that already has a live connection is
  rejected with HTTP 409 — retryable: the server heartbeats player
  sockets (websocket ping/pong, ~20s) and strike-closes dead seats, so
  a half-open stale connection clears within seconds and a retried
  reconnect then succeeds. A seat that disconnects may reconnect (any
  number of times) and resume at whatever tick the server sends next.

## Observations (510 bytes per hero, opaque)

The exact byte layout produced by upstream `compute_observations` in
`vendor/upstream/moba.h` (PufferAI/PufferLib @ `c5d3c637`): an 11x11
map crop around the hero (121 x 4 bytes) plus scalar hero state. Policies
trained on upstream Puffer MOBA consume these bytes unchanged — that is
the point of this port. Decode guidance for hand-written policies:
`players/scripted_player.py` documents the reliably-decodable fields.

## Actions (6 values per hero)

Upstream MultiDiscrete `[7, 7, 3, 2, 2, 2]`: `vel_y`, `vel_x` (0-6,
center 3 = zero velocity), target filter (0-2), and three skill buttons
(0/1). Values are truncated toward zero and clamped into range at the
sim boundary, exactly like upstream's C cast.

## Global viewer (`GET /global`, `GET /client/global`)

`/global` is a broadcast-only websocket: an initial
`{"type": "status", ...}` snapshot on connect, throttled `{"tick": t}`
progress messages while the episode runs, and the final
`{"done": true, "result": {...}}`. `/client/global` serves a minimal
HTML page over that feed. `GET /client/player?slot=N&token=T` serves a
token-checked seat page (play happens over the websocket, not the page).

## Runtime contract (Coworld)

The game container reads `COGAME_CONFIG_URI` (game config JSON, see the
manifest `config_schema`), writes `COGAME_RESULTS_URI` (results JSON,
see `results_schema`) and `COGAME_SAVE_REPLAY_URI` (binary replay), and
reports never-connected seats to `COGAME_PLAYER_FAILURE_URI`. It binds
`COGAME_HOST`:`COGAME_PORT` (default `0.0.0.0:8080`) and serves
`GET /healthz`. With `COGAME_LOAD_REPLAY_URI` set it runs in replay mode
instead: raw replay bytes at `GET /replay-data` and the wasm re-sim
viewer at `GET /client/replay`.

Wall-clock budget: the worst-case episode (`max_ticks x
tick_deadline_ms`) can far exceed the platform's
`episode_timeout_minutes` container kill, which would lose results and
replay. The engine therefore hard-stops a slow episode at
`wall_clock_budget_seconds` (config; default `min(0.9 x
episode_timeout_minutes x 60, max_ticks x tick_deadline_ms / 1000)`)
with `end_reason: "wall_clock"`, the same Ancient-health tiebreak as
`tick_cap`, and artifacts written normally.

## Replay format (binary, v1)

`MOBA` magic, u8 version, u32le header length, header JSON
(fully-resolved config incl. seed and player names, final result,
tick_count, sim wasm sha256), then `tick_count * 60` bytes of packed
post-clamp actions (10 heroes x 6 uint8 per tick). Seed + actions fully
determine the episode; the viewer re-simulates it with the same wasm sim.
Ground truth: `server/cogame_moba/replay.py`.
