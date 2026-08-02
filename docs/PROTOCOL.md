# cogame-nmmo wire protocol (v2)

The protocol a policy container speaks to play an episode, plus the
spectator and replay surfaces. The observation and action *encodings* are
deliberately opaque here: they are the upstream PufferLib Ocean NMMO3
encodings, transported verbatim (see "Observations" and "Actions" below
for the upstream ground truth).

Protocol v2 = the inherited cogame-moba lockstep protocol plus the
per-tick `resets` field (NMMO3 has per-agent partial episodes: agents
die and respawn mid-episode, so recurrent policies need per-agent done
flags every tick).

## Player websocket (`GET /player?slot=N&token=T`)

A player container receives its fully-formed connection URL in the
`COWORLD_PLAYER_WS_URL` environment variable (legacy alias
`COGAMES_ENGINE_WS_URL`), e.g.
`ws://game-host:8080/player?slot=3&token=abc123`. Connect and speak one
JSON text message per tick each way:

```
server -> player   {"tick": t, "obs": ["<base64>", ...], "resets": [bool, ...]}
player -> server   {"tick": t, "actions": [[a], ...]}
server -> player   {"done": true, "result": {...}}         episode end, then close
```

- `obs` has one entry per agent this seat controls (1 in the default
  variant; `heroes_per_seat` in general), each decoding to exactly
  **1707 bytes**. Agent order within a seat is ascending pid; seat `i`
  controls pids `[i*h, (i+1)*h)` with `h = heroes_per_seat`.
- `resets` is a parallel array of booleans, one per agent this seat
  controls: `resets[j]` is true when that agent's terminal fired on the
  **previous** sim step — it died (attack death) or hit the 500-tick
  stagnation reset, and respawned in place. The obs delivered alongside
  is the first observation of the agent's new life, so a recurrent
  policy must zero that agent's hidden state *before* consuming this
  tick's obs (exactly what upstream's demo `forward()` does with the
  env's done flags). All false at tick 0. Non-recurrent policies may
  ignore the field.
- `actions` is one single-int row per agent (the 26-way discrete action,
  see below): e.g. a 1-agent seat replies `{"tick": 5, "actions": [[17]]}`.
- The reply must echo the same `tick`. Wrong-tick, malformed, late
  (past `tick_deadline_ms`), or missing replies play the no-op action
  `[4]` for that seat's agents — the episode never stalls and never
  crashes on bad input.
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

## Observations (1707 bytes per agent, opaque)

The exact byte layout produced by upstream `compute_observations` in
`vendor/upstream/nmmo3.h` (PufferAI/PufferLib @ `c5d3c637`): an 11x15
egocentric tile window (11 rows x 15 cols x 10 bytes = 1650) + 47 self
scalars + 10 reward-block bytes = 1707. Policies trained on upstream
NMMO3 consume these bytes unchanged — that is the point of this port.

Two trained-on quirks policy authors MUST know (never "fixed", because
the pretrained net was trained on them):

- **Byte 1706 is never written.** The reward block writes 9 of its 10
  bytes; the last byte of every observation stays at whatever it held
  before (0 forever, in practice). Do not read meaning into it.
- **Stale entity bytes.** Bytes 4-9 of each tile cell (the entity
  fields) are only written when an entity currently occupies that tile,
  and the obs buffer is persistent and never cleared between ticks — a
  tile an entity walked away from keeps the entity's old bytes until
  another entity overwrites them. To know whether a tile's entity bytes
  are *current*, a decoder must not trust them in isolation. The
  upstream net was trained on exactly this residue.

Further decode guidance (window strides, scalar offsets) arrives with
the Phase-N3 scripted player, which derives its layout constants from
the vendored `nmmo3.h` factors table.

## Actions (1 value per agent)

Upstream discrete space of 26 actions (`Discrete(26)`, delivered to the
sim as one float cast to int per agent): movement, attacks, harvests,
item use, buy/sell. Action id 4 is the semantic no-op (`ATN_NOOP`);
several other ids are also semantic no-ops in most states. Values are
truncated toward zero and clamped to 0..25 at the sim boundary, exactly
like upstream's C cast — in-range integers pass through bit-exact.

## Global viewer (`GET /global`, `GET /client/global`)

`/global` is a broadcast-only websocket: an initial
`{"type": "status", ...}` snapshot on connect, throttled
`{"tick": t, "scores": [...]}` progress messages while the episode runs
(`scores` is the live per-seat standings, same ordering and semantics as
the results field), and the final `{"done": true, "result": {...}}`.
`/client/global` serves a minimal HTML page over that feed.
`GET /client/player?slot=N&token=T` serves a token-checked seat page
(play happens over the websocket, not the page).

## Results

`results.json` (closed key set, see the manifest `results_schema`):
`names`, `scores` (raw per-seat score, higher = better — cumulative
min(combat, profession) level over ended lives + the final life's min,
summed over the seat's agents; there is no winner field), `reward_sums`,
`end_reason` (`tick_cap` | `wall_clock` | `sim_fault`), `final_tick`,
`seed`, `state_digest` (u32 sim digest at episode end; a re-sim of the
replay reproduces it), `agent_stats` (per-agent score breakdown:
`cum_min_comb_prof`, `deaths`, `comb_lvl`, `prof_lvl`, `gold`,
`time_alive`), `noop_ticks`, `dead_seats`, `noop_causes`. A `sim_fault`
episode scores every seat 0.0 (equal = drawn; an infra fault is
nobody's loss).

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
with `end_reason: "wall_clock"`, scores read as they stand, and
artifacts written normally.

## Replay format (binary, v1)

`NMMO` magic (a fresh identity — the moba `MOBA` layout is not a
compatible ancestor), u8 version (= 1), u32le header length, header JSON
(fully-resolved config incl. seed and player names, final result,
tick_count, sim wasm sha256), then `tick_count * num_agents` bytes of
packed post-clamp actions (one uint8 26-way action per agent per tick;
`num_agents = len(players) x heroes_per_seat` from the header config).
Seed + actions fully determine the episode: a fresh sim stepped through
the recorded actions reproduces every obs/reward byte and the final
`state_digest`; the Phase-N4 viewer re-simulates replays with the same
wasm sim. Ground truth: `server/cogame_nmmo/replay.py`.
