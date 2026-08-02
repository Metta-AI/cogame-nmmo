# cogame-nmmo

A [Coworld](https://softmax.com) game that runs PufferLib's Ocean NMMO3
("Neural MMO 3") with bit-exact observation/action spaces and physics, so
policies RL-trained on the original environment play identically when
submitted to a Coworld league.

The upstream C sim ([PufferAI/PufferLib](https://github.com/PufferAI/PufferLib)
@ `c5d3c637`, MIT) is vendored pristine under `vendor/upstream/`, patched at
build time (`sim/patches/`), and compiled to WebAssembly with emscripten. A
Python server hosts the wasm sim via `wasmtime` and referees lockstep
websocket episodes; replays re-simulate deterministically in a static wasm
browser viewer.

## The fidelity guarantee

The whole point of this port is that the served environment **is** the
trained-on environment. The gate is `tests/test_fidelity.py`: it runs the
patched production sim and a pristine (behavior-patch-free) build of the
vendored source side by side for thousands of ticks of identical random
actions and asserts every observation and reward byte matches. Sim-touching
changes must keep this test green; the test itself is inviolable — fix the
patch, never the test.

## Quickstart

```sh
uv sync
bash sim/build_sim.sh      # sim wasm (requires emcc; brew install emscripten)
bash sim/build_brain.sh    # baseline MMONet brain wasm (requires xxd)
bash sim/build_viewer.sh   # browser replay viewer (downloads pinned raylib)
uv run pytest              # full suite, includes the fidelity gate
```

Run a local containerized episode (Docker required):

```sh
docker build --platform=linux/amd64 -t cogame-nmmo:local .
uv run coworld build --project . --version 0.1.0
uv run coworld run-episode dist/coworld_manifest.json
```

Watch a recorded replay:

```sh
uv run coworld replay dist/coworld_manifest.json <path/to/replay>
# or directly: COGAME_LOAD_REPLAY_URI=file://<replay> python -m cogame_nmmo.server
```

Viewer note: pressing ESC while the canvas has focus ends the viewer's wasm
runtime with `exit(0)` — that is upstream's own input handling (`c_render`
in `vendor/upstream/nmmo3.h`), kept unpatched under the fidelity discipline.
Reload the page to restart the viewer.

## The game

NMMO3 is a persistent 512x512 survival world. Eight free-for-all agents
(default variant) harvest resources, fight NPC enemies, level combat and
profession skills (1-40), equip tiered gear, and trade on a market. There is
no env-level episode end: death (or 500 ticks without leveling) respawns the
agent in place with reset levels, flagged per-agent on that tick. The server
truncates at `max_ticks`.

**Scoring** (higher = better): each seat scores the **mean
min(combat, profession) level per life**, summed over the seat's agents —
per agent, the cumulative min over ended lives plus the final life's min,
divided by lives (`deaths + 1`). Normalizing by lives is what kills the
suicide-farming exploit: under a raw cumulative total, dying every ~50 ticks
banks `min >= 1` per life and respawns, out-earning honest leveling — random
walkers would out-rank policies that actually play. Under the mean, a death
always hurts. Full rationale and edge cases: `docs/PROTOCOL.md` (Results).

## Protocol (for policy authors)

See [docs/PROTOCOL.md](docs/PROTOCOL.md). Short version: connect to the
websocket URL in `COWORLD_PLAYER_WS_URL`; each tick the server sends
`{"tick", "obs": [base64 x agents], "resets": [bool x agents]}` (each blob
is the upstream 1707-byte observation, per `compute_all_obs` in
`vendor/upstream/nmmo3.h`; `resets` are per-agent done flags for recurrent
policies) and you reply `{"tick", "actions": [[1 int] x agents]}` in the
upstream 26-way discrete action space (no-op = 4). Late, missing, or
malformed replies play no-op; the encodings are transported verbatim from
upstream.

## Players

| player | command | what it is |
| --- | --- | --- |
| random | `python -m players.random_player` | uniform-random in-range actions |
| baseline | `python -m players.baseline_player` | upstream pretrained MMONet weights (`nmmo3_weights.bin`) through the vendored network forward pass compiled to wasm — the bundled certification player (needs `build/nmmo3_brain.wasm`) |
| scripted | `python -m players.scripted_player` | hand-coded survival FSM over the decoded egocentric window (pure Python + aiohttp) |

## Repo layout

- `vendor/upstream/` — byte-pristine vendored upstream source (never edit;
  see `vendor/UPSTREAM.md`); all changes are patch files in `sim/patches/`
  (rationale in `vendor/PATCHES.md`)
- `sim/` — patches, wasm shims, build scripts (`apply_patches.sh`,
  `build_sim.sh`, `build_brain.sh`, `build_viewer.sh` -> `build/`,
  `viewer/dist/`)
- `server/cogame_nmmo/` — config, lockstep engine, websocket server, replay
  writer/reader, wasmtime sim host; entry `python -m cogame_nmmo.server`
- `players/` — see table above
- `tests/` — the full suite; `tests/test_fidelity.py` is the acceptance gate
- `Dockerfile`, `compose.yaml`, `coworld_manifest_template.json` — Coworld
  packaging (`uv run coworld build --project .`)

## Porting other PufferLib envs

This repo was ported by following
[docs/PORTING.md in cogame-moba](https://github.com/Metta-AI/cogame-moba/blob/main/docs/PORTING.md)
— the reusable recipe (maintained in the cogame-moba repo, the port this one
forked from) for turning a PufferLib Ocean env into a Coworld.

## Attribution

The simulation, renderer, sprite/tile assets, network weights, and puffernet
inference library are from [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib),
pinned at commit `c5d3c637446047a6efbcaa74c039c5295d201ab0`, MIT license
(`vendor/LICENSE-pufferlib`; asset license in
`vendor/upstream/resources/nmmo3/ASSETS_LICENSE.md`). This repo adds the
Coworld packaging around them.
