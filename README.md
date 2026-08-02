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

> **Port status:** this repo is a fork of cogame-moba being re-pointed at
> NMMO3 (plan: `docs/plans/2026-08-02-cogame-nmmo-implementation.md`).
> Vendoring (N0) and the sim wasm + fidelity/determinism gates (N1) are done;
> the server layer (N2), players (N3), viewer (N4), and packaging/docs (N5)
> are still moba-shaped and are adapted phase by phase.

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
uv run pytest              # full suite, includes the fidelity gate
```

(`sim/build_brain.sh` and `sim/build_viewer.sh` return in Phases N3/N4.)

## The game

NMMO3 is a persistent 512x512 survival world. Eight free-for-all agents
(default variant) harvest resources, fight NPC enemies, level combat and
profession skills (1-40), equip tiered gear, and trade on a market. There is
no env-level episode end: death (or 500 ticks without leveling) respawns the
agent in place with reset levels, flagged per-agent on that tick. The server
truncates at `max_ticks`; seats are ranked by cumulative
`min(combat_level, profession_level)` over all of a seat's lives.

## Protocol (for policy authors)

See [docs/PROTOCOL.md](docs/PROTOCOL.md). Short version: connect to the
websocket URL in `COWORLD_PLAYER_WS_URL`; each tick the server sends
`{"tick", "obs": [base64 x agents]}` (each blob is the upstream 1707-byte
observation, per `compute_all_obs` in `vendor/upstream/nmmo3.h`) and you
reply `{"tick", "actions": [[1 int] x agents]}` in the upstream 26-way
discrete action space (no-op = 4). Late, missing, or malformed replies play
no-op; the encodings are transported verbatim from upstream.

## Players

| player | command | what it is |
| --- | --- | --- |
| random | `python -m players.random_player` | uniform-random in-range actions |
| baseline | `python -m players.baseline_player` | upstream pretrained MMONet weights (`nmmo3_weights.bin`) compiled to wasm — the bundled certification player (Phase N3) |
| scripted | `python -m players.scripted_player` | survival FSM over the egocentric window (Phase N3) |

## Repo layout

- `vendor/upstream/` — byte-pristine vendored upstream source (never edit;
  see `vendor/UPSTREAM.md`); all changes are patch files in `sim/patches/`
  (rationale in `vendor/PATCHES.md`)
- `sim/` — patches, wasm shims, build scripts (`apply_patches.sh`,
  `build_sim.sh`, ... -> `build/`, `viewer/dist/`)
- `server/cogame_nmmo/` — config, lockstep engine, websocket server, replay
  writer/reader, wasmtime sim host; entry `python -m cogame_nmmo.server`
- `players/` — see table above
- `tests/` — the full suite; `tests/test_fidelity.py` is the acceptance gate
- `Dockerfile`, `compose.yaml`, `coworld_manifest_template.json` — Coworld
  packaging (`uv run coworld build --project .`)

## Porting other PufferLib envs

This repo follows [docs/PORTING.md](docs/PORTING.md) — the recipe (written
from the cogame-moba port, which this repo forked) for turning a PufferLib
Ocean env into a Coworld.

## Attribution

The simulation, renderer, sprite/tile assets, network weights, and puffernet
inference library are from [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib),
pinned at commit `c5d3c637446047a6efbcaa74c039c5295d201ab0`, MIT license
(`vendor/LICENSE-pufferlib`; asset license in
`vendor/upstream/resources/nmmo3/ASSETS_LICENSE.md`). This repo adds the
Coworld packaging around them.
