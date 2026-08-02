# Working in this repo

Conventions for agents (and humans) making changes here. The design and
implementation history live in `docs/plans/`; the porting recipe this repo
demonstrates is `docs/PORTING.md`.

## The two inviolable rules

1. **`vendor/upstream/` is byte-pristine.** It is the vendored PufferLib
   source at the pinned commit (`vendor/UPSTREAM.md` records the commit and
   per-file sha256s). Never edit anything under it. All source changes are
   patch files in `sim/patches/`, applied at build time into `build/src-*`
   by `sim/apply_patches.sh`, each documented in `vendor/PATCHES.md`.
2. **The fidelity gate is inviolable.** `tests/test_fidelity.py` proves the
   patched production sim is byte-identical (obs + rewards, thousands of
   ticks) to a pristine build of the vendored source. It must pass after
   every sim-touching change. If it fails, a patch changed physics — fix
   the patch, never the test. Weakening or skipping it is a failed task,
   not a passing build.

## Where things live

- Env-physics config values (world/entity counts, reward weights, obs
  windows) mirror upstream `config/nmmo3.ini` [env] + the `nmmo3.c` demo
  struct and live in `sim/shim_common.h` (`nmmo_configure`).
  Server-contract defaults (max_ticks, no-op action `[4]`, seat topology)
  live in `server/cogame_nmmo/defaults.py`. Keep the upstream citations
  next to the values.
- The 1707-byte obs and 26-way discrete action encodings are opaque
  contracts — transport them verbatim, never re-encode.
- Results keys are a CLOSED schema: `server/cogame_nmmo/server.py`
  `_results_doc` and the manifest template `results_schema` must list
  exactly the same keys. Adding a results field means updating both (and
  `tools/ci/docker_smoke.sh`'s expected-keys set).

## Build pipeline

```sh
bash sim/apply_patches.sh   # vendor + patches -> build/src-{pristine,patched}
bash sim/build_sim.sh       # -> build/nmmo3_sim.wasm, build/nmmo3_sim_pristine.wasm
bash sim/build_brain.sh     # -> build/nmmo3_brain.wasm (needs xxd; Phase N3 rewrite pending)
bash sim/build_viewer.sh    # -> viewer/dist/ + build/viewer_core.* (Phase N4 rewrite pending)
```

`build/`, `dist/`, and `viewer/dist/` are gitignored build outputs. The
Dockerfile runs the same four scripts in its wasm-builder stage; the emcc
pin (6.0.5) is recorded in `vendor/UPSTREAM.md` and must stay in sync
across the Dockerfile and `.github/workflows/ci.yml`.

## Testing and review discipline

- `uv run pytest` runs the full suite (fast — slow-marked tests are
  included in CI too). Run it before every commit that touches
  sim/server/players.
- TDD for behavior changes: failing test first, then the implementation.
- Commit in small, single-purpose units with pathspec `git add` (never
  `git add -A` in a shared tree).
- Packaging changes (Dockerfile, compose, manifest template) must keep
  `docker build` + `tools/ci/docker_smoke.sh` and
  `uv run coworld build --project . --version <v>` +
  `uv run coworld certify dist/coworld_manifest.json` green.

## Coworld platform contract

The server implements the Coworld runtime contract (`COGAME_*` env vars,
`/player` + `/global` websockets, `/client/*` pages, replay mode) — see
`docs/PROTOCOL.md` and the certifier probes in
`coworld.runner.runner.run_episode_containers`. The manifest template
declares a static replay viewer bundle (`static-replay-viewer`, built by
`tools/build_replay_viewer.sh` from the Dockerfile's wasm-builder stage),
which replaces the legacy replay-route certification probes; the server
still serves `/client/replay` for local viewing.

Uploads: the `upload-coworld` job in `.github/workflows/ci.yml`
(push-to-main, gated behind green `test` + `docker-smoke` jobs so a
red-test push can never publish; version = highest existing registry row
patch-bumped via `tools/ci/next_coworld_version.py` — never
`coworld next-version`, see its docstring). It no-op-skips until the
`SOFTMAX_TOKEN` repo secret exists.
