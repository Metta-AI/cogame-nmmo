# Working in this repo

Conventions for agents (and humans) making changes here. The design and
implementation history live in `docs/plans/`; the porting recipe this repo
followed is maintained in the cogame-moba repo
([docs/PORTING.md](https://github.com/Metta-AI/cogame-moba/blob/main/docs/PORTING.md)).

## The two inviolable rules

1. **`vendor/upstream/` is byte-pristine.** It is the vendored PufferLib
   source at the pinned commit (`vendor/UPSTREAM.md` records the commit and
   per-file sha256s). Never edit anything under it. All source changes are
   patch files in `sim/patches/`, applied at build time into `build/src-*`
   by `sim/apply_patches.sh`, each documented in `vendor/PATCHES.md`. The
   whole set is two patches — 0001 (render guard: compiles the sim headless)
   and 0002 (fault flag: replaces two unreachable `exit(1)` debug guards) —
   and it should stay that small; every new patch is a new fidelity risk.
2. **The fidelity gate is inviolable.** `tests/test_fidelity.py` proves the
   patched production sim is byte-identical (obs + rewards + terminals,
   thousands of ticks) to a pristine build of the vendored source. It must
   pass after every sim-touching change. If it fails, a patch changed
   physics — fix the patch, never the test. Weakening or skipping it is a
   failed task, not a passing build. (CI sets `COGAME_REQUIRE_WASM_BUILD=1`
   so a missing build artifact fails the gate instead of skipping it.)

## Obs quirks are contracts

The 1707-byte obs and 26-way discrete action encodings are opaque
contracts — transport them verbatim, never re-encode. That includes the
trained-on quirks documented in `docs/PROTOCOL.md`: the never-written byte
1706 and the per-window-cell stale entity bytes. They look like bugs; the
pretrained network was trained on them, so "fixing" them would break the
fidelity guarantee. Decode guidance lives in `players/scripted_player.py`
(constants pinned against upstream in `tests/test_scripted.py`).

## Where things live

- Env-physics config values (world/entity counts, reward weights, obs
  windows) mirror upstream `nmmo3.ini` [env] (vendored at
  `vendor/upstream/nmmo3.ini`) + the `nmmo3.c` demo struct and live in
  `sim/shim_common.h` (`nmmo_configure`).
  Server-contract defaults (max_ticks, no-op action `[4]`, seat topology,
  the 1024 total-agent cap) live in `server/cogame_nmmo/defaults.py`. Keep
  the upstream citations next to the values.
- Scoring is **mean min(comb, prof) per life** (anti-suicide-farming;
  rationale in `docs/PROTOCOL.md` Results). The wording is synced across
  `server/cogame_nmmo/engine.py`, the manifest `results_schema`, and
  `docs/PROTOCOL.md` — change one, change all.
- Results keys are a CLOSED schema: `server/cogame_nmmo/server.py`
  `_results_doc` and the manifest template `results_schema` must list
  exactly the same keys. Adding a results field means updating both (and
  `tools/ci/docker_smoke.sh`'s expected-keys set) — a triple-sync.

## Build pipeline

```sh
bash sim/apply_patches.sh   # vendor + patches -> build/src-{pristine,patched}
bash sim/build_sim.sh       # -> build/nmmo3_sim.wasm, build/nmmo3_sim_pristine.wasm
bash sim/build_brain.sh     # -> build/nmmo3_brain.wasm (needs xxd)
bash sim/build_viewer.sh    # -> viewer/dist/ + build/viewer_core.* (pinned raylib web)
```

`build/`, `dist/`, and `viewer/dist/` are gitignored build outputs. The
Dockerfile runs the last three scripts in its wasm-builder stage
(`apply_patches.sh` runs transitively inside `build_sim.sh` and
`build_viewer.sh`, and is idempotent); the emcc
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
- Review cadence for larger phases: implementer, then a spec-conformance
  pass (code vs. written plan), then a quality pass — reviewers read the
  code, not the implementer's report.

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
`coworld next-version`, see its docstring). It no-op-skips (with a
warning) until the `SOFTMAX_TOKEN` repo secret exists; once uploads are
live, set the `UPLOAD_REQUIRED` repo variable to `true` so a lost token
fails the job instead of silently skipping.
