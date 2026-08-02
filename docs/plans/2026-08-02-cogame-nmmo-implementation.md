# cogame-nmmo Implementation Plan

> Executed per docs/PORTING.md (since moved; canonical copy lives in Metta-AI/cogame-moba) with two-stage review (spec, then quality) per phase.
> Design (settled, do not re-litigate): docs/plans/2026-08-02-cogame-nmmo-design.md.
> This repo is a FORK of cogame-moba @ 2f24217 — the server layer, tooling, workflows, and test
> patterns are inherited; tasks below say what to SWAP. When in doubt, the moba twin of a file
> is the pattern to follow, and vendored nmmo3 source is ground truth for env facts.

**Working directory: /Users/daveey/code/cogame-nmmo. Never touch cogame-moba (another session
is actively working it), coworld-ctf, or metta (read-only references).**

**Global invariants:** vendor pristine + patches-at-build; obs (1707 B/agent) and action
(MultiDiscrete[26], floats-cast-to-int) encodings are opaque contracts; fidelity + determinism
gates green after every sim-layer change; env config = trained values (config/nmmo3.ini [env] +
nmmo3.c demo struct — cite per value); closed results schema triple-sync (server ↔ manifest ↔
docker_smoke).

## Phase N0: Re-point the fork

- Purge moba specifics: rm vendor/upstream/* (moba files), sim/patches/000{2,3,4}*, moba shim
  code specifics, players/scripted_player.py + its tests, moba README/PROTOCOL content (rewrite
  headers now, full docs in N5), manifest template game name → `nmmo`, source_url →
  github.com/Metta-AI/cogame-nmmo. Rename Python package server/cogame_moba → server/cogame_nmmo
  (adjust pyproject, imports, Dockerfile PYTHONPATH, entrypoints, workflows). Keep: engine,
  server, uris, replay, client, random player, tools/, workflows, test patterns — they get
  adapted in place in later phases; mark clearly-broken moba tests skipped with reason "N-phase
  pending" ONLY until their phase replaces them (no silent deletions without replacement).
- Vendor pinned nmmo3 files: ocean/nmmo3/{nmmo3.h,nmmo3.c,binding.c,simplex.h,tile_atlas.h},
  config/nmmo3.ini, resources/nmmo3/nmmo3_weights.bin, resources/nmmo3 render assets
  (~14 MB: merged_sheet.png, items_condensed.png, inventory PNGs, 50 character sheets,
  map/bloom shaders — enumerate from make_client nmmo3.h:2468-2515), resources/shared if the
  build preloads it. UPSTREAM.md: new sha256 table, same pin, same rules. LICENSE unchanged.
- Commit checkpoints per step; repo builds nothing yet — that's fine.

## Phase N1: Sim wasm + host + gates

- Patch 0001-render-guard: `#ifdef NMMO3_RENDER` around `#include "raylib.h"` (nmmo3.h:21) and
  lines 2134–EOF. PATCHES.md rewritten (single patch; document why 0002/0003 analogues are
  unnecessary — rng field seeding, no auto-reset).
- sim/shim.c + shim_common.h: nmmo_configure() with trained values (cited); moba_init analogue
  writes `env.rng = seed` BEFORE allocate/reset; exports: init(seed, num_agents), step, reset,
  obs_ptr (num_agents×1707), act_ptr (floats), rew_ptr, per-agent terminal flags (persist the
  tick's terminals before vec-style clearing — read how c_step writes them and expose
  per-tick), tick, per-agent score accessors (log fields: score/min_comb_prof, comb_lvl,
  prof_lvl, deaths — read add_player_log + Entity), state_digest (entity positions/hp/levels +
  rng), fault hook not needed unless source shows exit() guards — grep for exit( in the sim
  half; if any exist, replicate moba patch-0004 treatment.
- Build scripts adapted (memory: INITIAL 64 MB, ALLOW_GROWTH, MAXIMUM 1gb, ABORTING_MALLOC).
- server/cogame_nmmo/sim.py: NmmoSim host (moba sim.py pattern: module cache, _initialize,
  growth stub, copies, action validation — clamp to [0,26) int cast semantics).
- Tests: shapes; determinism (same seed twice identical, different seeds differ); fidelity gate
  (pristine vs patched, tick floor); obs-contract snapshot (hash of tick-0 obs for a fixed seed
  pinned in-test to catch accidental config drift); terminals surface test (kill/stagnation is
  hard to force quickly — at minimum verify flag plumbing with long random episodes; report
  observed death events).

## Phase N2: Server adaptation

- config.py/defaults.py: seats = num_agents (default variant 8×1), heroes_per_seat fixed 1 for
  now (drop moba's 5-hero batching or keep generalized — keep the generalized code, config
  validates product == num_agents), max_ticks default 5000, NOOP action = [4].
- Protocol v2: per-tick server→player message gains `"resets": [bool per hero]` (seat's
  agent(s) that hit done this tick — recurrent policies zero state); PROTOCOL.md updated.
- engine: per-seat cumulative score accumulation from sim accessors; EpisodeResult: scores =
  ranked score values (no winner/end_reason "ancient" — end_reason ∈ {tick_cap, wall_clock,
  sim_fault if applicable}); keep strike rule, wall-clock budget, lifecycle logging (inherited
  fixes).
- results doc: names, scores (the ranking score), score breakdown per seat (final comb/prof
  levels, deaths, respawns), noop/dead/cause tallies (inherited), seed. Manifest results_schema
  + docker_smoke sync.
- replay.py: body = num_agents bytes/tick (1 action byte per agent); header unchanged shape.
- Tests: adapt the moba server/engine/replay suites (ws episode 8 seats, no-show, malformed,
  strike/revive, replay round-trip + re-sim with digest).

## Phase N3: Players

- baseline: sim/brain_shim.c rebuilt around the demo's MMONet (port net construction + forward
  from vendored nmmo3.c:11-152 verbatim — it uses puffernet.h primitives; embed
  nmmo3_weights.bin via xxd; one net instance per seat-agent; expose reset_state(agent) and
  call it when the wire `resets` flag fires, mirroring nmmo3.c:71-78). Param-count check
  4,430,976. Sampling not argmax; document the libc caveat as in moba.
- random player: 26-way uniform.
- scripted_player.py (daveey's entry): survival FSM per design (window+scalars only; no global
  map). Obs-layout constants tripwired against vendored nmmo3.h (regex the factors table,
  window dims, scalar offsets). Behavioral test: scripted beats random on score; report
  scripted vs baseline (no bar).
- Tests: brain determinism + reset-on-done semantics; fuzz action validity; client resets
  plumbing.

## Phase N4: Viewer

- viewer_main.c adapted: -DNMMO3_RENDER build with assets preload (weights EXCLUDED from
  .data); re-sim from replay; same phase-lock treatment as moba's jitter fix (renderer
  interpolates via TICK_FRAMES=36 — read the render loop and phase-lock our accumulator to it
  from the start); headless node core + digest assertion; index.html reuse (names, scores
  endcard instead of winner).
- Budget note: .data ≈ 14 MB + wasm; verify hosted-viewer size acceptable; if the 50 character
  sheets bloat unreasonably, they are all required by make_client — keep, note size.

## Phase N5: Packaging, docs, publish

Carry-over doc debt flagged by earlier phases: tools/ci/docker_smoke.sh:95 comment ("scores are
raw values") and the manifest results_schema scores description are stale post-mean-per-life —
reword both during the N5 doc pass.

- Dockerfile/compose/manifest (name `nmmo`, default variant 8 seats, cert fixture 8 baseline
  players seed 42 max_ticks ~1500 — calibrate so cert runs in minutes), README/PROTOCOL/AGENTS
  rewritten for nmmo, CI adapted (same emsdk pin), upload workflow (name key `nmmo`), version
  picker reused. gh repo create Metta-AI/cogame-nmmo --public, push, CI green, local
  build+certify 10/10, watch one replay by eye (browser tools).
- SOFTMAX_TOKEN: needs its own repo secret (user mints or reuses pattern; workflow guard makes
  it safe to merge without).

## Phase N6: Hosted deploy

- upload-coworld (one-time local; record cow_ id); league seed (platform commissioner), single
  Competition division; ladder: swiss_neighbor, multiple_seats, elo (1500/k32/mean),
  enabled:false → review → true; trigger round.
- Players: PufferLib identity already exists (ply_767d5e55) — upload nmmo baseline as that
  player, submit; daveey scripted upload+submit as user default. NO credential minting (tokens
  handled; CI secret may lag — uploads skip safely).
- Verify goal: round completes with non-degenerate scores (FFA scores differ), Elo moves off
  1500 for both, hosted replay session serves, leaderboard publishes. Log everything to
  tmp/phase6_log.md.

## Review cadence

Same as moba: implementer → spec reviewer → quality reviewer per phase, fixes looped, final
five-lens review (tests, silent-failures, holistic, types, comments) before N6 completes.
