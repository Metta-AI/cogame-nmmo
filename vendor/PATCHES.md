# Patch set applied to the vendored sim

Files under `vendor/upstream/` are byte-pristine (see `vendor/UPSTREAM.md`).
All modifications are the patch files in `sim/patches/`, applied at build time
by `sim/apply_patches.sh` into `build/`:

- `build/src-pristine/` gets **0001 only** — the minimum required to compile
  the sim without raylib. This tree is the reference build for the fidelity
  test (`tests/test_fidelity.py`).
- `build/src-patched/` gets **all** patches. This is the production sim.

**Invariant:** patches must keep in-episode physics byte-identical. The
fidelity test drives both builds with identical seed + action logs and
requires byte-identical obs and reward streams. A patch that breaks it is
rejected. (Patch 0003 is obs-only and strictly opt-in per agent: with no
agent opted in — the fidelity test's configuration — obs remain
byte-identical; physics are untouched in every configuration.)

## 0001-render-guard.patch

Guards `#include "raylib.h"` (+ the render-only `GLSL_VERSION` block,
`nmmo3.h:21-27`) and the entire renderer section (everything from
`#define FRAME_RATE 60` at former line 2134 through end of file: render
constants, `Client`, `render_conversion`, `make_client`, `draw_*`,
`c_render`, `close_client`) behind `#ifdef NMMO3_RENDER`.

- Rationale: the server-side sim build must compile to STANDALONE_WASM with
  no raylib. Upstream's sim half (lines 1–2132) uses only libc/libm plus the
  vendored `simplex.h`/`tile_atlas.h`; the raylib include is unconditional,
  so without this guard nothing compiles headless.
- No sim lines change. The `Client* client` pointer in `struct MMO` stays:
  `Client` is forward-declared in the sim half (`nmmo3.h:677`) and only ever
  dereferenced in the renderer. `tile_atlas.h` (pure data) stays unguarded —
  the renderer's `render_conversion` uses it; the sim build just carries the
  unused array.
- The viewer build (Phase N4) compiles the same tree with `-DNMMO3_RENDER`.

## 0002-fault-flag.patch

Converts upstream's two **in-episode** debug-guard `exit(1)` calls in the sim
half into a recorded fault: a file-scope `static int nmmo_fault_code`
(site codes: 1 = `find_target` unknown held-weapon type, formerly
`assert(false); exit(1)` at `nmmo3.h:1331-1332`; 2 = `use_item` unknown item
type, formerly `exit(1)` at `nmmo3.h:1513`) set where upstream exited,
bailing out of the local operation only (no target found / item use no-ops).

- Rationale: `exit()` (or the live `assert(false)` abort — we build without
  `-DNDEBUG`, matching upstream's default Makefile-less build) inside the
  wasm raises a trap in the host and kills the episode process — results and
  replay are lost. With the flag, `sim/shim.c` exports `nmmo_fault()`; the
  engine polls it every tick and can end the episode cleanly with
  `end_reason: "sim_fault"`, writing results and the partial replay.
- Both guards are unreachable-in-practice states (every equippable weapon id
  is one of tool/bow/sword; every item id is one of the twelve known types);
  in-episode physics are unchanged unless a guard trips — at which point
  upstream would have aborted the process entirely. The fidelity gate is
  unaffected: the pristine build keeps upstream's `exit()` calls, and the
  gate's action stream never trips a guard (`nmmo_fault()` stays 0).
- The many other `assert()` invariant guards in the sim half (respawn-buffer
  capacity, spawn-count reconciliation in `c_reset`, obs-window bounds) are
  deliberately NOT converted: they guard init-time/structural invariants
  whose violation means corrupt state, and a loud wasm trap (a catchable
  wasmtime `Trap` in the host) is the honest outcome — identical to native
  upstream behavior.
- File-scope flag rather than an `MMO` struct field: no struct-layout change,
  TU-local (`static`), zero fidelity surface.

## Why there are no seed / done-flag patches (moba had them)

- **Seeding (moba patch 0002):** nmmo3 draws every random number through
  `rand_r(&env->rng)` — a plain `unsigned int` field of `struct MMO`
  consumed re-entrantly (`nmmo3.h:717`; e.g. `c_reset` terrain gen, spawn
  shuffles, teleportitis). It never calls `srand`/`rand`, and never seeds
  `env->rng` itself (zero-init = seed 0). The shim simply writes
  `env->rng = seed` before `c_reset` — pure host-side field write, no source
  change needed.
- **Done flag / auto-reset removal (moba patch 0003):** nmmo3 has no
  env-wide termination and no internal auto-reset — the world is persistent;
  agents individually die and respawn in place. Upstream already writes
  per-agent `env->terminals[pid] = 1.0f` in `add_player_log`
  (`nmmo3.h:762-764`) on attack death (`attack`, `nmmo3.h:1416`) and on the
  500-tick no-improvement stagnation reset (`c_step`, `nmmo3.h:1936`). The
  env never clears the flags (upstream's vecenv memsets them externally each
  step); the shim's `nmmo_step()` zeroes the buffer before `c_step`, so after
  each step it holds exactly that tick's done flags. Episode truncation at
  `max_ticks` is server-side, as the playbook requires.

## 0003-obs-clean-optin.patch

Per-agent opt-in for truthful entity bytes. Upstream's `compute_all_obs`
writes entity bytes 4..9 only while an entity occupies the cell and never
clears them (buffer calloc'd once at `allocate_mmo`) — stale "live enemy"
claims persist indefinitely and translate with the observer's window
(reported upstream: PufferAI/PufferLib#629). This patch adds
`MMO.obs_clean[num_agents]` (calloc'd 0 = legacy) and an
`else if (obs_clean_flag)` that zeroes bytes 4..9 of empty cells for opted-in
agents' windows only.

- Default path is the exact upstream code path: non-opted agents' obs are
  **bit-identical**, so existing policies (and the pretrained demo net) are
  untouched.
- Request channel: shim export `nmmo_set_obs_clean(pid, on)` (sim/shim.c),
  surfaced to policies as the `&obs=clean` query param on the `/player`
  websocket connect (docs/PROTOCOL.md).
- Simulation state, RNG, scoring, and replays are unaffected — the patch
  touches observation encoding only.
