# cogame-nmmo — Design

A Coworld running PufferLib's Ocean NMMO3 ("Neural MMO 3") with bit-exact obs/action/physics,
so policies RL-trained on the original environment play identically. Built by executing
`docs/PORTING.md` (this repo is forked from cogame-moba @ 2f24217, which includes the hardened
server layer); the moba design remains in `docs/plans/2026-08-01-cogame-moba-design.md` for
reference.

Decisions below derive from the Stage-1 research brief (pinned upstream source) and the goal
directive of 2026-08-02 (daveey): hosted NMMO league with a scripted daveey policy + the
pretrained PufferLib policy, Elo standings, rounds running, replays working, repo
`Metta-AI/cogame-nmmo`.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Upstream pin | `PufferAI/PufferLib` @ `c5d3c637` (same as moba) | toolchain consistency; nmmo3 present at pin (`ocean/nmmo3/`) |
| Sim core | vendored `nmmo3.h` sim (lines 1–2132) to STANDALONE_WASM, wasmtime-hosted | playbook |
| Patch set | **0001 render-guard only** (`#ifdef NMMO3_RENDER` around raylib include + lines 2134–end) | no auto-reset to remove (persistent world, no env-wide reset); no srand patch needed (seed = `env.rng` struct field, written by the shim before `c_reset`) |
| Seats | default variant: **8 seats × 1 character** (`num_agents=8`), FFA; `duo`-style variants deferred (YAGNI) | `num_agents` is elastic; upstream demo runs the trained net at `num_agents=1` on the trained world, so small populations are fidelity-sanctioned |
| World config | trained values unchanged: 512×512, 2048 enemies/resources, 1024 weapons, 512 gems, tiers 5, levels 40, teleportitis 0.001, windows 7/5 | trained-on contract; windows are hardcoded into the obs byte layout |
| Termination | server-side `max_ticks` truncation (default 5000; certification fixture lower). Per-seat done flags relayed every tick (death / 500-tick stagnation → in-place respawn) | env has no termination; per-agent partial episodes are a protocol extension vs moba |
| Scoring | per-seat score = **mean score per life**: (cumulative `min(comb_lvl, prof_lvl)` over ended lives + current life's min) ÷ (deaths + 1), derived engine-side from the shim's raw accessors; league ranking **Elo** over per-episode relative scores | REVISED 2026-08-02 after measurement: the raw cumulative sum rewards suicide-farming (random walkers bank min=1 every ~50-tick death and out-rank the trained net ~0.02/tick vs ~0.015/tick). Mean-per-life is upstream's own normalized eval metric (`log.min_comb_prof / log.n`) and is suicide-resistant. Raw cum + deaths stay in `agent_stats`. |
| Baseline player | upstream **MMONet** (nmmo3.c demo net: 59-ch one-hot conv encoder + byte embedding + 4×MinGRU(512) + 27-logit head, sampling) + `nmmo3_weights.bin` (4,430,976 f32) compiled to wasm; per-seat GRU state zeroed on that seat's done flag, exactly as the demo's `forward()` does | custom net — NOT moba's generic `make_puffernet`; `tests/test_nmmo3_encoder.py` upstream is the encoder spec |
| Scripted player | survival FSM from the 11×15 egocentric window + 47 scalars ONLY (procedural per-seed maps → no embedded world data, unlike moba): harvest level-appropriate resources, fight weaker enemies, equip upgrades, flee/heal at low HP, anti-stagnation wander | submitted as daveey's policy |
| Pacing / viewer / replay | unchanged from moba: pure lockstep, browser replay-only, replay = seed + per-tick action log (1 byte/agent/tick), static wasm viewer re-simulating; viewer preloads ~14 MB render assets, weights excluded from viewer bundle | playbook |
| Repo / league | `Metta-AI/cogame-nmmo`, coworld name `nmmo`; league: platform ladder, strategy `swiss_neighbor` (FFA), `insufficient_players: multiple_seats`, ranking elo | goal directive; platform auto-names the league (no rename surface — accepted) |

## Trained-on quirks preserved (never "fix")

- Obs byte 1706 never written (reward block writes 9 of 10 bytes).
- **Stale entity bytes**: tile entity fields (bytes 4–9 of each tile) are only written when an
  entity occupies the tile and the buffer is never cleared — the net was trained on the stale
  residue. One persistent obs buffer per env, exactly like upstream.
- 5 of the 26 action ids are semantic no-ops; actions cross the boundary as floats cast to int.
- Teleportitis (0.1 %/tick random teleport) stays on — trained-on dynamic.
- Missing `ManaSeedBody.ttf` (upstream renderer falls back to raylib default font) — do not add.

## Fidelity gates

1. Pristine (0001-only) vs patched wasm byte-identical obs/reward streams under identical seed +
   action logs, tick floor asserted — degenerate while the patch set is 0001-only, kept as the
   permanent guard for any future patch.
2. Determinism: same seed + actions twice → identical streams; different `env.rng` seeds differ.
3. Replay re-sim: recorded episode re-simulated from header seed+actions reaches identical final
   obs bytes and state digest.
4. Baseline behavioral: MMONet-wasm seats outperform random seats on score within a capped
   episode; per-seat GRU reset on done verified against the demo's `forward()` semantics.

## Structural deltas from the moba port (drive the implementation plan)

1. Per-seat done flags every tick in the wire protocol (protocol v2 message field `resets`).
2. Results doc: score-centric (no winner/ancient fields); closed-schema triple-sync rule applies.
3. Custom-net brain shim built from the demo's MMONet construction code, ~17 MB weights embedded.
4. Sim state ≈ 25–30 MB + growth headroom — same 1 GB wasm cap comfortably holds it.
5. Scripted bot has no static map to embed → no vendored-table tripwires; its tripwire is the
   obs-layout constants (window strides, scalar offsets) parsed from vendored `nmmo3.h`.
6. Sync task: port remaining cogame-moba review-fix items (B10+) once that repo's batch lands
   on GitHub (this fork predates them).
