# Consolidated review-fix batch (2026-08-02)

Findings from the five-agent full-repo review (tests, silent-failures, holistic, types,
comments), deduplicated and ordered. Sim-layer changes must keep the fidelity gate green.

## A. Critical — episode integrity

1. **Sim `exit()` containment** (silent-failures C1). Upstream debug guards (`spawn_player`
   move failure, dist>20 scan, tower reset, "should have reset") call `exit(0)` → empty
   `ExitTrap`, process death, results+replay lost. Add patch `0004-fault-flag`: convert these
   4 `exit(...)` sites to `env->fault = <site-id>` + return; shim exports `moba_fault()`;
   also add `env->tick` to the adjacent "glitch state" printfs. Engine: check `sim.fault()`
   per tick AND wrap sim calls in `try/except wasmtime errors`; on fault → end episode with
   `end_reason="sim_fault"` (add to results enum + manifest results_schema), write results +
   partial replay. `run_episode`: even on unexpected engine exception, attempt to write
   fault results + partial replay before re-raising. Fidelity gate: pristine build keeps
   upstream `exit()`s (patch applies to patched tree only) — gate unaffected since the
   random-action stream never trips the guards; add a unit test driving the fault path via
   a fake/forced condition if practical, else test the engine containment with a fake sim
   whose step() raises ExitTrap-like errors and one whose fault() returns nonzero.
2. **All-seats-dead event-loop starvation** (tests #1, empirically confirmed). Empty
   `asyncio.gather()` never yields → /healthz stalls → liveness kill. Fix: `await
   asyncio.sleep(0)` when no live seats (and consider a small sleep when all dead to avoid
   a busy loop). Regression test: all sources dead → heartbeat task keeps beating AND a
   source that becomes valid mid-episode revives.
3. **Stuck revival probe on disconnect** (holistic #1, confirmed by repro). `WsSeat` waiter
   parked forever when socket drops mid-probe; `_poll_dead_seat` never re-probes. Fix:
   `WsSeat.fail_waiter()` resolving the pending future with None, called from
   `_handle_player`'s finally. Test: dead seat → disconnect during probe → reconnect →
   revives.
4. **Network blip permanently kills a seat** (holistic #2 + silent-failures I2).
   `WebSocketResponse(heartbeat=20)` on /player so half-open sockets get reaped; client:
   move 409 from fatal to bounded-retry (403 stays fatal); server: on strike-death,
   force-close the seat's stale ws. Reconcile the two contradictory tests
   (test_server.py:243 vs test_players.py:86) to the single contract documented in
   PROTOCOL.md ("a seat may reconnect"); update PROTOCOL.md if wording needs it.
5. **Wall-clock budget** (holistic #3). `max_ticks × tick_deadline_ms` can reach 11h vs
   `episode_timeout_minutes: 60`. Add engine `wall_clock_budget_seconds` (config; default
   derived: min(episode_timeout×0.9, max_ticks×deadline)) → on expiry end episode
   `end_reason="wall_clock"` (add to enums/schema) and write artifacts normally. Update
   manifest variant configs to consistent values; document the relationship.

## B. Important — observability

6. **Seat lifecycle logging** (silent-failures C2): one stderr line w/ tick for connect,
   disconnect, 409-reject, strike-death, revival.
7. **Per-cause degrade counters** (silent-failures I1): count per seat
   {timeout, malformed, wrong_tick, disconnected, host_error}; add to results
   (`noop_causes`) + manifest schema; first-occurrence log per (seat, cause); never
   swallow non-player exceptions without logging type+message.
8. **Mid-episode progress heartbeat** (holistic #4): throttled stderr line every N ticks /
   30s: tick, per-seat strikes/noops.
9. **Client-side visibility** (silent-failures I3, M5): log each reconnect attempt
   (attempt #, reason, ticks answered); on done-received, exit 0 even if a trailing
   reconnect fails; malformed-obs message → PlayerError (not raw traceback).
10. **read_uri parity** (tests #3): retries (3× + backoff) + 30s timeout, per-attempt
    stderr on failure, non-2xx branch tested. write_uri: log per-attempt failures
    (silent-failures M4); fix `attempts=0` `raise None` edge.
11. **docker_smoke.sh**: dump game+player logs on assert failure (silent-failures I5).
12. **_wasm_sha256 fallback logs to stderr** (M3).

## C. Important — tripwires & tests

13. **Vendor pristineness test** (holistic #5): parse UPSTREAM.md sha table, re-hash
    vendor/upstream/**, assert equal.
14. **Manifest↔code sync test** (holistic #6 + tests #6): `set(_results_doc keys) ==
    set(manifest results_schema required)`; parse each manifest variant/cert game_config
    through GameConfig.from_dict.
15. **Fidelity gate cannot skip in CI** (tests #7): env var (set in ci.yml) that turns the
    build-missing skip into a failure.
16. **async_main startup tests** (tests #2): bad/missing config URI, malformed JSON (route
    through from_file_uri so ConfigError path is live, not dead code), replay-mode entry,
    exit codes.
17. **Multi-no-show failure reports** (tests #5 / silent-failures I6): pin behavior —
    write the LOWEST slot (first failure) rather than loop-order last; document; test with
    2 no-shows.

## D. Type/robustness one-liners (types review)

18. Mask seed to 32 bits in `GameConfig.from_dict` (canonical seed in replay header).
19. Range-check values in `ReplayWriter.append_tick` (reject >6 before astype wrap).
20. `MobaSim.num_agents` read-only property.
21. `end_reason: Literal[...]` type; timeout NaN rejection in config validation.

## E. Deferred/minor (do if cheap in passing)

22. Viewer: `Module.onAbort/onExit` → setStatus (silent-failures M6); 3-line clamp in
    `feed_and_step`; viewer compares header `sim_wasm_sha256` vs its own embedded hash →
    on-screen warning on mismatch (holistic minor).
23. Stale "Phase N" strings in shipped pages/comments (server.py:420,523,541,
    viewer_main.c:186); PORTING.md (since moved; canonical copy lives in
    Metta-AI/cogame-moba) upload-workflow filename (job lives in ci.yml);
    AGENTS.md "four scripts" → three; Dockerfile comment re apply_patches in build_brain.
24. Wasm stdio note in sim.py (unterminated printf never flushes); artifact-write retries
    delay done-broadcast — reorder: broadcast done first, then write with retries (check
    certifier tolerance) or shorten retry backoff.
25. UPSTREAM.md emcc line: distinguish brew `6.0.5-git` (local) vs `emscripten/emsdk:6.0.5`
    (Docker/CI) and state which is authoritative for released artifacts (the Docker one).

## F. Comment-accuracy fixes (comments review)

26. AGENTS.md + defaults.py: env-physics values live in `sim/shim_common.h` (`moba_configure`),
    not `sim/shim.c` — fix both citations (contributor contract points at wrong file).
27. scripted_player.py:67: towers do NOT out-range heroes (both scan at 5, shared attack
    gate) — the siege rationale is the damage asymmetry (110–175/hit); fix the comment
    (behavior unchanged).
28. brain_shim.c:41: "~4 MB" is wrong — ~80 KB of per-net buffers + one shared 382 KB weight
    blob ≈ 0.5 MB; fix the self-contradictory claim.
29. replay.py:53: "at most 40000 ticks" → "at the default tick cap" (max_ticks is
    configurable).
30. client.py:23: 409 wording — will be superseded by fix #4 (409 becomes retryable);
    rewrite the comment with the new contract.
31. scripted_player.py:75: cite kill_entity's body (moba.h:632-633) alongside the call site.

## G. Post-token additions

32. Now that SOFTMAX_TOKEN exists: implement silent-failures I4 — repo variable
    `UPLOAD_REQUIRED` (set to true via `gh variable set`) that turns the missing-secret
    skip path into a hard failure, so a deleted/expired token can't silently stop
    publishes. Guard: variable unset behaves as today (skip+warn).
33. Push coordination: pushes to main now trigger REAL hosted uploads. Before pushing this
    batch, check `uv run coworld list --json` for an existing `moba` row (Phase 6's local
    0.1.0 upload). If the registry is still empty, the upload job will fail on the version
    picker — either wait for Phase 6's upload or dispatch with an explicit version. Do not
    let a red upload job linger: re-run it once the registry has a row.
