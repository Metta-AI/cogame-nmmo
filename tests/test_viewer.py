"""Viewer verification without a browser (Phase N4).

Three layers:

- build outputs: sim/build_viewer.sh artifacts exist (skips with a clear
  message when the emscripten viewer build hasn't been run) and the
  browser bundle excludes the 17.7 MB policy weights;
- headless re-sim: the viewer core (viewer_main.c compiled WITHOUT
  NMMO3_RENDER, ENVIRONMENT=node) loads a real recorded replay under
  node and must reach the header's tick_count with the final-state
  digest matching the live recording — proving the viewer's replay
  parsing, step-scheduling and interpolation phase-lock logic with no
  pixels involved. The episode is scripted-vs-random seats so the
  recorded scores are non-degenerate;
- malformed input: viewer_load must reject bad magic/version, truncated
  and wasm32-wrapping header lengths, ragged bodies, and out-of-bounds
  agent counts.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cogame_nmmo import defaults, replay
from cogame_nmmo.config import GameConfig
from cogame_nmmo.engine import LockstepEngine
from cogame_nmmo.replay import ReplayWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIST = REPO_ROOT / "viewer" / "dist"
VIEWER_CORE_JS = REPO_ROOT / "build" / "viewer_core.js"
HARNESS = Path(__file__).parent / "viewer_core_harness.js"

NOT_BUILT = "viewer not built - run sim/build_viewer.sh first"

# One sim tick per 36 60Hz-equivalent frames at 1x (upstream TICK_FRAMES;
# see sim/viewer_main.c VIEWER_FRAMES_PER_TICK).
FRAMES_PER_TICK = 36

# Recorded-episode length: long enough for scripted seats to bank levels
# and random seats to die (non-degenerate scores; see the calibration in
# tests/test_scripted.py), short enough to keep the suite fast.
RECORD_TICKS = 1000


def _skip_or_fail_not_built():
    """Same CI rule as the fidelity gate (tests/test_fidelity.py): with
    COGAME_REQUIRE_WASM_BUILD set, a missing build artifact is a
    failure, never a silent skip."""
    if os.environ.get("COGAME_REQUIRE_WASM_BUILD"):
        pytest.fail(NOT_BUILT + " (COGAME_REQUIRE_WASM_BUILD is set)")
    pytest.skip(NOT_BUILT)


def test_build_viewer_outputs_exist():
    if not VIEWER_CORE_JS.exists():
        _skip_or_fail_not_built()
    for name in ("index.html", "nmmo3_viewer.js", "nmmo3_viewer.wasm",
                 "nmmo3_viewer.data", "sim_sha.js"):
        assert (VIEWER_DIST / name).exists(), f"viewer/dist/{name} missing"
    assert (REPO_ROOT / "build" / "viewer_core.wasm").exists()


def test_viewer_bundle_excludes_policy_weights():
    """The .data preload must carry the ~14 MB render assets but never
    nmmo3_weights.bin (17.7 MB the renderer never reads): the staged-
    asset-dir rule in sim/build_viewer.sh. A weights regression would
    show up as the .data ballooning past the raw asset size."""
    data = VIEWER_DIST / "nmmo3_viewer.data"
    if not data.exists():
        _skip_or_fail_not_built()
    weights = (REPO_ROOT / "vendor" / "upstream" / "resources" / "nmmo3"
               / "nmmo3_weights.bin")
    size = data.stat().st_size
    assert size < weights.stat().st_size, \
        f".data is {size} bytes - looks like the weights got preloaded"
    # sanity floor: the merged sheet alone is ~1 MB, all assets ~14 MB
    assert size > 10_000_000, f".data is only {size} bytes - assets missing?"


async def _record_replay(tmp_path: Path):
    """Record a real wasm episode via the engine: 4 scripted seats vs 4
    random seats (seed 17, the tests/test_scripted.py calibration family)
    so per-seat scores come out non-degenerate."""
    from cogame_nmmo.sim import NmmoSim
    from players.scripted_player import ScriptedPolicy

    class ScriptedSource:
        def __init__(self, seed):
            self.policy = ScriptedPolicy(seed=seed)

        async def get_actions(self, tick, obs, resets):
            rows = [np.asarray(row, dtype=np.uint8).tobytes() for row in obs]
            return self.policy(tick, rows, resets)

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(4000 + seat)

        async def get_actions(self, tick, obs, resets):
            return self.rng.integers(
                0, defaults.ACT_HIGH,
                size=(len(obs), defaults.ACTIONS_PER_AGENT)).tolist()

    cfg = GameConfig.from_dict({
        "players": [{"name": f"scripted-{i}"} for i in range(4)]
        + [{"name": f"random-{i}"} for i in range(4)],
        "tokens": [f"tok{i}" for i in range(8)],
        "seed": 17,
        "max_ticks": RECORD_TICKS,
        "tick_deadline_ms": 2000,
    })
    sim = NmmoSim(seed=cfg.seed, num_agents=cfg.num_agents)
    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    sources = [ScriptedSource(seed=3 + s) for s in range(4)] \
        + [RngSource(s) for s in range(4, 8)]
    engine = LockstepEngine(sim, cfg, sources, on_tick=writer.append_tick)
    result = await engine.run()
    data = writer.finalize({
        "scores": list(result.seat_scores),
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
    })
    path = tmp_path / "replay.bin"
    path.write_bytes(data)
    return path, result, sim


@pytest.mark.slow
async def test_headless_core_resimulates_recorded_replay(tmp_path):
    if not VIEWER_CORE_JS.exists():
        _skip_or_fail_not_built()
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")

    replay_path, result, sim = await _record_replay(tmp_path)
    assert result.final_tick == RECORD_TICKS
    # the endcard standings are only meaningful if scores differ across
    # seats (scripted banks levels, random mostly floors at ~1.0)
    assert len(set(result.seat_scores)) > 1, \
        f"degenerate recorded scores {result.seat_scores}"

    proc = subprocess.run(
        [node, str(HARNESS), str(VIEWER_CORE_JS), str(replay_path)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    out = json.loads(proc.stdout)

    # malformed bytes are rejected (-1), incl. the wasm32 wrap case and
    # out-of-bounds agent counts
    assert out["malformed"] == {
        "badMagic": -1, "badVersion": -1, "tooShort": -1,
        "truncatedHeader": -1, "wrappingHeaderLen": -1, "raggedBody": -1,
        "zeroAgents": -1, "hugeAgents": -1}

    # replay body parse: C-side tick count == header tick count, with the
    # stride derived from the header topology JS passed in
    assert out["numAgents"] == 8
    assert out["total"] == result.final_tick
    assert out["headerTickCount"] == result.final_tick

    # frame scheduling: 1 tick / 36 frames at 1x, 4 at 4x, none paused
    assert out["cadence1"] == 1
    assert out["cadence4"] == 4
    assert out["pausedTicks"] == 0

    # interpolation phase-lock (the moba jitter lesson, applied from the
    # start): at-tick (36) right after a seek, holds until the first tick
    # then sweeps 0,1,2,...; frozen across pause and continued (not
    # reset) on resume; pinned at-tick when one callback steps multiple
    # ticks (64x, 100ms => 10 ticks)
    assert out["phaseAfterSeek"] == FRAMES_PER_TICK
    assert out["phaseSweep"] == [0, 1, 2]
    assert out["phasePaused"] == 2
    assert out["phaseResumed"] == 3
    assert out["ticksMulti"] > 1
    assert out["phaseAtMulti"] == FRAMES_PER_TICK

    # time-based advance: 1 tick per 600ms of (speed-scaled) wall time,
    # with per-callback dt clamped to 100ms (no tab-switch burst: 5000ms
    # counts as 100)
    assert out["dtSteps"] == [0, 0, 0, 0, 0, 1]
    assert out["dtClamped"] == 0
    assert out["dtAfterClamp"] == [0, 0, 0, 0, 1]

    # seek: mid lands exactly, end reaches tick_count and pauses (no loop)
    assert out["midTick"] == result.final_tick // 2
    assert out["endTick"] == result.final_tick
    assert out["playingAtEnd"] == 0
    assert out["playAtEndRefused"] == 1

    # the re-simulated episode reproduces the recorded outcome for real:
    # the final-state digest (player r/c/hp/levels + rng + tick) matches
    # the live recording — and the recording matched the engine's own
    # digest at finalize time
    assert result.state_digest == sim.state_digest()
    assert out["stateDigest"] == sim.state_digest()
