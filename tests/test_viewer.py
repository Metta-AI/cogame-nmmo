# Inherited cogame-moba suite: exercises the moba-shaped modules this fork
# has not adapted yet. Skipped (not deleted) pending Phase N4 (viewer),
# which replaces it — see docs/plans/2026-08-02-cogame-nmmo-implementation.md.
import pytest

pytest.skip("moba-specific suite pending Phase N4 (viewer) rewrite",
            allow_module_level=True)

"""Viewer verification without a browser (Task 4.2).

Three layers:

- build outputs: sim/build_viewer.sh artifacts exist (skips with a clear
  message when the emscripten viewer build hasn't been run);
- headless re-sim: the viewer core (viewer_main.c compiled WITHOUT
  MOBA_RENDER, ENVIRONMENT=node) loads a real recorded replay under node
  and must reach the header's tick_count with the sim's winner AND
  final-state digest matching the live recording — proving the viewer's
  replay parsing and step-scheduling logic with no pixels involved. The
  episode is baseline-vs-random so it genuinely ends by ancient kill
  (done==1), making the winner comparison non-vacuous;
- malformed input: viewer_load must reject bad magic/version, truncated
  and wasm32-wrapping header lengths, and ragged bodies.
"""

import json
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


def test_build_viewer_outputs_exist():
    if not VIEWER_CORE_JS.exists():
        pytest.skip(NOT_BUILT)
    for name in ("index.html", "moba_viewer.js", "moba_viewer.wasm",
                 "moba_viewer.data"):
        assert (VIEWER_DIST / name).exists(), f"viewer/dist/{name} missing"
    assert (REPO_ROOT / "build" / "viewer_core.wasm").exists()


async def _record_replay(tmp_path: Path):
    """Record a real wasm episode: baseline (radiant) vs random (dire).

    Team variant, seed 13 — the calibration from tests/test_baseline.py:
    the pretrained policy destroys the dire ancient by tick ~1000-1200,
    so the episode ends by ancient kill and the recorded winner is a
    real outcome, not a tick-cap tiebreak.
    """
    from cogame_nmmo.sim import MobaSim
    from players.baseline_player import BaselinePolicy

    class BaselineSource:
        def __init__(self):
            self.policy = BaselinePolicy(seed=1)

        async def get_actions(self, tick, obs):
            return self.policy(tick, [row.tobytes() for row in obs])

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(4000 + seat)

        async def get_actions(self, tick, obs):
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(len(obs), 6)).tolist()

    cfg = GameConfig.from_dict({
        "players": [{"name": "baseline"}, {"name": "random"}],
        "tokens": ["tok0", "tok1"],
        "heroes_per_seat": 5,
        "seed": 13,
        "max_ticks": 5000,
        "tick_deadline_ms": 2000,
    })
    sim = MobaSim(seed=cfg.seed)
    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    engine = LockstepEngine(
        sim, cfg, [BaselineSource(), RngSource(1)],
        on_tick=writer.append_tick)
    result = await engine.run()
    data = writer.finalize({
        "winner": result.winner,
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
    })
    path = tmp_path / "replay.bin"
    path.write_bytes(data)
    return path, result, sim


@pytest.mark.slow
async def test_headless_core_resimulates_recorded_replay(tmp_path):
    if not VIEWER_CORE_JS.exists():
        pytest.skip(NOT_BUILT)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")

    replay_path, result, sim = await _record_replay(tmp_path)
    # non-vacuous winner check requires a real ancient kill; if this
    # starts failing after an emcc bump see the toolchain-drift note in
    # tests/test_baseline.py
    assert result.end_reason == "ancient", \
        "calibrated baseline-vs-random episode no longer ends by ancient"

    proc = subprocess.run(
        [node, str(HARNESS), str(VIEWER_CORE_JS), str(replay_path)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    out = json.loads(proc.stdout)

    # malformed bytes are rejected (-1), incl. the wasm32 wrap case
    assert out["malformed"] == {
        "badMagic": -1, "badVersion": -1, "tooShort": -1,
        "truncatedHeader": -1, "wrappingHeaderLen": -1, "raggedBody": -1}

    # replay body parse: C-side tick count == header tick count
    assert out["total"] == result.final_tick
    assert out["headerTickCount"] == result.final_tick

    # frame scheduling: 1 tick / 12 frames at 1x, 4 at 4x, none paused
    assert out["cadence1"] == 1
    assert out["cadence4"] == 4
    assert out["pausedTicks"] == 0

    # seek: mid lands exactly, end reaches tick_count and pauses (no loop)
    assert out["midTick"] == result.final_tick // 2
    assert out["endTick"] == result.final_tick
    assert out["playingAtEnd"] == 0
    assert out["playAtEndRefused"] == 1

    # the re-simulated episode reproduces the recorded outcome for real:
    # done==1 (ancient kill), same winner, and the final-state digest
    # (hero x/y/health + ancient healths) matches the live recording
    assert out["done"] == 1
    assert sim.done() == 1
    assert out["winner"] == result.winner
    assert out["stateDigest"] == sim.state_digest()
