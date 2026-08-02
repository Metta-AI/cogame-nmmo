"""THE acceptance gate (CI-enforced): patched sim == pristine upstream sim.

Drives build/nmmo3_sim.wasm (all patches) and build/nmmo3_sim_pristine.wasm
(patch 0001 render-guard only — the minimum that compiles) with the identical
random action log and asserts byte-identical 1707-byte observation streams,
float32 reward streams, and per-tick terminal flags every tick, plus equal
state digests at the end.

Seeding: both builds get the seed the same way — the shim writes the
env.rng struct field before c_reset (nmmo3 draws all randomness through
rand_r(&env->rng) and never seeds itself), so no seeding patch exists and
there is no seed-related divergence to reason about.

Termination: nmmo3 has no env-wide episode end and no internal auto-reset
(agents individually die and respawn in place, which both builds do
identically), so unlike the moba gate there is no early-stop branch — the
comparison runs the full TICKS ticks unconditionally, and the tick floor
assert keeps the gate from silently shrinking.

The only behavior patch is 0002 (fault flag): it replaces two
unreachable-in-practice exit(1) debug guards. This gate is degenerate while
that is the whole patch set — it exists as the permanent guard for any
future patch. A failure here means a patch changed in-episode physics: fix
the patch, never this test.
"""

import os

import numpy as np
import pytest

from cogame_nmmo.sim import (ACT_HIGH, DEFAULT_WASM_PATH, NUM_AGENTS,
                             PRISTINE_WASM_PATH, NmmoSim)

TICKS = 5000

NOT_BUILT = "sim wasm not built - run sim/build_sim.sh first"


def test_patched_matches_pristine():
    # Same COGAME_REQUIRE_WASM_BUILD rule as tests/test_viewer.py: the
    # local-convenience skip must never fire where the artifacts were
    # just built (CI sets the env var after its build step), or the
    # acceptance gate would silently stop gating.
    if not PRISTINE_WASM_PATH.exists():
        if os.environ.get("COGAME_REQUIRE_WASM_BUILD"):
            pytest.fail(NOT_BUILT + " (COGAME_REQUIRE_WASM_BUILD is set)")
        pytest.skip(NOT_BUILT)
    patched = NmmoSim(seed=1, wasm_path=DEFAULT_WASM_PATH)
    pristine = NmmoSim(seed=1, wasm_path=PRISTINE_WASM_PATH)

    assert patched.observations().tobytes() == pristine.observations().tobytes(), \
        "initial obs diverged (env.rng seeding mismatch?)"

    rng = np.random.default_rng(42)
    compared = 0
    for t in range(TICKS):
        acts = rng.integers(0, ACT_HIGH,
                            size=(NUM_AGENTS, 1)).astype(np.float32)
        for sim in (patched, pristine):
            sim.set_actions(acts)
            sim.step()
        assert patched.rewards().tobytes() == pristine.rewards().tobytes(), \
            f"rewards diverged at tick {t}"
        assert patched.terminals().tobytes() == pristine.terminals().tobytes(), \
            f"terminals diverged at tick {t}"
        assert patched.observations().tobytes() == pristine.observations().tobytes(), \
            f"obs diverged at tick {t}"
        compared += 1

    # Floor: the gate must not silently weaken.
    assert compared == TICKS, \
        f"only {compared}/{TICKS} ticks compared - gate coverage weakened"
    assert patched.tick() == TICKS and pristine.tick() == TICKS
    assert patched.state_digest() == pristine.state_digest()
    # the fault guards (patch 0002) must not have tripped on this stream
    assert patched.fault() == 0
