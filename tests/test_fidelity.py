"""THE acceptance gate (CI-enforced): patched sim == pristine upstream sim.

Drives build/moba_sim.wasm (all patches) and build/moba_sim_pristine.wasm
(patch 0001 render-guard only — the minimum that compiles) with the identical
random action log and asserts byte-identical 510-byte observation streams and
float32 reward streams every tick.

Seeding subtlety (patch 0002): the patched build calls srand(1) for seed=1;
the pristine build never calls srand, so it runs on libc's default rand()
state. Under emscripten's musl libc, srand(s) stores s - 1 and the initial
state is 0, so srand(1) reproduces the default stream exactly — verified by
this test passing from tick 0.

If the patched sim reports done (ancient death), the pristine sim has
auto-reset internally on that same tick (upstream behavior patch 0003
removes), so observation streams legitimately diverge there: the comparison
would stop at that tick after checking rewards, which are computed in
step_players before the win check and must still match. For THIS action
stream (numpy default_rng(42)) no win occurs within 5000 ticks, and the test
asserts exactly that at the end — the gate compares all 5000 ticks and must
fail loudly if a change to the action stream (or the sim) ever ends the
episode early and silently weakens the comparison.

A failure here means a patch changed in-episode physics. Fix the patch,
never this test.
"""

import numpy as np
import pytest

from cogame_nmmo.sim import (ACT_HIGH, DEFAULT_WASM_PATH, PRISTINE_WASM_PATH,
                             MobaSim)

TICKS = 5000


@pytest.mark.skipif(not PRISTINE_WASM_PATH.exists(),
                    reason="run sim/build_sim.sh first")
def test_patched_matches_pristine():
    patched = MobaSim(seed=1, wasm_path=DEFAULT_WASM_PATH)
    pristine = MobaSim(seed=1, wasm_path=PRISTINE_WASM_PATH)

    assert patched.observations().tobytes() == pristine.observations().tobytes(), \
        "initial obs diverged (seeding mismatch: srand(1) != default stream?)"

    rng = np.random.default_rng(42)
    compared = 0
    for t in range(TICKS):
        acts = rng.integers(0, ACT_HIGH, size=(10, 6)).astype(np.float32)
        for sim in (patched, pristine):
            sim.set_actions(acts)
            sim.step()
        assert patched.rewards().tobytes() == pristine.rewards().tobytes(), \
            f"rewards diverged at tick {t}"
        if patched.done():
            # pristine auto-reset this tick (patched skips it, patch 0003);
            # post-win obs legitimately differ - stop comparing here
            break
        assert patched.observations().tobytes() == pristine.observations().tobytes(), \
            f"obs diverged at tick {t}"
        compared += 1

    # Floor: the gate must not silently weaken. This action stream is known
    # not to end the episode; all TICKS ticks must have been fully compared.
    assert not patched.done(), \
        "episode ended early - action stream changed? re-verify gate coverage"
    assert compared == TICKS, \
        f"only {compared}/{TICKS} ticks compared - gate coverage weakened"
