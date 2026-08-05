"""Patch 0003 (per-seat opt-in truthful entity bytes) behavior gate.

Twin-run design: two instances of the PATCHED wasm, same seed, identical
scripted action log. Actions never depend on observations, so the two
runs' simulation states are identical by construction; the ONLY permitted
difference is the opted-in seat's observation encoding. That yields three
decisive assertions with no ground-truth oracle needed:

  1. NON-INTERFERENCE: every non-opted agent's obs stream is byte-identical
     between the runs (the compat guarantee, per tick, per byte).
  2. SUBTRACTIVE-ONLY: for the opted agent, wherever the clean run has
     NONZERO entity bytes they equal the legacy run's bytes at that cell —
     clean mode may only zero stale data, never alter live data.
  3. EFFECTIVE: there exist cells where the legacy run holds nonzero entity
     bytes and the clean run holds zeros (residue actually removed).

Plus state digests must match at the end (physics untouched).

The cross-build default-path guarantee (patched-with-0003, nobody opted, ==
pristine) is already enforced by tests/test_fidelity.py.
"""

import os

import numpy as np
import pytest

from cogame_nmmo.sim import (ACT_HIGH, DEFAULT_WASM_PATH, NUM_AGENTS,
                             NmmoSim)

TICKS = 600
OPTED = 3
NOT_BUILT = "sim wasm not built - run sim/build_sim.sh first"
NO_EXPORT = ("sim wasm predates patch 0003 (nmmo_set_obs_clean export "
             "missing) - run sim/build_sim.sh")

TILE_BYTES = 11 * 15 * 10


def _entity_mask() -> np.ndarray:
    """Bool mask over one 1707-byte obs selecting tile entity bytes 4..9."""
    mask = np.zeros(1707, dtype=bool)
    idx = np.arange(TILE_BYTES)
    mask[:TILE_BYTES] = (idx % 10) >= 4
    return mask


def test_opt_in_is_subtractive_and_isolated():
    if not os.path.exists(DEFAULT_WASM_PATH):
        if os.environ.get("COGAME_REQUIRE_WASM_BUILD"):
            pytest.fail(NOT_BUILT + " (COGAME_REQUIRE_WASM_BUILD is set)")
        pytest.skip(NOT_BUILT)

    legacy = NmmoSim(seed=7, wasm_path=DEFAULT_WASM_PATH)
    clean = NmmoSim(seed=7, wasm_path=DEFAULT_WASM_PATH)
    try:
        clean._exports["nmmo_set_obs_clean"]
    except KeyError:
        if os.environ.get("COGAME_REQUIRE_WASM_BUILD"):
            pytest.fail(NO_EXPORT + " (COGAME_REQUIRE_WASM_BUILD is set)")
        pytest.skip(NO_EXPORT)
    clean.set_obs_clean(OPTED, True)

    rng = np.random.default_rng(123)
    ent = _entity_mask()
    removed_any = False

    for tick in range(TICKS):
        obs_l = legacy.observations()
        obs_c = clean.observations()

        # 1. non-interference: non-opted agents byte-identical
        for pid in range(NUM_AGENTS):
            if pid == OPTED:
                continue
            assert obs_l[pid].tobytes() == obs_c[pid].tobytes(), \
                f"tick {tick}: non-opted agent {pid} obs diverged"

        # 2. subtractive-only on the opted agent's entity bytes; non-entity
        #    bytes (terrain/items/scalars) must be identical outright.
        l = obs_l[OPTED]
        c = obs_c[OPTED]
        assert l[~ent].tobytes() == c[~ent].tobytes(), \
            f"tick {tick}: opted agent non-entity bytes diverged"
        le, ce = l[ent], c[ent]
        nz = ce != 0
        assert np.array_equal(ce[nz], le[nz]), \
            f"tick {tick}: clean run altered (not just zeroed) entity bytes"
        if np.any((le != 0) & (ce == 0)):
            removed_any = True

        acts = rng.integers(0, ACT_HIGH,
                            size=(NUM_AGENTS, 1)).astype(np.float32)
        legacy.set_actions(acts)
        clean.set_actions(acts)
        legacy.step()
        clean.step()

    # 3. the patch actually removes residue for the opted seat
    assert removed_any, \
        "no residue was ever removed for the opted agent - patch inert?"

    assert legacy.state_digest() == clean.state_digest(), \
        "state digests diverged - patch touched physics, must be obs-only"
