"""Obs-contract snapshot: pins the byte-level observation contract.

The hex constants below are sha256 digests of the full (8, 1707) obs
buffer for seed 42 at tick 0 and at tick 100 under a fixed random action
script. They pin CONFIG + LAYOUT drift — an accidental change to any
trained-on env value in sim/shim_common.h (world size, entity counts,
windows, reward weights), to the obs byte layout, or to the seeding path
changes these bytes and fails loudly here. They do NOT prove physics
correctness against upstream — the fidelity gate (tests/test_fidelity.py)
does that.

If this fails after a DELIBERATE, design-approved config change,
recompute the constants: rebuild the wasm (`bash sim/build_sim.sh`),
run the ``_compute`` helper below (e.g. ``uv run python -c "from
tests.test_obs_contract import _compute; _compute()"``) and paste the
two printed digests over the constants. The pins assume the CI
toolchain (emsdk 6.0.5, musl libc): a different emcc/libc can change
the bytes without any config drift. If the test fails any other time,
the sim contract drifted — fix the code, not the constants.
"""

import hashlib

import numpy as np
import pytest

from cogame_nmmo.sim import (ACT_HIGH, DEFAULT_WASM_PATH, NUM_AGENTS,
                             OBS_SIZE, NmmoSim)

SNAPSHOT_SEED = 42
ACTION_SCRIPT_SEED = 123
SNAPSHOT_TICKS = 100

# sha256 over observations().tobytes() (8 agents x 1707 bytes, row-major)
TICK0_OBS_SHA256 = \
    "f126d207c27915ef23ed18917262f522caf748c6b7403a6d0826e7727f75e0ec"
TICK100_OBS_SHA256 = \
    "5aa71f8e1ac2e50fecf61fd8860d41f8b01594f96c29c8dcabfb3602052efd2c"
# state_digest() after the same 100 ticks (FNV-1a, sim/shim_common.h)
TICK100_STATE_DIGEST = 0x7865A5E0


def _compute():
    """Print fresh pin values (see the module docstring for when this is
    legitimate). Runs the exact snapshot script the test runs."""
    sim = NmmoSim(seed=SNAPSHOT_SEED)
    print("TICK0_OBS_SHA256 =",
          hashlib.sha256(sim.observations().tobytes()).hexdigest())
    rng = np.random.default_rng(ACTION_SCRIPT_SEED)
    for _ in range(SNAPSHOT_TICKS):
        acts = rng.integers(0, ACT_HIGH,
                            size=(NUM_AGENTS, 1)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
    print("TICK100_OBS_SHA256 =",
          hashlib.sha256(sim.observations().tobytes()).hexdigest())
    print(f"TICK100_STATE_DIGEST = 0x{sim.state_digest():08X}")


@pytest.mark.skipif(not DEFAULT_WASM_PATH.exists(),
                    reason="run sim/build_sim.sh first")
def test_obs_snapshot_pins_config_and_layout():
    sim = NmmoSim(seed=SNAPSHOT_SEED)
    obs0 = sim.observations()
    assert obs0.shape == (NUM_AGENTS, OBS_SIZE)
    assert hashlib.sha256(obs0.tobytes()).hexdigest() == TICK0_OBS_SHA256

    rng = np.random.default_rng(ACTION_SCRIPT_SEED)
    for _ in range(SNAPSHOT_TICKS):
        acts = rng.integers(0, ACT_HIGH,
                            size=(NUM_AGENTS, 1)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()

    obs100 = sim.observations()
    assert hashlib.sha256(obs100.tobytes()).hexdigest() == TICK100_OBS_SHA256
    assert sim.state_digest() == TICK100_STATE_DIGEST

    # Layout tripwire for the trained-on write quirk: the reward block
    # writes 9 of its 10 bytes; byte 1706 is allocated but never written
    # (compute_all_obs, nmmo3.h:1010-1018) and the buffer is calloc'd once,
    # so it must read 0 forever. If this ever flips, the obs layout moved.
    assert (obs0[:, 1706] == 0).all()
    assert (obs100[:, 1706] == 0).all()
