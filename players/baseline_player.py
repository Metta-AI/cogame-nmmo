"""Baseline cogame-nmmo player: ``python -m players.baseline_player``.

Runs the upstream pretrained NMMO3 policy (MMONet + vendored
nmmo3_weights.bin) through build/nmmo3_brain.wasm — the nmmo3.c demo net
ported to a wasm reactor with the weights embedded — hosted via wasmtime
exactly like the sim host (module cache, ``_initialize`` call,
memory-growth stub; see cogame_nmmo.sim.NmmoSim).

One wasm instance holds up to ``num_agents`` independent batch-1 nets,
one per hero index within the seat, each with its own 4-layer MinGRU
recurrent state. Protocol v2 ``resets`` flags drive per-hero state
zeroing: when ``resets[j]`` is true the policy calls
``brain_reset_state(j)`` BEFORE forwarding that tick's obs — exactly the
demo forward()'s terminals handling (vendored nmmo3.c:71-78).

Inference is stochastic: the demo samples from the softmax using the
wasm module's own libc rand() stream (see sim/brain_shim.c).
``COGAME_PLAYER_SEED`` seeds it; the default seed 1 reproduces THIS wasm
module's srand(1) stream (emscripten's musl libc). That matches a native
upstream demo run only if the native binary links the same libc —
glibc/macOS rand() differ, so a native-libc reference trace WILL diverge
in the sampled actions even with bit-exact logits (a libc difference,
not a port bug). Given a seed and a fixed call order the player is
fully deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from wasmtime import (Config, Engine, Func, FuncType, Linker, Module, Store,
                      ValType, WasiConfig)

from .client import PlayerError, run_policy_main, seed_from_env

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAIN_WASM_PATH = REPO_ROOT / "build" / "nmmo3_brain.wasm"

DEFAULT_NUM_BRAINS = 8    # net instances (>= heroes per seat; world default)
OBS_SIZE = 1707           # uint8 obs per hero (sim contract)
NUM_ATNS = 1              # one 26-way discrete action
EXPECTED_PARAMS = 4_430_976  # nmmo3_weights.bin float32 count (17,723,904 B)

DEFAULT_SEED = 1          # == upstream's unseeded libc rand stream

# brain_init failure codes (sim/brain_shim.c BRAIN_ERR_*), mapped to
# honest messages; a non-negative return is the loaded param count.
INIT_ERRORS = {
    -1: "brain already initialized in this wasm instance",
    -2: "num_agents out of range (must be 1..32)",
    -3: "embedded weights blob is not nmmo3_weights.bin (size mismatch)",
    -4: "allocation failure inside the brain wasm",
}

# Engine/Module compilation cached per wasm path (one NmmoBrain per
# process normally, but tests build several).
_MODULE_CACHE: dict[str, tuple[Engine, Module]] = {}


def _load_module(wasm_path: Path) -> tuple[Engine, Module]:
    key = str(wasm_path)
    if key not in _MODULE_CACHE:
        engine = Engine(Config())
        _MODULE_CACHE[key] = (engine, Module.from_file(engine, key))
    return _MODULE_CACHE[key]


class NmmoBrain:
    """wasmtime host for build/nmmo3_brain.wasm (see sim/brain_shim.c)."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_agents: int = DEFAULT_NUM_BRAINS,
                 wasm_path: str | Path = DEFAULT_BRAIN_WASM_PATH):
        wasm_path = Path(wasm_path)
        if not wasm_path.exists():
            raise FileNotFoundError(
                f"{wasm_path} not found - run sim/build_brain.sh first")
        self.num_agents = num_agents

        engine, module = _load_module(wasm_path)
        self._store = Store(engine)
        wasi = WasiConfig()
        wasi.inherit_stdout()
        wasi.inherit_stderr()
        self._store.set_wasi(wasi)

        linker = Linker(engine)
        linker.define_wasi()
        # -sALLOW_MEMORY_GROWTH emits this notification import; no-op stub
        linker.define(
            self._store, "env", "emscripten_notify_memory_growth",
            Func(self._store, FuncType([ValType.i32()], []), lambda _idx: None))
        instance = linker.instantiate(self._store, module)
        self._exports = instance.exports(self._store)
        self._memory = self._exports["memory"]

        # WASI reactor: run emscripten static constructors first
        self._exports["_initialize"](self._store)

        seed = seed & 0xFFFFFFFF
        if seed >= 1 << 31:  # u32 -> the signed i32 bit pattern wasm expects
            seed -= 1 << 32
        params = self._exports["brain_init"](self._store, seed, num_agents)
        if params < 0:
            detail = INIT_ERRORS.get(params, f"unknown error code {params}")
            raise RuntimeError(f"brain_init failed: {detail}")
        if params != EXPECTED_PARAMS:
            raise RuntimeError(
                f"brain_init reported {params} weight params, "
                f"expected {EXPECTED_PARAMS} - wasm/weights mismatch?")

        self._obs_ptrs = [self._exports["brain_obs_ptr"](self._store, i)
                          for i in range(num_agents)]
        self._act_ptrs = [self._exports["brain_act_ptr"](self._store, i)
                          for i in range(num_agents)]

    def _check_idx(self, agent_idx: int) -> None:
        if not 0 <= agent_idx < self.num_agents:
            raise ValueError(f"agent_idx must be 0..{self.num_agents - 1}, "
                             f"got {agent_idx}")

    def reset_state(self, agent_idx: int) -> None:
        """Zero one agent's MinGRU state (all layers) — the demo's
        terminals handling (nmmo3.c:71-78). Call when the wire ``resets``
        flag fires for this agent, before that tick's forward."""
        self._check_idx(agent_idx)
        rc = self._exports["brain_reset_state"](self._store, agent_idx)
        if rc != 0:
            raise RuntimeError(f"brain_reset_state({agent_idx}) failed ({rc})")

    def forward(self, agent_idx: int, obs_bytes: bytes) -> list[int]:
        """One inference step for one agent: 1707 obs bytes -> [action].

        Advances that agent's recurrent state and the module's shared
        rand() stream (call order matters for reproducibility).
        """
        self._check_idx(agent_idx)
        if len(obs_bytes) != OBS_SIZE:
            raise ValueError(
                f"obs must be {OBS_SIZE} bytes, got {len(obs_bytes)}")
        self._memory.write(self._store, obs_bytes, self._obs_ptrs[agent_idx])
        rc = self._exports["brain_forward"](self._store, agent_idx)
        if rc != 0:
            raise RuntimeError(f"brain_forward({agent_idx}) failed ({rc})")
        raw = self._memory.read(
            self._store, self._act_ptrs[agent_idx],
            self._act_ptrs[agent_idx] + NUM_ATNS * 4)
        return np.frombuffer(bytearray(raw), dtype=np.int32).tolist()


class BaselinePolicy:
    """policy(tick, obs_rows, resets): hero i in the seat -> brain i."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 num_agents: int = DEFAULT_NUM_BRAINS,
                 wasm_path: str | Path = DEFAULT_BRAIN_WASM_PATH):
        self.brain = NmmoBrain(seed=seed, num_agents=num_agents,
                               wasm_path=wasm_path)

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list:
        if len(obs_rows) > self.brain.num_agents:
            raise PlayerError(
                f"seat has {len(obs_rows)} heroes, brain built for "
                f"{self.brain.num_agents} - construct BaselinePolicy with "
                f"num_agents >= the seat's heroes_per_seat")
        actions = []
        for i, row in enumerate(obs_rows):
            if resets[i]:
                # zero hero i's recurrent state BEFORE consuming the
                # first obs of its new life (protocol v2 contract)
                self.brain.reset_state(i)
            actions.append(self.brain.forward(i, bytes(row)))
        return actions


def policy_from_env() -> BaselinePolicy:
    return BaselinePolicy(seed_from_env(default=DEFAULT_SEED))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
