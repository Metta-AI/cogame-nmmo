"""Baseline cogame-nmmo player: ``python -m players.baseline_player``.

Runs the upstream pretrained policy (vendored moba_weights.bin) through
build/moba_brain.wasm — puffernet compiled to wasm, weights embedded —
hosted via wasmtime exactly like the sim host (module cache,
``_initialize`` call, memory-growth stub; see cogame_nmmo.sim.MobaSim).

One wasm instance holds 10 independent batch-1 nets, one per possible
agent index, each with its own MinGRU recurrent state. The policy maps
seat heroes to brain instances by hero index within the seat, so a
5-hero team seat gets 5 isolated recurrent states from one process.

Inference is stochastic: puffernet's forward samples from the softmax
using the wasm module's own libc rand() stream (see sim/brain_shim.c).
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

from .client import run_policy_main, seed_from_env

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAIN_WASM_PATH = REPO_ROOT / "build" / "moba_brain.wasm"

NUM_BRAINS = 10        # independent net instances (one per agent index)
OBS_SIZE = 510         # uint8 obs per hero (sim contract)
NUM_ATNS = 6           # MultiDiscrete [7,7,3,2,2,2]
EXPECTED_PARAMS = 95_616  # moba_weights.bin float32 count (moba.c:5)

DEFAULT_SEED = 1       # == upstream's unseeded libc rand stream

# Engine/Module compilation cached per wasm path (one MobaBrain per
# process normally, but tests build several).
_MODULE_CACHE: dict[str, tuple[Engine, Module]] = {}


def _load_module(wasm_path: Path) -> tuple[Engine, Module]:
    key = str(wasm_path)
    if key not in _MODULE_CACHE:
        engine = Engine(Config())
        _MODULE_CACHE[key] = (engine, Module.from_file(engine, key))
    return _MODULE_CACHE[key]


class MobaBrain:
    """wasmtime host for build/moba_brain.wasm (see sim/brain_shim.c)."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 wasm_path: str | Path = DEFAULT_BRAIN_WASM_PATH):
        wasm_path = Path(wasm_path)
        if not wasm_path.exists():
            raise FileNotFoundError(
                f"{wasm_path} not found - run sim/build_brain.sh first")

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
        params = self._exports["brain_init"](self._store, seed)
        if params != EXPECTED_PARAMS:
            raise RuntimeError(
                f"brain_init reported {params} weight params, "
                f"expected {EXPECTED_PARAMS} - wasm/weights mismatch?")

        self._obs_ptrs = [self._exports["brain_obs_ptr"](self._store, i)
                          for i in range(NUM_BRAINS)]
        self._act_ptrs = [self._exports["brain_act_ptr"](self._store, i)
                          for i in range(NUM_BRAINS)]

    def forward(self, agent_idx: int, obs_bytes: bytes) -> list[int]:
        """One inference step for one agent: 510 obs bytes -> 6 action ints.

        Advances that agent's recurrent state and the module's shared
        rand() stream (call order matters for reproducibility).
        """
        if not 0 <= agent_idx < NUM_BRAINS:
            raise ValueError(f"agent_idx must be 0..{NUM_BRAINS - 1}, "
                             f"got {agent_idx}")
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
    """policy(tick, obs_rows): hero i within the seat -> brain instance i."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 wasm_path: str | Path = DEFAULT_BRAIN_WASM_PATH):
        self.brain = MobaBrain(seed=seed, wasm_path=wasm_path)

    def __call__(self, tick: int, obs_rows: list) -> list:
        return [self.brain.forward(i, bytes(row))
                for i, row in enumerate(obs_rows)]


def policy_from_env() -> BaselinePolicy:
    return BaselinePolicy(seed_from_env(default=DEFAULT_SEED))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
