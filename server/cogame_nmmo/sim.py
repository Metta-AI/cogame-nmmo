"""wasmtime host for the vendored Puffer MOBA sim compiled to wasm.

The wasm module is a WASI reactor (emscripten STANDALONE_WASM --no-entry)
built by sim/build_sim.sh. See sim/shim.c for the exported API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wasmtime import (Config, Engine, Func, FuncType, Linker, Module, Store,
                      ValType, WasiConfig)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WASM_PATH = REPO_ROOT / "build" / "moba_sim.wasm"
PRISTINE_WASM_PATH = REPO_ROOT / "build" / "moba_sim_pristine.wasm"

NUM_AGENTS = 10
OBS_SIZE = 510  # 11*11*4 map crop + 26 self features (opaque contract)
NUM_ATNS = 6    # MultiDiscrete [7,7,3,2,2,2] delivered as 6 floats

# MultiDiscrete action space highs (exclusive), per column
ACT_HIGH = (7, 7, 3, 2, 2, 2)
# MultiDiscrete no-op: center velocity (3,3 -> 0,0), scan-all filter, no skills
NOOP_ACTION = [3, 3, 0, 0, 0, 0]

# Engine/Module compilation cached per wasm path so per-episode instantiation
# (one MobaSim per episode) doesn't recompile the module every time.
_MODULE_CACHE: dict[str, tuple[Engine, Module]] = {}


def _load_module(wasm_path: Path) -> tuple[Engine, Module]:
    key = str(wasm_path)
    if key not in _MODULE_CACHE:
        engine = Engine(Config())
        _MODULE_CACHE[key] = (engine, Module.from_file(engine, key))
    return _MODULE_CACHE[key]


class MobaSim:
    """One MOBA episode simulator instance hosted in wasm.

    ``seed`` is a 32-bit unsigned value; wider Python ints are masked to
    32 bits before being handed to the sim.
    """

    def __init__(self, seed: int = 1, num_agents: int = NUM_AGENTS,
                 wasm_path: str | Path = DEFAULT_WASM_PATH):
        wasm_path = Path(wasm_path)
        if not wasm_path.exists():
            raise FileNotFoundError(
                f"{wasm_path} not found - run sim/build_sim.sh first")
        self.num_agents = num_agents

        engine, module = _load_module(wasm_path)
        self._store = Store(engine)
        wasi = WasiConfig()
        wasi.inherit_stdout()  # sim printfs (glitch-state warnings etc.)
        wasi.inherit_stderr()
        self._store.set_wasi(wasi)

        linker = Linker(engine)
        linker.define_wasi()
        # -sALLOW_MEMORY_GROWTH emits this notification import; no-op host stub
        linker.define(
            self._store, "env", "emscripten_notify_memory_growth",
            Func(self._store, FuncType([ValType.i32()], []), lambda _idx: None))
        instance = linker.instantiate(self._store, module)
        self._exports = instance.exports(self._store)
        self._memory = self._exports["memory"]

        # WASI reactor: run emscripten static constructors before anything else
        self._exports["_initialize"](self._store)
        # moba_init takes an unsigned 32-bit seed; mask, then re-encode as the
        # signed i32 bit pattern the wasm ABI expects for the u32 parameter.
        seed = seed & 0xFFFFFFFF
        if seed >= 1 << 31:
            seed -= 1 << 32
        self._exports["moba_init"](self._store, seed, num_agents)

        self._obs_ptr = self._exports["obs_ptr"](self._store)
        self._act_ptr = self._exports["act_ptr"](self._store)
        self._rew_ptr = self._exports["rew_ptr"](self._store)

    # -- lockstep API ------------------------------------------------------

    def observations(self) -> np.ndarray:
        """Fresh (num_agents, 510) uint8 copy of the current observations."""
        raw = self._memory.read(
            self._store, self._obs_ptr,
            self._obs_ptr + self.num_agents * OBS_SIZE)
        return np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(
            self.num_agents, OBS_SIZE)

    def set_actions(self, actions: np.ndarray) -> None:
        """Write per-agent actions into the sim's action buffer.

        Validation contract at the wasm boundary (upstream's action decode
        has no bounds checks, and a NaN would propagate into map indexing):

        - shape must be (num_agents, 6), else ValueError
        - non-finite values (NaN/Inf) raise ValueError
        - finite values are sanitized: truncated toward zero (matching the
          sim's C ``(int)`` cast) and clamped per column to the MultiDiscrete
          range ``0 .. ACT_HIGH[col]-1``

        In-range integer-valued floats (what trained policies emit) are
        written through byte-identical.
        """
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, NUM_ATNS):
            raise ValueError(
                f"actions must be ({self.num_agents}, {NUM_ATNS}), "
                f"got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("actions contain NaN or Inf")
        high = np.asarray(ACT_HIGH, dtype=np.float32) - 1.0
        actions = np.clip(np.trunc(actions), 0.0, high).astype(np.float32)
        self._memory.write(self._store, actions.tobytes(), self._act_ptr)

    def step(self) -> None:
        self._exports["moba_step"](self._store)

    def reset(self) -> None:
        self._exports["moba_reset"](self._store)

    def rewards(self) -> np.ndarray:
        """Fresh (num_agents,) float32 copy of the last step's rewards.

        The sim writes this buffer only during ``step()`` (step_players);
        after ``reset()`` the values are stale from the previous episode's
        final tick until the first ``step()`` of the new episode.
        """
        raw = self._memory.read(
            self._store, self._rew_ptr,
            self._rew_ptr + self.num_agents * 4)
        return np.frombuffer(bytearray(raw), dtype=np.float32)

    def done(self) -> int:
        return self._exports["moba_done"](self._store)

    def fault(self) -> int:
        """Patch-0004 fault flag: nonzero when an upstream in-episode
        debug guard tripped (the guards used to exit() the process). The
        engine polls this each tick and ends the episode with
        end_reason "sim_fault". Always 0 on the pristine build."""
        return self._exports["moba_fault"](self._store)

    def winner(self) -> int:
        return self._exports["moba_winner"](self._store)

    def tick(self) -> int:
        return self._exports["moba_tick"](self._store)

    def agent_stat(self, pid: int, which: int) -> int:
        return self._exports["agent_stat"](self._store, pid, which)

    def state_digest(self) -> int:
        """u32 FNV-1a digest of hero x/y/health + ancient healths at the
        current tick (sim/shim_common.h). A recorded episode's digest
        must equal the viewer core's viewer_state_digest() after
        re-simulating to the same tick."""
        return self._exports["state_digest"](self._store) & 0xFFFFFFFF

    def ancient_health(self, team: int) -> float:
        return self._exports["ancient_health"](self._store, team)
