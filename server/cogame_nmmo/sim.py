"""wasmtime host for the vendored Puffer NMMO3 sim compiled to wasm.

The wasm module is a WASI reactor (emscripten STANDALONE_WASM --no-entry)
built by sim/build_sim.sh. See sim/shim.c for the exported API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wasmtime import (Config, Engine, Func, FuncType, Linker, Module, Store,
                      ValType, WasiConfig)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WASM_PATH = REPO_ROOT / "build" / "nmmo3_sim.wasm"
PRISTINE_WASM_PATH = REPO_ROOT / "build" / "nmmo3_sim_pristine.wasm"

NUM_AGENTS = 8   # default seat count (num_agents is elastic upstream)
# 11x15 tile window x 10 bytes + 47 self scalars + 10 reward bytes
# (opaque contract; allocation site nmmo3.h:797)
OBS_SIZE = 11 * 15 * 10 + 47 + 10  # = 1707
NUM_ATNS = 1     # one 26-way discrete action, delivered as 1 float

# Discrete action high (exclusive)
ACT_HIGH = (26,)
# No-op: ATN_NOOP (nmmo3.h:50)
NOOP_ACTION = [4]

# Per-agent stat codes for agent_stat() (sim/shim.c)
STAT_CUM_MIN_COMB_PROF = 0
STAT_DEATHS = 1
STAT_COMB_LVL = 2
STAT_PROF_LVL = 3
STAT_LIFE_MIN_COMB_PROF = 4
STAT_GOLD = 5
STAT_TIME_ALIVE = 6
STAT_HP = 7

# Engine/Module compilation cached per wasm path so per-episode instantiation
# (one NmmoSim per episode) doesn't recompile the module every time.
_MODULE_CACHE: dict[str, tuple[Engine, Module]] = {}


def _load_module(wasm_path: Path) -> tuple[Engine, Module]:
    key = str(wasm_path)
    if key not in _MODULE_CACHE:
        engine = Engine(Config())
        _MODULE_CACHE[key] = (engine, Module.from_file(engine, key))
    return _MODULE_CACHE[key]


class NmmoSim:
    """One NMMO3 episode simulator instance hosted in wasm.

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
        wasi.inherit_stdout()  # sim printfs, if any
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
        # nmmo_init takes an unsigned 32-bit seed; mask, then re-encode as the
        # signed i32 bit pattern the wasm ABI expects for the u32 parameter.
        seed = seed & 0xFFFFFFFF
        if seed >= 1 << 31:
            seed -= 1 << 32
        self._exports["nmmo_init"](self._store, seed, num_agents)

        self._obs_ptr = self._exports["obs_ptr"](self._store)
        self._act_ptr = self._exports["act_ptr"](self._store)
        self._rew_ptr = self._exports["rew_ptr"](self._store)
        self._term_ptr = self._exports["term_ptr"](self._store)

    # -- lockstep API ------------------------------------------------------

    def observations(self) -> np.ndarray:
        """Fresh (num_agents, 1707) uint8 copy of the current observations.

        The underlying sim buffer is persistent and never cleared between
        ticks (trained-on quirk: per-tile entity bytes go stale when a tile
        empties); this method only copies, it must never zero anything.
        """
        raw = self._memory.read(
            self._store, self._obs_ptr,
            self._obs_ptr + self.num_agents * OBS_SIZE)
        return np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(
            self.num_agents, OBS_SIZE)

    def set_actions(self, actions: np.ndarray) -> None:
        """Write per-agent actions into the sim's action buffer.

        Validation contract at the wasm boundary (upstream's action decode
        has no bounds checks — `int action = env->actions[pid]` at
        nmmo3.h:1981 is a bare float->int cast):

        - shape must be (num_agents, 1), else ValueError
        - non-finite values (NaN/Inf) raise ValueError
        - finite values are sanitized: truncated toward zero (matching the
          sim's C ``(int)`` cast) and clamped to the discrete range 0..25

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
        self._exports["nmmo_step"](self._store)

    def reset(self) -> None:
        self._exports["nmmo_reset"](self._store)

    def rewards(self) -> np.ndarray:
        """Fresh (num_agents,) float32 copy of the last step's rewards."""
        raw = self._memory.read(
            self._store, self._rew_ptr,
            self._rew_ptr + self.num_agents * 4)
        return np.frombuffer(bytearray(raw), dtype=np.float32)

    def terminals(self) -> np.ndarray:
        """Fresh (num_agents,) float32 copy of THIS tick's done flags.

        1.0 means the agent's life ended this tick (attack death or 500-tick
        stagnation reset) and it respawns in place; recurrent policies must
        zero that agent's state. The shim clears the buffer before each
        c_step (upstream's vecenv did the same externally), so flags never
        accumulate across ticks. Valid after step(); all zeros at tick 0.
        """
        raw = self._memory.read(
            self._store, self._term_ptr,
            self._term_ptr + self.num_agents * 4)
        return np.frombuffer(bytearray(raw), dtype=np.float32)

    def dones(self) -> list[bool]:
        """terminals() as per-agent bools (the wire-protocol `resets`)."""
        return [bool(t) for t in self.terminals() != 0.0]

    def fault(self) -> int:
        """Patch-0002 fault flag: nonzero when an upstream in-episode
        debug guard tripped (the guards used to exit() the process). The
        engine polls this each tick and ends the episode with
        end_reason "sim_fault". Always 0 on the pristine build."""
        return self._exports["nmmo_fault"](self._store)

    def tick(self) -> int:
        return self._exports["nmmo_tick"](self._store)

    def agent_stat(self, pid: int, which: int) -> int:
        return self._exports["agent_stat"](self._store, pid, which)

    def score(self, pid: int) -> int:
        """Ranking score: cumulative min(comb_lvl, prof_lvl) over the
        agent's ended lives plus the current life's min (0 while dead-
        awaiting-respawn). See sim/shim.c nmmo_score()."""
        return self._exports["nmmo_score"](self._store, pid)

    def state_digest(self) -> int:
        """u32 FNV-1a digest of player r/c/hp/comb/prof + env rng + tick
        (sim/shim_common.h). A recorded episode's digest must equal the
        viewer core's digest after re-simulating to the same tick."""
        return self._exports["state_digest"](self._store) & 0xFFFFFFFF
