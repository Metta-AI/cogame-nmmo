"""ctypes wrapper around the natively-compiled patched sim shim.

Same C sources and export surface as the league wasm
(build/nmmo3_sim.wasm); compiled by:

    clang -O3 -I build/src-patched -dynamiclib sim/shim.c \
        -o train/nmmo3_sim_native.dylib

Bit-exactness vs the wasm build is asserted by tools/check_native_parity
(state_digest after N ticks, several seeds).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parent / "nmmo3_sim_native.dylib"


class NativeNmmo:
    def __init__(self, seed: int, num_agents: int):
        self.lib = ctypes.CDLL(str(LIB))
        lib = self.lib
        lib.nmmo_init.argtypes = [ctypes.c_uint, ctypes.c_int]
        lib.obs_size.restype = ctypes.c_int
        lib.obs_ptr.restype = ctypes.c_void_p
        lib.act_ptr.restype = ctypes.c_void_p
        lib.rew_ptr.restype = ctypes.c_void_p
        lib.term_ptr.restype = ctypes.c_void_p
        lib.agent_stat.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.agent_stat.restype = ctypes.c_int
        lib.nmmo_score.argtypes = [ctypes.c_int]
        lib.nmmo_score.restype = ctypes.c_int
        lib.state_digest.restype = ctypes.c_uint
        lib.nmmo_tick.restype = ctypes.c_int
        lib.nmmo_fault.restype = ctypes.c_int

        self.num_agents = num_agents
        lib.nmmo_init(seed, num_agents)
        self.obs_size = lib.obs_size()
        n = num_agents
        self.obs = np.ctypeslib.as_array(
            ctypes.cast(lib.obs_ptr(),
                        ctypes.POINTER(ctypes.c_ubyte)),
            shape=(n, self.obs_size))
        self.act = np.ctypeslib.as_array(
            ctypes.cast(lib.act_ptr(), ctypes.POINTER(ctypes.c_float)),
            shape=(n, 1))
        self.rew = np.ctypeslib.as_array(
            ctypes.cast(lib.rew_ptr(), ctypes.POINTER(ctypes.c_float)),
            shape=(n,))
        self.term = np.ctypeslib.as_array(
            ctypes.cast(lib.term_ptr(), ctypes.POINTER(ctypes.c_float)),
            shape=(n,))

    def step(self, actions: np.ndarray) -> None:
        self.act[:, 0] = actions
        self.lib.nmmo_step()

    def digest(self) -> int:
        return int(self.lib.state_digest())

    def score(self, pid: int) -> float:
        return self.lib.nmmo_score(pid)

    def stat(self, pid: int, which: int) -> int:
        return self.lib.agent_stat(pid, which)
