# Native link of the bit-exact league sim for Nim tests and benches.
#
# Compiles sim/shim.c against build/src-patched with rand_r renamed to
# the musl implementation (train/musl_rand.c) — the exact recipe whose
# state digest was verified identical to the league wasm
# (train/native_env.py provenance, 2026-08-04). Run sim/apply_patches.sh
# once before building.
#
# One sim instance per process (the shim's env is a C global).

import std/os

const
  RepoRoot = currentSourcePath().parentDir.parentDir.parentDir

{.passc: "-I" & RepoRoot & "/build/src-patched".}
{.compile(RepoRoot & "/sim/shim.c",
          "-O2 -Drand_r=cogame_musl_rand_r -Wno-unknown-attributes").}
{.compile: RepoRoot & "/train/musl_rand.c".}

proc nmmoInit*(seed: cuint, numAgents: cint) {.importc: "nmmo_init".}
proc nmmoStep*() {.importc: "nmmo_step".}
proc nmmoTick*(): cint {.importc: "nmmo_tick".}
proc nmmoFault*(): cint {.importc: "nmmo_fault".}
proc obsSizeC*(): cint {.importc: "obs_size".}
proc obsPtr*(): ptr UncheckedArray[uint8] {.importc: "obs_ptr".}
proc actPtr*(): ptr UncheckedArray[cfloat] {.importc: "act_ptr".}
proc rewPtr*(): ptr UncheckedArray[cfloat] {.importc: "rew_ptr".}
proc termPtr*(): ptr UncheckedArray[cfloat] {.importc: "term_ptr".}
proc agentStat*(pid, which: cint): cint {.importc: "agent_stat".}
proc nmmoScore*(pid: cint): cint {.importc: "nmmo_score".}
proc stateDigest*(): cuint {.importc: "state_digest".}
proc debugBufPtr*(): ptr UncheckedArray[cint] {.importc: "debug_buf_ptr".}
proc debugNearby*(pid, radius: cint): cint {.importc: "debug_nearby".}

type
  GtEnemy* = object
    dr*, dc*, level*, ranged*, hp*: int

proc groundTruth*(pid: int, radius = 6): seq[GtEnemy] =
  ## debug_nearby export: real enemies near pid (forensics/gates only).
  let n = debugNearby(cint(pid), cint(radius))
  let buf = debugBufPtr()
  for i in 0 ..< int(n):
    result.add GtEnemy(
      dr: int(buf[i * 5 + 0]), dc: int(buf[i * 5 + 1]),
      level: int(buf[i * 5 + 2]), ranged: int(buf[i * 5 + 3]),
      hp: int(buf[i * 5 + 4]))
