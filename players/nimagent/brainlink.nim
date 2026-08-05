# Baseline pretrained-net seats for the Nim bench: puffernet MMONet
# logits from brain_c.c + T=1.0 categorical sampling here.

import std/[math, os, random]

const RepoRoot = currentSourcePath().parentDir.parentDir.parentDir

{.passc: "-I" & RepoRoot & "/vendor/upstream".}
{.compile: currentSourcePath().parentDir / "brain_c.c".}

proc brainCreate(path: cstring, numAgents: cint): pointer
  {.importc: "brain_create".}
proc brainForward(handle: pointer, obs: ptr uint8, terminals: ptr cfloat,
                  outLogits: ptr cfloat) {.importc: "brain_forward".}

const WeightsPath* =
  RepoRoot / "vendor/upstream/resources/nmmo3/nmmo3_weights.bin"

type
  Brain* = object
    handle: pointer
    numAgents: int
    logits: seq[cfloat]
    terms: seq[cfloat]
    rng: Rand

proc initBrain*(numAgents: int, seed: int,
                path = WeightsPath): Brain =
  result.handle = brainCreate(cstring(path), cint(numAgents))
  doAssert result.handle != nil, "weights load failed: " & path
  result.numAgents = numAgents
  result.logits = newSeq[cfloat](numAgents * 27)
  result.terms = newSeq[cfloat](numAgents)
  result.rng = initRand(seed)

proc actions*(b: var Brain, obs: ptr UncheckedArray[uint8],
              resets: openArray[bool]): seq[int] =
  ## One action per agent from T=1.0 softmax over the 26 logits.
  ## obs: contiguous numAgents*1707 bytes.
  for i in 0 ..< b.numAgents:
    if resets[i]: b.terms[i] = 1.0
  brainForward(b.handle, addr obs[0], addr b.terms[0], addr b.logits[0])
  for i in 0 ..< b.numAgents:
    var mx = -1e30
    for k in 0 ..< 26:
      mx = max(mx, float(b.logits[i * 27 + k]))
    var probs: array[26, float]
    var z = 0.0
    for k in 0 ..< 26:
      probs[k] = exp(float(b.logits[i * 27 + k]) - mx)
      z += probs[k]
    var u = b.rng.rand(1.0) * z
    var act = 25
    for k in 0 ..< 26:
      u -= probs[k]
      if u <= 0: act = k; break
    result.add act
