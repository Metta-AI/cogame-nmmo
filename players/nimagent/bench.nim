# 8-seat interleaved bench in the bit-exact native sim:
# even seats (0,2,4,6) = pretrained baseline net, odd seats (1,3,5,7)
# = the Nim agent. Seat score = nmmo_score/(deaths+1) at the final
# tick (the league formula). Fatal-tick death classification for the
# Nim seats via the ground-truth export (same classes as
# tools/dancer_bench.py).
#
# Usage: bench [nseeds] [ticks] [seed0]

import std/[os, strformat, strutils, tables]
import simlink, percept, world, executive, brainlink

proc classifyDeath(gt: seq[GtEnemy], hpPrev: int): string =
  var mel, bow = 0
  var maxLvl = 0
  for e in gt:
    if abs(e.dr) + abs(e.dc) <= 1 and e.ranged == 0:
      inc mel
      maxLvl = max(maxLvl, e.level)
    if (e.dr == 0 or e.dc == 0) and max(abs(e.dr), abs(e.dc)) <= 4 and
        e.ranged == 1:
      inc bow
  if bow > 0 and (mel == 0 or hpPrev > 60): "bow"
  elif mel >= 2: "melee-multi"
  elif mel == 1:
    if maxLvl <= 4: "melee-lo"
    elif maxLvl <= 9: "melee-mid"
    else: "melee-hi"
  else: "unclear"

proc main() =
  let nseeds = if paramCount() >= 1: parseInt(paramStr(1)) else: 8
  let ticks = if paramCount() >= 2: parseInt(paramStr(2)) else: 1500
  let seed0 = if paramCount() >= 3: parseInt(paramStr(3)) else: 1

  var allNim, allBase: seq[float]
  var deathClasses = initCountTable[string]()
  var totDeaths = 0
  var maxComb = 0

  for si in 0 ..< nseeds:
    let seed = seed0 + si
    nmmoInit(cuint(seed), 8)
    let obs = obsPtr()
    let act = actPtr()
    let term = termPtr()
    let baseIdx = [0, 2, 4, 6]
    let nimIdx = [1, 3, 5, 7]
    var brain = initBrain(4, seed * 31 + 7)
    var minds: array[4, Mind]
    for i in 0 ..< 4:
      minds[i] = initMind(1000 + seed, i)
    var brainObs = newSeq[uint8](4 * ObsSize)
    var prevHp: array[4, int]
    for i in 0 ..< 4: prevHp[i] = 99
    var buf: array[ObsSize, uint8]

    for t in 0 ..< ticks:
      # baseline seats
      var resets: array[4, bool]
      for k in 0 ..< 4:
        let pid = baseIdx[k]
        copyMem(addr brainObs[k * ObsSize], addr obs[pid * ObsSize],
                ObsSize)
        resets[k] = term[pid] > 0.5
      let bacts = brain.actions(
        cast[ptr UncheckedArray[uint8]](addr brainObs[0]), resets)
      for k in 0 ..< 4:
        act[baseIdx[k]] = cfloat(bacts[k])
      # nim seats
      for k in 0 ..< 4:
        let pid = nimIdx[k]
        if term[pid] > 0.5:
          minds[k].tick = t
          minds[k].reset()
        copyMem(addr buf[0], addr obs[pid * ObsSize], ObsSize)
        act[pid] = cfloat(minds[k].act(buf))
      # pre-step ground truth for fatal-tick classification
      var gtNow: array[4, seq[GtEnemy]]
      for k in 0 ..< 4:
        gtNow[k] = groundTruth(nimIdx[k])
      nmmoStep()
      for k in 0 ..< 4:
        let hp = int(agentStat(cint(nimIdx[k]), 7))
        if hp == 0 and prevHp[k] > 0:
          deathClasses.inc classifyDeath(gtNow[k], prevHp[k])
        if term[nimIdx[k]] > 0.5 and hp == 99:
          deathClasses.inc "stagnation"
        prevHp[k] = hp

    var nimScores, baseScores: seq[float]
    var deaths: array[4, int]
    var combs: array[4, int]
    for k in 0 ..< 4:
      let pid = nimIdx[k]
      nimScores.add float(nmmoScore(cint(pid))) /
        float(agentStat(cint(pid), 1) + 1)
      deaths[k] = int(agentStat(cint(pid), 1))
      combs[k] = int(agentStat(cint(pid), 2))
      totDeaths += deaths[k]
      maxComb = max(maxComb, combs[k])
      let bp = baseIdx[k]
      baseScores.add float(nmmoScore(cint(bp))) /
        float(agentStat(cint(bp), 1) + 1)
    allNim.add nimScores
    allBase.add baseScores
    var nMean, bMean: float
    for s in nimScores: nMean += s / 4
    for s in baseScores: bMean += s / 4
    echo &"seed {seed}: nim={nMean:.2f} base={bMean:.2f} " &
      &"deaths={deaths} combs={combs}"

  var nMean, bMean: float
  for s in allNim: nMean += s / float(allNim.len)
  for s in allBase: bMean += s / float(allBase.len)
  echo &"\nnim mean={nMean:.3f} vs baseline {bMean:.3f} | " &
    &"deaths/agent/{ticks}t = " &
    &"{float(totDeaths) / float(allNim.len):.1f} | max comb={maxComb}"
  deathClasses.sort()
  echo "death classes: ", deathClasses

main()
