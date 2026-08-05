# 8-seat interleaved bench in the bit-exact native sim:
# even seats (0,2,4,6) = pretrained baseline net, odd seats (1,3,5,7)
# = the Nim agent. Seat score = nmmo_score/(deaths+1) at the final
# tick (the league formula). Fatal-tick death classification for the
# Nim seats via the ground-truth export (same classes as
# tools/dancer_bench.py).
#
# Usage: bench [nseeds] [ticks] [seed0] [-probe]
#   -probe: per-death forensics — killer-tracked x branch aggregate +
#   the last 8 decision rows of each death (dancer_probe.py port).

import std/[os, strformat, strutils, tables]
import simlink, percept, world, executive, brainlink

type ProbeRow = object
  tick, hp, mn, life: int
  branch: string
  act: int
  tracked: string

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
  var args: seq[string]
  var probe = false
  for i in 1 .. paramCount():
    if paramStr(i) == "-probe": probe = true
    else: args.add paramStr(i)
  let nseeds = if args.len >= 1: parseInt(args[0]) else: 8
  let ticks = if args.len >= 2: parseInt(args[1]) else: 1500
  let seed0 = if args.len >= 3: parseInt(args[2]) else: 1

  var allNim, allBase: seq[float]
  var deathClasses = initCountTable[string]()
  var baseDeathClasses = initCountTable[string]()
  var probeAgg = initCountTable[string]()
  var probeReports = 0
  var totDeaths = 0
  var totBaseDeaths = 0
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
    var prevHpBase: array[4, int]
    for i in 0 ..< 4:
      prevHp[i] = 99
      prevHpBase[i] = 99
    var buf: array[ObsSize, uint8]
    var hist: array[4, seq[ProbeRow]]
    var branchTicks: array[4, CountTable[string]]
    # chain funnel: per life, which stage was reached
    var lifeComb, lifeTool, lifeSword, lifeMin: array[4, int]
    var lifeDef: array[4, int]
    var lifeProf: array[4, int]
    var funnel: array[5, int]   # lives, kill, tool, sword, min>=4
    var armedCombSum, armedProfSum, armedLives: int
    var armor24, armor48, t2tools: int

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
          inc funnel[0]
          if lifeComb[k] >= 2: inc funnel[1]
          if lifeTool[k] >= 1: inc funnel[2]
          if lifeTool[k] >= 2: inc t2tools
          if lifeSword[k] >= 1: inc funnel[3]
          if lifeMin[k] >= 4: inc funnel[4]
          if lifeDef[k] >= 24: inc armor24
          if lifeDef[k] >= 48: inc armor48
          if lifeSword[k] >= 1:
            armedCombSum += lifeComb[k]
            armedProfSum += lifeProf[k]
            inc armedLives
          lifeComb[k] = 0; lifeTool[k] = 0
          lifeSword[k] = 0; lifeMin[k] = 0; lifeProf[k] = 0
          lifeDef[k] = 0
          minds[k].tick = t
          minds[k].reset()
        copyMem(addr buf[0], addr obs[pid * ObsSize], ObsSize)
        let p = initPercept(buf)
        lifeComb[k] = max(lifeComb[k], p.combLvl)
        lifeTool[k] = max(lifeTool[k], p.maxToolTier)
        lifeSword[k] = max(lifeSword[k], max(p.bestSwordTier,
                                             p.bestBowTier))
        lifeMin[k] = max(lifeMin[k], min(p.combLvl, p.profLvl))
        lifeProf[k] = max(lifeProf[k], p.profLvl)
        lifeDef[k] = max(lifeDef[k], p.eqDef)
        let a = minds[k].act(buf)
        act[pid] = cfloat(a)
        if probe:
          branchTicks[k].inc minds[k].branch.split("[")[0]
          var tr = ""
          for e in minds[k].world.enemies(tkMelee):
            tr.add &"m({e.r - CenterRow},{e.c - CenterCol},d{e.t.delta}," &
              &"a{minds[k].tick - 1 - e.t.lastConfirmed}) "
          for e in minds[k].world.enemies(tkBow):
            tr.add &"b({e.r - CenterRow},{e.c - CenterCol}) "
          hist[k].add ProbeRow(
            tick: t, hp: p.hp, mn: min(p.combLvl, p.profLvl),
            life: 0, branch: minds[k].branch, act: a, tracked: tr)
          if hist[k].len > 8: hist[k].delete(0)
      # pre-step ground truth for fatal-tick classification
      var gtNow: array[4, seq[GtEnemy]]
      var gtBase: array[4, seq[GtEnemy]]
      for k in 0 ..< 4:
        gtNow[k] = groundTruth(nimIdx[k])
        gtBase[k] = groundTruth(baseIdx[k])
      nmmoStep()
      for k in 0 ..< 4:
        let hp = int(agentStat(cint(baseIdx[k]), 7))
        if hp == 0 and prevHpBase[k] > 0:
          baseDeathClasses.inc classifyDeath(gtBase[k], prevHpBase[k])
        if term[baseIdx[k]] > 0.5 and hp == 99:
          baseDeathClasses.inc "stagnation"
        prevHpBase[k] = hp
      for k in 0 ..< 4:
        let hp = int(agentStat(cint(nimIdx[k]), 7))
        if hp == 0 and prevHp[k] > 0:
          let cls = classifyDeath(gtNow[k], prevHp[k])
          deathClasses.inc cls
          if probe:
            # killer-tracked classification (dancer_probe port)
            var killers: seq[GtEnemy]
            for e in gtNow[k]:
              if (abs(e.dr) + abs(e.dc) <= 1 and e.ranged == 0) or
                  ((e.dr == 0 or e.dc == 0) and
                   max(abs(e.dr), abs(e.dc)) <= 4 and e.ranged == 1):
                killers.add e
            var nMatch = 0
            for e in killers:
              var hit = false
              for tk in minds[k].world.enemies(tkMelee):
                if abs(tk.r - CenterRow - e.dr) <= 1 and
                    abs(tk.c - CenterCol - e.dc) <= 1:
                  hit = true; break
              if not hit:
                for tk in minds[k].world.enemies(tkBow):
                  if abs(tk.r - CenterRow - e.dr) <= 1 and
                      abs(tk.c - CenterCol - e.dc) <= 1:
                    hit = true; break
              if hit: inc nMatch
            let tcls =
              if killers.len == 0: "no-killer-visible"
              elif nMatch == killers.len: "all-tracked"
              elif nMatch > 0: "some-untracked"
              else: "untracked"
            let lastBranch =
              if hist[k].len > 0: hist[k][^1].branch else: "?"
            probeAgg.inc tcls & "|" & lastBranch
            if probeReports < 40:
              inc probeReports
              echo &"=== death seed={seed} t={t} seat={k} [{tcls}/{cls}]"
              for e in killers:
                echo &"  killer rel=({e.dr},{e.dc}) L{e.level} " &
                  &"ranged={e.ranged}"
              for row in hist[k]:
                echo &"  t{row.tick} hp={row.hp} mn={row.mn} " &
                  &"{row.branch} act={row.act} {row.tracked}"
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
      combs[k] = min(int(agentStat(cint(pid), 2)),
                     int(agentStat(cint(pid), 3)))  # min(comb,prof)
      totDeaths += deaths[k]
      maxComb = max(maxComb, combs[k])
      let bp = baseIdx[k]
      baseScores.add float(nmmoScore(cint(bp))) /
        float(agentStat(cint(bp), 1) + 1)
      totBaseDeaths += int(agentStat(cint(bp), 1))
    allNim.add nimScores
    allBase.add baseScores
    var nMean, bMean: float
    for s in nimScores: nMean += s / 4
    for s in baseScores: bMean += s / 4
    echo &"seed {seed}: nim={nMean:.2f} base={bMean:.2f} " &
      &"deaths={deaths} mins={combs}"
    if probe:
      for k in 0 ..< 4:
        branchTicks[k].sort()
        echo &"  seat{k} d={deaths[k]} min={combs[k]} branches: ",
          branchTicks[k]
    echo &"  funnel: lives={funnel[0]} kill={funnel[1]} " &
      &"tool={funnel[2]} sword={funnel[3]} min4={funnel[4]} " &
      &"armed(comb,prof)=({armedCombSum},{armedProfSum})/{armedLives} " &
      &"armor24={armor24} armor48={armor48} t2tool={t2tools}"

  var nMean, bMean: float
  for s in allNim: nMean += s / float(allNim.len)
  for s in allBase: bMean += s / float(allBase.len)
  echo &"\nnim mean={nMean:.3f} vs baseline {bMean:.3f} | " &
    &"deaths/agent/{ticks}t = " &
    &"{float(totDeaths) / float(allNim.len):.1f} | max comb={maxComb}"
  deathClasses.sort()
  echo "death classes: ", deathClasses
  baseDeathClasses.sort()
  echo &"baseline deaths/agent/{ticks}t = " &
    &"{float(totBaseDeaths) / float(allBase.len):.1f} | classes: ",
    baseDeathClasses
  if probeAgg.len > 0:
    probeAgg.sort()
    echo "(killer-tracking | last-branch): ", probeAgg

main()
