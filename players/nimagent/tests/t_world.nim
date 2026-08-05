# L1 gate: tracker precision/recall vs sim ground truth, PLUS port
# equivalence — this harness mirrors tools/world_gate_ref.py exactly
# (same deterministic action script on the digest-identical sim), so
# its counters must match the python tracker's counters. Bars are the
# honest session-4 values from tests/test_dancer_world.py.

import std/strutils
import ../simlink, ../percept, ../world

proc lcg(x: uint32): uint32 =
  x * 1664525'u32 + 1013904223'u32

proc scriptAction(agent, tick: int): int =
  ## Deterministic per-agent action stream, identical in the python
  ## reference: mostly walks with pauses.
  let h = lcg(uint32(agent * 7919 + tick * 104729 + 12345))
  let v = int(h shr 8) mod 10
  if v < 6: v mod 4 else: AtnNoop

type Counters = object
  gt, hit, closeGt, closeHit, tracks, trueT, kindPairs, kindOk: int

proc runSeed(seed: int, ticks = 600): Counters =
  nmmoInit(cuint(seed), 8)
  var w = initWorldModel()
  let obs = obsPtr()
  let act = actPtr()
  let term = termPtr()
  var prevGtFrame: seq[tuple[r, c: int]]
  var buf: array[ObsSize, uint8]

  for tick in 0 ..< ticks:
    copyMem(addr buf[0], addr obs[0], ObsSize)
    let p = initPercept(buf)
    if tick > 0 and term[0] > 0.5:
      w.reset(tick)
    w.observe(p, tick)
    let act0 = scriptAction(0, tick)
    w.noteAction(act0)

    # -- score BEFORE stepping ------------------------------------
    let gt = groundTruth(0)
    var gtFrame: seq[tuple[r, c: int]]
    for g in gt:
      gtFrame.add (w.pos.r + g.dr, w.pos.c + g.dc)
    if not w.teleported and tick > 5:
      var trkClose: seq[tuple[r, c: int, kind: TrackKind]]
      for e in w.enemies(tkMelee):
        if max(abs(e.r - CenterRow), abs(e.c - CenterCol)) <= 4:
          trkClose.add (e.r - CenterRow, e.c - CenterCol, tkMelee)
      for e in w.enemies(tkBow):
        if max(abs(e.r - CenterRow), abs(e.c - CenterCol)) <= 4:
          trkClose.add (e.r - CenterRow, e.c - CenterCol, tkBow)
      for gi, g in gt:
        if max(abs(g.dr), abs(g.dc)) > 3 or g.hp <= 0: continue
        let fr = gtFrame[gi]
        var settled, frozen = false
        for pf in prevGtFrame:
          if max(abs(fr.r - pf.r), abs(fr.c - pf.c)) <= 1: settled = true
          if fr == pf: frozen = true
        if not settled or frozen: continue
        var matched = false
        var kindAt = tkMelee
        var exact = false
        for t in trkClose:
          if max(abs(t.r - g.dr), abs(t.c - g.dc)) <= 1:
            matched = true
          if t.r == g.dr and t.c == g.dc and not exact:
            exact = true
            kindAt = t.kind
        inc result.gt
        if matched: inc result.hit
        if max(abs(g.dr), abs(g.dc)) <= 2:
          inc result.closeGt
          if matched: inc result.closeHit
        if exact:
          inc result.kindPairs
          let realKind = if g.ranged != 0: tkBow else: tkMelee
          if kindAt == realKind: inc result.kindOk
      # precision
      for t in trkClose:
        inc result.tracks
        var isTrue = false
        for g in gt:
          if g.hp > 0 and max(abs(t.r - g.dr), abs(t.c - g.dc)) <= 1:
            isTrue = true; break
        if isTrue: inc result.trueT
    prevGtFrame = gtFrame

    for i in 0 ..< 8:
      act[i] = cfloat(scriptAction(i, tick))
    nmmoStep()

var agg: Counters
for seed in [11, 12, 13]:
  let s = runSeed(seed)
  agg.gt += s.gt; agg.hit += s.hit
  agg.closeGt += s.closeGt; agg.closeHit += s.closeHit
  agg.tracks += s.tracks; agg.trueT += s.trueT
  agg.kindPairs += s.kindPairs; agg.kindOk += s.kindOk

let recall = agg.hit / max(agg.gt, 1)
let closeRecall = agg.closeHit / max(agg.closeGt, 1)
let precision = agg.trueT / max(agg.tracks, 1)
let kindAcc = agg.kindOk / max(agg.kindPairs, 1)
echo "tracker: recall=", recall.formatFloat(ffDecimal, 3),
  " (", agg.hit, "/", agg.gt, ")",
  " close=", closeRecall.formatFloat(ffDecimal, 3),
  " (", agg.closeHit, "/", agg.closeGt, ")",
  " precision=", precision.formatFloat(ffDecimal, 3),
  " (", agg.trueT, "/", agg.tracks, ")",
  " kind=", kindAcc.formatFloat(ffDecimal, 3),
  " (", agg.kindOk, "/", agg.kindPairs, ")"

doAssert recall >= 0.86, "recall " & $recall
doAssert closeRecall >= 0.82, "close recall " & $closeRecall
doAssert precision >= 0.64, "precision " & $precision
doAssert kindAcc == 1.0, "kind acc " & $kindAcc
echo "L1 GATE OK"
