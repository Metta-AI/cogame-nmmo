# Residue-liveness measurement: for every window cell carrying
# enemy-signature bytes, tally P(real enemy within 1 | features)
# against the ground-truth export. Features: anim byte, last-diff age
# bucket, element (bow/melee), Chebyshev distance band.
#
# Output: counts table to bake calibrated constants into the tracker.

import std/[strformat, tables]
import simlink, percept

proc lcg(x: uint32): uint32 = x * 1664525'u32 + 1013904223'u32

proc scriptAction(agent, tick: int): int =
  let h = lcg(uint32(agent * 7919 + tick * 104729 + 12345))
  let v = int(h shr 8) mod 10
  if v < 6: v mod 4 else: AtnNoop

var tally = initTable[string, tuple[live, total: int]]()

for seed in [11, 12, 13, 14]:
  nmmoInit(cuint(seed), 8)
  let obs = obsPtr()
  let act = actPtr()
  var buf: array[ObsSize, uint8]
  var prev: array[ObsSize, uint8]
  var lastDiff: array[WindowRows * WindowCols, int]
  for i in 0 ..< lastDiff.len: lastDiff[i] = -100
  var havePrev = false

  for tick in 0 ..< 600:
    copyMem(addr buf[0], addr obs[0], ObsSize)
    let p = initPercept(buf)
    # per-cell diff age (window-anchored, like the tracker's evidence)
    if havePrev:
      for r in 0 ..< WindowRows:
        for c in 0 ..< WindowCols:
          let base = (r * WindowCols + c) * TileBytes
          for b in TbEntType .. TbEntDir:
            if buf[base + b] != prev[base + b]:
              lastDiff[r * WindowCols + c] = tick
              break
    prev = buf
    havePrev = true

    if tick > 20:
      let gt = groundTruth(0, 7)
      for r in 0 ..< WindowRows:
        for c in 0 ..< WindowCols:
          if p.tile(r, c, TbEntType) != EntityEnemy: continue
          if p.tile(r, c, TbEntAnim) == AnimDeath: continue
          let dr = r - CenterRow
          let dc = c - CenterCol
          let dist = max(abs(dr), abs(dc))
          if dist > 5 or dist == 0: continue
          var isLive = false
          for e in gt:
            if abs(e.dr - dr) <= 1 and abs(e.dc - dc) <= 1 and e.hp > 0:
              isLive = true; break
          let age = tick - lastDiff[r * WindowCols + c]
          let ageB =
            if age == 0: "diff-now"
            elif age <= 3: "age1-3"
            elif age <= 10: "age4-10"
            else: "age>10"
          let kind = if p.tile(r, c, TbEntElement) != 0: "bow" else: "mel"
          let anim = p.tile(r, c, TbEntAnim)
          let key = &"{kind}|anim{anim}|{ageB}"
          var cur = tally.getOrDefault(key, (0, 0))
          cur.total.inc
          if isLive: cur.live.inc
          tally[key] = cur

    for i in 0 ..< 8:
      act[i] = cfloat(scriptAction(i, tick))
    nmmoStep()

echo "P(live within 1) by residue features:"
for key, v in tally.pairs:
  if v.total >= 30:
    echo &"  {key}: {v.live}/{v.total} = {100 * v.live div v.total}%"
