# Scaffold smoke: sim links, steps, obs parses, digest is stable for a
# fixed seed+action script (pins native/wasm parity provenance).

import ../simlink, ../percept

nmmoInit(1, 8)
doAssert obsSizeC() == ObsSize

let obs = obsPtr()
let act = actPtr()
var digests: seq[uint32]
for t in 0 ..< 100:
  for i in 0 ..< 8:
    act[i] = cfloat((t + i * 3) mod 26)
  nmmoStep()
  digests.add uint32(stateDigest())

doAssert nmmoFault() == 0
doAssert nmmoTick() == 100

# parse agent 0's obs
var buf: array[ObsSize, uint8]
copyMem(addr buf[0], addr obs[0], ObsSize)
let p = initPercept(buf)
doAssert p.hp >= 0 and p.hp <= 99
doAssert p.combLvl >= 1
doAssert p.profLvl >= 1
doAssert p.passable(CenterRow, CenterCol)

var enemies = 0
for e in p.enemyCells: inc enemies
echo "smoke ok: tick=", nmmoTick(), " hp=", p.hp, " comb=", p.combLvl,
  " enemyCells=", enemies, " digest100=", digests[^1]
