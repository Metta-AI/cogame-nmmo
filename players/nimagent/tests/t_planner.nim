# L2 gate: planMelee vs MiniMelee — port of tests/test_dancer_planner.py
# (the python suite passed 13/13 on 2026-08-03; the 1v1 scenarios run
# through planMelee with n=1, exactly gate (d) there).

import ../percept, ../planner, ../executive

type
  MiniEnemy = object
    r, c: int
    dmg: float
    hp: int
    ranged: bool

  MiniMelee = object
    pr, pc: int
    enemies: seq[MiniEnemy]
    ourDmg: float
    sword: bool
    bow: bool
    walls: seq[tuple[r, c: int]]
    hitsDealt, hitsTaken: int
    dmgTaken: float
    movedTicks: int

proc initMini(pr, pc: int, enemies: seq[MiniEnemy], ourDmg: float,
              sword: bool, walls: seq[tuple[r, c: int]] = @[]): MiniMelee =
  MiniMelee(pr: pr, pc: pc, enemies: enemies, ourDmg: ourDmg,
            sword: sword, walls: walls)

proc passable(m: MiniMelee, r, c: int): bool = (r, c) notin m.walls

proc inArc(m: MiniMelee, r, c: int): bool =
  let ar = abs(r - m.pr)
  let ac = abs(c - m.pc)
  if m.bow:
    return (ar == 0 or ac == 0) and max(ar, ac) <= 4 and ar + ac > 0
  ar + ac == 1 or (m.sword and ar == 1 and ac == 1)

proc occ(m: MiniMelee): seq[tuple[r, c: int]] =
  for e in m.enemies:
    if e.hp > 0: result.add (e.r, e.c)

proc step(m: var MiniMelee, action: int, target: int) =
  ## target: index into m.enemies or -1. Player first, then each live
  ## melee in list order (attack at man-1 else largest-gap-axis step,
  ## tie -> column, blocked -> stall; vacated cells free up).
  var o = m.occ()
  if action == AtnAttack:
    if target >= 0 and m.enemies[target].hp > 0 and
        m.inArc(m.enemies[target].r, m.enemies[target].c):
      m.enemies[target].hp -= int(m.ourDmg)
      inc m.hitsDealt
  elif isMove(action) or isRun(action):
    let run = isRun(action)
    let d = moveDelta(action)
    let steps = if run: 2 else: 1
    let mid = (m.pr + d.dr, m.pc + d.dc)
    let dest = (m.pr + d.dr * steps, m.pc + d.dc * steps)
    if m.passable(mid[0], mid[1]) and mid notin o and
        (not run or (m.passable(dest[0], dest[1]) and dest notin o)):
      m.pr = dest[0]
      m.pc = dest[1]
      inc m.movedTicks
  # enemies respond (aggro cutoff: outside the +-4 box they idle -
  # enemy_ai scans only its own box, nmmo3.h:1592)
  o = m.occ()
  for i in 0 ..< m.enemies.len:
    if m.enemies[i].hp <= 0: continue
    let r = m.enemies[i].r
    let c = m.enemies[i].c
    let dr = m.pr - r
    let dc = m.pc - c
    if max(abs(dr), abs(dc)) > 4: continue
    var nr = r
    var nc = c
    if m.enemies[i].ranged:
      if dr == 0 or dc == 0:
        m.dmgTaken += m.enemies[i].dmg
        inc m.hitsTaken
        continue
      if abs(dr) > abs(dc):
        nc = c + (if dc > 0: 1 else: -1)
      else:
        nr = r + (if dr > 0: 1 else: -1)
    elif abs(dr) + abs(dc) == 1:
      m.dmgTaken += m.enemies[i].dmg
      inc m.hitsTaken
      continue
    else:
      if abs(dr) > abs(dc):
        nr = r + (if dr > 0: 1 else: -1)
      else:
        nc = c + (if dc > 0: 1 else: -1)
    if m.passable(nr, nc) and (nr, nc) != (m.pr, m.pc) and
        (nr, nc) notin o:
      let idx = o.find((r, c))
      if idx >= 0: o.del(idx)
      o.add (nr, nc)
      m.enemies[i].r = nr
      m.enemies[i].c = nc

proc drive(m: var MiniMelee, target: int, intentKill: bool,
           ticks = 80) =
  ## Replan each tick over the live enemy set; stops at target death.
  for _ in 0 ..< ticks:
    var enemies: seq[PlanEnemy]
    var tidx = -1
    var liveCount = 0
    for i, e in m.enemies:
      if e.hp <= 0: continue
      if i == target: tidx = enemies.len
      enemies.add PlanEnemy(r: e.r, c: e.c, dmg: e.dmg, hp: e.hp,
                            ranged: e.ranged)
      inc liveCount
    if liveCount == 0: return
    let mm = m   # capture snapshot for the closure
    let act = planMelee(m.pr, m.pc, enemies, tidx, m.ourDmg, m.sword,
                        intentKill,
                        proc (r, c: int): bool = mm.passable(r, c),
                        bowHeld = m.bow)
    m.step(act, target)
    if target >= 0 and m.enemies[target].hp <= 0: return

# --- 1v1 gates (n=1 regression) ---------------------------------------

block danceKillsWithoutDamage:
  for start in [(3, 0), (0, 3), (2, 2), (4, 1), (-3, -2)]:
    var m = initMini(0, 0,
      @[MiniEnemy(r: start[0], c: start[1], dmg: 45.0, hp: 99)],
      30.0, sword = true)
    m.drive(0, intentKill = true)
    doAssert m.enemies[0].hp <= 0, "no kill from " & $start
    doAssert m.dmgTaken == 0, "took " & $m.dmgTaken & " from " & $start
  echo "dance-kill: OK"

block danceEfficiency:
  var m = initMini(0, 0, @[MiniEnemy(r: 3, c: 1, dmg: 45.0, hp: 99)],
                   25.0, sword = true)
  var ticks = 0
  for _ in 0 ..< 80:
    var es = @[PlanEnemy(r: m.enemies[0].r, c: m.enemies[0].c,
                         dmg: 45.0, hp: m.enemies[0].hp)]
    let mm = m
    let act = planMelee(m.pr, m.pc, es, 0, 25.0, true, true,
                        proc (r, c: int): bool = mm.passable(r, c))
    inc ticks
    m.step(act, 0)
    if m.enemies[0].hp <= 0: break
  doAssert m.enemies[0].hp <= 0
  doAssert ticks <= 4 * 4 + 6, "kill took " & $ticks & " ticks"
  doAssert m.dmgTaken == 0
  echo "dance-efficiency: OK (", ticks, " ticks)"

block escapeUnhittable:
  var m = initMini(0, 0, @[MiniEnemy(r: 2, c: 1, dmg: 80.0, hp: 99)],
                   0.0, sword = false)
  m.drive(0, intentKill = false, ticks = 60)
  doAssert m.dmgTaken == 0, "took " & $m.dmgTaken & " escaping"
  echo "escape-unhittable: OK"

block bareHandsTrade:
  var m = initMini(0, 0, @[MiniEnemy(r: 1, c: 0, dmg: 18.0, hp: 99)],
                   37.0, sword = false)
  m.drive(0, intentKill = true)
  doAssert m.enemies[0].hp <= 0
  doAssert m.dmgTaken <= 4 * 18, "trade cost " & $m.dmgTaken
  echo "bare-hands-trade: OK (cost ", m.dmgTaken, ")"

block cornerNoCatastrophe:
  var walls: seq[tuple[r, c: int]]
  for r in -4 .. 4:
    walls.add (r, -2)
    walls.add (r, 2)
  for c in -2 .. 2:
    walls.add (-3, c)
  var m = initMini(0, 0, @[MiniEnemy(r: 2, c: 0, dmg: 50.0, hp: 99)],
                   20.0, sword = true, walls = walls)
  m.drive(0, intentKill = true, ticks = 50)
  doAssert m.enemies[0].hp <= 0 or m.dmgTaken <= 150,
    "corner cost " & $m.dmgTaken
  echo "corner-1v1: OK"

# --- multi-enemy gates -------------------------------------------------

block twoChaserOpenEscape:
  var m = initMini(0, 0,
    @[MiniEnemy(r: 2, c: 1, dmg: 60.0, hp: 99),
      MiniEnemy(r: -2, c: -1, dmg: 60.0, hp: 99)],
    0.0, sword = false)
  m.drive(-1, intentKill = false, ticks = 60)
  doAssert m.hitsTaken <= 1,
    "pincer: " & $m.hitsTaken & " hits (" & $m.dmgTaken & " dmg)"
  echo "two-chaser-escape: OK (", m.hitsTaken, " hits)"

block duelWithBystander:
  var m = initMini(0, 0,
    @[MiniEnemy(r: 3, c: 0, dmg: 45.0, hp: 99),
      MiniEnemy(r: 0, c: 8, dmg: 60.0, hp: 99)],
    30.0, sword = true)
  m.drive(0, intentKill = true, ticks = 80)
  doAssert m.enemies[0].hp <= 0, "target survived"
  doAssert m.hitsTaken <= 1,
    "bystander: " & $m.hitsTaken & " hits (" & $m.dmgTaken & " dmg)"
  echo "duel-with-bystander: OK (", m.hitsTaken, " hits)"

block twoChaserCorner:
  var walls: seq[tuple[r, c: int]]
  for r in -4 .. 6:
    walls.add (r, -2)
    walls.add (r, 2)
  for c in -2 .. 2:
    walls.add (-3, c)
  var m = initMini(0, 0,
    @[MiniEnemy(r: 4, c: 0, dmg: 50.0, hp: 99),
      MiniEnemy(r: 5, c: 1, dmg: 50.0, hp: 99)],
    25.0, sword = true, walls = walls)
  m.drive(0, intentKill = true, ticks = 40)
  var bothDead = true
  for e in m.enemies:
    if e.hp > 0: bothDead = false
  doAssert bothDead or m.dmgTaken <= 250.0,
    "cornered: " & $m.dmgTaken & " dmg"
  doAssert m.movedTicks >= 5, "stand-and-tank: moved " & $m.movedTicks
  echo "two-chaser-corner: OK (dmg ", m.dmgTaken, ", moved ",
    m.movedTicks, ")"

block bowKiteKill:
  # held bow (on-axis <=4): kill ANY hittable melee with ZERO damage
  # in the open - the chaser aligns itself, we shoot aligned and step
  # away otherwise; it can never reach Manhattan-1
  for start in [(4, 0), (0, 4), (3, 2), (-2, -3)]:
    var m = initMini(0, 0,
      @[MiniEnemy(r: start[0], c: start[1], dmg: 60.0, hp: 99)],
      30.0, sword = false)
    m.bow = true
    m.drive(0, intentKill = true, ticks = 80)
    doAssert m.enemies[0].hp <= 0, "bow-kite no kill from " & $start
    doAssert m.dmgTaken == 0,
      "bow-kite took " & $m.dmgTaken & " from " & $start
  echo "bow-kite-kill: OK"

block bowKiteBystander:
  # kite one melee while a second approaches: kill, <=1 hit
  var m = initMini(0, 0,
    @[MiniEnemy(r: 4, c: 0, dmg: 45.0, hp: 99),
      MiniEnemy(r: 0, c: 8, dmg: 60.0, hp: 99)],
    30.0, sword = false)
  m.bow = true
  m.drive(0, intentKill = true, ticks = 80)
  doAssert m.enemies[0].hp <= 0, "target survived"
  doAssert m.hitsTaken <= 1, "bystander hits " & $m.hitsTaken
  echo "bow-kite-bystander: OK"

block bowCrossfireEscape:
  # melee chaser + off-axis bow: escape with <=1 arrow taken
  var m = initMini(0, 0,
    @[MiniEnemy(r: 2, c: 1, dmg: 60.0, hp: 99),
      MiniEnemy(r: -2, c: 3, dmg: 80.0, hp: 99, ranged: true)],
    0.0, sword = false)
  m.drive(-1, intentKill = false, ticks = 50)
  doAssert m.hitsTaken <= 1,
    "crossfire: " & $m.hitsTaken & " hits (" & $m.dmgTaken & ")"
  echo "bow-crossfire-escape: OK (", m.hitsTaken, " hits)"

# kill-the-bow: PARITY-REFUTED in the open (tie->row realignment
# locks our turns to aligned-adjacent states; each sword hit costs one
# arrow ~80 - unprofitable at any reachable gear). Armor is the only
# bow answer; terrain-assisted kills are niche future work.

echo "L2 GATE OK"
