# L4 Executive: phases + goals on top of WorldModel and planMelee.
# Port of players/dancer/executive.py (with the session-4 fixes:
# melee-aware bow-flee, fresh-tracks-as-bodies / stale-as-hazards, no
# double-counting) — plus A* routing (astar.nim) in place of the
# greedy one-step safe_step filter.
#
# All combat goes through planMelee (the one authority); all movement
# through the A* cost-field router or the flee filter. Recovery is a
# gate, never a control-stealing branch.

import std/random
import percept, world, planner, astar

const
  StagWindow = 500
  HerbHp = 45
  HerbHpCombat = 65
  RecoverFloor = 60
  RecoverUntil = 90
  BowAvoid = 4    # a bow only ACTS on players inside its +-4 box
                  # (enemy_ai scan, nmmo3.h:1592); range >= 5 is
                  # harmless leashed wander - fleeing it was measured
                  # pure cost (pushed agents into melee)
  EngageR = 4
  HerbStock = 2

  BaseAttack = 40
  EnemyBase = 15
  LevelMul = 2
  EnemyLevelMul = 5

type
  Mind* = object
    world*: WorldModel
    rng: Rand
    seed: int
    agentIdx: int
    generation: int
    tick*: int
    lifeTick: int
    windowStartMin: int
    recovering: bool
    prevComb: int
    lootSweepLeft: int
    selling: bool
    lastAction: int
    wanderAction: int
    wanderLeft: int
    idleTicks: int
    branch*: string

proc reset*(m: var Mind) =
  inc m.generation
  m.rng = initRand(m.seed * 1_000_003 + m.agentIdx * 7919 +
                   m.generation + 1)
  m.world.reset(m.tick)
  m.lifeTick = 0
  m.windowStartMin = 1
  m.recovering = false
  m.prevComb = 1
  m.lootSweepLeft = 0
  m.selling = false
  m.lastAction = AtnNoop
  m.wanderAction = AtnDown
  m.wanderLeft = 0
  m.idleTicks = 0
  m.branch = ""

proc initMind*(seed, agentIdx: int): Mind =
  result.seed = seed
  result.agentIdx = agentIdx
  result.generation = -1
  result.world = initWorldModel()
  result.tick = 0
  result.reset()

proc deltaBounds(myLvl, delta: int): tuple[lo, hi: int] =
  if delta >= 4:
    (myLvl + 8, myLvl + 40)
  else:
    (max(1, myLvl + 2 * delta), myLvl + 2 * delta + 1)

# -- inventory helpers --------------------------------------------------

proc herbSlot(p: Percept): int =
  for i in 0 ..< NumKeySlots:
    let item = p.inventory(i)
    if item != 0 and itemType(item) == IHerb: return i
  -1

proc herbCount(p: Percept): int =
  for i in 0 ..< InventorySlots:
    let item = p.inventory(i)
    if item != 0 and itemType(item) == IHerb: inc result

proc slotOf(p: Percept, rawId: int, equipped: bool): int =
  for i in 0 ..< NumKeySlots:
    if p.inventory(i) == rawId and p.isEquipped(i) == equipped:
      return i
  -1

proc bestUnequipped(p: Percept, t: int): tuple[slot, tier: int] =
  result = (-1, 0)
  for i in 0 ..< NumKeySlots:
    let item = p.inventory(i)
    if item != 0 and itemType(item) == t and not p.isEquipped(i):
      if result.slot < 0 or itemTier(item) > result.tier:
        result = (i, itemTier(item))

proc maxToolTier(p: Percept): int =
  for i in 0 ..< InventorySlots:
    let item = p.inventory(i)
    if item != 0 and itemType(item) == ITool:
      result = max(result, itemTier(item))
  let held = p.equipment(SlotHeld)
  if held != 0 and itemType(held) == ITool:
    result = max(result, itemTier(held))

proc bestSwordTier(p: Percept): int =
  for i in 0 ..< InventorySlots:
    let item = p.inventory(i)
    if item != 0 and itemType(item) == ISword:
      result = max(result, itemTier(item))
  let held = p.equipment(SlotHeld)
  if held != 0 and itemType(held) == ISword:
    result = max(result, itemTier(held))

proc allInventoryFull(p: Percept): bool =
  for i in 0 ..< InventorySlots:
    if p.inventory(i) == 0: return false
  true

proc junkSlot(p: Percept): int =
  let bestTool = p.maxToolTier
  let bestSword = p.bestSwordTier
  result = -1
  var bestTier = 99
  for i in 0 ..< NumKeySlots:
    let item = p.inventory(i)
    if item == 0 or p.isEquipped(i): continue
    let t = itemType(item)
    let tier = itemTier(item)
    if t == IHerb: continue
    if t == ITool and tier >= bestTool: continue
    if t == ISword and tier >= bestSword: continue
    if isArmorType(t):
      let cur = p.equipment(armorSlot(t))
      if cur == 0 or itemTier(cur) < tier: continue
    if tier < bestTier:
      bestTier = tier
      result = i

# -- stagnation ---------------------------------------------------------

proc urgentAny(m: Mind, p: Percept): bool =
  let mn = min(p.combLvl, p.profLvl)
  if mn > m.windowStartMin: return false
  (StagWindow - (m.lifeTick mod StagWindow)) <= 150

proc urgentComb(m: Mind, p: Percept): bool =
  m.urgentAny(p) and p.combLvl <= p.profLvl

# -- movement helpers ---------------------------------------------------

proc lastMoveBlocked(m: Mind, p: Percept): bool =
  (isMove(m.lastAction) or isRun(m.lastAction)) and p.anim == 0

proc bowDanger*(r, c, br, bc: int): bool =
  ## Cell (r,c) is in bow (br,bc)'s danger zone: aligned within 5 (an
  ## aligned bow at range 5 is one wander-step from a 99-dmg shot) or
  ## one-step-from-aligned within its 4-box.
  let dr = abs(r - br)
  let dc = abs(c - bc)
  (min(dr, dc) == 0 and max(dr, dc) <= 5) or
    (min(dr, dc) == 1 and max(dr, dc) <= 4)

proc flee(m: var Mind, p: Percept,
          cells: seq[tuple[r, c: int]],
          bows, melee: seq[tuple[r, c: int]],
          occ: seq[tuple[r, c: int]]): int =
  ## Best single step away from `cells`, melee- and bow-aware.
  var best = AtnNoop
  var bestKey = (-1, -1, -1)
  for ai in 0 .. 3:
    let d = MoveDeltas[ai]
    let nr = CenterRow + d.dr
    let nc = CenterCol + d.dc
    if not p.passable(nr, nc) or (nr, nc) in occ: continue
    var bowOk = 1
    for b in bows:
      if bowDanger(nr, nc, b.r, b.c):
        bowOk = 0; break
    var meleeOk = 1
    for mm in melee:
      if abs(nr - mm.r) + abs(nc - mm.c) <= 1:
        meleeOk = 0; break
    var dist = 9
    for cell in cells:
      dist = min(dist, max(abs(nr - cell.r), abs(nc - cell.c)))
    let key = (meleeOk, bowOk, dist)
    if key > bestKey:
      bestKey = key
      best = ai
  best   # walks only

proc wander(m: var Mind, p: Percept,
            bows, melee, occ: seq[tuple[r, c: int]],
            roam: bool): int =
  if not roam and m.wanderLeft <= 0:
    inc m.idleTicks
    # MOVEMENT DIET: the baseline net takes 1.2 deaths/1500t to our
    # 6.9 in the same worlds (bow 3 vs 35) purely by near-zero
    # roaming - every fresh aggro/bow box entered is a dice roll.
    # Overwatch long, wander rarely and briefly.
    if m.idleTicks < 120: return AtnNoop
  let blocked = m.lastMoveBlocked(p)
  var options: seq[int]
  for ai in 0 .. 3:
    if blocked and ai == m.wanderAction: continue
    let d = MoveDeltas[ai]
    let nr = CenterRow + d.dr
    let nc = CenterCol + d.dc
    if not p.passable(nr, nc) or (nr, nc) in occ: continue
    var bad = false
    for b in bows:
      if bowDanger(nr, nc, b.r, b.c):
        bad = true; break
    if bad: continue
    for mm in melee:
      if abs(nr - mm.r) + abs(nc - mm.c) <= 1:
        bad = true; break
    if bad: continue
    options.add ai
  if m.wanderLeft <= 0 or blocked or m.wanderAction notin options:
    if options.len == 0: return AtnNoop
    m.wanderAction = options[m.rng.rand(options.len - 1)]
    m.wanderLeft = 3 + m.rng.rand(2)
  dec m.wanderLeft
  if m.wanderLeft <= 0: m.idleTicks = 0
  if m.wanderLeft mod 3 == 0: return AtnNoop
  m.wanderAction

# -- equipment / harvest ------------------------------------------------

proc equipAction(p: Percept, swordHeld: bool): int =
  ## ATN_ONE+slot to fix equipment, or -1 when nothing to do.
  for atype in [IHelm, IChest, ILegs]:
    let slot = armorSlot(atype)
    let best = bestUnequipped(p, atype)
    let cur = p.equipment(slot)
    if best.slot >= 0 and (cur == 0 or best.tier > itemTier(cur)):
      if cur == 0: return AtnOne + best.slot
      let s = slotOf(p, cur, equipped = true)
      if s >= 0: return AtnOne + s
  let want =
    if p.bestSwordTier > 0 and p.combLvl <= p.profLvl: ISword
    else: ITool
  let best = bestUnequipped(p, want)
  let held = p.equipment(SlotHeld)
  let heldType = if held != 0: itemType(held) else: -1
  let heldTier = if held != 0: itemTier(held) else: 0
  if heldType == want:
    if best.slot >= 0 and best.tier > heldTier:
      let s = slotOf(p, held, equipped = true)
      if s >= 0: return AtnOne + s
    return -1
  if best.slot >= 0:
    if held != 0:
      let s = slotOf(p, held, equipped = true)
      if s >= 0: return AtnOne + s
    return AtnOne + best.slot
  -1

proc harvestTarget(p: Percept): tuple[ok: bool, r, c: int] =
  let heldTier = p.heldToolTier
  let herbs = p.herbCount
  if p.allInventoryFull: return (false, 0, 0)
  let needSword = p.bestSwordTier < max(heldTier, 1)
  var armorNeeds: seq[int]
  for atype in [IHelm, IChest, ILegs]:
    let cur = p.equipment(armorSlot(atype))
    if cur == 0 or itemTier(cur) < heldTier: armorNeeds.add atype
  var bestKey = (99.0, 999)
  result = (false, 0, 0)
  for it in p.itemTiles:
    var priority: float
    if it.itype == ITool:
      if it.tier <= p.maxToolTier: continue
      priority = 0.0
    elif it.itype == IOre or it.itype == IWood or it.itype == IHilt or
        it.itype == IHerb or
        (it.itype >= IGemFirst and it.itype <= IGemLast):
      if it.tier > heldTier: continue
      if it.itype == IHilt and needSword:
        priority = 0.5
      elif float(p.profLvl) < tierLevel(it.tier):
        priority = 1.0 - 0.1 * float(it.tier)
      elif it.itype == IHerb and herbs < HerbStock:
        priority = 2.0
      elif it.itype == IOre and armorNeeds.len > 0:
        priority = 3.0
      else:
        continue
    else:
      continue
    let dist = abs(it.r - CenterRow) + abs(it.c - CenterCol)
    if dist > 5: continue    # movement diet: distant items are not
                             # worth the aggro/bow boxes on the way
    let key = (priority, dist)
    if key < bestKey:
      bestKey = key
      result = (true, it.r, it.c)

# -- decision -----------------------------------------------------------

proc decide(m: var Mind, p: Percept): int =
  var bows: seq[tuple[r, c: int]]
  for e in m.world.enemiesExtended(tkBow, margin = 4):
    bows.add (e.r, e.c)
  var melee: seq[tuple[r, c: int, t: Track]]
  for e in m.world.enemies(tkMelee):
    melee.add e
  var meleeCells: seq[tuple[r, c: int]]
  for e in melee: meleeCells.add (e.r, e.c)
  let occ = m.world.occupancy()

  # market UI
  m.branch = "ui"
  if p.uiMode != ModePlay:
    if m.selling and not p.inCombat:
      if p.uiMode == ModeSellSelect:
        let junk = p.junkSlot
        if junk >= 0: return AtnOne + junk
      elif p.uiMode == ModeSellPrice:
        m.selling = false
        return AtnNine
    m.selling = false
    return AtnNoop
  m.selling = false

  # herb - but NEVER while standing in an unarmored bow danger zone:
  # eating mid-lane means taking the next arrow at heal-tick (measured
  # death t970 seed 1). The bow-flee branch below breaks alignment
  # first; the herb fires next tick.
  var inBowLane = false
  block laneChk:
    for b in bows:
      if bowDanger(CenterRow, CenterCol, b.r, b.c):
        inBowLane = true
        break laneChk
  let limit = if p.inCombat: HerbHpCombat else: HerbHp
  if p.hp < limit and not (inBowLane and p.eqDef < 40):
    let herb = p.herbSlot
    if herb >= 0:
      m.branch = "herb"
      return AtnOne + herb

  # recovery bookkeeping
  if p.hp < RecoverFloor: m.recovering = true
  if p.hp >= RecoverUntil: m.recovering = false

  # kill detection -> loot sweep
  if p.combLvl > m.prevComb: m.lootSweepLeft = 16
  m.prevComb = p.combLvl

  let held = p.equipment(SlotHeld)
  let swordHeld = held != 0 and itemType(held) == ISword
  let eqAtk = p.eqAtk
  let eqDef = p.eqDef
  let armored = eqDef >= 40

  # bow safety (hard until armored); melee-aware flee
  var bowClose: seq[tuple[r, c: int]]
  var hereFunnel = false
  for b in bows:
    if max(abs(b.r - CenterRow), abs(b.c - CenterCol)) <= 5:
      bowClose.add b
    if bowDanger(CenterRow, CenterCol, b.r, b.c):
      hereFunnel = true
  var dNow = 99
  for b in bowClose:
    dNow = min(dNow, max(abs(CenterRow - b.r), abs(CenterCol - b.c)))
  if not armored and (hereFunnel or dNow <= 4):
    m.branch = "bow-flee"
    return m.flee(p, bowClose, bows, meleeCells, occ)

  # melee engagement via the planner
  let engageR = EngageR + (if m.recovering: 2 else: 0)
  var near: seq[tuple[r, c: int, t: Track]]
  for e in melee:
    if max(abs(e.r - CenterRow), abs(e.c - CenterCol)) <= engageR:
      near.add e
  if near.len > 0:
    proc dmgVs(t: Track): float =
      let (_, hiRaw) = deltaBounds(p.combLvl, t.delta)
      let hi = min(hiRaw, 14)
      float(BaseAttack + LevelMul * p.combLvl + eqAtk -
            EnemyLevelMul * hi)
    proc theirDmgVs(t: Track): float =
      let (_, hiRaw) = deltaBounds(p.combLvl, t.delta)
      let hi = min(hiRaw, 14)
      max(float(EnemyBase + EnemyLevelMul * hi - LevelMul * p.combLvl -
                eqDef), 0.0)
    # target: the SOFTEST worthwhile enemy nearby, else nearest
    var pool: seq[tuple[r, c: int, t: Track]]
    for e in near:
      if dmgVs(e.t) >= 12: pool.add e
    if pool.len == 0: pool = near
    var tgt = pool[0]
    var tgtKey = (999, 0.0)
    for e in pool:
      let key = (max(abs(e.r - CenterRow), abs(e.c - CenterCol)),
                 -dmgVs(e.t))
      if key < tgtKey:
        tgtKey = key
        tgt = e
    let ourDmg = dmgVs(tgt.t)
    let theirDmg = theirDmgVs(tgt.t)
    let eHp = tgt.t.hpBucket * 20 + 10
    var second = false
    for e in near:
      if e.t != tgt.t: second = true; break
    let wantComb = p.combLvl <= p.profLvl or p.heldToolTier == 0 or
      m.urgentComb(p)
    let hitsNeeded =
      if ourDmg > 0: (eHp + int(ourDmg) - 1) div int(ourDmg) else: 99
    let fightable = ourDmg >= 12 and hitsNeeded <= 8 and wantComb
    var hpOk = p.hp >= (if theirDmg * 3 < float(p.hp): 55 else: 85)
    if p.heldToolTier == 0 and not swordHeld:
      # bootstrap trades are bare-handed hp-for-hp: only decisive,
      # isolated, full-bar fights
      var farOk = true
      for e in near:
        if e.t != tgt.t and
            max(abs(e.r - CenterRow), abs(e.c - CenterCol)) <= 5:
          farOk = false; break
      hpOk = p.hp >= 95 and farOk
    let intentKill = fightable and hpOk and not second and
      not m.recovering and bowClose.len == 0
    # pre-fight herb: do not enter reach below one-hit headroom
    if theirDmg > 0 and float(p.hp) < min(95.0, theirDmg + 25.0):
      let herb = p.herbSlot
      if herb >= 0:
        m.branch = "herb-prefight"
        return AtnOne + herb
    # FRESH tracks are exact bodies; STALE ones degrade to soft
    # hazard reach-cells (phantom bodies cage the dance - measured)
    var enemies: seq[PlanEnemy]
    var tidx = -1
    var stale: seq[tuple[r, c: int]]
    for e in melee:
      let fresh = (m.tick - e.t.lastConfirmed) <= 3
      if e.t == tgt.t:
        tidx = enemies.len
      elif not fresh:
        stale.add (e.r, e.c)
        continue
      enemies.add PlanEnemy(r: e.r, c: e.c, dmg: theirDmgVs(e.t),
                            hp: e.t.hpBucket * 20 + 10)
    var occSoft: seq[tuple[r, c: int]]
    for cell in occ:
      var isEnemy = false
      for e in enemies:
        if (e.r, e.c) == cell: isEnemy = true; break
      if not isEnemy: occSoft.add cell
    var hazards: seq[tuple[r, c: int]]
    for b in bows:
      for rr in b.r - 5 .. b.r + 5:
        for cc in b.c - 5 .. b.c + 5:
          if bowDanger(rr, cc, b.r, b.c): hazards.add (rr, cc)
    for s in stale:
      hazards.add s
      for d in MoveDeltas:
        hazards.add (s.r + d.dr, s.c + d.dc)
    m.branch = "engage"
    let pp = p   # snapshot for closure
    return planMelee(CenterRow, CenterCol, enemies, tidx, ourDmg,
                     swordHeld, intentKill,
                     proc (r, c: int): bool = pp.passable(r, c),
                     occSoft, hazards, ourHp = float(p.hp))

  # loot sweep (A*-routed)
  if m.lootSweepLeft > 0:
    dec m.lootSweepLeft
    var best = (999, 0, 0)
    for it in p.itemTiles:
      if it.itype != ITool or it.tier <= p.maxToolTier: continue
      let d = abs(it.r - CenterRow) + abs(it.c - CenterCol)
      if d <= 6 and not p.allInventoryFull:
        if d < best[0]: best = (d, it.r, it.c)
    if best[0] < 999:
      let cm = initCostMap(p, bows, meleeCells, occ)
      let (ok, act) = cm.safeRoute(best[1], best[2])
      if ok:
        m.branch = "loot"
        return act

  # recovery: sit and regen
  if m.recovering:
    m.branch = "recover"
    return AtnNoop

  # equipment
  let eact = equipAction(p, swordHeld)
  if eact >= 0 and not p.inCombat:
    m.branch = "equip"
    return eact

  # harvest (A*-routed)
  let target = harvestTarget(p)
  if target.ok:
    let cm = initCostMap(p, bows, meleeCells, occ)
    let (ok, act) = cm.safeRoute(target.r, target.c)
    if ok:
      m.branch = "harvest"
      return act
  elif p.allInventoryFull and not p.inCombat and p.junkSlot >= 0:
    m.selling = true
    m.branch = "sell"
    return AtnSell

  # overwatch / roam
  m.branch = "wander"
  m.wander(p, bows, meleeCells, occ, roam = m.urgentAny(p))

proc act*(m: var Mind, obs: openArray[uint8]): int =
  let p = initPercept(obs)
  m.world.lastAction = m.lastAction
  m.world.observe(p, m.tick)
  inc m.lifeTick
  if m.lifeTick mod StagWindow == 1:
    m.windowStartMin = min(p.combLvl, p.profLvl)
  let action = m.decide(p)
  m.world.noteAction(action)
  m.lastAction = action
  inc m.tick
  action
