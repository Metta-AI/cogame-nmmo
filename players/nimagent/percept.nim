# Obs layout + decoded view of one 1707-byte observation.
# Port of players/scripted_player.py's constants and Percept — the
# python side is tripwired against vendor/upstream in
# tests/test_scripted.py; keep byte-for-byte in sync with it.

const
  WindowRows* = 11
  WindowCols* = 15
  TileBytes* = 10
  CenterRow* = 5
  CenterCol* = 7
  NumScalars* = 47
  ObsSize* = WindowRows * WindowCols * TileBytes + NumScalars + 10
  ScalarsOff* = WindowRows * WindowCols * TileBytes   # 1650
  RewardOff* = ScalarsOff + NumScalars                # 1697

  # tile-cell byte indices
  TbSeason* = 0
  TbTerrain* = 1
  TbItemType* = 2
  TbItemTier* = 3
  TbEntType* = 4
  TbEntElement* = 5
  TbEntDelta* = 6
  TbEntHp* = 7
  TbEntAnim* = 8
  TbEntDir* = 9

  TerrainGrass* = 0
  TerrainDirt* = 1
  TerrainStone* = 2
  TerrainWater* = 3

  # self-scalar offsets
  SComb* = 1
  SElement* = 2
  SDir* = 3
  SAnim* = 4
  SHp* = 5
  SHpMax* = 6
  SProf* = 7
  SUiMode* = 8
  SMarketTier* = 9
  SSellIdx* = 10
  SGold* = 11
  SInCombat* = 12
  SEquipment* = 13     # 5 slots: helm, chest, legs, held, gem
  SInventory* = 18     # 12 slots raw item ids
  SIsEquipped* = 30    # 12 parallel bools
  SRanged* = 43
  SEquipAttack* = 45
  SEquipDefense* = 46

  InventorySlots* = 12
  NumKeySlots* = 9     # only AtnOne..AtnNine exist: slots 0-8 usable

  SlotHelm* = 0
  SlotChest* = 1
  SlotLegs* = 2
  SlotHeld* = 3
  SlotGem* = 4

  # actions
  AtnDown* = 0
  AtnUp* = 1
  AtnRight* = 2
  AtnLeft* = 3
  AtnNoop* = 4
  AtnAttack* = 5
  AtnOne* = 8
  AtnNine* = 16
  AtnBuy* = 20
  AtnSell* = 21
  AtnDownShift* = 22
  RunOffset* = AtnDownShift - AtnDown

  AnimIdle* = 0
  AnimMove* = 1
  AnimDeath* = 5
  AnimRun* = 6

  ModePlay* = 0
  ModeBuyTier* = 1
  ModeBuyItem* = 2
  ModeSellSelect* = 3
  ModeSellPrice* = 4

  EntityPlayer* = 1
  EntityEnemy* = 2

  # item types (ground forms: ore/hilt/wood become armor/sword/bow)
  IHelm* = 1
  IChest* = 2
  ILegs* = 3
  ISword* = 4
  IBow* = 5
  ITool* = 6
  IGemFirst* = 7
  IGemLast* = 10
  IHerb* = 11
  IOre* = 12
  IWood* = 13
  IHilt* = 14
  IN* = 17

  TierExpBase* = 8

const
  MoveDeltas*: array[4, tuple[dr, dc: int]] = [
    (1, 0),    # AtnDown
    (-1, 0),   # AtnUp
    (0, 1),    # AtnRight
    (0, -1)]   # AtnLeft

proc isMove*(a: int): bool = a >= AtnDown and a <= AtnLeft
proc isRun*(a: int): bool =
  a >= AtnDown + RunOffset and a <= AtnLeft + RunOffset
proc moveDelta*(a: int): tuple[dr, dc: int] =
  ## delta of a walk OR run action (caller scales runs by 2)
  if isMove(a): MoveDeltas[a]
  elif isRun(a): MoveDeltas[a - RunOffset]
  else: (0, 0)

proc itemType*(id: int): int = id mod IN
proc itemTier*(id: int): int = id div IN + 1

proc tierLevel*(tier: int): float =
  ## Level equivalent of an item tier: 8 * 2^(tier-1).
  var v = float(TierExpBase)
  for _ in 1 ..< tier: v *= 2.0
  v

proc isArmorType*(t: int): bool = t == IHelm or t == IChest or t == ILegs
proc armorSlot*(t: int): int =
  case t
  of IHelm: SlotHelm
  of IChest: SlotChest
  of ILegs: SlotLegs
  else: -1

type
  Percept* = object
    obs*: array[ObsSize, uint8]

proc initPercept*(buf: openArray[uint8]): Percept =
  doAssert buf.len == ObsSize
  copyMem(addr result.obs[0], unsafeAddr buf[0], ObsSize)

template tile*(p: Percept, r, c, b: int): int =
  int(p.obs[(r * WindowCols + c) * TileBytes + b])

template scalar*(p: Percept, i: int): int =
  int(p.obs[ScalarsOff + i])

proc hp*(p: Percept): int = p.scalar(SHp)
proc combLvl*(p: Percept): int = p.scalar(SComb)
proc profLvl*(p: Percept): int = p.scalar(SProf)
proc uiMode*(p: Percept): int = p.scalar(SUiMode)
proc anim*(p: Percept): int = p.scalar(SAnim)
proc inCombat*(p: Percept): bool = p.scalar(SInCombat) > 0
proc gold*(p: Percept): int = p.scalar(SGold)
proc eqAtk*(p: Percept): int = p.scalar(SEquipAttack)
proc eqDef*(p: Percept): int = p.scalar(SEquipDefense)

proc equipment*(p: Percept, slot: int): int = p.scalar(SEquipment + slot)
proc inventory*(p: Percept, slot: int): int = p.scalar(SInventory + slot)
proc isEquipped*(p: Percept, slot: int): bool =
  p.scalar(SIsEquipped + slot) > 0

proc heldToolTier*(p: Percept): int =
  let held = p.equipment(SlotHeld)
  if held != 0 and itemType(held) == ITool: itemTier(held) else: 0

proc passable*(p: Percept, r, c: int): bool =
  ## Terrain passability; out-of-window counts blocked.
  if r < 0 or r >= WindowRows or c < 0 or c >= WindowCols:
    return false
  let t = p.tile(r, c, TbTerrain)
  t == TerrainGrass or t == TerrainDirt

iterator enemyCells*(p: Percept):
    tuple[r, c, delta, hpBucket, element: int] =
  ## Window tiles whose (stale-prone!) entity bytes claim a live enemy;
  ## corpse anims filtered. element != 0 <=> bow <=> level >= 15.
  for r in 0 ..< WindowRows:
    for c in 0 ..< WindowCols:
      if p.tile(r, c, TbEntType) == EntityEnemy and
          p.tile(r, c, TbEntAnim) != AnimDeath:
        yield (r, c, p.tile(r, c, TbEntDelta), p.tile(r, c, TbEntHp),
               p.tile(r, c, TbEntElement))

iterator itemTiles*(p: Percept): tuple[r, c, itype, tier: int] =
  ## Ground items in the window (byte 2/3 of each tile cell).
  for r in 0 ..< WindowRows:
    for c in 0 ..< WindowCols:
      let t = p.tile(r, c, TbItemType)
      if t != 0:
        yield (r, c, t, p.tile(r, c, TbItemTier) + 1)
