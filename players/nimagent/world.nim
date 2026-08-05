# L1 WorldModel: object tracks from the egocentric obs stream.
# Port of players/dancer/world.py with every session-4 verdict baked in
# (see cogames/nmmo learnings/dancer-v4-verdicts.md):
#
# - The obs entity bytes are WINDOW-anchored residue (a stale obs
#   buffer): never frame-align the diff, never trust imprints without a
#   track.
# - Tracks are created/confirmed ONLY by an observed byte change;
#   between confirmations melee tracks propagate by the exact chase
#   model (verified 33/33) and age out.
# - Absence proofs RE-ANCHOR to the last observed position instead of
#   deleting (chase propagation drifts on unseen blockers).
# - Ghost-strike inference: hp dropped with no adjacent melee track and
#   no aligned bow -> materialize an inferred melee from adjacent
#   residue (the byte-identical-rewrite blindness is real).

import std/tables
import percept

const
  NpcAggro* = 4
  TeleportChangedCells = 45
  TrackExpiryNear = 6
  TrackExpiryFar = 20
  TrackExpiryOut = 40

type
  TrackKind* = enum
    tkMelee = "melee", tkBow = "bow", tkPlayer = "player"

  Track* = ref object
    pos*: tuple[r, c: int]        # frame coords
    obsPos*: tuple[r, c: int]     # last OBSERVED position
    kind*: TrackKind
    delta*: int
    hpBucket*: int
    element*: int
    lastConfirmed*: int
    born*: int
    inferred*: bool
    ghost*: bool                  # born from ghost-strike inference:
                                  # expires fast, no residue-signature
                                  # lease (its own residue would renew
                                  # it forever)

  WorldModel* = object
    pos*: tuple[r, c: int]        # our frame position
    tracks*: seq[Track]
    prevTiles: array[WindowRows * WindowCols * TileBytes, uint8]
    havePrev: bool
    lastAction*: int
    teleported*: bool
    prevHp: int
    originTick*: int
    stillTicks: int               # consecutive ticks with no own move
    lastCorpse*: tuple[tick, wr, wc: int]  # most recent corpse diff
                                  # (window coords at sighting)
    dangerAnchors*: seq[tuple[r, c: int]]  # frame positions of bows
                                  # that actually HIT us this life -
                                  # leashed shooters, static all life

proc reset*(w: var WorldModel, tick: int) =
  w.originTick = tick
  w.pos = (0, 0)
  w.tracks = @[]
  w.havePrev = false
  w.lastAction = AtnNoop
  w.teleported = false
  w.prevHp = -1
  w.stillTicks = 0
  w.dangerAnchors = @[]
  w.lastCorpse = (-999, 0, 0)

proc initWorldModel*(): WorldModel =
  result.reset(0)

# -- helpers -----------------------------------------------------------

proc cheb(a, b: tuple[r, c: int]): int =
  max(abs(a.r - b.r), abs(a.c - b.c))

proc chebW(w: tuple[r, c: int]): int =
  max(abs(w.r - CenterRow), abs(w.c - CenterCol))

proc toWindow*(w: WorldModel, fpos: tuple[r, c: int]):
    tuple[ok: bool, r, c: int] =
  let r = fpos.r - w.pos.r + CenterRow
  let c = fpos.c - w.pos.c + CenterCol
  if r >= 0 and r < WindowRows and c >= 0 and c < WindowCols:
    (true, r, c)
  else:
    (false, 0, 0)

proc toFrame*(w: WorldModel, r, c: int): tuple[r, c: int] =
  (w.pos.r + r - CenterRow, w.pos.c + c - CenterCol)

proc chaseStep(w: WorldModel, epos: tuple[r, c: int]): tuple[r, c: int] =
  ## Exact chase rule, one enemy move toward us (frame coords).
  let dr = w.pos.r - epos.r
  let dc = w.pos.c - epos.c
  if abs(dr) + abs(dc) <= 1:
    return epos                    # adjacent: it attacks
  if abs(dr) > abs(dc):
    (epos.r + (if dr > 0: 1 else: -1), epos.c)
  else:
    (epos.r, epos.c + (if dc > 0: 1 else: -1))

proc ownShift(w: WorldModel, p: Percept): tuple[r, c: int] =
  var walk = w.lastAction
  if isRun(walk): walk -= RunOffset
  if not isMove(walk): return (0, 0)
  let d = MoveDeltas[walk]
  if p.anim == AnimMove: (d.dr, d.dc)
  elif p.anim == AnimRun: (2 * d.dr, 2 * d.dc)
  else: (0, 0)

# -- per-tick update ---------------------------------------------------

proc observe*(w: var WorldModel, p: Percept, tick: int) =
  ## Advance the model with this tick's percept. Call exactly once per
  ## tick, BEFORE reading views, AFTER handling a life reset.
  let shift = w.ownShift(p)
  w.teleported = false
  if shift == (0, 0): inc w.stillTicks
  else: w.stillTicks = 0

  # entity-byte window diff (bytes 4..9 per cell); center cell is us
  var changed: array[WindowRows * WindowCols, bool]
  var nChanged = 0
  if w.havePrev:
    for r in 0 ..< WindowRows:
      for c in 0 ..< WindowCols:
        if r == CenterRow and c == CenterCol: continue
        let base = (r * WindowCols + c) * TileBytes
        for b in TbEntType .. TbEntDir:
          if p.obs[base + b] != w.prevTiles[base + b]:
            changed[r * WindowCols + c] = true
            inc nChanged
            break

  # teleportitis: window content jumps wholesale; start clean
  if nChanged >= TeleportChangedCells:
    copyMem(addr w.prevTiles[0], unsafeAddr p.obs[0],
            WindowRows * WindowCols * TileBytes)
    w.havePrev = true
    w.tracks = @[]
    w.pos = (0, 0)
    w.originTick = tick
    w.teleported = true
    w.prevHp = p.hp
    return

  w.pos = (w.pos.r + shift.r, w.pos.c + shift.c)

  # 1. chase-propagate melee tracks near us, unless another player
  # track contests the chase target (then hold position)
  var playerPos: seq[tuple[r, c: int]]
  for t in w.tracks:
    if t.kind == tkPlayer: playerPos.add t.pos
  for t in w.tracks:
    t.inferred = true
    if t.kind == tkMelee:
      let win = w.toWindow(t.pos)
      if win.ok and chebW((win.r, win.c)) <= NpcAggro:
        var contested = false
        for pp in playerPos:
          if cheb(t.pos, pp) <= 5: contested = true; break
        if contested: continue
        let np = w.chaseStep(t.pos)
        let nw = w.toWindow(np)
        if not nw.ok or p.passable(nw.r, nw.c):
          t.pos = np

  # 2. observation update: every changed cell with entity bytes is a
  # confirmed presence NOW; match <=2 frame cells to a same-kind track
  type ObsHit = tuple[r, c, entType: int]
  var hits: seq[ObsHit]
  for r in 0 ..< WindowRows:
    for c in 0 ..< WindowCols:
      if not changed[r * WindowCols + c]: continue
      let et = p.tile(r, c, TbEntType)
      if et == 0: continue
      if p.tile(r, c, TbEntAnim) == AnimDeath:
        # corpse frame: kill the matching track if any. Loot drops at
        # kill sites and despawns in 20 ticks - note the sighting.
        let fpos = w.toFrame(r, c)
        var kept: seq[Track]
        for t in w.tracks:
          if cheb(t.pos, fpos) > 1: kept.add t
        w.tracks = kept
        w.lastCorpse = (tick, r, c)
        continue
      hits.add (r, c, et)

  var claimed: seq[bool] = newSeq[bool](w.tracks.len)
  for h in hits:
    let fpos = w.toFrame(h.r, h.c)
    let kind =
      if h.entType == EntityPlayer: tkPlayer
      elif p.tile(h.r, h.c, TbEntElement) != 0: tkBow
      else: tkMelee
    var bestI = -1
    var bestD = 99
    for i, t in w.tracks:
      if claimed[i] or t.kind != kind: continue
      let d = cheb(t.pos, fpos)
      if d <= 2 and d < bestD:
        bestI = i; bestD = d
    if bestI >= 0:
      let t = w.tracks[bestI]
      claimed[bestI] = true
      t.pos = fpos
      t.obsPos = fpos
      t.inferred = false
      t.ghost = false
      t.lastConfirmed = tick
      if kind != tkPlayer:
        t.delta = p.tile(h.r, h.c, TbEntDelta)
        t.hpBucket = p.tile(h.r, h.c, TbEntHp)
        t.element = p.tile(h.r, h.c, TbEntElement)
    else:
      w.tracks.add Track(
        pos: fpos, obsPos: fpos, kind: kind,
        delta: p.tile(h.r, h.c, TbEntDelta),
        hpBucket: p.tile(h.r, h.c, TbEntHp),
        element: p.tile(h.r, h.c, TbEntElement),
        lastConfirmed: tick, born: tick, inferred: false)
      # tracks grow mid-loop; keep the claim mask parallel (a track
      # created by an earlier hit IS a legal match target for later
      # hits this tick - python-tracker semantics)
      claimed.add false

  # 3. expiry + re-anchor + dedup
  var kept: seq[Track]
  for t in w.tracks:
    let win = w.toWindow(t.pos)
    let age = tick - t.lastConfirmed
    if not win.ok:
      if age <= TrackExpiryOut: kept.add t
      continue
    let man = abs(win.r - CenterRow) + abs(win.c - CenterCol)
    var nearPlayer = false
    for pp in playerPos:
      if cheb(t.pos, pp) <= 2: nearPlayer = true; break
    # absence proofs disprove the PROPAGATED position, not the enemy:
    # re-anchor to obsPos; delete only when the proof fires there too.
    # MOTION-GATED: "an adjacent melee would be attacking (in_combat)"
    # is only evidence while WE are stationary - a chaser legally
    # alternates moves with a moving player and never attacks, so the
    # proof fired on real chasers during walk-flees and teleported
    # their tracks away (measured kill chain, session 5 probe).
    let stationary = w.stillTicks >= 2
    if t.kind == tkMelee and man == 1 and not p.inCombat and
        age >= 2 and not nearPlayer and stationary:
      if t.pos != t.obsPos:
        t.pos = t.obsPos
        kept.add t
      continue
    if t.kind == tkMelee and chebW((win.r, win.c)) <= 2 and
        t.inferred and age >= 3 and not p.inCombat and not nearPlayer and
        stationary:
      if t.pos != t.obsPos:
        t.pos = t.obsPos
        kept.add t
      continue
    # soft-confirm: a stationary enemy is diff-invisible; if the cell
    # still carries this track's exact signature, extend the lease
    let sigOk = p.tile(win.r, win.c, TbEntType) != 0 and
      p.tile(win.r, win.c, TbEntElement) == t.element and
      p.tile(win.r, win.c, TbEntHp) == t.hpBucket
    var limit =
      if chebW((win.r, win.c)) <= 4:
        if p.inCombat: 20
        elif sigOk: 14
        else: TrackExpiryNear
      else:
        TrackExpiryFar
    if t.ghost:
      # ghost-born tracks sit on their own residue: the signature
      # lease would renew them forever. Byte-confirmation clears the
      # flag; otherwise they live 4 ticks.
      limit = 4
    if age <= limit: kept.add t
  # dedup: same-kind tracks on one cell merge (keep fresher)
  var byCell = initTable[(int, int, TrackKind), Track]()
  for t in kept:
    let key = (t.pos.r, t.pos.c, t.kind)
    if key notin byCell or t.lastConfirmed > byCell[key].lastConfirmed:
      byCell[key] = t
  w.tracks = @[]
  for t in byCell.values: w.tracks.add t

  # ghost-strike inference: hp dropped, yet no tracked melee adjacent
  # and no tracked bow aligned in range - the attacker is
  # diff-invisible; materialize an inferred melee from adjacent residue
  if w.prevHp >= 0 and p.hp < w.prevHp:
    var adjMelee, bowAligned = false
    for t in w.tracks:
      let win = w.toWindow(t.pos)
      if not win.ok: continue
      if t.kind == tkMelee and
          abs(win.r - CenterRow) + abs(win.c - CenterCol) <= 1:
        adjMelee = true
      if t.kind == tkBow and chebW((win.r, win.c)) <= 4 and
          (win.r == CenterRow or win.c == CenterCol):
        bowAligned = true
        if t.pos notin w.dangerAnchors:
          w.dangerAnchors.add t.pos
    if not adjMelee and not bowAligned:
      for d in MoveDeltas:
        let r = CenterRow + d.dr
        let c = CenterCol + d.dc
        if p.tile(r, c, TbEntType) != EntityEnemy or
            p.tile(r, c, TbEntElement) != 0:
          continue
        let fpos = w.toFrame(r, c)
        var exists = false
        for t in w.tracks:
          if t.kind == tkMelee and t.pos == fpos: exists = true; break
        if exists: continue
        w.tracks.add Track(
          pos: fpos, obsPos: fpos, kind: tkMelee,
          delta: p.tile(r, c, TbEntDelta),
          hpBucket: p.tile(r, c, TbEntHp), element: 0,
          lastConfirmed: tick, born: tick, inferred: true,
          ghost: true)
      # bow ghost-strike: a stationary aligned bow shoots from
      # stillness with zero diffs (never tracked). After an
      # unexplained hit, aligned bow-signature residue within its
      # 4-range IS the shooter - materialize it.
      for dr in -4 .. 4:
        for dc in -4 .. 4:
          if dr != 0 and dc != 0: continue
          if dr == 0 and dc == 0: continue
          let r = CenterRow + dr
          let c = CenterCol + dc
          if r < 0 or r >= WindowRows or c < 0 or c >= WindowCols:
            continue
          if p.tile(r, c, TbEntType) != EntityEnemy or
              p.tile(r, c, TbEntElement) == 0 or
              p.tile(r, c, TbEntAnim) == AnimDeath:
            continue
          let fpos = w.toFrame(r, c)
          var exists = false
          for t in w.tracks:
            if t.kind == tkBow and cheb(t.pos, fpos) <= 1:
              exists = true; break
          if exists: continue
          w.tracks.add Track(
            pos: fpos, obsPos: fpos, kind: tkBow,
            delta: p.tile(r, c, TbEntDelta),
            hpBucket: p.tile(r, c, TbEntHp),
            element: p.tile(r, c, TbEntElement),
            lastConfirmed: tick, born: tick, inferred: true,
            ghost: true)
          if fpos notin w.dangerAnchors:
            w.dangerAnchors.add fpos
  w.prevHp = p.hp

  copyMem(addr w.prevTiles[0], unsafeAddr p.obs[0],
          WindowRows * WindowCols * TileBytes)
  w.havePrev = true

proc noteAction*(w: var WorldModel, action: int) =
  w.lastAction = action

# -- views -------------------------------------------------------------

iterator enemies*(w: WorldModel, kind = tkMelee):
    tuple[r, c: int, t: Track] =
  ## In-window enemy tracks of one kind (window coords).
  for t in w.tracks:
    if t.kind != kind: continue
    let win = w.toWindow(t.pos)
    if win.ok:
      yield (win.r, win.c, t)

iterator enemiesExtended*(w: WorldModel, kind = tkBow, margin = 4):
    tuple[r, c: int, t: Track] =
  ## Like enemies() but includes tracks up to margin cells OUTSIDE the
  ## window (coords may exceed bounds) - run-safety for shallow rows.
  for t in w.tracks:
    if t.kind != kind: continue
    let r = t.pos.r - w.pos.r + CenterRow
    let c = t.pos.c - w.pos.c + CenterCol
    if r >= -margin and r < WindowRows + margin and
        c >= -margin and c < WindowCols + margin:
      yield (r, c, t)

proc occupancy*(w: WorldModel): seq[tuple[r, c: int]] =
  ## Window cells our move() would bounce off (all tracks).
  for t in w.tracks:
    let win = w.toWindow(t.pos)
    if win.ok: result.add (win.r, win.c)
