# L2 planner: exact-model search for the local melee game.
# Port of players/dancer/planner.py plan_melee (the N-enemy search that
# passed every MiniMelee gate and stress variant on 2026-08-03).
#
# Verified rules: players act first within a tick; each melee attacks
# at Manhattan-1 on its turn, else steps 1 along the strictly-largest
# gap axis (tie -> column), stalling when blocked by terrain, the
# player, or another enemy; enemy-enemy collisions block (a cell
# vacated earlier the same tick is free).

import percept

const
  Depth* = 5
  KillBonus = 10.0
  HitCredit = 0.6
  OccPenalty = 2.5
  HazardPenalty = 14.0
  WKill = 5.0
  WAvoid = 9.0
  MaxEnemies* = 8

type
  PlanEnemy* = object
    r*, c*: int
    dmg*: float          # their damage to us per hit
    hp*: int
    ranged*: bool        # bow enemy: shoots when aligned <=4, else
                         # steps along the SHORTEST axis toward
                         # alignment; NO melee attack (enemy_ai)

  Cell = tuple[r, c: int]

  PlanCtx = object
    enemies: seq[PlanEnemy]
    targetIdx: int       # -1 = pure escape
    dmg: float           # our damage per hit (0 = cannot hurt)
    sword: bool
    bow: bool            # held bow: attack arc is ON-AXIS <= 4
                         # (ATTACK_BOW, nmmo3.h:891) - the kite weapon
    intentKill: bool
    w: float
    hp: float            # our hp: cumulative damage >= hp is DEATH,
                         # not a recoverable cost
    passable: proc (r, c: int): bool {.closure.}
    occupied: seq[Cell]
    hazards: seq[Cell]

proc inArc(ctx: PlanCtx, pr, pc, r, c: int): bool =
  let ar = abs(r - pr)
  let ac = abs(c - pc)
  if ctx.bow:
    return (ar == 0 or ac == 0) and max(ar, ac) <= 4 and ar + ac > 0
  if ar + ac == 1: return true
  ctx.sword and ar == 1 and ac == 1

proc stepEnemies(ctx: PlanCtx, pr, pc: int, epos: var seq[Cell],
                 alive: seq[bool]): float =
  ## All live enemies act vs player at (pr,pc): returns damage taken;
  ## mutates epos in place. List order; vacated cells free up.
  ## AGGRO CUTOFF (enemy_ai, nmmo3.h:1592): an enemy only acts when a
  ## player is inside its +-4 box - at Chebyshev >= 5 it reverts to
  ## leashed wandering (modeled as a stall). Escape is absolute.
  var occ: seq[Cell]
  for i in 0 ..< epos.len:
    if alive[i]: occ.add epos[i]
  for i in 0 ..< epos.len:
    if not alive[i]: continue
    let (r, c) = epos[i]
    let dr = pr - r
    let dc = pc - c
    if max(abs(dr), abs(dc)) > 4:
      continue                       # outside its aggro box: idle
    var nr = r
    var nc = c
    if ctx.enemies[i].ranged:
      # bow: shoot when aligned (incl. adjacent); else step along the
      # SHORTEST axis toward alignment. No melee attack.
      if dr == 0 or dc == 0:
        result += ctx.enemies[i].dmg
        continue
      if abs(dr) > abs(dc):
        nc = c + (if dc > 0: 1 else: -1)
      else:
        nr = r + (if dr > 0: 1 else: -1)
    else:
      if abs(dr) + abs(dc) == 1:
        result += ctx.enemies[i].dmg
        continue
      if abs(dr) > abs(dc):
        nr = r + (if dr > 0: 1 else: -1)
      else:
        nc = c + (if dc > 0: 1 else: -1)
    if ctx.passable(nr, nc) and (nr, nc) != (pr, pc) and
        (nr, nc) notin occ:
      let idx = occ.find((r, c))
      if idx >= 0: occ.del(idx)
      occ.add (nr, nc)
      epos[i] = (nr, nc)

proc leafValue(ctx: PlanCtx, pr, pc: int, epos: seq[Cell],
               alive: seq[bool]): float =
  var d = 9
  for i in 0 ..< epos.len:
    if alive[i]:
      d = min(d, max(abs(epos[i].r - pr), abs(epos[i].c - pc)))
  result = float(d) * (if ctx.intentKill: 0.05 else: 0.8)
  if d >= 5 and not ctx.intentKill:
    result += 2.0       # outside every aggro box: fully disengaged

proc search(ctx: PlanCtx, pr, pc: int, epos: seq[Cell],
            alive: seq[bool], thp: float, d: int,
            hpLeft: float): tuple[v: float, a: int] =
  if ctx.targetIdx >= 0 and thp <= 0:
    return (KillBonus + float(d) * 2.0, AtnNoop)
  if d == 0:
    return (ctx.leafValue(pr, pc, epos, alive), AtnNoop)
  var bestV = -1e18
  var bestA = AtnNoop
  # candidates: (action, npr, npc, attacked)
  var cands: seq[tuple[a, npr, npc: int, attacked: bool]]
  cands.add (AtnNoop, pr, pc, false)
  if ctx.dmg > 0 and ctx.targetIdx >= 0 and alive[ctx.targetIdx] and
      ctx.inArc(pr, pc, epos[ctx.targetIdx].r, epos[ctx.targetIdx].c):
    cands.add (AtnAttack, pr, pc, true)
  for ai in 0 .. 3:
    let (mr, mc) = MoveDeltas[ai]
    let nr = pr + mr
    let nc = pc + mc
    var blockedByEnemy = false
    for i in 0 ..< epos.len:
      if alive[i] and epos[i] == (nr, nc): blockedByEnemy = true; break
    if ctx.passable(nr, nc) and not blockedByEnemy:
      cands.add (ai, nr, nc, false)
      if mr == 0:
        # horizontal dance runs only (vertical runs can outrun the
        # shallow +-5-row view)
        let n2r = pr + 2 * mr
        let n2c = pc + 2 * mc
        var blocked2 = false
        for i in 0 ..< epos.len:
          if alive[i] and epos[i] == (n2r, n2c): blocked2 = true; break
        if ctx.passable(n2r, n2c) and not blocked2 and
            (nr, nc) notin ctx.occupied:
          cands.add (ai + RunOffset, n2r, n2c, false)
  for cand in cands:
    var val: float
    let nthp = if cand.attacked: thp - ctx.dmg else: thp
    if ctx.targetIdx >= 0 and nthp <= 0:
      val = KillBonus + ctx.dmg * HitCredit + float(d) * 2.0
    else:
      var nepos = epos
      let taken = ctx.stepEnemies(cand.npr, cand.npc, nepos, alive)
      let nhpLeft = hpLeft - taken
      if nhpLeft <= 0:
        # this line kills us: terminal, catastrophic, sooner is worse
        val = -1000.0 - float(d)
      else:
        let sub = ctx.search(cand.npr, cand.npc, nepos, alive, nthp,
                             d - 1, nhpLeft)
        val = sub.v +
          (if cand.attacked: ctx.dmg * HitCredit else: 0.0) -
          ctx.w * taken / 10.0
    if (cand.npr, cand.npc) != (pr, pc):
      if (cand.npr, cand.npc) in ctx.occupied: val -= OccPenalty
      if (cand.npr, cand.npc) in ctx.hazards: val -= HazardPenalty
    if val > bestV:
      bestV = val
      bestA = cand.a
  (bestV, bestA)

proc planMelee*(ourR, ourC: int,
                enemies: seq[PlanEnemy],
                targetIdx: int,          # -1 = pure escape
                ourDmg: float,
                sword: bool,
                intentKill: bool,
                passable: proc (r, c: int): bool {.closure.},
                occupied: seq[Cell] = @[],
                hazards: seq[Cell] = @[],
                depth = Depth,
                ourHp = 99.0,
                bowHeld = false): int =
  ## Best action for the local melee state. Any consistent integer
  ## frame (window cells work).
  var ctx = PlanCtx(
    enemies: enemies, targetIdx: targetIdx,
    dmg: max(ourDmg, 0.0), sword: sword, intentKill: intentKill,
    w: (if intentKill and ourDmg > 0: WKill else: WAvoid),
    bow: bowHeld,
    hp: ourHp,
    passable: passable, occupied: occupied, hazards: hazards)
  var epos: seq[Cell]
  var alive: seq[bool]
  for e in enemies:
    epos.add (e.r, e.c)
    alive.add true
  let thp = if targetIdx >= 0: float(enemies[targetIdx].hp) else: 1.0
  ctx.search(ourR, ourC, epos, alive, thp, depth, ourHp).a
