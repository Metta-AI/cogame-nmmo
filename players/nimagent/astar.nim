# A* router over the 11x15 window with threat-cost fields.
#
# Replaces the greedy one-step safe_step filter of the python dancer:
# greedy vetoed unsafe steps and stalled; A* routes AROUND bow funnel
# lines, melee reach and occupied cells, only crossing a threat cell
# when no clean path exists (threat costs are soft, high). Walks only —
# every measured one-shot death involved a run chosen while the killer
# was untracked (session-4 verdict).

import std/heapqueue
import percept

const
  CostStep = 10          # base cost per step (int-scaled)
  CostOcc* = 25          # another entity's cell (moves bounce off)
  CostBowFunnel* = 120   # aligned-with-bow cells within shot range
  CostMeleeReach* = 200  # Manhattan-1 of a tracked melee
  CostMeleeNear* = 40    # Chebyshev-1 ring around melee reach
  Infinity = 1_000_000

type
  CostMap* = object
    cost*: array[WindowRows * WindowCols, int32]  # additional per-cell
    blocked*: array[WindowRows * WindowCols, bool]

proc idx(r, c: int): int {.inline.} = r * WindowCols + c

proc initCostMap*(p: Percept,
                  bows: openArray[tuple[r, c: int]],
                  melee: openArray[tuple[r, c: int]],
                  occ: openArray[tuple[r, c: int]]): CostMap =
  ## Threat-cost field over the current window. Bow coords may lie
  ## outside the window (extended tracks); their funnel still projects
  ## into it.
  for r in 0 ..< WindowRows:
    for c in 0 ..< WindowCols:
      if not p.passable(r, c):
        result.blocked[idx(r, c)] = true
  for cell in occ:
    if cell.r in 0 ..< WindowRows and cell.c in 0 ..< WindowCols:
      result.cost[idx(cell.r, cell.c)] += CostOcc
  for b in bows:
    # funnel: same row or column within Chebyshev 6, one-step-to-
    # aligned included via the |min|<=1 band (python _veto rule)
    for r in 0 ..< WindowRows:
      for c in 0 ..< WindowCols:
        let dr = abs(r - b.r)
        let dc = abs(c - b.c)
        if max(dr, dc) <= 6 and min(dr, dc) <= 1:
          result.cost[idx(r, c)] += int32(CostBowFunnel)
  for m in melee:
    for r in max(0, m.r - 2) .. min(WindowRows - 1, m.r + 2):
      for c in max(0, m.c - 2) .. min(WindowCols - 1, m.c + 2):
        let man = abs(r - m.r) + abs(c - m.c)
        if man <= 1:
          result.cost[idx(r, c)] += int32(CostMeleeReach)
        elif max(abs(r - m.r), abs(c - m.c)) <= 2:
          result.cost[idx(r, c)] += int32(CostMeleeNear)

proc route*(cm: CostMap, tr, tc: int):
    tuple[found: bool, firstAct: int, cost: int] =
  ## A* from the window center to (tr, tc); returns the FIRST action of
  ## the cheapest path. Manhattan-distance heuristic (admissible with
  ## CostStep scaling).
  if tr < 0 or tr >= WindowRows or tc < 0 or tc >= WindowCols or
      cm.blocked[idx(tr, tc)]:
    return (false, AtnNoop, 0)
  var g: array[WindowRows * WindowCols, int32]
  var firstAct: array[WindowRows * WindowCols, int8]
  for i in 0 ..< g.len: g[i] = Infinity
  var pq = initHeapQueue[(int32, int16)]()   # (f, cellIdx)
  let start = idx(CenterRow, CenterCol)
  g[start] = 0
  firstAct[start] = int8(AtnNoop)
  proc h(r, c: int): int32 =
    int32((abs(r - tr) + abs(c - tc)) * CostStep)
  pq.push (h(CenterRow, CenterCol), int16(start))
  while pq.len > 0:
    let (f, cellI) = pq.pop()
    let cell = int(cellI)
    let r = cell div WindowCols
    let c = cell mod WindowCols
    if f - h(r, c) > g[cell]: continue    # stale entry
    if r == tr and c == tc:
      return (true, int(firstAct[cell]), int(g[cell]))
    for ai in 0 .. 3:
      let d = MoveDeltas[ai]
      let nr = r + d.dr
      let nc = c + d.dc
      if nr < 0 or nr >= WindowRows or nc < 0 or nc >= WindowCols:
        continue
      let ni = idx(nr, nc)
      if cm.blocked[ni]: continue
      let ng = g[cell] + int32(CostStep) + cm.cost[ni]
      if ng < g[ni]:
        g[ni] = ng
        firstAct[ni] = if cell == start: int8(ai) else: firstAct[cell]
        pq.push (ng + h(nr, nc), int16(ni))
  (false, AtnNoop, 0)

proc safeRoute*(cm: CostMap, tr, tc: int,
                maxThreatCost = CostBowFunnel - 1):
    tuple[ok: bool, act: int] =
  ## Route whose total added threat cost stays under the given bound —
  ## i.e. a path that never crosses a bow funnel or melee reach (the
  ## greedy filter's veto semantics, but with detours). Falls back to
  ## "no move" like safe_step returning None.
  let (found, act, cost) = cm.route(tr, tc)
  if not found: return (false, AtnNoop)
  let man = abs(tr - CenterRow) + abs(tc - CenterCol)
  let threatCost = cost - man * CostStep
  if threatCost > maxThreatCost: return (false, AtnNoop)
  if act == AtnNoop and man > 0: return (false, AtnNoop)
  (true, act)
