"""L2 DuelPlanner: exact-model search for the 1v1 melee game.

The melee response is deterministic (verified 33/33): attack at
Manhattan-1 on its own turn, else one step reducing the strictly
largest gap axis (tie -> column), stalling when blocked. Players act
first within a tick. So the local duel is perfectly simulable: search
our action tree DEPTH plies deep with the exact enemy response and play
the best line. The search rediscovers the diagonal sword dance, the
safe-stall escape, first-strike holds, corner play and trade decisions
from the rules alone.

Intent comes from the executive:
  kill  - low damage aversion, kill bonus dominates
  avoid - high damage aversion, distance shaping dominates

Occupancy cells (other tracks) are soft penalties, never hard walls:
phantom occupancy must not cage the search (measured v3 failure).
"""

from __future__ import annotations

from ..scripted_player import (ATN_ATTACK, ATN_NOOP, MOVE_DELTAS,
                               RUN_OFFSET)

DEPTH = 5
KILL_BONUS = 10.0     # small closure bonus; the real reward is per-hit
HIT_CREDIT = 0.6      # value per point of damage dealt - proportional
                      # credit keeps partial progress visible inside the
                      # horizon (a flat kill bonus made tanking beat the
                      # dance: only the tank-kill fits in 5 plies)
OCC_PENALTY = 2.5
HAZARD_PENALTY = 14.0   # stepping into another enemy's reach costs a
                        # real hit (40-70 pre-armor); the duel target is
                        # modeled exactly, every OTHER threat is a
                        # hazard cell supplied by the executive
W_KILL = 5.0          # per-hit-taken cost > per-hit credit: prefer
                      # clean lines; free-hit dances still land kills
W_AVOID = 9.0


def plan_duel(our_pos: tuple[int, int],
              enemy_pos: tuple[int, int],
              enemy_hp: int,
              our_dmg: int,
              their_dmg: int,
              sword: bool,
              intent_kill: bool,
              passable,          # (r, c) -> bool  (terrain, window-clamped)
              occupied: set[tuple[int, int]] = frozenset(),
              hazards: set[tuple[int, int]] = frozenset(),
              vruns: dict | None = None,
              depth: int = DEPTH) -> int:
    """Best action (int) for the duel state. Coordinates are any
    consistent integer frame (window cells work)."""
    W = W_KILL if intent_kill and our_dmg > 0 else W_AVOID
    dmg = max(our_dmg, 0)

    def in_arc(pr, pc, r, c):
        ar, ac = abs(r - pr), abs(c - pc)
        if ar + ac == 1:
            return True
        return sword and ar == 1 and ac == 1

    def enemy_step(pr, pc, r, c):
        """(nr, nc, attacked) exact chase response."""
        dr, dc = pr - r, pc - c
        if abs(dr) + abs(dc) == 1:
            return r, c, True
        if abs(dr) > abs(dc):
            nr, nc = r + (1 if dr > 0 else -1), c
        else:
            nr, nc = r, c + (1 if dc > 0 else -1)
        if passable(nr, nc) and (nr, nc) != (pr, pc) \
                and (nr, nc) not in occupied:
            return nr, nc, False
        return r, c, False

    def leaf_value(pr, pc, r, c):
        d = max(abs(r - pr), abs(c - pc))
        return d * (0.8 if not intent_kill else 0.05)

    def search(pr, pc, r, c, ehp, d):
        if ehp <= 0:
            return KILL_BONUS + d * 2.0, ATN_NOOP  # already credited
        if d == 0:
            return leaf_value(pr, pc, r, c), ATN_NOOP
        best_v, best_a = -1e18, ATN_NOOP
        cands = [(ATN_NOOP, pr, pc, False)]
        if dmg > 0 and in_arc(pr, pc, r, c):
            cands.append((ATN_ATTACK, pr, pc, True))
        for atn, (mr, mc) in MOVE_DELTAS.items():
            nr, nc = pr + mr, pc + mc
            if passable(nr, nc) and (nr, nc) != (r, c):
                cands.append((atn, nr, nc, False))
                # horizontal dance runs only (local geometry is
                # modeled; vertical runs can outrun the shallow view)
                if mr == 0:
                    n2r, n2c = pr + 2 * mr, pc + 2 * mc
                    if passable(n2r, n2c) and (n2r, n2c) != (r, c) \
                            and (nr, nc) not in occupied:
                        cands.append((atn + RUN_OFFSET, n2r, n2c, False))
        for atn, npr, npc, attacked in cands:
            nehp = ehp - dmg if attacked else ehp
            if nehp <= 0:
                val = KILL_BONUS + dmg * HIT_CREDIT + d * 2.0
            else:
                ner, nec, ehit = enemy_step(npr, npc, r, c)
                sub, _ = search(npr, npc, ner, nec, nehp, d - 1)
                val = sub + (dmg * HIT_CREDIT if attacked else 0.0) \
                    - (W * their_dmg / 10.0 if ehit else 0.0)
            if (npr, npc) != (pr, pc):
                if (npr, npc) in occupied:
                    val -= OCC_PENALTY
                if (npr, npc) in hazards:
                    val -= HAZARD_PENALTY
            if val > best_v:
                best_v, best_a = val, atn
        return best_v, best_a

    _v, act = search(our_pos[0], our_pos[1],
                     enemy_pos[0], enemy_pos[1], enemy_hp, depth)
    return act


def plan_melee(our_pos: tuple[int, int],
               enemies: list,      # [(pos, dmg_to_us, hp)] all tracked melee
               target_idx: int | None,
               our_dmg: int,
               sword: bool,
               intent_kill: bool,
               passable,
               occupied: set[tuple[int, int]] = frozenset(),
               hazards: set[tuple[int, int]] = frozenset(),
               depth: int = DEPTH) -> int:
    """Multi-enemy generalization of plan_duel: every tracked melee is
    stepped by the exact chase model each ply; damage sums over all
    adjacent attackers. target_idx selects which enemy earns attack
    credit (None = pure escape). This is the one search that handles
    duels, multi-chaser escapes, corners and third-party geometry
    together - the single-enemy planner's blind spot was every death
    class that remained."""
    W = W_KILL if intent_kill and our_dmg > 0 else W_AVOID
    dmg = max(our_dmg, 0)
    n = len(enemies)

    def in_arc(pr, pc, r, c):
        ar, ac = abs(r - pr), abs(c - pc)
        if ar + ac == 1:
            return True
        return sword and ar == 1 and ac == 1

    def step_enemies(pr, pc, epos):
        """All enemies act: returns (new_positions, damage_taken)."""
        taken = 0.0
        out = list(epos)
        occ_now = set(out)
        for i in range(n):
            r, c = out[i]
            dr, dc = pr - r, pc - c
            if abs(dr) + abs(dc) == 1:
                taken += enemies[i][1]
                continue
            if abs(dr) > abs(dc):
                nr, nc = r + (1 if dr > 0 else -1), c
            else:
                nr, nc = r, c + (1 if dc > 0 else -1)
            if passable(nr, nc) and (nr, nc) != (pr, pc) \
                    and (nr, nc) not in occ_now:
                occ_now.discard((r, c))
                occ_now.add((nr, nc))
                out[i] = (nr, nc)
        return tuple(out), taken

    def leaf_value(pr, pc, epos):
        d = min((max(abs(r - pr), abs(c - pc)) for r, c in epos),
                default=9)
        return d * (0.8 if not intent_kill else 0.05)

    def search(pr, pc, epos, thp, d):
        if target_idx is not None and thp <= 0:
            return KILL_BONUS + d * 2.0, ATN_NOOP
        if d == 0:
            return leaf_value(pr, pc, epos), ATN_NOOP
        best_v, best_a = -1e18, ATN_NOOP
        tpos = epos[target_idx] if target_idx is not None else None
        cands = [(ATN_NOOP, pr, pc, False)]
        if dmg > 0 and tpos is not None and in_arc(pr, pc, *tpos):
            cands.append((ATN_ATTACK, pr, pc, True))
        epos_set = set(epos)
        for atn, (mr, mc) in MOVE_DELTAS.items():
            nr, nc = pr + mr, pc + mc
            if passable(nr, nc) and (nr, nc) not in epos_set:
                cands.append((atn, nr, nc, False))
                if mr == 0:
                    n2r, n2c = pr + 2 * mr, pc + 2 * mc
                    if passable(n2r, n2c) and (n2r, n2c) not in epos_set \
                            and (nr, nc) not in occupied:
                        cands.append((atn + RUN_OFFSET, n2r, n2c, False))
        for atn, npr, npc, attacked in cands:
            nthp = thp - dmg if attacked else thp
            if target_idx is not None and nthp <= 0:
                val = KILL_BONUS + dmg * HIT_CREDIT + d * 2.0
            else:
                nepos, taken = step_enemies(npr, npc, epos)
                sub, _ = search(npr, npc, nepos, nthp, d - 1)
                val = sub + (dmg * HIT_CREDIT if attacked else 0.0) \
                    - W * taken / 10.0
            if (npr, npc) != (pr, pc):
                if (npr, npc) in occupied:
                    val -= OCC_PENALTY
                if (npr, npc) in hazards:
                    val -= HAZARD_PENALTY
            if val > best_v:
                best_v, best_a = val, atn
        return best_v, best_a

    epos0 = tuple(tuple(e[0]) for e in enemies)
    thp0 = enemies[target_idx][2] if target_idx is not None else 1
    _v, act = search(our_pos[0], our_pos[1], epos0, thp0, depth)
    return act
