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
            if (npr, npc) != (pr, pc) and (npr, npc) in occupied:
                val -= OCC_PENALTY
            if val > best_v:
                best_v, best_a = val, atn
        return best_v, best_a

    _v, act = search(our_pos[0], our_pos[1],
                     enemy_pos[0], enemy_pos[1], enemy_hp, depth)
    return act
