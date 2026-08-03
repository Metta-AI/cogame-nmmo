"""L1 WorldModel: object tracks from the egocentric NMMO3 obs stream.

The obs entity bytes are write-only residue: a window cell's entity
bytes are rewritten every tick an entity stands on the tile under it,
and NEVER cleared. Cell-level liveness classification was measured
unreliable (v2/v3); this tracker works at the object level instead:

- A track is created/confirmed ONLY by an observed byte change (a
  change at a window offset proves a write, which proves presence).
- Between confirmations a track is propagated by the exact chase model
  (melee within aggro range chase us deterministically - verified
  33/33) or held stationary, aging until expiry.
- Imprints with no track are residue and are exposed nowhere.

Frame: cumulative-offset coordinates anchored at this life's spawn.
Our own position advances by the verified own-shift rule (anim proves
last move success + speed; the action supplies direction). Teleportitis
breaks the anchor - detected via a changed-cell explosion and handled
by a full reset.

Consumers get window-relative views each tick: melee tracks, bow
tracks (element byte != 0 <=> level >= 15 <=> ranged), player tracks,
and an occupancy set for move planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scripted_player import (ANIM_DEATH, CENTER_COL, CENTER_ROW,
                               ENTITY_ENEMY, ENTITY_PLAYER, MOVE_DELTAS,
                               Percept, RUN_OFFSET, WINDOW_COLS, WINDOW_ROWS)

ANIM_IDLE = 0
ANIM_MOVE = 1
ANIM_RUN = 6

NPC_AGGRO = 4
TELEPORT_CHANGED_CELLS = 45   # diff explosion => our anchor jumped
TRACK_EXPIRY_NEAR = 6         # unconfirmed ticks for a track within cheb 4
                              # (extended while in_combat: fight evidence)
TRACK_EXPIRY_FAR = 20         # unconfirmed ticks before a far track dies
TRACK_EXPIRY_OUT = 40         # ... once outside the window entirely


@dataclass
class Track:
    pos: tuple[int, int]          # frame coords
    kind: str                     # "melee" | "bow" | "player"
    delta: int                    # last seen delta bucket (enemies)
    hp_bucket: int
    element: int
    last_confirmed: int           # tick of last byte-change confirmation
    born: int
    inferred: bool = False        # position from model, not observation
    obs_pos: tuple[int, int] = (0, 0)   # last OBSERVED position


class WorldModel:
    def __init__(self):
        self.reset(0)

    def reset(self, tick: int) -> None:
        self.origin_tick = tick
        self.pos = (0, 0)                 # our frame position
        self.tracks: list[Track] = []
        self.prev_tiles: np.ndarray | None = None
        self.last_action: int = 4
        self.teleported = False

    # -- per-tick update -------------------------------------------------

    def observe(self, p: Percept, tick: int) -> None:
        self._in_combat = p.in_combat
        """Advance the model with this tick's percept. Call exactly once
        per tick, BEFORE reading any view, AFTER a life reset was
        handled by calling reset()."""
        shift = self._own_shift(p)
        self.teleported = False

        if self.prev_tiles is None:
            changed = np.zeros((WINDOW_ROWS, WINDOW_COLS), dtype=bool)
        else:
            changed = (p.tiles[:, :, 4:]
                       != self.prev_tiles[:, :, 4:]).any(axis=2)
            changed[CENTER_ROW, CENTER_COL] = False   # that's us
        n_changed = int(changed.sum())

        # teleportitis: the window content jumps wholesale; our anchor
        # and every track are garbage. Start clean.
        if n_changed >= TELEPORT_CHANGED_CELLS:
            self.prev_tiles = p.tiles.copy()
            self.tracks = []
            self.pos = (0, 0)
            self.origin_tick = tick
            self.teleported = True
            return

        self.pos = (self.pos[0] + shift[0], self.pos[1] + shift[1])

        # 1. model-propagate melee tracks that are inside our aggro
        # neighborhood: they chase deterministically - but only when WE
        # are unambiguously the chase target. Enemies attack the first
        # player in their row-major scan box; if another player track is
        # near the enemy, its behavior is unpredictable: hold position.
        # A predicted step onto impassable terrain stalls (move() fails).
        player_pos = [t.pos for t in self.tracks if t.kind == "player"]
        for t in self.tracks:
            t.inferred = True
            if t.kind == "melee":
                w = self._to_window(t.pos)
                if w is not None and self._cheb(w) <= NPC_AGGRO:
                    contested = any(self._dist(t.pos, pp) <= 5
                                    for pp in player_pos)
                    if contested:
                        continue
                    np_pos = self._chase_step(t.pos)
                    nw = self._to_window(np_pos)
                    if nw is None or p.passable(nw[0], nw[1]):
                        t.pos = np_pos

        # 2. observation update: every changed cell with entity bytes is
        # a confirmed presence NOW. Match to nearest track (<= 2 frame
        # cells - one own move + one enemy move), else new track.
        obs_hits = []
        for r, c in zip(*np.nonzero(changed)):
            ent_type = int(p.tiles[r, c, 4])
            if ent_type == 0:
                continue
            anim = int(p.tiles[r, c, 8])
            if anim == ANIM_DEATH:
                # corpse frame: kill the matching track if any
                fpos = self._to_frame((int(r), int(c)))
                self.tracks = [t for t in self.tracks
                               if self._dist(t.pos, fpos) > 1]
                continue
            obs_hits.append((int(r), int(c), ent_type))

        claimed: set[int] = set()
        for r, c, ent_type in obs_hits:
            fpos = self._to_frame((r, c))
            kind = ("player" if ent_type == ENTITY_PLAYER else
                    ("bow" if int(p.tiles[r, c, 5]) != 0 else "melee"))
            best_i, best_d = None, 99
            for i, t in enumerate(self.tracks):
                if i in claimed or t.kind != kind:
                    continue
                d = self._dist(t.pos, fpos)
                if d <= 2 and d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                t = self.tracks[best_i]
                claimed.add(best_i)
                t.pos = fpos
                t.obs_pos = fpos
                t.inferred = False
                t.last_confirmed = tick
                if kind != "player":
                    t.delta = int(p.tiles[r, c, 6])
                    t.hp_bucket = int(p.tiles[r, c, 7])
                    t.element = int(p.tiles[r, c, 5])
            else:
                self.tracks.append(Track(
                    pos=fpos, obs_pos=fpos, kind=kind,
                    delta=int(p.tiles[r, c, 6]),
                    hp_bucket=int(p.tiles[r, c, 7]),
                    element=int(p.tiles[r, c, 5]),
                    last_confirmed=tick, born=tick, inferred=False))

        # 3. expiry + dedup. Near tracks are the dangerous ones AND the
        # phantom-prone ones: a chase-propagated track whose enemy left
        # converges on us and would sit forever. Evidence rules:
        #  - a track at Manhattan-1 while we are NOT in combat for 2+
        #    ticks is provably absent (adjacent melee attack every tick)
        #  - otherwise near tracks live TRACK_EXPIRY_NEAR unconfirmed
        #    ticks (extended while in_combat: the fight is the proof)
        kept: list[Track] = []
        for t in self.tracks:
            w = self._to_window(t.pos)
            age = tick - t.last_confirmed
            if w is None:
                if age <= TRACK_EXPIRY_OUT:
                    kept.append(t)
                continue
            man = abs(w[0] - CENTER_ROW) + abs(w[1] - CENTER_COL)
            near_player = any(self._dist(t.pos, pp) <= 2
                              for pp in player_pos)
            if t.kind == "melee" and man == 1 and not p.in_combat \
                    and age >= 2 and not near_player:
                continue
            # an inferred melee track at point-blank range that has not
            # been confirmed for several ticks is a phantom: a real one
            # would be attacking (in_combat) or moving (diff-visible)
            if t.kind == "melee" and self._cheb(w) <= 2 and t.inferred \
                    and age >= 3 and not p.in_combat and not near_player:
                continue
            # soft-confirm: a stationary enemy writes byte-identical
            # imprints (diff-invisible); if the cell still carries this
            # track's exact signature, extend its lease
            sig_ok = (int(p.tiles[w[0], w[1], 4]) != 0
                      and int(p.tiles[w[0], w[1], 5]) == t.element
                      and int(p.tiles[w[0], w[1], 7]) == t.hp_bucket)
            if self._cheb(w) <= 4:
                limit = 20 if p.in_combat else (
                    14 if sig_ok else TRACK_EXPIRY_NEAR)
            else:
                limit = TRACK_EXPIRY_FAR
            if age <= limit:
                kept.append(t)
        # dedup: two same-kind tracks on one cell merge (keep fresher)
        by_cell: dict[tuple[int, int, str], Track] = {}
        for t in kept:
            key = (t.pos[0], t.pos[1], t.kind)
            cur = by_cell.get(key)
            if cur is None or t.last_confirmed > cur.last_confirmed:
                by_cell[key] = t
        self.tracks = list(by_cell.values())

        self.prev_tiles = p.tiles.copy()

    def note_action(self, action: int) -> None:
        """Record the action we RETURNED this tick (used by next tick's
        own-shift estimation)."""
        self.last_action = action

    # -- views -----------------------------------------------------------

    def enemies(self, kind: str | None = None):
        """[(wr, wc, track)] for enemy tracks currently in-window."""
        out = []
        for t in self.tracks:
            if t.kind == "player":
                continue
            if kind is not None and t.kind != kind:
                continue
            w = self._to_window(t.pos)
            if w is not None:
                out.append((w[0], w[1], t))
        return out

    def occupancy(self) -> set[tuple[int, int]]:
        """Window cells our move() would bounce off (all tracks)."""
        occ = set()
        for t in self.tracks:
            w = self._to_window(t.pos)
            if w is not None:
                occ.add(w)
        return occ

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _cheb(w: tuple[int, int]) -> int:
        return max(abs(w[0] - CENTER_ROW), abs(w[1] - CENTER_COL))

    @staticmethod
    def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _to_window(self, fpos) -> tuple[int, int] | None:
        r = fpos[0] - self.pos[0] + CENTER_ROW
        c = fpos[1] - self.pos[1] + CENTER_COL
        if 0 <= r < WINDOW_ROWS and 0 <= c < WINDOW_COLS:
            return (r, c)
        return None

    def _to_frame(self, w) -> tuple[int, int]:
        return (self.pos[0] + w[0] - CENTER_ROW,
                self.pos[1] + w[1] - CENTER_COL)

    def _chase_step(self, epos) -> tuple[int, int]:
        """Exact chase rule, one enemy move toward us (frame coords).
        Terrain/entity blocking is unknown here; the planner re-checks."""
        dr = self.pos[0] - epos[0]
        dc = self.pos[1] - epos[1]
        if abs(dr) + abs(dc) <= 1:
            return epos                       # adjacent: it attacks
        if abs(dr) > abs(dc):
            return (epos[0] + (1 if dr > 0 else -1), epos[1])
        return (epos[0], epos[1] + (1 if dc > 0 else -1))

    def _own_shift(self, p: Percept) -> tuple[int, int]:
        walk = self.last_action
        if walk - RUN_OFFSET in MOVE_DELTAS:
            walk -= RUN_OFFSET
        if walk not in MOVE_DELTAS:
            return (0, 0)
        dr, dc = MOVE_DELTAS[walk]
        if p.anim == ANIM_MOVE:
            return (dr, dc)
        if p.anim == ANIM_RUN:
            return (2 * dr, 2 * dc)
        return (0, 0)
