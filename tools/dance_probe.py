"""Verify the diagonal melee dance against ground truth.

Model (from vendored nmmo3.h enemy_ai + c_step ordering):
- players act before enemies within a tick;
- melee enemies attack only at Manhattan-1 (on their turn), else move
  1 tile along the LONGEST axis toward the player (tie -> column move);
- ATN_ATTACK with a sword held hits the 8-neighborhood (ATTACK_SWORD
  covers diagonals); bare/tool ATTACK hits orthogonal only.

Claimed cycle (vector = us - enemy, in (row, col) magnitudes):
  at (1,1): ATTACK  -> free hit (enemy not adjacent on its turn: it
            tie-moves along the column to (1,0))
  at (1,0): RUN 2 rows away -> (3,0); enemy row-moves -> (2,0)
  at (2,0): step col -> (2,1); enemy row-moves (row longest) -> (1,1)
  at (1,1): ATTACK ...

This probe drives agent 0 with a hand-rolled dance controller against
the nearest melee enemy found via the ground-truth debug export, and
reports hits landed vs damage taken. All other seats play baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))

from cogame_nmmo.sim import NmmoSim
from players.baseline_player import BaselinePolicy

ATN_DOWN, ATN_UP, ATN_RIGHT, ATN_LEFT, ATN_NOOP, ATN_ATTACK = 0, 1, 2, 3, 4, 5
RUN = 22  # + direction offset

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TICKS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

sim = NmmoSim(seed=SEED, num_agents=8)

def nearby(pid, radius=7):
    n = sim._exports["debug_nearby"](sim._store, pid, radius)
    ptr = sim._exports["debug_buf_ptr"](sim._store)
    raw = sim._memory.read(sim._store, ptr, ptr + n*5*4)
    return np.frombuffer(bytearray(raw), dtype=np.int32).reshape(n, 5)

pols = [None] + [BaselinePolicy(seed=1000+i, num_agents=1) for i in range(1, 8)]
resets = [False]*8

hits_landed = 0
dmg_taken = 0
kills = 0
prev_hp = 99
prev_target_hp = None
state = "seek"
for tick in range(TICKS):
    obs = sim.observations()
    acts = np.zeros((8, 1), dtype=np.float32)
    for i in range(1, 8):
        a = pols[i](tick, [obs[i]], [resets[i]])
        acts[i, 0] = a[0][0]

    me_hp = sim.agent_stat(0, 7)
    if me_hp < prev_hp and not resets[0]:
        dmg_taken += prev_hp - me_hp
    prev_hp = me_hp if not resets[0] else 99

    ens = nearby(0)
    # target: nearest melee (ranged==0) within 7, no other enemy within 3 of us
    melee = [e for e in ens if e[3] == 0 and e[4] > 0]
    others_close = [e for e in ens if max(abs(e[0]), abs(e[1])) <= 3]
    act = ATN_NOOP
    if melee is not None and len(melee) and len(others_close) <= 1:
        t = min(melee, key=lambda e: max(abs(e[0]), abs(e[1])))
        dr, dc = int(t[0]), int(t[1])   # enemy - us? debug gives enemy-player: dr = e.r - p.r
        # vector from enemy to us is (-dr, -dc); use magnitudes + our move choices
        adr, adc = abs(dr), abs(dc)
        if prev_target_hp is not None and t[4] > prev_target_hp:
            pass  # regen; ignore
        if adr <= 1 and adc <= 1 and adr + adc > 0:
            # 8-neighborhood: if diagonal -> free attack; orthogonal-adjacent -> reposition
            if adr == 1 and adc == 1:
                act = ATN_ATTACK
                hits_landed += 1
            else:
                # orthogonal adjacent: run 2 away along the adjacency axis
                if adr == 1 and adc == 0:
                    act = RUN + (ATN_DOWN if dr < 0 else ATN_UP)   # away from enemy
                else:
                    act = RUN + (ATN_RIGHT if dc < 0 else ATN_LEFT)
        elif (adr, adc) in ((2, 0), (0, 2)):
            # step perpendicular to make (2,1)/(1,2)
            act = ATN_RIGHT if adr == 2 else ATN_DOWN
        elif adr + adc <= 7:
            # approach: step toward reducing longest axis
            if adr >= adc and adr > 1:
                act = ATN_DOWN if dr > 0 else ATN_UP
            elif adc > 1:
                act = ATN_RIGHT if dc > 0 else ATN_LEFT
        prev_target_hp = int(t[4])
    acts[0, 0] = act
    sim.set_actions(acts)
    sim.step()
    resets = sim.dones()

print(f"seed {SEED}: hits_landed={hits_landed} dmg_taken={dmg_taken} "
      f"comb={sim.agent_stat(0,2)} deaths={sim.agent_stat(0,1)} "
      f"gold={sim.agent_stat(0,5)}")
