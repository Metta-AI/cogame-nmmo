"""Death forensics: run one episode with v2 seats, record per-tick
context for one agent, dump the window before each death."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))

from cogame_nmmo.sim import NmmoSim, STAT_DEATHS
from players.scripted_player_v2 import ScriptedPolicyV2, MindV2
from players.scripted_player import Percept, CENTER_ROW, CENTER_COL
from players.baseline_player import BaselinePolicy

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TICKS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
WATCH = 0

sim = NmmoSim(seed=SEED, num_agents=8)
pols = [ScriptedPolicyV2(1000 + i) if i < 4 else
        BaselinePolicy(seed=1000 + i, num_agents=1) for i in range(8)]

hist: list[dict] = []
resets = [False] * 8
deaths_seen = 0
for tick in range(TICKS):
    obs = sim.observations()
    p = Percept(bytes(obs[WATCH]))
    mind: MindV2 | None = pols[WATCH].minds.get(0)
    row = {
        "tick": tick, "hp": p.hp, "comb": p.comb_lvl, "prof": p.prof_lvl,
        "in_combat": p.in_combat, "ui": p.ui_mode,
        "held": p.equipment[3],
        "target": None if mind is None else mind.fight_target,
        "life_tick": None if mind is None else mind.life_tick,
    }
    acts = np.zeros((8, 1), dtype=np.float32)
    for i, pol in enumerate(pols):
        a = pol(tick, [obs[i]], [resets[i]])
        acts[i, 0] = a[0][0]
    row["action"] = int(acts[WATCH, 0])
    if mind is not None:
        row["hints"] = [(int(r), int(c), int(d), int(h))
                        for r, c, d, h in mind._live_hints(p)]
    hist.append(row)
    sim.set_actions(acts)
    sim.step()
    resets = sim.dones()
    if resets[WATCH]:
        deaths_seen += 1
        print(f"=== death #{deaths_seen} at tick {tick} "
              f"(life_tick={row['life_tick']}) ===")
        for h in hist[-25:]:
            print(" ", h)
        if deaths_seen >= 4:
            break
print("total deaths stat:", sim.agent_stat(WATCH, STAT_DEATHS))
