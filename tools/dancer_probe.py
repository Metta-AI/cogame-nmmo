"""Dancer death forensics: run seeds with 4 dancer seats, and at every
fatal tick (hp -> 0) dump ground truth (debug_nearby) vs the world
model's tracks vs the recent decision branches. Aggregates per-death:
was the killer TRACKED, and which branch was active when it landed.

Usage: uv run python tools/dancer_probe.py [seed0] [nseeds] [ticks] [-v]
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))

from cogame_nmmo.sim import NmmoSim
from players.baseline_player import BaselinePolicy
from players.dancer.policy import DancerPolicy
from players.scripted_player import CENTER_COL, CENTER_ROW, Percept

SEED0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TICKS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
VERBOSE = "-v" in sys.argv

HIST = 10


def run_seed(seed):
    sim = NmmoSim(seed=seed, num_agents=8)

    def nearby(pid, radius=6):
        n = sim._exports["debug_nearby"](sim._store, pid, radius)
        ptr = sim._exports["debug_buf_ptr"](sim._store)
        raw = sim._memory.read(sim._store, ptr, ptr + n * 5 * 4)
        return np.frombuffer(bytearray(raw), dtype=np.int32).reshape(n, 5)

    pols = [DancerPolicy(1000 + i) if i < 4 else
            BaselinePolicy(seed=1000 + i, num_agents=1) for i in range(8)]
    resets = [False] * 8
    hists = {i: deque(maxlen=HIST) for i in range(4)}
    prev_hp = {i: 99 for i in range(4)}
    prev_gt = {i: [] for i in range(4)}
    agg = Counter()
    reports = []
    for tick in range(TICKS):
        obs = sim.observations()
        acts = np.zeros((8, 1), dtype=np.float32)
        rows = {}
        for i, pol in enumerate(pols):
            a = pol(tick, [obs[i]], [resets[i]])
            acts[i, 0] = a[0][0]
            if i < 4:
                mind = pol.minds[0]
                p = Percept(bytes(obs[i]))
                tracked = [(r - CENTER_ROW, c - CENTER_COL,
                            t.kind, t.delta, tick - t.last_confirmed)
                           for r, c, t in mind.world.enemies()]
                rows[i] = dict(tick=tick, hp=p.hp, comb=p.comb_lvl,
                               prof=p.prof_lvl, life=mind.life_tick,
                               branch=mind.branch, act=int(a[0][0]),
                               tracked=tracked)
        gt_now = {i: nearby(i) for i in range(4)}
        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()
        for i in range(4):
            hists[i].append(rows[i])
            hp_now = sim.agent_stat(i, 7)
            if hp_now == 0 and prev_hp[i] > 0:
                gt = prev_gt[i]      # killer in place at the pre-fatal obs
                gtf = gt_now[i]      # and at the fatal tick itself
                killers = [e for e in gtf
                           if (abs(e[0]) + abs(e[1]) <= 1 and e[3] == 0)
                           or ((e[0] == 0 or e[1] == 0)
                               and max(abs(e[0]), abs(e[1])) <= 4
                               and e[3] == 1)]
                tracked = rows[i]["tracked"]
                n_match = 0
                for e in killers:
                    hit = any(abs(tr - e[0]) <= 1 and abs(tc - e[1]) <= 1
                              for tr, tc, _k, _d, _a in tracked)
                    n_match += hit
                cls = ("all-tracked" if killers and n_match == len(killers)
                       else "some-untracked" if n_match else
                       "untracked" if killers else "no-killer-visible")
                branches = [h["branch"].split("[")[0] for h in hists[i]]
                agg[(cls, branches[-1] if branches else "?")] += 1
                reports.append((seed, tick, i, cls, killers,
                                list(hists[i])))
            prev_hp[i] = hp_now
            prev_gt[i] = gt_now[i]
    return agg, reports


total = Counter()
all_reports = []
for s in range(SEED0, SEED0 + NSEEDS):
    agg, reports = run_seed(s)
    total.update(agg)
    all_reports += reports
    print(f"seed {s}: {sum(agg.values())} deaths", flush=True)

print("\n(killer-tracking, last-branch) counts:")
for k, v in total.most_common():
    print(f"  {k}: {v}")

if VERBOSE:
    for seed, tick, i, cls, killers, hist in all_reports[:12]:
        print(f"\n=== seed {seed} tick {tick} agent {i} [{cls}] "
              f"killers={[list(map(int, e)) for e in killers]}")
        for h in hist:
            print(f"  t{h['tick']} hp={h['hp']} m={min(h['comb'], h['prof'])} "
                  f"life={h['life']} {h['branch']} act={h['act']} "
                  f"tracked={h['tracked']}")
