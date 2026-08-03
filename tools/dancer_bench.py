"""Parallel dancer benchmark: deaths-by-class + seat scores.

Usage: uv run python tools/dancer_bench.py [nseeds] [ticks] [seed0] [kind]
"""
from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))


def run_one(args):
    seed, ticks, kind = args
    import numpy as np
    from cogame_nmmo.sim import NmmoSim
    from players.baseline_player import BaselinePolicy
    if kind == "dancer":
        from players.dancer.policy import DancerPolicy as K
    elif kind == "netstrap":
        from players.dancer.netstrap import NetstrapPolicy as K
    elif kind == "netburst":
        from players.dancer.netstrap import NetstrapPolicy
        def K(seed):
            return NetstrapPolicy(seed, style="burst")
    elif kind == "v2":
        from players.scripted_player_v2 import ScriptedPolicyV2 as K
    else:
        raise ValueError(kind)

    sim = NmmoSim(seed=seed, num_agents=8)

    def nearby(pid, radius=6):
        n = sim._exports["debug_nearby"](sim._store, pid, radius)
        ptr = sim._exports["debug_buf_ptr"](sim._store)
        raw = sim._memory.read(sim._store, ptr, ptr + n * 5 * 4)
        return np.frombuffer(bytearray(raw), dtype=np.int32).reshape(n, 5)

    pols = [K(1000 + i) if i < 4 else
            BaselinePolicy(seed=1000 + i, num_agents=1) for i in range(8)]
    resets = [False] * 8
    kinds = Counter()
    prev = {i: (99, []) for i in range(4)}
    for tick in range(ticks):
        obs = sim.observations()
        acts = np.zeros((8, 1), dtype=np.float32)
        for i, pol in enumerate(pols):
            a = pol(tick, [obs[i]], [resets[i]])
            acts[i, 0] = a[0][0]
        near_now = {i: nearby(i) for i in range(4)}
        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()
        for i in range(4):
            if resets[i] and sim.agent_stat(i, 7) == 99:
                kinds["stagnation"] += 1
        for i in range(4):
            hp_now = sim.agent_stat(i, 7)
            hp_prev, near = prev[i]
            # classify at the FATAL tick (hp hits 0), not at the reset
            # flag 2 ticks later - the killer is still in place here
            if hp_now == 0 and hp_prev > 0:
                mel = [e for e in near
                       if abs(e[0]) + abs(e[1]) <= 1 and e[3] == 0]
                bow = [e for e in near
                       if (e[0] == 0 or e[1] == 0)
                       and max(abs(e[0]), abs(e[1])) <= 4 and e[3] == 1]
                if bow and (not mel or hp_prev > 60):
                    kinds["bow"] += 1
                elif len(mel) >= 2:
                    kinds["melee-multi"] += 1
                elif mel:
                    lv = max(e[2] for e in mel)
                    kinds[f"melee-{'lo' if lv <= 4 else 'mid' if lv <= 9 else 'hi'}"] += 1
                else:
                    kinds["unclear"] += 1
            prev[i] = (hp_now, near_now[i])
    scores = [sim.score(i) / (sim.agent_stat(i, 1) + 1) for i in range(4)]
    bscores = [sim.score(i) / (sim.agent_stat(i, 1) + 1) for i in range(4, 8)]
    stats = [dict(comb=sim.agent_stat(i, 2), prof=sim.agent_stat(i, 3),
                  deaths=sim.agent_stat(i, 1)) for i in range(4)]
    return dict(seed=seed, scores=scores, bscores=bscores, deaths=kinds,
                stats=stats)


def main():
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    kind = sys.argv[4] if len(sys.argv) > 4 else "dancer"
    jobs = [(seed0 + k, ticks, kind) for k in range(nseeds)]
    import statistics as st
    all_scores, all_b, agg = [], [], Counter()
    tot_deaths = 0
    combs = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run_one, jobs):
            all_scores += r["scores"]
            all_b += r["bscores"]
            agg.update(r["deaths"])
            tot_deaths += sum(s["deaths"] for s in r["stats"])
            combs += [s["comb"] for s in r["stats"]]
            print(f"seed {r['seed']}: {kind}={st.mean(r['scores']):.2f} "
                  f"deaths={[s['deaths'] for s in r['stats']]} "
                  f"combs={[s['comb'] for s in r['stats']]}", flush=True)
    n_agents = len(all_scores)
    print(f"\n{kind}: mean={st.mean(all_scores):.3f} vs baseline "
          f"{st.mean(all_b):.3f} | deaths/agent/{ticks}t = "
          f"{tot_deaths / n_agents:.1f} | max comb={max(combs)}")
    print("death classes:", dict(agg.most_common()))


if __name__ == "__main__":
    main()
