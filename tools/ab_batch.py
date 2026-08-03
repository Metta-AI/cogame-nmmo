"""Powered A/B batch: interleaved seats, many seeds, multiprocess.

Usage: uv run python tools/ab_batch.py <kindA> <kindB> <seed0> <nseeds> \
           [ticks] [out.json]
Seats 0,2,4,6 play kindA; seats 1,3,5,7 play kindB (interleaving controls
seat-index effects; the world is shared so arms face identical conditions).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))


def make(kind, seed):
    if kind == "baseline":
        from players.baseline_player import BaselinePolicy
        return BaselinePolicy(seed=seed, num_agents=1)
    if kind.startswith("baseline_t"):
        from players.baseline_player import BaselinePolicy
        t = int(kind[len("baseline_t"):]) / 100.0
        return BaselinePolicy(seed=seed, num_agents=1, temperature=t)
    if kind == "hybrid":
        from players.hybrid_player import HybridPolicy
        return HybridPolicy(seed=seed, num_agents=1)
    if kind == "hybrid_herb":
        from players.hybrid_player import HybridPolicy
        return HybridPolicy(seed=seed, num_agents=1,
                            enable_stuck_reset=False, enable_rescue=False)
    if kind == "v2":
        from players.scripted_player_v2 import ScriptedPolicyV2
        return ScriptedPolicyV2(seed)
    if kind == "v3":
        from players.scripted_player_v3 import ScriptedPolicyV3
        return ScriptedPolicyV3(seed)
    if kind == "scripted":
        from players.scripted_player import ScriptedPolicy
        return ScriptedPolicy(seed)
    raise ValueError(kind)


def run_episode(args):
    kind_a, kind_b, seed, ticks = args
    import numpy as np
    from cogame_nmmo.sim import NmmoSim
    sim = NmmoSim(seed=seed, num_agents=8)
    pols = [make(kind_a if i % 2 == 0 else kind_b, 1000 + i)
            for i in range(8)]
    resets = [False] * 8
    for tick in range(ticks):
        obs = sim.observations()
        acts = np.zeros((8, 1), dtype=np.float32)
        for i, pol in enumerate(pols):
            a = pol(tick, [obs[i]], [resets[i]])
            acts[i, 0] = a[0][0]
        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()
    scores = [sim.score(i) / (sim.agent_stat(i, 1) + 1) for i in range(8)]
    return {"seed": seed,
            "a": [scores[i] for i in range(0, 8, 2)],
            "b": [scores[i] for i in range(1, 8, 2)]}


def main():
    kind_a, kind_b = sys.argv[1], sys.argv[2]
    seed0, nseeds = int(sys.argv[3]), int(sys.argv[4])
    ticks = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
    out = sys.argv[6] if len(sys.argv) > 6 else None
    jobs = [(kind_a, kind_b, seed0 + k, ticks) for k in range(nseeds)]
    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run_episode, jobs):
            results.append(r)
            import statistics as st
            a = [x for row in results for x in row["a"]]
            b = [x for row in results for x in row["b"]]
            print(f"seed {r['seed']} done | cum {kind_a}: n={len(a)} "
                  f"mean={st.mean(a):.3f} | {kind_b}: mean={st.mean(b):.3f}",
                  flush=True)
    if out:
        Path(out).write_text(json.dumps(
            {"kind_a": kind_a, "kind_b": kind_b, "ticks": ticks,
             "results": results}, indent=1))
    import statistics as st
    a = [x for row in results for x in row["a"]]
    b = [x for row in results for x in row["b"]]
    da = st.mean(b) - st.mean(a)
    # paired-by-episode t on per-episode means
    pa = [st.mean(r["a"]) for r in results]
    pb = [st.mean(r["b"]) for r in results]
    diffs = [y - x for x, y in zip(pa, pb)]
    sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
    import math
    t = st.mean(diffs) / (sd / math.sqrt(len(diffs))) if sd else float("nan")
    print(f"FINAL {kind_a} mean={st.mean(a):.3f} vs {kind_b} "
          f"mean={st.mean(b):.3f} | delta={da:+.3f} | paired t={t:.2f} "
          f"(n={len(diffs)} episodes)")


if __name__ == "__main__":
    main()
