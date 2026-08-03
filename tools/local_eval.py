"""Local A/B eval harness: in-process seeded episodes, scripted vs baseline.

Runs the sim directly (no server/websockets), seats policies onto agent
slots, and reports the exact league seat score (sim.score(pid) /
(deaths+1), matching engine._seat_score) plus diagnostic stats.

Usage:
  uv run python tools/local_eval.py --seeds 1,2,3 --ticks 5000 \
      --arrangement half            # seats 0-3 scripted, 4-7 baseline
  uv run python tools/local_eval.py --seeds 1..8 --scripted-only

Per-seat counterfactual discipline: same seed + same arrangement, swap
only the policy under test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))

from cogame_nmmo.sim import (NmmoSim, STAT_COMB_LVL, STAT_CUM_MIN_COMB_PROF,
                             STAT_DEATHS, STAT_GOLD, STAT_PROF_LVL,
                             STAT_TIME_ALIVE)


def make_policy(kind: str, seed: int):
    if kind == "scripted":
        from players.scripted_player import ScriptedPolicy
        return ScriptedPolicy(seed)
    if kind == "v2":
        from players.scripted_player_v2 import ScriptedPolicyV2
        return ScriptedPolicyV2(seed)
    if kind == "v3":
        from players.scripted_player_v3 import ScriptedPolicyV3
        return ScriptedPolicyV3(seed)
    if kind == "dancer":
        from players.dancer.policy import DancerPolicy
        return DancerPolicy(seed)
    if kind == "baseline":
        from players.baseline_player import BaselinePolicy
        return BaselinePolicy(seed=seed, num_agents=1)
    if kind == "noop":
        class Noop:
            def __call__(self, tick, obs_rows, resets):
                return [[4] for _ in obs_rows]
        return Noop()
    raise ValueError(kind)


def run_episode(seed: int, kinds: list[str], ticks: int,
                trace: bool = False) -> dict:
    """kinds[i] = policy kind for seat i (1 hero per seat)."""
    n = len(kinds)
    sim = NmmoSim(seed=seed, num_agents=n)
    policies = [make_policy(k, seed=seed * 1000 + i)
                for i, k in enumerate(kinds)]
    resets = [False] * n
    t0 = time.time()
    trace_rows = []
    for tick in range(ticks):
        obs = sim.observations()
        actions = np.zeros((n, 1), dtype=np.float32)
        for i, pol in enumerate(policies):
            act = pol(tick, [obs[i]], [resets[i]])
            actions[i, 0] = act[0][0]
        sim.set_actions(actions)
        sim.step()
        dones = sim.dones()
        resets = [r or d for r, d in zip(resets, dones)]
        # engine clears the pending flag when the seat replies; in-process
        # every seat replies every tick, so resets = this step's dones
        resets = dones
        if trace and tick % 250 == 249:
            trace_rows.append({
                "tick": tick + 1,
                "stats": [agent_row(sim, i) for i in range(n)]})
    dt = time.time() - t0
    out = {
        "seed": seed, "ticks": ticks, "kinds": kinds, "wall_s": round(dt, 1),
        "seats": []}
    for i in range(n):
        row = agent_row(sim, i)
        row["kind"] = kinds[i]
        row["seat_score"] = sim.score(i) / (row["deaths"] + 1)
        out["seats"].append(row)
    if trace:
        out["trace"] = trace_rows
    return out


def agent_row(sim: NmmoSim, pid: int) -> dict:
    return {
        "pid": pid,
        "cum_min": sim.agent_stat(pid, STAT_CUM_MIN_COMB_PROF),
        "score_raw": sim.score(pid),
        "deaths": sim.agent_stat(pid, STAT_DEATHS),
        "comb": sim.agent_stat(pid, STAT_COMB_LVL),
        "prof": sim.agent_stat(pid, STAT_PROF_LVL),
        "gold": sim.agent_stat(pid, STAT_GOLD),
        "time_alive": sim.agent_stat(pid, STAT_TIME_ALIVE),
    }


def parse_seeds(spec: str) -> list[int]:
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1..4")
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--arrangement", default="half",
                    choices=["half", "alt", "scripted-only", "baseline-only"])
    ap.add_argument("--scripted-kind", default="scripted",
                    help="module kind for the 'scripted' seats")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    s = args.scripted_kind
    arrangements = {
        "half": [s] * 4 + ["baseline"] * 4,
        "alt": [s, "baseline"] * 4,
        "scripted-only": [s] * 8,
        "baseline-only": ["baseline"] * 8,
    }
    kinds = arrangements[args.arrangement]

    results = []
    for seed in parse_seeds(args.seeds):
        r = run_episode(seed, kinds, args.ticks, trace=args.trace)
        results.append(r)
        by_kind: dict[str, list[float]] = {}
        for seat in r["seats"]:
            by_kind.setdefault(seat["kind"], []).append(seat["seat_score"])
        summ = {k: round(float(np.mean(v)), 3) for k, v in by_kind.items()}
        print(f"seed {seed}: {summ}  ({r['wall_s']}s)")
        for seat in r["seats"]:
            print(f"   pid{seat['pid']} {seat['kind'][:9]:9s} "
                  f"score={seat['seat_score']:.2f} raw={seat['score_raw']} "
                  f"deaths={seat['deaths']} comb={seat['comb']} "
                  f"prof={seat['prof']} gold={seat['gold']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
