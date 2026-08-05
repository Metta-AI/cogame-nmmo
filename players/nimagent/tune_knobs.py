#!/usr/bin/env python3
"""Joint knob-space tuner for the nim agent: hill-climb with random
coordinate perturbations (single + pairs) against the deterministic
8-seed bench. Hand-probing tested each knob alone; this explores
interactions. Logs every eval; keeps the best config in best.json.

Run: python3 tune_knobs.py [hours]
"""
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "tune.log"
BEST = HERE / "tune_best.json"

PARAMS = {  # name: (incumbent, min, max, step)
    "KillBonusI": (100, 40, 240, 20),
    "HitCreditI": (6, 2, 14, 1),
    "OccPenaltyI": (25, 5, 65, 10),
    "HazardPenaltyI": (140, 60, 320, 30),
    "WKillI": (50, 20, 100, 10),
    "WAvoidI": (90, 50, 150, 10),
    "HerbHp": (45, 35, 65, 5),
    "HerbHpCombat": (65, 50, 85, 5),
    "RecoverFloor": (60, 40, 80, 5),
    "RecoverUntil": (90, 70, 98, 4),
    "EngageR": (6, 5, 7, 1),
    "HerbStock": (2, 1, 3, 1),
    "IdleTicksLimit": (120, 70, 220, 20),
    "QuietSettle": (20, 10, 34, 4),
}


def evaluate(cfg: dict) -> float:
    flags = " ".join(f"-d:{k}={v}" for k, v in cfg.items())
    r = subprocess.run(
        f"nim c -d:release --hints:off --warnings:off {flags} "
        f"-o:bench_tune bench.nim",
        shell=True, cwd=HERE, capture_output=True, text=True,
        timeout=600)
    if r.returncode != 0:
        return -1.0
    r = subprocess.run("./bench_tune 8 1500 1", shell=True, cwd=HERE,
                       capture_output=True, text=True, timeout=1800)
    m = re.search(r"nim mean=([0-9.]+)", r.stdout)
    return float(m.group(1)) if m else -1.0


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    deadline = time.time() + hours * 3600
    cur = {k: v[0] for k, v in PARAMS.items()}
    best_score = evaluate(cur)
    best = dict(cur)
    with open(LOG, "a") as f:
        f.write(f"START incumbent={best_score:.3f} {json.dumps(cur)}\n")
    evals = 1
    while time.time() < deadline:
        cand = dict(best)
        npick = random.choice([1, 1, 1, 2, 2, 3])
        for name in random.sample(list(PARAMS), npick):
            lo, hi, step = PARAMS[name][1:]
            delta = random.choice([-2, -1, 1, 2]) * step
            cand[name] = max(lo, min(hi, cand[name] + delta))
        if cand == best:
            continue
        score = evaluate(cand)
        evals += 1
        tag = ""
        if score > best_score:
            best_score = score
            best = cand
            with open(BEST, "w") as f:
                json.dump({"score": score, "cfg": best}, f)
            tag = " *** NEW BEST"
        with open(LOG, "a") as f:
            f.write(f"eval {evals}: {score:.3f}{tag} "
                    f"{json.dumps(cand)}\n")
    with open(LOG, "a") as f:
        f.write(f"DONE best={best_score:.3f} {json.dumps(best)}\n")


if __name__ == "__main__":
    main()
