"""Paired seat evaluation of two weight bins in the bit-exact native env.

Mirrors tools/ab_batch.py's protocol: 8 seats per episode, seats
0,2,4,6 play bin A, seats 1,3,5,7 play bin B, seat score =
nmmo_score/(deaths+1) at the final tick, paired per-episode means, then
the paired t over episodes. Torch forward (parity 1.4e-4 vs the C
brain), T=1.0 categorical sampling.

Usage:
  train/.venv/bin/python train/eval_seats.py A.bin B.bin \
      [seed0] [neps] [ticks]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.mmonet import load_puffer_bin  # noqa: E402
from train.native_env import NativeNmmo  # noqa: E402


@torch.no_grad()
def run_episode(net_a, net_b, seed: int, ticks: int, gen: torch.Generator):
    env = NativeNmmo(seed=seed, num_agents=8)
    a_idx, b_idx = [0, 2, 4, 6], [1, 3, 5, 7]
    st_a = net_a.initial_state(4)
    st_b = net_b.initial_state(4)
    for _t in range(ticks):
        obs = torch.from_numpy(env.obs.copy())
        term = env.term.copy()
        acts = np.zeros(8, dtype=np.float32)
        for net, idx, st_name in ((net_a, a_idx, "a"),
                                  (net_b, b_idx, "b")):
            st = st_a if st_name == "a" else st_b
            done = torch.tensor([term[i] > 0.5 for i in idx])
            if done.any():
                st[:, done, :] = 0.0
            logits, _v, st = net(obs[idx], st)
            act = torch.multinomial(
                torch.softmax(logits, dim=-1), 1, generator=gen)
            acts[idx] = act.squeeze(1).numpy()
            if st_name == "a":
                st_a = st
            else:
                st_b = st
        env.step(acts)
    sa = [env.score(i) / (env.stat(i, 1) + 1) for i in a_idx]
    sb = [env.score(i) / (env.stat(i, 1) + 1) for i in b_idx]
    return float(np.mean(sa)), float(np.mean(sb))


def main():
    bin_a, bin_b = sys.argv[1], sys.argv[2]
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    neps = int(sys.argv[4]) if len(sys.argv) > 4 else 18
    ticks = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
    net_a = load_puffer_bin(bin_a).eval()
    net_b = load_puffer_bin(bin_b).eval()
    gen = torch.Generator().manual_seed(97)
    da, db = [], []
    for k in range(neps):
        a, b = run_episode(net_a, net_b, seed0 + k, ticks, gen)
        da.append(a)
        db.append(b)
        print(f"seed {seed0+k}: A={a:.2f} B={b:.2f} | "
              f"cum A={np.mean(da):.3f} B={np.mean(db):.3f}", flush=True)
    diff = np.array(db) - np.array(da)
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)) + 1e-12)
    print(f"FINAL A({Path(bin_a).name})={np.mean(da):.3f} "
          f"B({Path(bin_b).name})={np.mean(db):.3f} "
          f"delta={diff.mean():+.3f} paired t={t:.2f} (n={len(diff)})")


if __name__ == "__main__":
    main()
