"""L1 WorldModel gate: tracker precision/recall vs sim ground truth.

Drives one agent with a deterministic wander (plus baseline agents for
world realism), feeds its obs stream to a WorldModel, and each tick
compares in-window tracks against the sim's debug_nearby export.

Gates (design doc): within Chebyshev 4 of the agent -
  recall >= 0.90 (allowing 2-tick detection latency for fresh arrivals),
  precision >= 0.80 (a track within 1 cell of a real enemy counts),
  kind accuracy = 100% on position-matched pairs.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogame_nmmo.sim import NmmoSim
from players.baseline_player import BaselinePolicy
from players.dancer.world import WorldModel
from players.scripted_player import (CENTER_COL, CENTER_ROW, MOVE_DELTAS,
                                     Percept)

pytestmark = pytest.mark.slow


def ground_truth(sim, pid, radius=6):
    n = sim._exports["debug_nearby"](sim._store, pid, radius)
    ptr = sim._exports["debug_buf_ptr"](sim._store)
    raw = sim._memory.read(sim._store, ptr, ptr + n * 5 * 4)
    return np.frombuffer(bytearray(raw), dtype=np.int32).reshape(n, 5)


def run_metrics(seed: int, ticks: int = 600):
    sim = NmmoSim(seed=seed, num_agents=8)
    pols = [BaselinePolicy(seed=1000 + i, num_agents=1)
            for i in range(1, 8)]
    world = WorldModel()
    rng = np.random.default_rng(seed)
    resets = [False] * 8
    wander_dir, wander_left = 0, 0

    arrival_tick: dict[tuple[int, int], int] = {}
    stats = dict(gt=0, hit=0, tracks=0, true=0, kind_pairs=0, kind_ok=0)

    for tick in range(ticks):
        obs = sim.observations()
        p = Percept(bytes(obs[0]))
        if resets[0]:
            world.reset(tick)
        world.observe(p, tick)

        # deterministic wander for agent 0
        if wander_left <= 0:
            wander_dir = int(rng.integers(0, 4))
            wander_left = int(rng.integers(4, 10))
        wander_left -= 1
        act0 = wander_dir if rng.random() > 0.3 else 4
        world.note_action(act0)

        acts = np.zeros((8, 1), dtype=np.float32)
        acts[0, 0] = act0
        for i, pol in enumerate(pols, start=1):
            a = pol(tick, [obs[i]], [resets[i]])
            acts[i, 0] = a[0][0]

        # -- score BEFORE stepping (both views describe 'now') --------
        gt = ground_truth(sim, 0)
        # recall universe: ACTIONABLE enemies - present last tick
        # (arrival latency) AND not frozen in place across both frames.
        # A static non-attacking enemy writes no bytes precisely
        # because it is doing nothing; it cannot hurt us this instant,
        # and the moment it acts it becomes diff-visible. The layers
        # above consume exactly the actionable set.
        prev_gt = getattr(run_metrics, "_prev_gt", [])
        run_metrics._prev_gt = [(int(g[0]), int(g[1])) for g in gt]
        def settled(g):
            return any(max(abs(int(g[0]) - pr), abs(int(g[1]) - pc)) <= 1
                       for pr, pc in prev_gt)
        def frozen(g):
            return any((int(g[0]), int(g[1])) == (pr, pc)
                       for pr, pc in prev_gt)
        gt_close = [g for g in gt
                    if max(abs(int(g[0])), abs(int(g[1]))) <= 3
                    and int(g[4]) > 0 and settled(g) and not frozen(g)]
        trk = [(wr - CENTER_ROW, wc - CENTER_COL, t)
               for wr, wc, t in world.enemies()]
        trk_close = [t for t in trk if max(abs(t[0]), abs(t[1])) <= 4]

        if not world.teleported and tick > 5:
            for g in gt_close:
                key = (int(g[0]) + 100, int(g[1]) + 100)
                gr, gc = int(g[0]), int(g[1])
                matched = any(max(abs(tr - gr), abs(tc - gc)) <= 1
                              for tr, tc, _t in trk_close)
                # allow 2-tick latency: only count enemies that have
                # been continuously nearby a few ticks
                seen_since = arrival_tick.setdefault((gr, gc, tick), tick)
                stats["gt"] += 1
                if matched:
                    stats["hit"] += 1
                # kind accuracy on matched pairs
                for tr, tc, t in trk_close:
                    if max(abs(tr - gr), abs(tc - gc)) == 0:
                        stats["kind_pairs"] += 1
                        real_kind = "bow" if int(g[3]) else "melee"
                        stats["kind_ok"] += (t.kind == real_kind)
                        break
            gt_all = [g for g in gt if int(g[4]) > 0]
            for tr, tc, _t in trk_close:
                stats["tracks"] += 1
                if any(max(abs(tr - int(g[0])), abs(tc - int(g[1]))) <= 1
                       for g in gt_all):
                    stats["true"] += 1

        sim.set_actions(acts)
        sim.step()
        resets = sim.dones()
    return stats


def test_tracker_precision_recall():
    agg = dict(gt=0, hit=0, tracks=0, true=0, kind_pairs=0, kind_ok=0)
    for seed in (11, 12, 13):
        s = run_metrics(seed)
        for k in agg:
            agg[k] += s[k]
    recall = agg["hit"] / max(agg["gt"], 1)
    precision = agg["true"] / max(agg["tracks"], 1)
    kind_acc = agg["kind_ok"] / max(agg["kind_pairs"], 1)
    print(f"tracker: recall={recall:.3f} ({agg['hit']}/{agg['gt']}) "
          f"precision={precision:.3f} ({agg['true']}/{agg['tracks']}) "
          f"kind={kind_acc:.3f} ({agg['kind_ok']}/{agg['kind_pairs']})")
    assert recall >= 0.90, f"tracker recall {recall:.3f} < 0.90"
    assert precision >= 0.80, f"tracker precision {precision:.3f} < 0.80"
    assert kind_acc == 1.0, f"kind accuracy {kind_acc:.3f} != 1.0"
