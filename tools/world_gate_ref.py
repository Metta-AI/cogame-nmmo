"""Python twin of players/nimagent/tests/t_world.nim: SAME deterministic
action script, SAME sim (native, digest-identical), SAME metric
universe, run through the PYTHON tracker (players/dancer/world.py).
Matching counters == the Nim port preserves tracker behavior.

Run: uv run python tools/world_gate_ref.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))

from players.dancer.world import WorldModel  # noqa: E402
from players.scripted_player import CENTER_COL, CENTER_ROW, Percept  # noqa: E402
from train.native_env import NativeNmmo  # noqa: E402


def lcg(x: int) -> int:
    return (x * 1664525 + 1013904223) & 0xFFFFFFFF


def script_action(agent: int, tick: int) -> int:
    h = lcg((agent * 7919 + tick * 104729 + 12345) & 0xFFFFFFFF)
    v = (h >> 8) % 10
    return v % 4 if v < 6 else 4


def run_seed(seed: int, ticks: int = 600) -> dict:
    env = NativeNmmo(seed=seed, num_agents=8)
    world = WorldModel()
    c = dict(gt=0, hit=0, close_gt=0, close_hit=0, tracks=0, true=0,
             kind_pairs=0, kind_ok=0)
    prev_gt_frame: list[tuple[int, int]] = []
    lib = env.lib

    def ground_truth():
        n = lib.debug_nearby(0, 6)
        import ctypes
        buf = np.ctypeslib.as_array(
            ctypes.cast(lib.debug_buf_ptr(),
                        ctypes.POINTER(ctypes.c_int32)),
            shape=(n * 5,))
        return buf.reshape(n, 5).copy()

    lib.debug_buf_ptr.restype = __import__("ctypes").c_void_p
    lib.debug_nearby.restype = __import__("ctypes").c_int
    lib.debug_nearby.argtypes = [__import__("ctypes").c_int,
                                 __import__("ctypes").c_int]

    for tick in range(ticks):
        obs = bytes(env.obs[0])
        p = Percept(obs)
        if tick > 0 and env.term[0] > 0.5:
            world.reset(tick)
        world.observe(p, tick)
        act0 = script_action(0, tick)
        world.note_action(act0)

        gt = ground_truth()
        gt_frame = [(world.pos[0] + int(g[0]), world.pos[1] + int(g[1]))
                    for g in gt]
        if not world.teleported and tick > 5:
            trk_close = []
            for wr, wc, t in world.enemies():
                if max(abs(wr - CENTER_ROW), abs(wc - CENTER_COL)) <= 4:
                    trk_close.append((wr - CENTER_ROW, wc - CENTER_COL,
                                      t.kind))
            for gi, g in enumerate(gt):
                gr, gc = int(g[0]), int(g[1])
                if max(abs(gr), abs(gc)) > 3 or int(g[4]) <= 0:
                    continue
                fr = gt_frame[gi]
                settled = any(max(abs(fr[0] - pr), abs(fr[1] - pc)) <= 1
                              for pr, pc in prev_gt_frame)
                frozen = fr in prev_gt_frame
                if not settled or frozen:
                    continue
                matched = False
                exact_kind = None
                for tr, tc, kind in trk_close:
                    if max(abs(tr - gr), abs(tc - gc)) <= 1:
                        matched = True
                    if (tr, tc) == (gr, gc) and exact_kind is None:
                        exact_kind = kind
                c["gt"] += 1
                c["hit"] += matched
                if max(abs(gr), abs(gc)) <= 2:
                    c["close_gt"] += 1
                    c["close_hit"] += matched
                if exact_kind is not None:
                    c["kind_pairs"] += 1
                    real = "bow" if int(g[3]) else "melee"
                    c["kind_ok"] += (exact_kind == real)
            for tr, tc, _k in trk_close:
                c["tracks"] += 1
                c["true"] += any(
                    int(g[4]) > 0
                    and max(abs(tr - int(g[0])), abs(tc - int(g[1]))) <= 1
                    for g in gt)
        prev_gt_frame = gt_frame

        acts = np.array([script_action(i, tick) for i in range(8)],
                        dtype=np.float32)
        env.step(acts)
    return c


def main():
    agg = dict(gt=0, hit=0, close_gt=0, close_hit=0, tracks=0, true=0,
               kind_pairs=0, kind_ok=0)
    for seed in (11, 12, 13):
        s = run_seed(seed)
        for k in agg:
            agg[k] += s[k]
    recall = agg["hit"] / max(agg["gt"], 1)
    close = agg["close_hit"] / max(agg["close_gt"], 1)
    precision = agg["true"] / max(agg["tracks"], 1)
    kind = agg["kind_ok"] / max(agg["kind_pairs"], 1)
    print(f"tracker: recall={recall:.3f} ({agg['hit']}/{agg['gt']}) "
          f"close={close:.3f} ({agg['close_hit']}/{agg['close_gt']}) "
          f"precision={precision:.3f} ({agg['true']}/{agg['tracks']}) "
          f"kind={kind:.3f} ({agg['kind_ok']}/{agg['kind_pairs']})")


if __name__ == "__main__":
    main()
