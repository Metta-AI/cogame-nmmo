# NMMO post-training

`tools/export_posttrain.py` runs the production Wasm simulator with the
maintained scripted policy for each of eight seats. It decodes each seat's
1,707-byte observation into scalars, observed terrain rows, items, and nearby
entity hints. The hints can contain stale bytes, as the hosted observation does.
The simulator consumes the same 26-way action used by league players.
Whole seeded episodes stay in one split.

```sh
bash sim/build_sim.sh
uv sync
uv run python tools/test_posttrain.py
uv run python tools/export_posttrain.py /tmp/nmmo-data 10 --max-ticks 256
```

The certified variant runs for 5,000 ticks. `--max-ticks 256` creates a
shorter training curriculum with the same dynamics and observation/action
contract. Omit the flag to export full-length games. Ten 256-tick games
yielded 16,384 training and 4,096 validation decisions. The largest prompt
used 2,123 tokens with a local Qwen2.5 tokenizer, within 4,096 tokens. One
CPU optimizer step on a tiny local model reduced held-out loss from 5.5803
to 5.4859. This verifies the post-training path, not policy quality.

From a Metta checkout with `metta-posttrain` installed:

```sh
uv run --package metta-posttrain --extra train python -m metta_posttrain.train \
  --dataset /tmp/nmmo-data --output /tmp/nmmo-adapter \
  --model Qwen/Qwen3-0.6B --max-steps 100 --max-length 4096
```

## Numeric reinforcement learning

`tools/train_bridge.py` exposes the original 1,707 observation bytes as
numeric values and all 26 original actions. The bridge keeps every opponent
on a scripted policy and advances the production Wasm simulator only after
all eight seats choose. `semantic_view` carries the exact base64 observation
and per-life reset flag; `messages` carry the decoded text view above.

```sh
uv run python tools/test_train_bridge.py
```

From a Metta checkout with the Coworld training stack, pass the bridge
command, absolute manifest path, and `default` variant to
`recipes.external.coworld.train` for native PufferLib or
`recipes.external.coworld_metta_rl.train` for Metta RL. Use `players=8`,
`max_decisions=40000`, and a timestep limit for the certified 5,000-tick
game. A fourth bridge argument sets a shorter tick cap for curriculum runs;
set `max_decisions` to eight times that cap.

A complete certified-length bridge game finished all 40,000 seat decisions
with nonzero native scores for every seat. A 32-tick curriculum completed 512
Metta RL timesteps. A native PufferLib run on one RTX 4090 completed 4,096
timesteps, saved and reloaded its checkpoint, and evaluated four games on
each held-out seed. The learner's native score was 1.000 on both seeds; its
pairwise performance was 0.393 and 0.446. These short curriculum runs verify
the training and reload paths, not policy quality at 5,000 ticks.
