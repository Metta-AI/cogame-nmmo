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
