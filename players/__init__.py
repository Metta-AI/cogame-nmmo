"""cogame-nmmo player clients.

- ``players.client``: reusable async websocket harness (URL from env,
  obs decode, action send, bounded reconnects).
- ``python -m players.random_player``: uniform-random policy.
- ``python -m players.baseline_player``: upstream pretrained MMONet
  policy via the wasm-compiled brain.

NOTE (Phase N3 pending): random_player and baseline_player are still
moba-shaped (6-int action rows, moba brain wasm); Phase N3 adapts them
to the NMMO3 26-way single-action protocol and adds the scripted
survival-FSM player.
"""
