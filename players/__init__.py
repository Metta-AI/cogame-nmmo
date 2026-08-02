"""cogame-nmmo player clients.

- ``players.client``: reusable async websocket harness (URL from env,
  obs decode, action send, bounded reconnects).
- ``python -m players.random_player``: uniform-random policy.
- ``python -m players.baseline_player``: upstream pretrained policy via
  the wasm-compiled puffernet brain.
- ``python -m players.scripted_player``: hand-coded lane-push reference
  bot (finite-state, deterministic, no wasm dependency).
"""
