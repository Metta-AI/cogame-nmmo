"""Jev-backed NMMO3 player.

Jev judges the 26 legal NMMO3 actions from a compact decode of the 1707-byte
observation.  The websocket transport remains the same as every other player.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from .client import run_policy_main, seed_from_env

ACTION_NAMES = {
    0: "move_down", 1: "move_up", 2: "move_right", 3: "move_left",
    4: "noop", 5: "attack", 6: "noop_6", 7: "noop_7",
    **{i: f"use_slot_{i - 8}" for i in range(8, 17)},
    17: "noop_17", 18: "noop_18", 19: "noop_19",
    20: "buy", 21: "sell", 22: "run_down", 23: "run_up",
    24: "run_right", 25: "run_left",
}


def _decode_observation(obs: bytes, reset: bool) -> dict[str, object]:
    if len(obs) != 1707:
        raise ValueError(f"NMMO observation must be 1707 bytes, got {len(obs)}")
    scalars = obs[1650:1697]
    return {
        "reset": reset,
        "combat_level": scalars[1],
        "element": scalars[2],
        "direction": scalars[3],
        "animation": scalars[4],
        "hp": scalars[5],
        "hp_max": scalars[6],
        "profession_level": scalars[7],
        "ui_mode": scalars[8],
        "gold": scalars[11],
        "in_combat": scalars[12],
        "equipment": list(scalars[13:18]),
        "inventory": list(scalars[18:30]),
        "visible_items": [
            {"type": obs[i + 2], "tier": obs[i + 3]}
            for i in range(0, 1650, 10) if obs[i + 2]
        ],
    }


def _ask_jev(state: dict[str, object], api_key: str, base_url: str, model: str) -> int:
    criteria = {str(i): f"{name} (action id {i})" for i, name in ACTION_NAMES.items()}
    body = json.dumps({
        "state": {"game": "Neural MMO 3", "observation": state},
        "model": model,
        "questions": {"action": {
            "type": "choice",
            "instructions": "Choose the action most likely to increase survival and the lower of combat or profession level. NOOP is valid when no action is safe.",
            "criteria": criteria,
        }},
    }).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/systemone", data=body,
        headers={"accept": "application/json", "authorization": f"Bearer {api_key}", "content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    answer = result["answers"]["action"]
    action = int(answer["choice"])
    if action not in ACTION_NAMES:
        raise ValueError(f"Jev selected invalid NMMO action {action}")
    print(json.dumps({"kind": "jev_judgment", "model": result["model"], "choice": ACTION_NAMES[action], "confidence": answer["confidence"], "probabilities": answer["probabilities"], "inputTokens": result["usage"].get("input_tokens", 0), "outputTokens": result["usage"].get("output_tokens", 0)}), flush=True)
    return action


class JevPolicy:
    def __init__(self) -> None:
        self.api_key = os.environ["TYPESAFE_API_KEY"]
        self.base_url = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        self.model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")

    def __call__(self, tick: int, obs_rows: list, resets: list) -> list[list[int]]:
        return [[_ask_jev(_decode_observation(obs, reset), self.api_key, self.base_url, self.model)] for obs, reset in zip(obs_rows, resets, strict=True)]


def policy_from_env() -> JevPolicy:
    seed_from_env(default=None)
    return JevPolicy()


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
