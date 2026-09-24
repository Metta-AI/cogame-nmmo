#!/usr/bin/env python3
"""Persistent numeric decisions over the production NMMO3 Wasm simulator."""

import base64
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT))

from cogame_nmmo.sim import OBS_SIZE, NmmoSim  # noqa: E402
from players.scripted_player import ScriptedPolicy  # noqa: E402
from tools.export_posttrain import SYSTEM_PROMPT, observed_view  # noqa: E402


class Bridge:
    def __init__(self, manifest: Path, variant: str, max_ticks: int | None = None) -> None:
        document = json.loads(manifest.read_text())
        config = next(entry["game_config"] for entry in document["variants"] if entry["id"] == variant)
        self.players = config["num_agents"]
        assert variant == "default" and self.players == 8
        self.max_ticks = config["max_ticks"] if max_ticks is None else max_ticks
        assert 1 <= self.max_ticks <= config["max_ticks"]
        self.sim: NmmoSim
        self.policies: list[ScriptedPolicy]
        self.observations: np.ndarray
        self.resets: list[bool]
        self.actions: list[int]
        self.cursor = 0
        self.decision_id = 0

    def current(self) -> dict:
        row = self.observations[self.cursor]
        view = observed_view(row, self.sim.tick(), self.resets[self.cursor])
        return {
            "kind": "decision", "game": "nmmo", "decision_id": self.decision_id,
            "seat": self.cursor, "engine_seat": self.cursor, "turn": self.sim.tick(),
            "semantic_view": {
                "obs": base64.b64encode(row.tobytes()).decode(),
                "reset": self.resets[self.cursor],
            },
            "inbox": [],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(view)},
            ],
            "speech_messages": [],
            "action_schema": {"type": "object", "properties": {
                "action": {"enum": list(range(26))}}, "required": ["action"]},
            "typed_question": None,
        }

    def encode(self) -> dict:
        values = self.observations[self.cursor].tolist()
        assert len(values) == OBS_SIZE
        return {"decision_id": self.decision_id, "values": values,
                "actions": [{"action": action} for action in range(26)]}

    def reset(self, request: dict) -> dict:
        assert request["players"] == self.players
        seed = zlib.crc32(request["seed"].encode()) & 0xFFFFFFFF
        self.sim = NmmoSim(seed=seed, num_agents=self.players)
        self.policies = [ScriptedPolicy(seed=seed + seat) for seat in range(self.players)]
        self.observations = self.sim.observations()
        self.resets = [False] * self.players
        self.actions = [4] * self.players
        self.cursor = 0
        self.decision_id = 0
        return self.current()

    def teacher(self) -> dict:
        action = self.policies[self.cursor](
            self.sim.tick(), [self.observations[self.cursor]], [self.resets[self.cursor]]
        )[0][0]
        return {"response": json.dumps({"action": action})}

    def step(self, request: dict) -> dict:
        assert request["decision_id"] == self.decision_id
        action = json.loads(request["response"])
        assert set(action) == {"action"} and action["action"] in range(26)
        self.actions[self.cursor] = action["action"]
        self.cursor += 1
        self.decision_id += 1
        if self.cursor == self.players:
            self.sim.set_actions(np.asarray(self.actions, dtype=np.float32).reshape(self.players, 1))
            self.sim.step()
            assert self.sim.fault() == 0
            self.cursor = 0
            self.resets = self.sim.dones()
            if self.sim.tick() == self.max_ticks:
                scores = {
                    str(seat): self.sim.score(seat) / (self.sim.agent_stat(seat, 1) + 1)
                    for seat in range(self.players)
                }
                observation = {"kind": "terminal", "scores": scores}
            else:
                self.observations = self.sim.observations()
                observation = self.current()
        else:
            observation = self.current()
        return {"kind": "accepted", "action": action, "observation": observation}


if __name__ == "__main__":
    assert len(sys.argv) in (3, 4), "usage: train_bridge.py MANIFEST VARIANT [MAX_TICKS]"
    bridge = Bridge(Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else None)
    for line in sys.stdin:
        request = json.loads(line)
        response = {
            "reset": bridge.reset,
            "encode": lambda _request: bridge.encode(),
            "teacher": lambda _request: bridge.teacher(),
            "step": bridge.step,
        }[request["kind"]](request)
        print(json.dumps(response, separators=(",", ":")), flush=True)
