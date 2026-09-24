"""Exercise complete NMMO games through the numeric bridge."""

import base64
import json
import random
import subprocess
import sys
from pathlib import Path


manifest = Path(__file__).resolve().parents[1] / "coworld_manifest_template.json"
for policy in ("teacher", "random"):
    with subprocess.Popen(
        [sys.executable, "tools/train_bridge.py", str(manifest), "default", "32"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as bridge:
        assert bridge.stdin is not None and bridge.stdout is not None

        def request(payload):
            bridge.stdin.write(json.dumps(payload) + "\n")
            bridge.stdin.flush()
            return json.loads(bridge.stdout.readline())

        observation = request({"kind": "reset", "seed": f"nmmo-{policy}", "players": 8})
        rng = random.Random(42)
        decisions = 0
        while observation["kind"] == "decision":
            view = observation["semantic_view"]
            raw = base64.b64decode(view["obs"])
            assert len(raw) == 1707 and isinstance(view["reset"], bool)
            assert "terrain_rows" in json.loads(observation["messages"][1]["content"])
            encoded = request({"kind": "encode"})
            assert encoded["decision_id"] == observation["decision_id"]
            assert encoded["values"] == list(raw)
            assert encoded["actions"] == [{"action": action} for action in range(26)]
            if policy == "teacher":
                action = json.loads(request({"kind": "teacher"})["response"])
            else:
                action = rng.choice(encoded["actions"])
            result = request(
                {"kind": "step", "decision_id": observation["decision_id"], "response": json.dumps(action)}
            )
            assert result["kind"] == "accepted" and result["action"] == action
            observation = result["observation"]
            decisions += 1
        assert decisions == 32 * 8
        assert set(observation["scores"]) == {str(seat) for seat in range(8)}
        assert all(score >= 0 for score in observation["scores"].values())
        bridge.stdin.close()
        assert bridge.wait() == 0
    print(f"NMMO {policy}: {decisions} decisions, 1707 raw values")
