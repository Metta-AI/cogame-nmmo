"""Check complete, seed-separated NMMO training episodes."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


with TemporaryDirectory() as temporary:
    output = Path(temporary) / "nmmo"
    subprocess.run([sys.executable, "tools/export_posttrain.py", str(output), "5", "--max-ticks", "32"], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert manifest["train_examples"] == len(train) == 4 * 32 * 8
    assert manifest["validation_examples"] == len(validation) == 32 * 8
    assert len(manifest["runs"]) == 5
    assert all(run["ticks"] == 32 and len(run["scores"]) == 8 for run in manifest["runs"])
    assert {row["seed"] for row in train}.isdisjoint({row["seed"] for row in validation})
    for row in train + validation:
        assert row["game"] == "nmmo"
        assert [part["role"] for part in row["prompt"]] == ["system", "user"]
        view = json.loads(row["prompt"][1]["content"])
        assert len(view["self_scalars"]) == 47
        assert len(view["terrain_rows"]) == 11
        assert all(len(line) == 15 for line in view["terrain_rows"])
        assert all(abs(hint[0] - 5) <= 4 and abs(hint[1] - 7) <= 4 for hint in view["entity_hints"])
        action = json.loads(row["completion"][0]["content"])["action"]
        assert 0 <= action < 26
    print(f"NMMO: {len(train)} train, {len(validation)} validation decisions")
