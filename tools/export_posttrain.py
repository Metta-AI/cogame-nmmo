"""Export complete native NMMO episodes as decoded action conversations."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from cogame_nmmo.sim import NmmoSim
from players.scripted_player import (
    SCALARS_OFF,
    ScriptedPolicy,
    TILE_BYTES,
    WINDOW_COLS,
    WINDOW_ROWS,
)


SYSTEM_PROMPT = (
    "Control one NMMO3 hero. Choose one integer action from 0 through 25. "
    "Actions 0-3 move down, up, right, left; 4 waits; 5 attacks; 8-16 "
    "select inventory slots; 20 buys; 21 sells; 22-25 run in the four "
    "movement directions. Return JSON with one action field. Terrain rows "
    "contain observed terrain ids. Entity hints may be stale."
)


def observed_view(row: np.ndarray, tick: int, reset: bool) -> dict:
    tiles = row[:SCALARS_OFF].reshape(WINDOW_ROWS, WINDOW_COLS, TILE_BYTES)
    return {
        "tick": tick,
        "reset": reset,
        "self_scalars": row[SCALARS_OFF:SCALARS_OFF + 47].tolist(),
        "terrain_rows": ["".join(str(int(value)) for value in line) for line in tiles[:, :, 1]],
        "items": [
            [int(r), int(c), int(tiles[r, c, 2]), int(tiles[r, c, 3])]
            for r, c in np.argwhere(tiles[:, :, 2] != 0)
        ],
        "entity_hints": [
            [int(r), int(c), *map(int, tiles[r, c, 4:9])]
            for r, c in np.argwhere(tiles[:, :, 4] != 0)
            if abs(r - WINDOW_ROWS // 2) <= 4 and abs(c - WINDOW_COLS // 2) <= 4
        ],
    }


def export(output: Path, episodes: int, ticks: int) -> None:
    if episodes < 5 or ticks < 1 or ticks > 5000:
        raise ValueError("Require at least five games and 1..5000 ticks")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = json.loads(Path("coworld_manifest_template.json").read_text())
    variant = next(entry for entry in manifest["variants"] if entry["id"] == "default")
    players = variant["game_config"]["num_agents"]
    assert players == 8 and variant["game_config"]["max_ticks"] == 5000
    train_count = 0
    validation_count = 0
    runs = []
    with (output / "train.jsonl").open("w") as train, (output / "validation.jsonl").open("w") as validation:
        for seed in range(1, episodes + 1):
            sim = NmmoSim(seed=seed, num_agents=players)
            policies = [ScriptedPolicy(seed=seed * 100 + seat) for seat in range(players)]
            resets = [False] * players
            sink = validation if seed % 5 == 0 else train
            for tick in range(ticks):
                observations = sim.observations()
                actions = []
                for seat, row in enumerate(observations):
                    action = policies[seat](tick, [row], [resets[seat]])[0][0]
                    actions.append(action)
                    sink.write(json.dumps({
                        "episode_id": f"nmmo-default-{seed}",
                        "seed": f"nmmo-default-{seed}",
                        "decision_id": tick * players + seat,
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(observed_view(row, tick, resets[seat]))},
                        ],
                        "completion": [{"role": "assistant", "content": json.dumps({"action": action})}],
                        "game": "nmmo", "action_schema_revision": "nmmo3-26-v1",
                    }) + "\n")
                sim.set_actions(np.asarray(actions, dtype=np.float32).reshape(players, 1))
                sim.step()
                assert sim.fault() == 0
                resets = sim.dones()
            scores = [sim.score(seat) / (sim.agent_stat(seat, 1) + 1) for seat in range(players)]
            runs.append({"seed": seed, "ticks": sim.tick(), "scores": scores})
            if seed % 5 == 0:
                validation_count += ticks * players
            else:
                train_count += ticks * players
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "game": "nmmo", "variant": "default",
        "source_revision": revision, "teacher": "scripted",
        "max_ticks": ticks, "train_examples": train_count,
        "validation_examples": validation_count, "runs": runs,
    }, indent=2) + "\n")
    print(f"train={train_count} validation={validation_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("episodes", type=int)
    parser.add_argument("--max-ticks", type=int, default=5000)
    args = parser.parse_args()
    export(args.output, args.episodes, args.max_ticks)
