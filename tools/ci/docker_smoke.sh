#!/usr/bin/env bash
# Raw-Docker episode smoke (Coworld Cookbook shape): one game container
# + one player container per seat on the coworld-local network with
# file:// artifact URIs. Asserts the episode completes and writes
# results.json (with the manifest's result keys) and the replay.
#
# usage: tools/ci/docker_smoke.sh [image]   (default cogame-nmmo:ci)
set -euo pipefail

image="${1:-cogame-nmmo:ci}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nmmo-smoke.XXXXXX")"
run_id="$$"

cleanup() {
  docker ps -aq --filter "name=nmmo-smoke-${run_id}" | xargs -r docker rm -f >/dev/null 2>&1 || true
  rm -rf "${work_dir}"
}
trap cleanup EXIT

# Team variant (2 seats x 5 heroes) keeps the smoke to 3 containers.
cat > "${work_dir}/config.json" <<'JSON'
{
  "seed": 7,
  "max_ticks": 200,
  "heroes_per_seat": 5,
  "tick_deadline_ms": 1000,
  "player_connect_timeout_seconds": 120,
  "players": [{"name": "smoke-radiant"}, {"name": "smoke-dire"}],
  "tokens": ["token-0", "token-1"]
}
JSON
chmod 777 "${work_dir}"

docker network inspect coworld-local >/dev/null 2>&1 || docker network create coworld-local

docker run -d --name "nmmo-smoke-${run_id}-game" \
  --network coworld-local --network-alias "nmmo-smoke-${run_id}" \
  -e COGAME_HOST=0.0.0.0 -e COGAME_PORT=8080 \
  -e COGAME_CONFIG_URI=file:///coworld/config.json \
  -e COGAME_RESULTS_URI=file:///coworld/results.json \
  -e COGAME_SAVE_REPLAY_URI=file:///coworld/replay \
  -e COGAME_PLAYER_FAILURE_URI=file:///coworld/player_failure.json \
  -v "${work_dir}:/coworld:rw" \
  "${image}" >/dev/null

for slot in 0 1; do
  docker run -d --name "nmmo-smoke-${run_id}-p${slot}" --network coworld-local \
    -e COWORLD_PLAYER_WS_URL="ws://nmmo-smoke-${run_id}:8080/player?slot=${slot}&token=token-${slot}" \
    "${image}" python -m players.baseline_player >/dev/null
done

echo "waiting for the episode (game container exit) ..."
deadline=$((SECONDS + 300))
while docker ps -q --filter "name=nmmo-smoke-${run_id}-game" | grep -q .; do
  if (( SECONDS > deadline )); then
    echo "FAIL: game container did not exit within 300s" >&2
    docker logs "nmmo-smoke-${run_id}-game" 2>&1 | tail -30 >&2
    exit 1
  fi
  sleep 2
done

exit_code="$(docker inspect -f '{{.State.ExitCode}}' "nmmo-smoke-${run_id}-game")"
if [[ "${exit_code}" != "0" ]]; then
  echo "FAIL: game container exited ${exit_code}" >&2
  docker logs "nmmo-smoke-${run_id}-game" 2>&1 | tail -30 >&2
  exit 1
fi

python3 - "${work_dir}" <<'EOF'
import json, sys
from pathlib import Path

work = Path(sys.argv[1])
results = json.loads((work / "results.json").read_text())
expected = {
    "names", "scores", "win", "team", "winner", "end_reason", "final_tick",
    "seed", "reward_sums", "ancient_healths", "agent_stats", "noop_ticks",
    "dead_seats", "noop_causes",
}
assert set(results) == expected, f"results keys drifted: {sorted(set(results) ^ expected)}"
assert len(results["scores"]) == 2, results["scores"]
# win (1.0 + 0.0) and draw (0.5 + 0.5) both sum to 1.0
assert sum(results["scores"]) == 1.0, results["scores"]
# Both players must have actually played every tick: a broken player
# entrypoint would show up as noop fallbacks / a strike-rule dead seat,
# and must fail the smoke rather than ride a NOOP-vs-NOOP episode.
assert results["noop_ticks"] == [0, 0], results["noop_ticks"]
assert results["dead_seats"] == [False, False], results["dead_seats"]
replay = (work / "replay").read_bytes()
assert replay[:4] == b"MOBA", replay[:8]
print(f"smoke OK: end_reason={results['end_reason']} winner={results['winner']} "
      f"final_tick={results['final_tick']} replay={len(replay)}B")
EOF
