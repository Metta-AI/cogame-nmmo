#!/usr/bin/env bash
# Replay-viewer bundle build hook for `coworld build` (spec 0068).
#
# coworld.bundle._build_replay_viewer_bundle invokes this with one
# argument: the absolute bundle output directory (named after
# game.replay_viewer.bundle, "static-replay-viewer"). It must produce
# index.html there. We reuse the Dockerfile's wasm-builder stage (hot
# cache right after `coworld build`'s compose build) and copy out
# viewer/dist — the same bundle the game image serves at /client/replay.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/static-replay-viewer" >&2
  exit 1
fi

requested_output="$1"

if [[ "${requested_output}" != /* || "$(basename "${requested_output}")" != "static-replay-viewer" ]]; then
  echo "unsafe bundle output: ${requested_output}" >&2
  exit 1
fi

output_parent="$(cd "$(dirname "${requested_output}")" && pwd -P)"
output_dir="${output_parent}/static-replay-viewer"
if [[ "${output_dir}" != "${repo_dir}"/* || -L "${output_dir}" ]]; then
  echo "unsafe bundle output: ${requested_output}" >&2
  exit 1
fi

rm -rf "${output_dir}"
mkdir -p "${output_dir}"

image_tag="cogame-nmmo-viewer-build:$$"
container_id=""
cleanup() {
  if [[ -n "${container_id}" ]]; then
    docker rm "${container_id}" >/dev/null 2>&1 || true
  fi
  docker image rm "${image_tag}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The wasm-builder stage runs on $BUILDPLATFORM (wasm output is
# architecture-independent), so no --platform pin here.
docker build \
  --file "${repo_dir}/Dockerfile" \
  --target wasm-builder \
  --tag "${image_tag}" \
  "${repo_dir}"
container_id="$(docker create "${image_tag}")"
docker cp "${container_id}:/src/viewer/dist/." "${output_dir}"

test -f "${output_dir}/index.html"
