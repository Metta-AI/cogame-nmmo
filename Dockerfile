# cogame-nmmo Coworld image: game server + bundled players in one image.
#
# Stage 1 (wasm-builder) compiles the vendored PufferLib nmmo3 sim with
# emscripten. Wasm artifacts are architecture-independent, so this stage
# runs on the build host's native platform ($BUILDPLATFORM) — no qemu
# emulation for the compile on ARM hosts.
#
# NOTE (phase status, mirrors .github/workflows/ci.yml): only
# sim/build_sim.sh runs here for now. build_brain.sh returns in Phase N3
# (nmmo3 MMONet brain shim + its xxd-embedded weights and the raylib
# prefetch it doesn't need) and build_viewer.sh in Phase N4 (nmmo3
# renderer viewer bundle + raylib web prefetch layer + viewer/dist COPY);
# until those phases land their shim sources are still moba-shaped and
# the scripts fail deliberately at compile.
#
# Stage 2 is the linux/amd64 runtime: python:3.11-slim + locked deps via
# uv, with the repo layout preserved at /workspace (server code resolves
# build/*.wasm and viewer/dist relative to the repo root, so the project
# is NOT pip-installed into site-packages).
#
# Entrypoints (Coworld manifest `run`):
#   game            python -m cogame_nmmo.server
#   baseline player python -m players.baseline_player   (Phase N3)
#   random player   python -m players.random_player     (Phase N3 adapts)
#
# Build: docker build --platform=linux/amd64 -t cogame-nmmo:local .

# Pin matches the emcc used for local builds (vendor/UPSTREAM.md: 6.0.5).
FROM --platform=$BUILDPLATFORM emscripten/emsdk:6.0.5 AS wasm-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl libdigest-sha-perl patch unzip xxd && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY vendor/ vendor/
COPY sim/ sim/

# apply_patches.sh runs inside the build script (idempotent). Phases
# N3/N4 append build_brain.sh and build_viewer.sh here — see the NOTE
# at the top of this file.
RUN bash sim/build_sim.sh


FROM python:3.11-slim

WORKDIR /workspace

# Locked runtime deps only (aiohttp/numpy/wasmtime): the project itself
# stays at /workspace via PYTHONPATH so repo-root-relative wasm/viewer
# paths keep working. uv is bind-mounted from its distribution image for
# this RUN only — a COPY'd binary would persist in its layer even after
# a later `rm`, so it never becomes a layer at all.
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.18,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --no-dev --no-install-project

ENV PATH="/workspace/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/server:/workspace" \
    PYTHONUNBUFFERED=1

COPY server/ server/
COPY players/ players/
COPY --from=wasm-builder /src/build/nmmo3_sim.wasm build/
# Phase N3 restores build/nmmo3_brain.wasm; Phase N4 restores viewer/dist/.

CMD ["python", "-m", "cogame_nmmo.server"]
