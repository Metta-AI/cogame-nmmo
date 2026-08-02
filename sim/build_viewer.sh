#!/usr/bin/env bash
# Build the replay viewer from sim/viewer_main.c (patched tree):
#
#   viewer/dist/nmmo3_viewer.{js,wasm,data} + index.html   (browser bundle,
#       -DNMMO3_RENDER, raylib web, preloaded staged render assets)
#   build/viewer_core.{js,wasm}                           (headless core,
#       no raylib, ENVIRONMENT=node — tests/test_viewer.py re-sim check)
#
# raylib: the exact prebuilt artifact upstream build.sh --web pins
# (raylib-5.5_webassembly.zip), fetched once into build/raylib-web/ and
# cached; sha256-verified. See vendor/UPSTREAM.md "Build-time dependency".
#
# Memory flags match sim/build_sim.sh rationale: grow to 1 GB, abort
# loudly on OOM instead of NULL-write corruption.
#
# Assets: the ~14 MB of render assets make_client loads (nmmo3.h:
# 2456-2531) are STAGED into build/viewer-assets/ by explicit list and
# preloaded from there — never the raw vendor resources dir, which also
# holds nmmo3_weights.bin (17.7 MB of policy weights the renderer never
# reads; shipping them would more than double the bundle).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v emcc >/dev/null 2>&1; then
    echo "error: emcc not found on PATH - install emscripten (brew install emscripten)" >&2
    exit 1
fi

./sim/apply_patches.sh

# -- raylib 5.5 web (prebuilt, cached) ---------------------------------------
RAYLIB_DIR=build/raylib-web
RAYLIB_ZIP_URL="https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_webassembly.zip"
RAYLIB_ZIP_SHA256="798b6bea650e78a60fe49f106a15d92ea4e33efd3aa1b3efa34b0438a14bbf2c"

# Cache guard includes the pin: a RAYLIB_ZIP_SHA256 bump invalidates a
# stale build/raylib-web/ (the stamp file records which zip it came from).
RAYLIB_STAMP="$RAYLIB_DIR/.zip-sha256"
if [ ! -f "$RAYLIB_DIR/lib/libraylib.a" ] || \
   [ "$(cat "$RAYLIB_STAMP" 2>/dev/null)" != "$RAYLIB_ZIP_SHA256" ]; then
    echo "Fetching raylib-5.5_webassembly ..."
    # trailing X's for BSD/GNU mktemp portability; unzip ignores the
    # missing .zip extension, and the trap reaps failed downloads
    tmpzip="$(mktemp "${TMPDIR:-/tmp}/raylib-web.zip.XXXXXX")"
    trap 'rm -f "$tmpzip"' EXIT
    curl -fsSL --retry 3 "$RAYLIB_ZIP_URL" -o "$tmpzip"
    echo "$RAYLIB_ZIP_SHA256  $tmpzip" | shasum -a 256 -c - >/dev/null
    rm -rf "$RAYLIB_DIR" build/raylib-5.5_webassembly
    (cd build && unzip -q "$tmpzip")
    mv build/raylib-5.5_webassembly "$RAYLIB_DIR"
    printf '%s\n' "$RAYLIB_ZIP_SHA256" > "$RAYLIB_STAMP"
    rm -f "$tmpzip"
    trap - EXIT
fi

# -- stage render assets (weights EXCLUDED) ----------------------------------
# Exactly what make_client() loads on the web (nmmo3.h:2456-2531), plus
# the assets license. GLSL 330 is desktop-only; the web build resolves
# map_shader_100.fs (GLSL_VERSION 100 without PLATFORM_DESKTOP).
# ManaSeedBody.ttf is intentionally absent upstream (raylib falls back to
# its default font) — do not add it. A missing listed file is a hard
# error: a silently thinner .data would die at runtime asset load.
ASSETS_SRC=vendor/upstream/resources/nmmo3
ASSETS_DIR=build/viewer-assets
rm -rf "$ASSETS_DIR"
mkdir -p "$ASSETS_DIR/resources/nmmo3"
VIEWER_ASSETS=(
    map_shader_100.fs
    merged_sheet.png
    items_condensed.png
    inventory_64.png
    inventory_64_selected.png
    inventory_64_press.png
    ASSETS_LICENSE.md
)
for element in neutral fire water earth air; do
    for i in 0 1 2 3 4 5 6 7 8 9; do
        VIEWER_ASSETS+=("${element}_${i}.png")
    done
done
for f in "${VIEWER_ASSETS[@]}"; do
    cp "$ASSETS_SRC/$f" "$ASSETS_DIR/resources/nmmo3/$f"
done

# -- staging-completeness guard ----------------------------------------------
# Logic lives in sim/viewer_asset_guard.sh (sourced function) so
# tests/test_viewer_guard.py can exercise its failure path without an
# emcc build; see that file's header comment for the full rationale.
NMMO3_H=build/src-patched/nmmo3.h
# shellcheck source=sim/viewer_asset_guard.sh
. "$REPO_ROOT/sim/viewer_asset_guard.sh"
viewer_asset_guard || exit 1

VIEWER_EXPORTS=_viewer_load,_viewer_seek,_viewer_advance,_viewer_advance_frame,_viewer_render_phase,_viewer_tick,_viewer_total_ticks,_viewer_set_speed,_viewer_get_speed,_viewer_set_playing,_viewer_playing,_viewer_state_digest,_malloc,_free
# Camera-follow controls exist only in the render build (they poke the
# raylib client); the headless core has no client to follow with.
RENDER_EXPORTS=_main,_viewer_set_follow,_viewer_get_follow,$VIEWER_EXPORTS

MEM_FLAGS=(-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sABORTING_MALLOC=1
           -sINITIAL_MEMORY=512MB -sSTACK_SIZE=512KB)

# -- browser bundle (render build) -------------------------------------------
mkdir -p viewer/dist
emcc -O2 -DNMMO3_RENDER -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
    -I build/src-patched -I "$RAYLIB_DIR/include" \
    sim/viewer_main.c "$RAYLIB_DIR/lib/libraylib.a" \
    -sUSE_GLFW=3 -sUSE_WEBGL2=1 \
    "${MEM_FLAGS[@]}" \
    -sENVIRONMENT=web \
    -sEXPORTED_FUNCTIONS="$RENDER_EXPORTS" \
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8 \
    --preload-file "$ASSETS_DIR/resources@resources" \
    -o viewer/dist/nmmo3_viewer.js
cp viewer/index.html viewer/dist/index.html

# Embed the sha of the sim wasm this viewer was built alongside (index
# warns on screen when a replay header's sim_wasm_sha256 differs). The
# build order (build_sim.sh before build_viewer.sh, as in Dockerfile and
# CI) makes build/nmmo3_sim.wasm the matching sim; absent (viewer-only
# dev build), the warning is simply disabled.
if [ -f build/nmmo3_sim.wasm ]; then
    sim_sha="$(shasum -a 256 build/nmmo3_sim.wasm | cut -d' ' -f1)"
    printf 'window.SIM_WASM_SHA256 = "%s";\n' "$sim_sha" \
        > viewer/dist/sim_sha.js
else
    echo "build/nmmo3_sim.wasm absent: sim-sha mismatch warning disabled" >&2
    printf 'window.SIM_WASM_SHA256 = null;\n' > viewer/dist/sim_sha.js
fi

# -- headless core (node verification build) ---------------------------------
emcc -O2 \
    -I build/src-patched \
    sim/viewer_main.c \
    --no-entry \
    "${MEM_FLAGS[@]}" \
    -sENVIRONMENT=node \
    -sMODULARIZE=1 -sEXPORT_NAME=createViewerCore \
    -sEXPORTED_FUNCTIONS="$VIEWER_EXPORTS" \
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8 \
    -o build/viewer_core.js

ls -la viewer/dist build/viewer_core.*
echo "build_viewer: OK"
