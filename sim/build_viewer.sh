#!/usr/bin/env bash
# Build the replay viewer from sim/viewer_main.c (patched tree):
#
#   viewer/dist/nmmo3_viewer.{js,wasm,data} + index.html   (browser bundle,
#       -DNMMO3_RENDER, raylib web, preloaded vendor render assets)
#   build/viewer_core.{js,wasm}                           (headless core,
#       no raylib, ENVIRONMENT=node — tests/test_viewer.py re-sim check)
#
# raylib: the exact prebuilt artifact upstream build.sh --web pins
# (raylib-5.5_webassembly.zip), fetched once into build/raylib-web/ and
# cached; sha256-verified. See vendor/UPSTREAM.md "Build-time dependency".
#
# Memory flags match sim/build_sim.sh rationale: the sim's ai_paths BFS
# cache is a 256 MB calloc (and viewer_seek re-allocates it); grow to
# 1 GB, abort loudly on OOM instead of NULL-write corruption.
# NOTE (Phase N4 pending): sim/viewer_main.c is still the moba viewer;
# Phase N4 rewrites it for the nmmo3 renderer (-DNMMO3_RENDER). Until
# then this script fails at the compile step.
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

VIEWER_EXPORTS=_viewer_load,_viewer_seek,_viewer_advance_frame,_viewer_tick,_viewer_total_ticks,_viewer_set_speed,_viewer_get_speed,_viewer_set_playing,_viewer_playing,_viewer_done,_viewer_winner,_viewer_state_digest,_malloc,_free

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
    -sEXPORTED_FUNCTIONS="_main,$VIEWER_EXPORTS" \
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8 \
    --preload-file vendor/upstream/resources@resources \
    -o viewer/dist/nmmo3_viewer.js
cp viewer/index.html viewer/dist/index.html

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
