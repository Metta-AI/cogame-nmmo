#!/usr/bin/env bash
# Materialize patched source trees from the pristine vendor copy.
#
#   build/src-pristine/  = vendor/upstream + patch 0001 ONLY
#                          (0001 is the render guard, required to compile the
#                          sim at all without raylib; it changes no sim lines)
#   build/src-patched/   = vendor/upstream + ALL patches in sim/patches/
#
# vendor/upstream/ itself is never modified (byte-pristine rule, see
# vendor/UPSTREAM.md). Patch rationale lives in vendor/PATCHES.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$REPO_ROOT/vendor/upstream"
PATCHES="$REPO_ROOT/sim/patches"
BUILD="$REPO_ROOT/build"

rm -rf "$BUILD/src-pristine" "$BUILD/src-patched"
mkdir -p "$BUILD/src-pristine" "$BUILD/src-patched"
# Top-level source files only: vendor/upstream/resources/ holds render
# assets (preloaded by sim/build_viewer.sh), not compiled sources.
find "$UPSTREAM" -maxdepth 1 -type f -exec cp {} "$BUILD/src-pristine/" \;
find "$UPSTREAM" -maxdepth 1 -type f -exec cp {} "$BUILD/src-patched/" \;

# --fuzz=0: a patch that no longer applies at its exact context must
# fail loudly, never fuzz-apply against drifted vendor source.

# Pristine tree: 0001 only.
patch -p1 -s --fuzz=0 -d "$BUILD/src-pristine" < "$PATCHES/0001-render-guard.patch"

# Patched tree: every patch, in order.
for p in "$PATCHES"/*.patch; do
    patch -p1 -s --fuzz=0 -d "$BUILD/src-patched" < "$p"
done

echo "apply_patches: OK (pristine=0001 only, patched=$(ls "$PATCHES" | wc -l | tr -d ' ') patches)"
