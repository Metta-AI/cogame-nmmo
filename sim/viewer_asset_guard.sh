# shellcheck shell=bash
# Staging-completeness guard for the viewer asset preload, sourced by
# sim/build_viewer.sh (and exercised directly, no emcc build needed, by
# tests/test_viewer_guard.py).
#
# Every resources/nmmo3/ path the (patched) render code references must
# be covered by VIEWER_ASSETS: a vendor bump that starts loading a new
# asset must fail the build loudly, not ship a silently thinner .data
# that renders blank sprites. TextFormat patterns are expanded (%i ->
# GLSL_VERSION, 100 on web; %d -> the 10-frame element loop). An
# unrecognized new pattern falls through to the membership check and
# fails there — extend the expansion when upstream adds one.
#
# Caller-provided inputs (globals):
#   VIEWER_ASSETS  bash array of filenames staged into the .data preload
#   NMMO3_H        path to the nmmo3.h to scan for asset references
#   ASSETS_SRC     vendor resources dir (ManaSeedBody.ttf decision check)
#
# Returns nonzero (with error lines on stderr) on any uncovered asset.
viewer_asset_guard() {
    local referenced
    # `|| true`: a zero-match grep exits 1, which under the caller's
    # `set -eo pipefail` would kill the script HERE, silently — before
    # the loud guard-broken error below ever printed.
    referenced="$(grep -o 'resources/nmmo3/[A-Za-z0-9_.%]*' "$NMMO3_H" \
        | sed 's|resources/nmmo3/||' | sort -u || true)"
    if [ -z "$referenced" ]; then
        echo "error: no resources/nmmo3/ references found in $NMMO3_H - guard broken?" >&2
        return 1
    fi
    local guard_fail=0 ref expanded f i
    for ref in $referenced; do
        case "$ref" in
            map_shader_%i.fs) expanded="map_shader_100.fs" ;;
            *_%d.png)
                expanded=""
                for i in 0 1 2 3 4 5 6 7 8 9; do
                    expanded+="${ref/\%d/$i} "
                done ;;
            ManaSeedBody.ttf)
                # Intentionally absent upstream (raylib falls back to its
                # default font). If a vendor bump ships the file, staging it
                # becomes a decision: fail loudly so a human makes it.
                if [ -e "$ASSETS_SRC/ManaSeedBody.ttf" ]; then
                    echo "error: ManaSeedBody.ttf now exists in $ASSETS_SRC - decide whether to add it to VIEWER_ASSETS" >&2
                    guard_fail=1
                fi
                continue ;;
            *) expanded="$ref" ;;
        esac
        for f in $expanded; do
            if ! printf '%s\n' "${VIEWER_ASSETS[@]}" | grep -qxF "$f"; then
                echo "error: nmmo3.h loads resources/nmmo3/$f but VIEWER_ASSETS does not stage it" >&2
                guard_fail=1
            fi
        done
    done
    return "$guard_fail"
}
