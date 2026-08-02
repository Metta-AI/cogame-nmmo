"""Staging-completeness guard negative tests (no emcc build needed).

sim/build_viewer.sh sources ``viewer_asset_guard`` from
sim/viewer_asset_guard.sh; these tests invoke that function directly
under bash with a synthetic nmmo3.h and a caller-controlled
VIEWER_ASSETS list, proving the failure path actually fails (nonzero
exit + loud stderr) — the full build only ever exercises the passing
path. Fast: pure bash subprocess, no emcc, no build/ artifacts.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_SH = REPO_ROOT / "sim" / "viewer_asset_guard.sh"

# Synthetic header covering every expansion branch the guard knows:
# a literal asset, the %i shader pattern, a %d 10-frame pattern, and
# the intentionally-absent font.
FAKE_HEADER = """
    LoadTexture("resources/nmmo3/merged_sheet.png");
    TextFormat("resources/nmmo3/map_shader_%i.fs", GLSL_VERSION)
    TextFormat("resources/nmmo3/neutral_%d.png", frame)
    LoadFont("resources/nmmo3/ManaSeedBody.ttf");
"""

COMPLETE_ASSETS = (
    ["merged_sheet.png", "map_shader_100.fs"]
    + [f"neutral_{i}.png" for i in range(10)]
)


def _run_guard(tmp_path, assets, header_text=FAKE_HEADER, with_font=False):
    """Source the guard function with the given VIEWER_ASSETS and a
    synthetic nmmo3.h; return the CompletedProcess."""
    header = tmp_path / "nmmo3.h"
    header.write_text(header_text)
    assets_src = tmp_path / "assets-src"
    assets_src.mkdir(exist_ok=True)
    if with_font:
        (assets_src / "ManaSeedBody.ttf").write_bytes(b"\x00font")
    quoted = " ".join(f"'{a}'" for a in assets)
    script = tmp_path / "run_guard.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"VIEWER_ASSETS=({quoted})\n"
        f"NMMO3_H='{header}'\n"
        f"ASSETS_SRC='{assets_src}'\n"
        f". '{GUARD_SH}'\n"
        "viewer_asset_guard\n"
    )
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=30
    )


def test_guard_passes_on_complete_list(tmp_path):
    proc = _run_guard(tmp_path, COMPLETE_ASSETS)
    assert proc.returncode == 0, proc.stderr


def test_guard_fails_loudly_on_thinned_list(tmp_path):
    """A vendor bump that references an unstaged asset must fail the
    build with one loud stderr line per uncovered file — never ship a
    silently thinner .data."""
    thinned = [a for a in COMPLETE_ASSETS
               if a not in ("merged_sheet.png", "neutral_3.png")]
    proc = _run_guard(tmp_path, thinned)
    assert proc.returncode != 0
    assert ("nmmo3.h loads resources/nmmo3/merged_sheet.png but "
            "VIEWER_ASSETS does not stage it") in proc.stderr
    assert "resources/nmmo3/neutral_3.png" in proc.stderr
    # both misses reported, not just the first
    assert proc.stderr.count("does not stage it") == 2


def test_guard_fails_when_header_has_no_references(tmp_path):
    """An nmmo3.h with zero resources/nmmo3/ references means the scan
    itself broke (wrong path, upstream restructure): hard error, not a
    vacuous pass."""
    proc = _run_guard(tmp_path, COMPLETE_ASSETS,
                      header_text="// no asset loads here\n")
    assert proc.returncode != 0
    assert "guard broken?" in proc.stderr


def test_guard_flags_manaseed_font_appearance(tmp_path):
    """ManaSeedBody.ttf is intentionally absent upstream; if a vendor
    bump ships it, the guard must fail so a human decides whether to
    stage it."""
    proc = _run_guard(tmp_path, COMPLETE_ASSETS, with_font=True)
    assert proc.returncode != 0
    assert "ManaSeedBody.ttf" in proc.stderr
    assert "decide whether" in proc.stderr
