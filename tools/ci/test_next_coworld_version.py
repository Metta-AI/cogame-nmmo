#!/usr/bin/env python3
"""Fixture tests for the next_coworld_version picker (stdlib only, no network).

Run from anywhere: python3 tools/ci/test_next_coworld_version.py
The upload workflows run this before every version computation.
"""

import sys
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import next_coworld_version as picker  # noqa: E402

compute_next = picker.compute_next


def row(name, version, canonical=False, rid="cow_test"):
    return {"id": rid, "name": name, "version": version, "canonical": canonical}


def expect_exit(fn, fragment):
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        assert fragment in msg, f"expected {fragment!r} in error, got: {msg}"
        return
    raise AssertionError(f"expected SystemExit containing {fragment!r}, none raised")


# THE defect scenario (2026-07-30/31 wedge): canonical 0.7.127, orphan
# NON-canonical 0.7.128 above it. canonical-based picker returns 0.7.128
# and 409s forever; this picker must return 0.7.129.
orphan_rows = [
    row("ctf", "0.7.128", canonical=False),  # the orphan (newest-first)
    row("ctf", "0.7.127", canonical=True),
    row("ctf", "0.7.126", canonical=False),
    row("paintbot", "0.7.138", canonical=True),
]
assert compute_next(orphan_rows, "ctf") == "0.7.129"

# Clean registry: max row IS the canonical -> plain patch bump.
assert compute_next(orphan_rows, "paintbot") == "0.7.139"

# Numeric (not lexicographic) ordering: 0.7.9 < 0.7.10 < 0.7.100.
numeric_rows = [
    row("ctf", "0.7.9", canonical=True),
    row("ctf", "0.7.100", canonical=False),
    row("ctf", "0.7.10", canonical=False),
]
assert compute_next(numeric_rows, "ctf") == "0.7.101"

# Other names never leak into the computation.
mixed = orphan_rows + [row("speedrun-wow", "9.9.9", canonical=True)]
assert compute_next(mixed, "ctf") == "0.7.129"

# Under-read guards: a fetch that misses the canonical row must hard-fail,
# never emit a number that can re-collide. (This is the ONLY under-read
# signal the endpoint offers: canonical rows are a subset of the fetched
# rows, so a "max fetched < max canonical" comparison can never fire and
# deliberately does not exist.)
expect_exit(lambda: compute_next([row("ctf", "0.7.5")], "ctf"), "no canonical row")
expect_exit(lambda: compute_next([], "ctf"), "no rows for coworld")
expect_exit(lambda: compute_next(orphan_rows, "nosuch"), "no rows for coworld")

# Unparseable version for our name is a hard failure, not a silent skip —
# a skipped max row would re-collide.
expect_exit(
    lambda: compute_next([row("ctf", "0.7.x"), row("ctf", "0.7.1", canonical=True)], "ctf"),
    "non-semver",
)


class Response(BytesIO):
    def __init__(self, rows, cursor=None):
        super().__init__(json.dumps(rows).encode())
        self.headers = {"X-Next-Cursor": cursor} if cursor else {}


requests = []


def fake_urlopen(req, timeout):
    assert timeout == 60
    requests.append(parse_qs(urlsplit(req.full_url).query))
    if len(requests) == 1:
        return Response([row("nmmo", "0.1.5"), row("other", "1.0.0")], cursor="next/page")
    return Response([row("nmmo", "0.1.4", canonical=True)])


with patch.object(picker, "PAGE_SIZE", 2), patch.object(picker.urllib.request, "urlopen", side_effect=fake_urlopen):
    rows = picker.fetch_all_rows("test-token")
assert len(rows) == 3
assert requests == [{"limit": ["2"]}, {"limit": ["2"], "cursor": ["next/page"]}]
assert compute_next(rows, "nmmo") == "0.1.6"

print("test_next_coworld_version: all assertions passed")
