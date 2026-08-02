#!/usr/bin/env python3
"""Compute the next coworld publish version from the registry's highest existing row.

Usage: next_coworld_version.py <coworld-name>
Env:   SOFTMAX_TOKEN (Bearer token for the registry API)

Why this exists (2026-07-31): `coworld next-version <name>` is CANONICAL-based.
A partially-failed publish (row uploaded, certification/canonicalize failed)
leaves an orphan NON-canonical row above the canonical one, and every
subsequent publish then recomputes the same taken number and dies with
"409: already exists" until someone clears the orphan. Seven consecutive
publishes failed overnight 2026-07-30/31 on orphan ctf:0.7.128 this way.
This picker takes max(existing rows for <name>) + 1 patch instead, so an
orphan row advances the counter rather than wedging it.

Known trade (accepted by the operator, decision 1217042089600759): a failed
certification now burns a version number, i.e. MORE orphan rows over time.
That is fine — rows are cheap, wedges are not. NOTE for readers: a publish
is NOT a seat; the league era oracle is rounds / league.game.coworld_id,
never "the newest registry row" (the fleet once mislabelled 0.7.125/0.7.126
as live this way).

Registry API facts this code is built on (verified 2026-07-31):
  - GET /v2/coworlds returns a bare JSON list, newest-first.
  - `limit` hard-caps at 500 (501 -> HTTP 422); the registry already holds
    more than 500 rows, so a single page silently under-reads.
  - `?name=` is IGNORED by the server; filter client-side.
  - `?offset=` works; page until a short page.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("SOFTMAX_API_BASE", "https://softmax.com/api/observatory")
PAGE_SIZE = 500  # server-side hard cap
MAX_PAGES = 200  # safety stop; ~100k rows before this trips
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _fetch_json(req, retries=1, backoff_seconds=5):
    """GET+parse with one bounded retry on transient failures (5xx/URLError).

    4xx responses raise immediately: they are contract/auth errors a retry
    cannot fix, and this tool must fail loudly rather than guess.
    """
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt >= retries:
                raise
            print(f"registry returned {exc.code}; retrying in "
                  f"{backoff_seconds}s", file=sys.stderr)
        except urllib.error.URLError:
            if attempt >= retries:
                raise
            print(f"registry unreachable; retrying in {backoff_seconds}s",
                  file=sys.stderr)
        time.sleep(backoff_seconds)


def fetch_all_rows(token):
    rows = []
    for page in range(MAX_PAGES):
        url = f"{BASE}/v2/coworlds?limit={PAGE_SIZE}&offset={page * PAGE_SIZE}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                # The site's edge blocks python's default "Python-urllib/x.y"
                # User-Agent with a 403 (verified 2026-07-31); any explicit UA passes.
                "User-Agent": "cogame-nmmo-ci/next_coworld_version",
            },
        )
        batch = _fetch_json(req)
        if not isinstance(batch, list):
            raise SystemExit(f"unexpected response shape from {url}: {type(batch)}")
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
    raise SystemExit(f"registry did not terminate within {MAX_PAGES} pages; refusing to guess")


def parse_version(row):
    m = SEMVER_RE.match(row.get("version") or "")
    if not m:
        # Never skip silently: an unparseable row for our name could be the max.
        raise SystemExit(
            f"row {row.get('id')} for {row.get('name')} has non-semver version "
            f"{row.get('version')!r}; refusing to compute a next version past it"
        )
    return tuple(int(g) for g in m.groups())


def compute_next(rows, name):
    """Return next version string for <name>: highest existing row, patch + 1.

    Truncated-fetch guard: hard-fails when the fetched set has no rows for
    <name> at all, or has rows but no canonical one. A published coworld
    always has exactly one canonical row, so its absence means the fetch
    under-read (or the name is wrong) — refuse to pick a number that could
    re-collide. A truncated fetch that still contains the canonical row
    cannot be detected from this endpoint alone (pagination is the only
    read the registry offers), so no stronger invariant is promised here.
    """
    mine = [r for r in rows if r.get("name") == name]
    if not mine:
        raise SystemExit(f"no rows for coworld {name!r} in {len(rows)} fetched rows")
    versions = [(parse_version(r), r) for r in mine]
    max_ver, max_row = max(versions, key=lambda vr: vr[0])
    if not any(r.get("canonical") for _, r in versions):
        raise SystemExit(
            f"no canonical row for {name!r} among {len(mine)} fetched rows; "
            "fetch likely truncated — refusing to pick a version"
        )
    nxt = f"{max_ver[0]}.{max_ver[1]}.{max_ver[2] + 1}"
    print(
        f"{name}: {len(mine)} rows, max existing {max_row.get('version')} "
        f"(canonical: {max_row.get('canonical')}), next -> {nxt}",
        file=sys.stderr,
    )
    return nxt


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <coworld-name>")
    token = os.environ.get("SOFTMAX_TOKEN")
    if not token:
        raise SystemExit("SOFTMAX_TOKEN is not set")
    rows = fetch_all_rows(token)
    print(f"fetched {len(rows)} registry rows", file=sys.stderr)
    print(compute_next(rows, sys.argv[1]))


if __name__ == "__main__":
    main()
