# Hosted deployment (Softmax)

The nmmo Coworld runs as a live platform-ladder league on Softmax. This file records the
production identifiers, the league configuration, and how to re-verify the deployment.
Deployed 2026-08-02 (Phase N6).

## Identifiers

| Thing | Id |
| --- | --- |
| Coworld (bootstrap upload, 0.1.0) | `cow_f0fb2e20-4fb1-4907-8db4-e826bff446d7` |
| Coworld (CI dispatch, 0.1.1) | `cow_c51249c7-49fe-4123-aca1-3f3dd3ad00cf` |
| Coworld (CI push picker, 0.1.2) | `cow_e66fe6ae-8a62-4497-bf9e-a3ebd0266abf` |
| League seed | `lseed_81fee93b-26dd-47a5-9a5f-106052218ecb` |
| League (display name "Nmmo") | `league_d7e141f9-ee29-4a46-b15c-5bbdb1639a44` |
| Game | `game_a41502d3-16da-4fcd-98b2-7c50b80c34b5` |
| Division (Competition, level 1) | `div_5f259000-d229-4205-bba7-28292e53a25d` |
| PufferLib player | `ply_767d5e55-d5d2-4bfd-93b8-acd89dcf057d` |
| daveey player | `ply_44ae9048-3242-4654-881f-6d9d43347fa3` |
| nmmo-baseline:v1 policy version | `77323624-430a-46f9-9c22-ee4096a0c67e` |
| nmmo-scripted:v1 policy version | `90817388-2c87-4499-b94e-7e2339cd65f8` |
| baseline submission / membership | `sub_06bea3d3-90f6-4748-a46d-576c2fa58ace` / `lpm_51e8bb9a-996f-4b7c-864e-b36a4ac3b877` |
| scripted submission / membership | `sub_87fc1774-f74c-49ed-a374-a2c1dcd379fc` / `lpm_edfcd2f8-ec94-4671-87cd-11e6c05dbc6f` |

The canonical Coworld version advances on every green push to main (see CI below); the league
resolves the coworld by name, so this table's cow_ ids are historical snapshots.

## League configuration

Platform ladder (Temporal), commissioner_key `platform`, single Competition division.
`settings` document (POST /v2/leagues/{league_id}/settings — POST replaces the whole document;
GET → merge first, and preserve siblings such as `counterfactual_eval`):

- Top-level `round_interval_minutes: 10`
- `ladder.enabled: true`
- scheduler: `swiss_neighbor`, `insufficient_players: multiple_seats`, `min_episodes_per_entrant: 8`
  (8 FFA seats per episode; with fewer than 8 champions the platform duplicates real policies into
  filler-marked seats — duplicates earn no credit)
- fulfillment: `allowed_failures: 0.05`, `retry_times: 2`
- ranking: `elo`, initial 1500, k 32, `round_scoring_rule: mean`
- divisions: Competition with `disqualify_after_consecutive_failures: 3`

Rounds are unpaused; the ladder parent workflow (`ladder-league_d7e141f9-…`, task queue
`league-ladder`) self-starts a round every ~10 minutes. Each round runs 8 episodes
(5000 ticks, `end_reason: tick_cap`).

League routes need a team credential with `X-Use-Elevated-Privileges: true`.

## CI publishing

- `SOFTMAX_TOKEN` repo secret: set (non-expiring CI credential).
- `UPLOAD_REQUIRED` repo variable: `true` — a missing/expired token now hard-fails the upload
  job instead of silently skipping.
- Every green push to main publishes the next patch version (highest-registry-row picker);
  `workflow_dispatch` with an explicit `version` input overrides. Verified live:
  run 30759401094 (dispatch, 0.1.1) and run 30759870568 (push picker, 0.1.2), both green with
  real (non-skipped) upload jobs.

## Deployment verification evidence (2026-08-02)

- Bootstrap upload 0.1.0 hosted-certified: all 10 steps pass (matriculate, source-resolves,
  images-reachable, fixture-conforms, smoke-episode, results-conform, replay-present,
  replay-loadable, players-run, supporting-roles).
- Round 1 (`round_1f3c4f24-6917-419d-b90b-d92d45ef21ac`): 8/8 episodes completed, every episode
  with a replay_url. Example scores (ereq_7e5fddb8, seed 2646520577): baseline seats
  6.14-7.80, scripted seats 1.14-1.34 mean-min(combat,profession)-per-life — matching the local
  scripted-vs-baseline expectation, non-degenerate across all 8 seats.
- Rounds 1-6 all completed, self-paced at exact 10-minute intervals (17:44:54, 17:54:55,
  18:04:55, 18:14:55, 18:24:56, 18:34:56) — only round 1 was manually triggered.
- Elo trajectory (div_5f259000): after round 3 PufferLib 1543.75 / daveey 1456.25; after round 6
  PufferLib 1576.96 (rank 1, win_rate 1.0) / daveey 1423.04 — decisively off 1500 with a
  monotonically widening gap under the consistent baseline winner.
- Hosted replay session (`coworld replay-open <ereq_> --hosted`): viewer URL served HTTP 200;
  in-browser the wasm viewer rendered the NMMO3 world, its standings panel matched the episode
  scores exactly, and seeking re-simulated deterministically to the target tick.

## How to re-verify

```bash
uv run coworld list                    # nmmo versions; exactly one canonical
uv run coworld status <cow_id>         # hosted certification verdict
uv run coworld rounds -l league_d7e141f9-ee29-4a46-b15c-5bbdb1639a44          # rounds self-starting ~10 min
uv run coworld episodes -r <round_id>  # 8 completed episodes, replay URLs
uv run coworld episode-results <ereq_> # scores: baseline ~5-8, scripted ~1.1-1.4 per seat
uv run coworld results div_5f259000-d229-4205-bba7-28292e53a25d               # Elo standings
uv run coworld replay-open <ereq_> --hosted   # hosted replay viewer
```

Pause / resume: `POST /v2/leagues/{league_id}/rounds-paused {"paused": true|false}`.
Retire: disable the seed (see the platform retire-seeded-league procedure).
