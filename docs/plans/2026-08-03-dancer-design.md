# Dancer (scripted v4) — design

Goal: a scripted league policy that beats the pretrained net's ~6.7-7.2 mean
seat score by exploiting mechanics the net never learned. Target: min(comb,
prof) ~12-15 on one long life, deaths ≈ 0.

## Verified foundations (do not re-derive; see commit bd1a6ab + campaign learnings)

- Melee chase rule is exactly deterministic (33/33 ground-truth prediction):
  attack at Manhattan-1 on its turn; else 1 step reducing the strictly-largest
  gap axis (tie → column); blocked steps stall. Players act before enemies.
- Diagonal sword dance: free hit every ~3 ticks, zero return damage, vs any
  melee the damage formula can hurt (dmg = 40 + 2L + equip_atk − 5L′ > 0).
- element byte ≠ 0 ⟺ bow ⟺ level ≥ 15 (perfect discriminator; survives respawn).
- Damage: enemy→us 15 + 5L′ − 2L − equip_def; herb heals 50 + 10·tier in combat;
  armor def 8·2^(t−1)/piece; sword atk 3·8·2^(t−1); kills of ≥-level enemies
  are +1 comb and drop tier level_tier(L′) tools; tools never spawn on the map.
- Stagnation: min must strictly rise every 500 life-ticks or the life is
  banked (score divided) and gear wiped.

## Root causes of v2/v3 failure (each an architectural fix here)

1. Cell-classification (live/ghost maps) is unreliable → v4 tracks OBJECTS.
   Imprints with no track are residue and never enter decisions.
2. Five competing movement implementations → v4 has ONE safe-move filter that
   every consumer goes through.
3. Combat logic scattered across branches → v4 has ONE authority (the exact-
   model planner); the executive only chooses WHETHER/WHAT to engage, never HOW.
4. Recovery as a control-stealing branch starved the economy → v4 recovery is
   an intent modifier (planner weights + goal gating), not a branch.
5. Blind iteration on integrated behavior → v4 layers are built bottom-up,
   each with a falsifiable ground-truth test before the next builds on it.

## Layers & files (players/dancer/)

| Layer | File | Contract | Test (gate) |
|---|---|---|---|
| L1 WorldModel | world.py | obs+actions → entity tracks (frame coords, class, delta, hp, age), terrain/item views, occupancy, own-shift; teleport reset | tests/test_dancer_world.py: vs debug_nearby ground truth — recall ≥ 0.9 within cheb 4 (≤2-tick latency), precision ≥ 0.9, class accuracy 100% on matched |
| L2 DuelPlanner | planner.py | (our pos, target, terrain, occupancy, dmg params, intent) → action; depth-5 search with exact enemy response | tests/test_dancer_planner.py: vs MiniDuelSim (pure-python mirror of verified rules) — dance yields ≥ 1 hit / 4 ticks with 0 damage taken in open terrain; escape from unhittable melee takes 0 damage over 60 ticks; corner cases degrade gracefully |
| L3 BowSafety | safety.py | bow tracks → forbidden-cell predicate for moves (aligned ≤ 5 or one-step-to-aligned ≤ 5) | folded into world/planner tests + bench arrow-damage metric |
| L4 Executive | executive.py, policy.py | phases BOOTSTRAP→ECONOMY⇄FARM, RECOVER overlay, stagnation clock, equip policy, goal selection; all movement through safe filter; all combat through planner | tools/dancer_bench.py: deaths ≤ 2/1500 ticks, then score vs baseline seats; ship gate = ab_batch ≥ +0.3, t > 2, n ≥ 18 episodes |

## Executive policy (fixed spec)

- BOOTSTRAP (no tool): full-hp trade vs isolated d=0 melee (planner, intent
  kill, first-strike hold discovered by search). Goal: first kill → tool.
- ECONOMY (tool): goals = hilt-for-sword > prof-leveling resource (highest
  tier ≤ tool) > ore for missing/upgradable armor > herbs to 2. Defense-only
  engagements.
- FARM (sword, comb ≤ prof or comb-bound stagnation): dance-kill best isolated
  melee with our_dmg > 0 (prefer d ≥ 1 = guaranteed +1 comb); loot sweep after
  kills (tier-2 tools from L9+ kills are the economy jump).
- RECOVER overlay (hp < 60 until ≥ 90): planner weight W high, no offense, no
  goals; sit-regen when clear.
- Stagnation urgency (< 150 ticks to a predicted bank with no improvement):
  relax isolation one notch on the binding skill's activity.
- Equip: phase-appropriate held item with swap hysteresis (only out of combat,
  no threat within 3). Armor always. Never gems.

## Non-goals (v4.0)

Bow hunting (comb > 15), market buying, multi-agent coordination, map memory
beyond the track frame. All are post-ship levers.
