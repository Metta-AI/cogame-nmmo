// cogame-nmmo wasm shim around the vendored Puffer NMMO3 sim.
//
// Compiled twice by sim/build_sim.sh:
//   - against build/src-patched  -> build/nmmo3_sim.wasm           (production)
//   - against build/src-pristine -> build/nmmo3_sim_pristine.wasm  (-DPRISTINE,
//     fidelity-test reference; that tree lacks the patch-0002 fault flag, so
//     its reference is guarded out here — nothing else differs between trees)
//
// NMMO3_RENDER is never defined here: the renderer half of nmmo3.h is
// guarded out by patch 0001 and this shim links no raylib.

#include <stdlib.h>
#include <string.h>
#include "shim_common.h"  // nmmo_configure(): trained env values (+ nmmo3.h)

static MMO env;

// ---- per-agent scoring shadow (host-side bookkeeping, never fed back into
// the sim) --------------------------------------------------------------
//
// Upstream aggregates all agents into ONE env-wide Log (env->log,
// add_player_log nmmo3.h:733-765): per ended life it adds
// min(comb_lvl, prof_lvl) to log->min_comb_prof and sets
// log->score = log->min_comb_prof. Per-seat ranking needs that same
// quantity attributed per agent, so the shim tracks it here:
//
//   cum_min_comb_prof[pid]  sum of min(comb_lvl, prof_lvl) over pid's ENDED
//                           lives — the exact value add_player_log recorded
//   deaths[pid]             number of ended lives (terminal flags seen)
//   last_alive_min[pid]     min(comb_lvl, prof_lvl) at the end of the last
//                           tick pid was alive
//
// Attribution is exact, by cause (both causes set terminals[pid]=1.0f in
// add_player_log, nmmo3.h:762-764):
//  - Attack death (attack(), nmmo3.h:1408-1416): the entity keeps hp==0 and
//    its death-time levels until the respawn branch runs on a LATER tick
//    (c_step nmmo3.h:1891-1903), so reading min(comb,prof) right after
//    c_step equals the value add_player_log recorded.
//  - Stagnation reset (c_step nmmo3.h:1926-1956): spawn() runs the same
//    tick and resets levels/hp, so post-step state is gone — but levels
//    never decrease within a life, and stagnation means min(comb,prof) was
//    flat for 500 ticks, so the recorded value equals last tick's
//    last_alive_min[pid]. (Distinguished from attack death by hp: the
//    stagnation respawn leaves hp==99, attack death leaves hp==0.)
//  - Both-in-one-tick is impossible: the stagnation respawn lands on a
//    safe_tile with no entity within Chebyshev distance 5 (spawn ->
//    safe_tile(env,5), nmmo3.h:1061), and enemies later in the same tick
//    move at most 1 tile with attack reach <= 4 — they cannot reach it.
static int* cum_min_comb_prof;
static int* deaths;
static int* last_alive_min;

static int min_comb_prof_now(int pid) {
    const Entity* e = &env.players[pid];
    return (e->prof_lvl < e->comb_lvl) ? e->prof_lvl : e->comb_lvl;
}

static void scoring_reset(void) {
    for (int pid = 0; pid < env.num_agents; pid++) {
        cum_min_comb_prof[pid] = 0;
        deaths[pid] = 0;
        // c_reset leaves everyone alive at comb=prof=1 (nmmo3.h:1783-1796)
        last_alive_min[pid] = min_comb_prof_now(pid);
    }
}

__attribute__((export_name("nmmo_init")))
void nmmo_init(unsigned int seed, int num_agents) {
    nmmo_configure(&env, seed, num_agents);  // includes env.rng = seed
    // allocate_mmo() (nmmo3.h:795-802) callocs observations
    // (num_agents * 1707 B), rewards, terminals, actions, then init()
    // allocates the world. The obs buffer is allocated ONCE and never
    // cleared afterwards — stale entity bytes between ticks are a
    // trained-on quirk (see the design doc); never re-zero it.
    allocate_mmo(&env);
    c_reset(&env);
    cum_min_comb_prof = calloc(num_agents, sizeof(int));
    deaths = calloc(num_agents, sizeof(int));
    last_alive_min = calloc(num_agents, sizeof(int));
    scoring_reset();
}

__attribute__((export_name("nmmo_step")))
void nmmo_step(void) {
    // Upstream's env code only ever SETS terminals (add_player_log,
    // nmmo3.h:762-764) and never clears them — the training vecenv memsets
    // the buffer externally each step. Reproduce that here so the buffer
    // holds exactly THIS tick's per-agent done flags after c_step.
    memset(env.terminals, 0, env.num_agents * sizeof(float));
    c_step(&env);
    for (int pid = 0; pid < env.num_agents; pid++) {
        if (env.terminals[pid] != 0.0f) {
            deaths[pid] += 1;
            cum_min_comb_prof[pid] += (env.players[pid].hp == 0)
                ? min_comb_prof_now(pid)   // attack death: levels intact
                : last_alive_min[pid];     // stagnation: respawned in-tick
        }
    }
    for (int pid = 0; pid < env.num_agents; pid++) {
        if (env.players[pid].hp > 0) {
            last_alive_min[pid] = min_comb_prof_now(pid);
        }
    }
}

__attribute__((export_name("nmmo_reset")))
void nmmo_reset(void) {
#ifndef PRISTINE
    nmmo_fault_code = 0;  // patch 0002 fault flag is host-cleared
#endif
    // env.rng is NOT rewritten: c_reset consumes the current stream, so a
    // second reset produces a fresh world. The server creates one sim
    // instance per episode and never calls this; it exists for parity with
    // upstream's reusable-env shape.
    c_reset(&env);
    scoring_reset();
}

// Patch-0002 fault flag: nonzero when an upstream in-episode debug guard
// tripped (site codes documented at nmmo_fault_code in the patched
// nmmo3.h and vendor/PATCHES.md). The host polls this each tick and ends
// the episode with end_reason "sim_fault" instead of losing the process
// to exit(). The pristine build keeps upstream's exit() calls (no flag).
__attribute__((export_name("nmmo_fault")))
int nmmo_fault(void) {
#ifndef PRISTINE
    return nmmo_fault_code;
#else
    return 0;
#endif
}

__attribute__((export_name("obs_ptr")))
unsigned char* obs_ptr(void) { return env.observations; }

__attribute__((export_name("act_ptr")))
float* act_ptr(void) { return env.actions; }

__attribute__((export_name("rew_ptr")))
float* rew_ptr(void) { return env.rewards; }

// This tick's per-agent done flags (num_agents floats, 1.0 = this agent's
// life ended this tick and it respawned / is respawning in place). Valid
// after nmmo_step(); see the memset there.
__attribute__((export_name("term_ptr")))
float* term_ptr(void) { return env.terminals; }

__attribute__((export_name("nmmo_tick")))
int nmmo_tick(void) { return env.tick; }

// Per-agent scoring stats. `which` codes:
//   0 cum_min_comb_prof   sum of min(comb,prof) over ended lives
//                         (== upstream log.score contribution, nmmo3.h:744,758)
//   1 deaths              ended lives (attack deaths + stagnation resets)
//   2 comb_lvl            current life's combat level (Entity, nmmo3.h:527)
//   3 prof_lvl            current life's profession level (Entity, nmmo3.h:533)
//   4 current-life min(comb,prof) contribution: 0 while dead-awaiting-
//     respawn (hp==0 — that life is already in cum_min_comb_prof)
//   5 gold                (Entity, nmmo3.h:537)
//   6 time_alive          ticks since this life spawned (Entity, nmmo3.h:553)
//   7 hp                  (Entity, nmmo3.h:531)
__attribute__((export_name("agent_stat")))
int agent_stat(int pid, int which) {
    if (pid < 0 || pid >= env.num_agents)
        return 0;
    const Entity* e = &env.players[pid];
    switch (which) {
        case 0:  return cum_min_comb_prof[pid];
        case 1:  return deaths[pid];
        case 2:  return e->comb_lvl;
        case 3:  return e->prof_lvl;
        case 4:  return (e->hp == 0) ? 0 : min_comb_prof_now(pid);
        case 5:  return e->gold;
        case 6:  return e->time_alive;
        case 7:  return e->hp;
        default: return 0;
    }
}

// The design-doc ranking score: cumulative min(comb,prof) over ended lives
// plus the current (unfinished) life's min(comb,prof). A dead-awaiting-
// respawn agent contributes 0 for the current life — its ended life is
// already counted in the cumulative part.
__attribute__((export_name("nmmo_score")))
int nmmo_score(int pid) {
    if (pid < 0 || pid >= env.num_agents)
        return 0;
    return agent_stat(pid, 0) + agent_stat(pid, 4);
}

// State digest (FNV-1a over player r/c/hp/levels + env rng + tick; see
// sim/shim_common.h). Recorded episodes compare this against the Phase-N4
// viewer core's digest at the same tick.
__attribute__((export_name("state_digest")))
unsigned int state_digest(void) {
    return nmmo_state_digest(&env);
}
