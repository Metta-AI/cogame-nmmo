// cogame-nmmo wasm shim around the vendored Puffer MOBA sim.
//
// Compiled twice by sim/build_sim.sh:
//   - against build/src-patched  -> build/moba_sim.wasm           (production)
//   - against build/src-pristine -> build/moba_sim_pristine.wasm  (-DPRISTINE,
//     fidelity-test reference; that tree lacks the seed/done/winner fields
//     added by patches 0002/0003, so those references are guarded out here)
//
// MOBA_RENDER is never defined here: the renderer half of moba.h is guarded
// out by patch 0001 and this shim links no raylib.

#include <stdlib.h>
#include "shim_common.h"  // moba_configure(): shared env defaults (+ moba.h)

static MOBA env;

__attribute__((export_name("moba_init")))
void moba_init(unsigned int seed, int num_agents) {
    moba_configure(&env, seed, num_agents);
    // allocate_moba() allocates obs/actions/rewards/terminals/truncations,
    // ai_path_buffer, and the 256 MB ai_paths cache, then calls init_moba().
    allocate_moba(&env);
    c_reset(&env);
}

__attribute__((export_name("moba_step")))
void moba_step(void) { c_step(&env); }

__attribute__((export_name("moba_reset")))
void moba_reset(void) {
#ifndef PRISTINE
    env.done = 0;   // patch 0003 fields are cleared here, not in c_reset
    env.winner = 0;
    moba_fault_code = 0;  // patch 0004 fault flag, likewise host-cleared
#endif
    c_reset(&env);
}

// Patch-0004 fault flag: nonzero when an upstream in-episode debug guard
// tripped (site codes documented at moba_fault_code in the patched
// moba.h). The host polls this each tick and ends the episode with
// end_reason "sim_fault" instead of losing the process to exit().
// The pristine build keeps upstream's exit() calls (no flag to read).
__attribute__((export_name("moba_fault")))
int moba_fault(void) {
#ifndef PRISTINE
    return moba_fault_code;
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

__attribute__((export_name("moba_done")))
int moba_done(void) {
#ifndef PRISTINE
    return env.done;
#else
    return 0;       // pristine sim auto-resets internally and never reports
#endif
}

__attribute__((export_name("moba_winner")))
int moba_winner(void) {
#ifndef PRISTINE
    return env.winner;
#else
    return 0;
#endif
}

__attribute__((export_name("moba_tick")))
int moba_tick(void) { return env.tick; }

// Per-agent scoring stats. `which` codes (see Entity / PlayerLog in moba.h):
//   0 level            1 kills            2 deaths
//   3 towers_killed    4 creeps_killed    5 neutrals_killed
//   6 xp               7 damage_dealt     8 damage_received
//   9 healing_dealt   10 healing_received
__attribute__((export_name("agent_stat")))
int agent_stat(int pid, int which) {
    if (pid < 0 || pid >= NUM_PLAYERS)
        return 0;
    Entity* e = &env.entities[pid];
    PlayerLog* pl = &env.player_logs[pid];
    switch (which) {
        case 0:  return e->level;
        case 1:  return (int)pl->kills;
        case 2:  return (int)pl->deaths;
        case 3:  return (int)pl->towers_killed;
        case 4:  return (int)pl->creeps_killed;
        case 5:  return (int)pl->neutrals_killed;
        case 6:  return e->xp;
        case 7:  return (int)pl->damage_dealt;
        case 8:  return (int)pl->damage_received;
        case 9:  return (int)pl->healing_dealt;
        case 10: return (int)pl->healing_received;
        default: return 0;
    }
}

// Final-state digest (FNV-1a over hero x/y/health + ancient healths; see
// sim/shim_common.h). Recorded episodes compare this against the viewer
// core's viewer_state_digest() at the same tick.
__attribute__((export_name("state_digest")))
unsigned int state_digest(void) {
    return moba_state_digest(&env);
}

// Ancient health, for draw tiebreaks on tick-cap. team 0 = radiant
// (entity idx TOWER_OFFSET+23, pid 205), team 1 = dire (TOWER_OFFSET+22,
// pid 204) — matching c_step's radiant_pid/dire_pid.
__attribute__((export_name("ancient_health")))
float ancient_health(int team) {
    int idx = (team == 0) ? TOWER_OFFSET + 23 : TOWER_OFFSET + 22;
    Entity* ancient = &env.entities[idx];
    // kill_entity() zeroes health and sets pid = -1 together, so checking
    // pid == -1 alone suffices for "dead"; the explicit 0.0 return just
    // guards against any future entity-slot reuse leaving stale health.
    if (ancient->pid == -1)
        return 0.0f;
    return ancient->health;
}
