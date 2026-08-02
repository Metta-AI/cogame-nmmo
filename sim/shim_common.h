// Shared NMMO3 env configuration used by every host shim (sim/shim.c and,
// from Phase N4, sim/viewer_main.c). One definition so the viewer's
// re-simulation can never drift from the server sim's init path.
//
// Values are the trained-on env defaults, identical in upstream
// config/nmmo3.ini [env] and the nmmo3.c demo struct (nmmo3.c:158-179).
// Do not change: policies were trained on these. The obs byte layout
// hardcodes 11x15x10 window strides (compute_all_obs, nmmo3.h:943), so
// x_window/y_window especially are not tunable.
#ifndef COGAME_SHIM_COMMON_H
#define COGAME_SHIM_COMMON_H

#include "nmmo3.h"  // vendored sim (render half compiled out, patch 0001)

// Per-agent observation size in bytes: 11x15 egocentric tile window x 10
// bytes/tile + 47 self scalars + 10 reward bytes (only 9 ever written;
// byte 1706 stays 0 forever). Single allocation site: nmmo3.h:797.
#define NMMO_OBS_SIZE (11*15*10 + 47 + 10)

static inline void nmmo_configure(MMO* env, unsigned int seed,
                                  int num_agents) {
    env->num_agents = num_agents;        // elastic: ini 1024, demo runs 1; we default 8 (server layer)
    env->width = 512;                    // nmmo3.ini [env] width / nmmo3.c:161
    env->height = 512;                   // nmmo3.ini [env] height / nmmo3.c:160
    env->num_enemies = 2048;             // nmmo3.ini [env] num_enemies / nmmo3.c:163
    env->num_resources = 2048;           // nmmo3.ini [env] num_resources / nmmo3.c:164
    env->num_weapons = 1024;             // nmmo3.ini [env] num_weapons / nmmo3.c:165
    env->num_gems = 512;                 // nmmo3.ini [env] num_gems / nmmo3.c:166
    env->tiers = 5;                      // nmmo3.ini [env] tiers / nmmo3.c:167
    env->levels = 40;                    // nmmo3.ini [env] levels / nmmo3.c:168
    env->teleportitis_prob = 0.001f;     // nmmo3.ini [env] teleportitis_prob / nmmo3.c:169
    env->enemy_respawn_ticks = 2;        // nmmo3.ini [env] enemy_respawn_ticks / nmmo3.c:170
    env->item_respawn_ticks = 100;       // nmmo3.ini [env] item_respawn_ticks / nmmo3.c:171
    env->x_window = 7;                   // nmmo3.ini [env] x_window / nmmo3.c:172 (obs-layout hardcoded)
    env->y_window = 5;                   // nmmo3.ini [env] y_window / nmmo3.c:173 (obs-layout hardcoded)
    env->reward_combat_level = 1.0f;     // nmmo3.ini [env] reward_combat_level / nmmo3.c:174
    env->reward_prof_level = 1.0f;       // nmmo3.ini [env] reward_prof_level / nmmo3.c:175
    env->reward_item_level = 1.0f;       // nmmo3.ini [env] reward_item_level / nmmo3.c:176
    env->reward_market = 0.0f;           // nmmo3.ini [env] reward_market / nmmo3.c:177
    env->reward_death = -1.0f;           // nmmo3.ini [env] reward_death / nmmo3.c:178
    // Seeding: the sim draws ALL randomness via rand_r(&env->rng)
    // (nmmo3.h:717 field; consumed re-entrantly everywhere) and never
    // seeds the field itself — writing it here IS the seeding, no
    // srand patch needed (see vendor/PATCHES.md).
    env->rng = seed;
}

// Cheap state digest for replay certification: FNV-1a (32-bit) over each
// player's (r, c, hp, comb_lvl, prof_lvl) int bits, then env->rng and
// env->tick. Pure read of entity/env state — never touches the obs/reward
// path. Shared so the server host (sim/shim.c) and the Phase-N4 viewer
// core can never diverge; a recorded episode's live digest must equal the
// viewer's re-sim digest at the same tick.
static inline unsigned int nmmo_fnv1a_u32(unsigned int h, unsigned int v) {
    for (unsigned int i = 0; i < 4; i++) {
        h ^= (v >> (8 * i)) & 0xFFu;
        h *= 16777619u;
    }
    return h;
}

static inline unsigned int nmmo_state_digest(const MMO* env) {
    unsigned int h = 2166136261u;  // FNV-1a offset basis
    for (int pid = 0; pid < env->num_agents; pid++) {
        const Entity* e = &env->players[pid];
        h = nmmo_fnv1a_u32(h, (unsigned int)e->r);
        h = nmmo_fnv1a_u32(h, (unsigned int)e->c);
        h = nmmo_fnv1a_u32(h, (unsigned int)e->hp);
        h = nmmo_fnv1a_u32(h, (unsigned int)e->comb_lvl);
        h = nmmo_fnv1a_u32(h, (unsigned int)e->prof_lvl);
    }
    h = nmmo_fnv1a_u32(h, env->rng);
    h = nmmo_fnv1a_u32(h, (unsigned int)env->tick);
    return h;
}

#endif  // COGAME_SHIM_COMMON_H
