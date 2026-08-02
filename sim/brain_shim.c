// cogame-nmmo wasm brain shim: the upstream pretrained NMMO3 policy
// (MMONet) compiled to a STANDALONE_WASM reactor, weights embedded.
//
// The net is ported from the vendored demo (vendor/upstream/nmmo3.c:11-152)
// VERBATIM in structure: same puffernet.h primitive calls in the same
// order, because the flat weight blob (nmmo3_weights.bin) is consumed
// sequentially by the make_* constructors — any reorder silently
// misassigns every later tensor. Construction (init_mmonet, nmmo3.c:30-50)
// and its weight consumption, in order:
//
//   map_conv1    make_conv2d(w, N, 15, 11, 59, 128, 5, 3)   188,800 + 128
//   map_conv2    make_conv2d(w, N,  4,  3, 128, 128, 3, 1)  147,456 + 128
//   player_embed make_embedding(w, N*47, 128, 32)             4,096
//   proj         make_affine(w, N, 1817, 512)               930,304 + 512
//   decoder      make_linear(w, N, 512, 26+1)                13,824 (aligned)
//   mingru       make_mingru(w, N, 512, 4)               4 x 786,432 (aligned)
//                                                    total  4,430,976 floats
//
// (make_linear/make_mingru read via get_weights_aligned — 8-float
// alignment — but every preceding tensor count lands 8-aligned already,
// so no padding is consumed and the total equals the file size exactly:
// 4,430,976 x 4 B = 17,723,904 B.)
//
// forward() (nmmo3.c:70-152): 59-channel one-hot of the 11x15x10 tile
// window (factors {4,4,17,5,3,5,5,5,7,4}) -> conv1 -> relu -> conv2;
// the 47 self scalars go through BOTH a byte embedding (as discrete ids)
// and raw float passthrough; concat [conv 256 | embed 47*32 | scalars 47 |
// reward 10] = 1817 -> affine 512 -> relu -> 4-layer MinGRU(512) ->
// 27-logit decoder (26 action logits + fused value head).
//
// Sampling (confirmed by reading vendored puffernet.h + nmmo3.c:151): the
// demo calls softmax_multidiscrete, which SAMPLES stochastically from the
// softmax using C rand()/RAND_MAX (_softmax_multidiscrete, puffernet.h)
// — it does NOT argmax. Upstream never calls srand, so its stream is its
// libc's default, identical to srand(1) in that libc. brain_init(seed, n)
// seeds THIS wasm module's rand stream (emscripten's musl libc): any seed
// is deterministic because the RNG lives entirely inside the module, and
// it matches a NATIVE upstream demo run only if that binary links the
// same libc — glibc and macOS rand() use different algorithms, so a
// native-libc reference trace WILL diverge in the sampled actions even
// with bit-exact logits (a libc difference, NOT a port bug). All nets
// share the single libc rand stream, so hosts must issue brain_forward
// calls in a deterministic order for reproducibility.
//
// Per-agent recurrence: upstream runs ONE net at batch num_agents (MinGRU
// state is per-batch-row, so its agents already have isolated state) and
// the demo runs num_agents=1. Here we build `num_agents` independent
// batch-1 nets — one per agent index — so a host can reset one agent's
// state without touching the others. Per-row math in puffernet is
// batch-independent (conv2d, embedding, linear and MinGRU all treat rows
// independently), so a batch-1 forward is numerically identical to one
// row of a batch-N forward. All nets share one read-only weight buffer
// (Weights.idx is rewound between init_mmonet calls); each net owns its
// activation/state buffers.
//
// Reset semantics (nmmo3.c:71-78): when an env terminal fires for batch
// row b, the demo's forward() zeroes ONLY that row's MinGRU state — for
// every layer l, the hidden_size floats at
//   mingru->state + l*batch*hidden + b*hidden
// — before running the net on the first obs of the new life. Nothing
// else in MMONet is recurrent. brain_reset_state(i) reproduces exactly
// that for net i (batch 1 -> the whole state buffer, all 4 layers). The
// host must call it when the wire `resets[j]` flag is true, BEFORE that
// tick's brain_forward.
//
// Memory (documented for build_brain.sh's INITIAL_MEMORY): the embedded
// weight array is ~17.7 MB of .data; brain_init heap-copies it once into
// a Weights struct (+17.7 MB) and each batch-1 net owns ~188 KB of
// activations (ob_map 39 KB + conv1 buffer 84.5 KB — make_conv2d
// over-allocates at in_height*in_width — + conv2/embed/proj/relu/mingru
// buffers ~64 KB), so 8 nets ~= 1.5 MB. Total ~= 37 MB + emscripten
// runtime; INITIAL_MEMORY=64mb avoids growth churn, ALLOW_GROWTH +
// MAXIMUM_MEMORY=1gb + ABORTING_MALLOC for flag parity with the sim.

#include <stdlib.h>
#include <string.h>
#include "puffernet.h"

#define BRAIN_MAX_AGENTS 32
#define BRAIN_OBS_SIZE (11*15*10 + 47 + 10)   // 1707, sim obs contract
#define BRAIN_HIDDEN 512
#define BRAIN_GRU_LAYERS 4
#define BRAIN_NUM_ACTIONS 26
#define EXPECTED_PARAMS 4430976               // nmmo3_weights.bin / 4

// Generated at build time by sim/build_brain.sh (xxd -i over the vendored
// nmmo3_weights.bin); never committed.
extern const unsigned char nmmo3_weights_bin[];
extern const unsigned int nmmo3_weights_bin_len;

// MMONet, ported from vendor/upstream/nmmo3.c:11-28 (minus the raw
// num_agents obs staging pointers the demo reads from the env struct —
// the host writes obs into obs_u8 below instead).
typedef struct MMONet MMONet;
struct MMONet {
    int num_agents;
    float* ob_map;
    int* ob_player_discrete;
    float* ob_player_continuous;
    float* ob_reward;
    Conv2D* map_conv1;
    ReLU* map_relu;
    Conv2D* map_conv2;
    Embedding* player_embed;
    float* proj_buffer;
    Affine* proj;
    ReLU* proj_relu;
    Linear* decoder;
    MinGRU* mingru;
    Multidiscrete* multidiscrete;
};

// init_mmonet, nmmo3.c:30-50 — verbatim (the weight-consumption order
// documented in the header comment lives HERE; do not reorder).
static MMONet* init_mmonet(Weights* weights, int num_agents) {
    MMONet* net = calloc(1, sizeof(MMONet));
    int hidden = BRAIN_HIDDEN;
    net->num_agents = num_agents;
    net->ob_map = calloc(num_agents*11*15*59, sizeof(float));
    net->ob_player_discrete = calloc(num_agents*47, sizeof(int));
    net->ob_player_continuous = calloc(num_agents*47, sizeof(float));
    net->ob_reward = calloc(num_agents*10, sizeof(float));
    net->map_conv1 = make_conv2d(weights, num_agents, 15, 11, 59, 128, 5, 3);
    net->map_relu = make_relu(num_agents, 128*3*4);
    net->map_conv2 = make_conv2d(weights, num_agents, 4, 3, 128, 128, 3, 1);
    net->player_embed = make_embedding(weights, num_agents*47, 128, 32);
    net->proj_buffer = calloc(num_agents*1817, sizeof(float));
    net->proj = make_affine(weights, num_agents, 1817, hidden);
    net->proj_relu = make_relu(num_agents, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, 26 + 1);
    net->mingru = make_mingru(weights, num_agents, hidden, 4);
    int logit_sizes[1] = {26};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

// forward, nmmo3.c:70-152 — verbatim EXCEPT: (a) the terminals-driven
// state-zeroing loop (nmmo3.c:71-78) is factored out into
// brain_reset_state (the host drives it from the wire `resets` flags
// instead of an in-process terminals buffer); (b) obs bytes feeding
// array indices (one-hot channel, embedding row) are clamped to their
// table sizes — sim-produced obs are always in range (every factor bound
// was verified against compute_all_obs, nmmo3.h:926-1020), so for
// contract-valid obs the clamp never fires and the output is identical;
// it only stops a malicious/corrupt obs blob from writing outside the
// one-hot buffer or reading outside the embedding table in-wasm.
static void mmonet_forward(MMONet* net, unsigned char* observations, float* actions) {
    memset(net->ob_map, 0, net->num_agents*11*15*59*sizeof(float));

    // CNN subnetwork
    int factors[10] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    float (*ob_map)[59][11][15] = (float (*)[59][11][15])net->ob_map;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*(11*15*10 + 47 + 10);
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 15; j++) {
                int f_offset = 0;
                for (int f = 0; f < 10; f++) {
                    int ob = observations[b_offset + i*15*10 + j*10 + f];
                    if (ob >= factors[f]) {
                        ob = factors[f] - 1;  // unreachable for sim obs
                    }
                    int obs_idx = f_offset + ob;
                    ob_map[b][obs_idx][i][j] = 1;
                    f_offset += factors[f];
                }
            }
        }
    }
    conv2d(net->map_conv1, net->ob_map);
    relu(net->map_relu, net->map_conv1->output);
    conv2d(net->map_conv2, net->map_relu->output);

    // Player embedding subnetwork
    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < 47; i++) {
            unsigned char ob = observations[b*(11*15*10 + 47 + 10) + 11*15*10 + i];
            net->ob_player_discrete[b*47 + i] = (ob < 128) ? ob : 127;
            net->ob_player_continuous[b*47 + i] = ob;
        }
    }
    embedding(net->player_embed, net->ob_player_discrete);

    // Rewards
    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < 10; i++) {
            net->ob_reward[b*10 + i] = observations[b*(11*15*10 + 47 + 10) + 11*15*10 + 47 + i];
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b*1817;
        for (int i = 0; i < 256; i++) {
            net->proj_buffer[b_offset + i] = net->map_conv2->output[b*256 + i];
        }

        b_offset += 256;
        for (int i = 0; i < 47*32; i++) {
            net->proj_buffer[b_offset + i] = net->player_embed->output[b*47*32 + i];
        }

        b_offset += 47*32;
        for (int i = 0; i < 47; i++) {
            net->proj_buffer[b_offset + i] = net->ob_player_continuous[b*47 + i];
        }

        b_offset += 47;
        for (int i = 0; i < 10; i++) {
            net->proj_buffer[b_offset + i] = net->ob_reward[b*10 + i];
        }
    }

    affine(net->proj, net->proj_buffer);
    relu(net->proj_relu, net->proj->output);

    mingru(net->mingru, net->proj_relu->output);
    linear(net->decoder, net->mingru->output);

    softmax_multidiscrete(net->multidiscrete, net->decoder->output, actions);
}

static MMONet* nets[BRAIN_MAX_AGENTS];
static int brain_num_agents = 0;
static unsigned char obs_u8[BRAIN_MAX_AGENTS][BRAIN_OBS_SIZE];
static int act_i32[BRAIN_MAX_AGENTS][1];

// brain_init(seed, num_agents) failure codes — the host maps each to an
// honest message (players/baseline_player.py INIT_ERRORS):
#define BRAIN_ERR_REINIT      (-1)  // already initialized in this instance
#define BRAIN_ERR_NUM_AGENTS  (-2)  // num_agents outside 1..BRAIN_MAX_AGENTS
#define BRAIN_ERR_WEIGHTS_LEN (-3)  // embedded blob is not nmmo3_weights.bin
#define BRAIN_ERR_ALLOC       (-4)  // allocation failure

// brain_init(seed, num_agents): build num_agents independent batch-1
// nets. Returns the weight param count (4,430,976) so the host can
// sanity-check the embedded blob, or a BRAIN_ERR_* code (< 0). Call
// exactly once per instance, before any forward: a second call would
// leak the first nets and reset recurrent state, so it is rejected.
// (The allocation NULL checks are unreachable under -sABORTING_MALLOC=1,
// which traps on OOM; kept as belt-and-braces.)
__attribute__((export_name("brain_init")))
int brain_init(unsigned int seed, int num_agents) {
    if (brain_num_agents != 0)
        return BRAIN_ERR_REINIT;
    if (num_agents < 1 || num_agents > BRAIN_MAX_AGENTS)
        return BRAIN_ERR_NUM_AGENTS;
    if (nmmo3_weights_bin_len != EXPECTED_PARAMS * sizeof(float))
        return BRAIN_ERR_WEIGHTS_LEN;
    srand(seed);  // seeds THIS module's musl rand stream (see header note)

    // Replicate load_weights() (puffernet.h:39-63) minus the FILE* I/O:
    // same +7-float over-allocation so get_weights_aligned never reads
    // past the buffer, same size bookkeeping.
    size_t num_weights = nmmo3_weights_bin_len / sizeof(float);
    Weights* weights = (Weights*)calloc(
        1, sizeof(Weights) + (num_weights + 7) * sizeof(float));
    if (weights == NULL)
        return BRAIN_ERR_ALLOC;
    weights->data = (float*)(weights + 1);
    memcpy(weights->data, nmmo3_weights_bin, num_weights * sizeof(float));
    weights->size = num_weights + 7;

    for (int i = 0; i < num_agents; i++) {
        weights->idx = 0;  // nets share the weight buffer, own their state
        nets[i] = init_mmonet(weights, 1);
        if (nets[i] == NULL)
            return BRAIN_ERR_ALLOC;
    }
    brain_num_agents = num_agents;
    return (int)num_weights;
}

// 1707 uint8 obs in for one agent index (validated against the built
// net count, same as brain_forward/brain_reset_state).
__attribute__((export_name("brain_obs_ptr")))
unsigned char* brain_obs_ptr(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= brain_num_agents)
        return NULL;
    return obs_u8[agent_idx];
}

// 1 int32 action out for one agent index (validated likewise).
__attribute__((export_name("brain_act_ptr")))
int* brain_act_ptr(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= brain_num_agents)
        return NULL;
    return act_i32[agent_idx];
}

// Zero one agent's recurrent state (all MinGRU layers), exactly what the
// demo's forward() does for a terminal batch row (nmmo3.c:71-78; see the
// header comment). Call when the wire `resets` flag fires for this
// agent, BEFORE forwarding that tick's obs. Returns 0 on success, -1 on
// bad agent_idx or missing brain_init.
__attribute__((export_name("brain_reset_state")))
int brain_reset_state(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= brain_num_agents)
        return -1;
    MinGRU* g = nets[agent_idx]->mingru;
    // batch_size == 1, so row 0 of every layer == the whole state buffer
    for (int l = 0; l < g->num_layers; l++) {
        memset(g->state + l * g->batch_size * g->hidden_size,
               0, g->hidden_size * sizeof(float));
    }
    return 0;
}

// One inference step for one agent: obs_u8[agent] -> act_i32[agent][0].
// Advances that agent's MinGRU state and the shared rand() stream.
// Returns 0 on success, -1 on bad agent_idx or missing brain_init.
__attribute__((export_name("brain_forward")))
int brain_forward(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= brain_num_agents)
        return -1;

    float act_f[1];
    mmonet_forward(nets[agent_idx], obs_u8[agent_idx], act_f);

    // The sim's action decode casts each float with (int); sampled values
    // are exact small non-negative integers, so this is lossless.
    act_i32[agent_idx][0] = (int)act_f[0];
    return 0;
}
