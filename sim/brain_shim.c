// cogame-nmmo wasm brain shim: the upstream pretrained MOBA policy served
// through the vendored pure-C puffernet, compiled to a WASI reactor module
// (build/moba_brain.wasm, built by sim/build_brain.sh).
//
// Mirrors vendored moba.c demo() inference EXACTLY:
//   - net:  make_puffernet(weights, batch, 510, 64, 5, {7,7,3,2,2,2}, 6)
//     (moba.c line 9; batch here is 1 per net instead of upstream's 5 —
//     see below)
//   - obs preprocessing: plain uint8 -> float cast, NO normalization
//     (moba.c lines 28-29: obs_f[i] = (float)env.observations[i])
//   - action extraction: forward_puffernet writes one float per action
//     column; the sim's decode does a C (int) cast, replicated here.
//
// Sampling (confirmed by reading vendored puffernet.h): the policy head is
// multidiscrete (logit_sizes are not all 1, so is_continuous == 0), and
// forward_puffernet -> softmax_multidiscrete SAMPLES stochastically from
// the softmax using C rand()/RAND_MAX (puffernet.h _softmax_multidiscrete)
// — it does NOT argmax. Upstream never calls srand, so its stream is its
// libc's default, identical to srand(1) in that libc. brain_init(seed)
// seeds THIS wasm module's rand stream (emscripten's musl libc): seed 1
// reproduces this module's srand(1) sequence, and any seed is
// deterministic because the RNG lives entirely inside the module. It
// matches a NATIVE upstream demo run only if that binary links the same
// libc — glibc and macOS rand() use different algorithms, so a
// native-libc reference trace WILL diverge in the sampled actions even
// with bit-exact logits. Such a divergence is a libc rand() difference,
// NOT a port bug (this matters when the baseline is the certification
// fixture). NOTE: all nets share the single libc rand stream, so hosts
// must issue brain_forward calls in a deterministic order for
// reproducibility.
//
// Per-agent recurrence: upstream runs one net with batch 5 (MinGRU state
// is per-batch-row, so its 5 heroes already have isolated state). Here we
// build BRAIN_AGENTS independent batch-1 nets — one per possible agent
// index — so a host can serve any subset of heroes with per-hero MinGRU
// state isolation. Per-row math in puffernet is batch-independent (linear
// layers and MinGRU treat rows independently), so a batch-1 forward is
// numerically identical to one row of a batch-5 forward. All nets share
// one read-only weight buffer (Weights.idx is rewound between
// make_puffernet calls); each net owns its state/output buffers
// (95,616 f32 params; 10 nets of buffers ~= 4 MB total).

#include <stdlib.h>
#include <string.h>
#include "puffernet.h"

#define BRAIN_AGENTS 10
#define BRAIN_OBS_SIZE 510
#define BRAIN_HIDDEN 64
#define BRAIN_GRU_LAYERS 5
#define BRAIN_NUM_ACTIONS 6

// Generated at build time by sim/build_brain.sh (xxd -i over the vendored
// moba_weights.bin); never committed.
extern const unsigned char moba_weights_bin[];
extern const unsigned int moba_weights_bin_len;

static PufferNet* nets[BRAIN_AGENTS];
static unsigned char obs_u8[BRAIN_AGENTS][BRAIN_OBS_SIZE];
static int act_i32[BRAIN_AGENTS][BRAIN_NUM_ACTIONS];

// brain_init(seed): build the nets. Returns the weight param count
// (95,616) so the host can sanity-check the embedded blob, or -1 on
// failure. Call exactly once per instance, before any forward: a second
// call would leak the first nets and reset recurrent state, so it is
// rejected. (The allocation NULL checks below are unreachable under
// -sABORTING_MALLOC=1, which traps on OOM; kept as belt-and-braces.)
__attribute__((export_name("brain_init")))
int brain_init(unsigned int seed) {
    if (nets[0] != NULL)
        return -1;  // already initialized
    srand(seed);  // seeds THIS module's musl rand stream (see header note)

    // Replicate load_weights() (puffernet.h) minus the FILE* I/O: same
    // +7-float over-allocation so get_weights_aligned never reads past
    // the buffer, same size bookkeeping.
    size_t num_weights = moba_weights_bin_len / sizeof(float);
    Weights* weights = (Weights*)calloc(
        1, sizeof(Weights) + (num_weights + 7) * sizeof(float));
    if (weights == NULL)
        return -1;
    weights->data = (float*)(weights + 1);
    memcpy(weights->data, moba_weights_bin, num_weights * sizeof(float));
    weights->size = num_weights + 7;

    int logit_sizes[BRAIN_NUM_ACTIONS] = {7, 7, 3, 2, 2, 2};  // moba.c:8
    for (int i = 0; i < BRAIN_AGENTS; i++) {
        weights->idx = 0;  // nets share the weight buffer, own their state
        nets[i] = make_puffernet(weights, 1, BRAIN_OBS_SIZE, BRAIN_HIDDEN,
                                 BRAIN_GRU_LAYERS, logit_sizes,
                                 BRAIN_NUM_ACTIONS);
        if (nets[i] == NULL)
            return -1;
    }
    return (int)num_weights;
}

// 510 uint8 obs in for one agent index.
__attribute__((export_name("brain_obs_ptr")))
unsigned char* brain_obs_ptr(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= BRAIN_AGENTS)
        return NULL;
    return obs_u8[agent_idx];
}

// 6 int32 actions out for one agent index.
__attribute__((export_name("brain_act_ptr")))
int* brain_act_ptr(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= BRAIN_AGENTS)
        return NULL;
    return act_i32[agent_idx];
}

// One inference step for one agent: obs_u8[agent] -> act_i32[agent].
// Advances that agent's MinGRU state and the shared rand() stream.
// Returns 0 on success, -1 on bad agent_idx or missing brain_init.
__attribute__((export_name("brain_forward")))
int brain_forward(int agent_idx) {
    if (agent_idx < 0 || agent_idx >= BRAIN_AGENTS)
        return -1;
    if (nets[agent_idx] == NULL)
        return -1;

    // moba.c:28-29 preprocessing: plain uint8 -> float cast.
    float obs_f[BRAIN_OBS_SIZE];
    for (int i = 0; i < BRAIN_OBS_SIZE; i++)
        obs_f[i] = (float)obs_u8[agent_idx][i];

    float act_f[BRAIN_NUM_ACTIONS];
    forward_puffernet(nets[agent_idx], obs_f, act_f);

    // The sim's action decode casts each float with (int); sampled values
    // are exact small non-negative integers, so this is lossless.
    for (int a = 0; a < BRAIN_NUM_ACTIONS; a++)
        act_i32[agent_idx][a] = (int)act_f[a];
    return 0;
}
