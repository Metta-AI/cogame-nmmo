// cogame-nmmo replay viewer: re-simulates a recorded episode in the
// browser and renders it with the upstream raylib client.
//
// Built twice by sim/build_viewer.sh, always against build/src-patched:
//   - with -DNMMO3_RENDER (raylib + emscripten main loop)
//       -> viewer/dist/nmmo3_viewer.{js,wasm,data}    (browser bundle)
//   - without NMMO3_RENDER (headless core, ENVIRONMENT=node, --no-entry)
//       -> build/viewer_core.{js,wasm}                (node verification)
//
// The core API (viewer_load / viewer_seek / viewer_advance / ...) is
// identical in both builds; only the raylib main loop is render-only.
// This is what lets tests/test_viewer.py prove the re-sim logic under
// node without pixels.
//
// Replay format v1 (server/cogame_nmmo/replay.py is the authority):
//   bytes 0-3  magic "NMMO"; byte 4 version u8 == 1;
//   bytes 5-8  header_len u32le; then header JSON (parsed JS-side);
//   then tick_count * num_agents bytes (one uint8 26-way action per
//   agent per tick, post-clamp).
// C never parses the JSON: JS passes num_agents (config players x
// heroes_per_seat) to viewer_load, and tick_count == body_len /
// num_agents by construction.
//
// Determinism: viewer_seek() rebuilds the sim through the exact fresh-
// instance path the server host uses (zeroed struct -> nmmo_configure
// [env.rng = seed] -> allocate_mmo -> c_reset) and replays the recorded
// actions from tick 0. Replay values are stored post-clamp, so a direct
// uint8 -> float cast matches the server's set_actions byte-for-byte.
//
// RNG hazard (render build only): upstream's make_client() runs
// render_conversion() which draws from env->rng — the SAME stream the
// sim consumes. The recording host never creates a client, so letting
// c_render() create one lazily would silently fork the rng stream and
// every subsequent tick would diverge from the recording. frame()
// therefore creates the client itself with the rng saved and restored
// around the call; c_render() then always sees env->client != NULL and
// never runs make_client. sim_fresh() preserves the client across
// rebuilds (same seed => same terrain, so the conversion stays valid).

#include <stdlib.h>
#include <string.h>

#include "shim_common.h"  // nmmo_configure(): shared env defaults (+ nmmo3.h)

#ifdef NMMO3_RENDER
#include <emscripten.h>
#endif

#define VIEWER_MAGIC_LEN 9        // magic(4) + version u8 + header_len u32le
#define VIEWER_MAX_AGENTS 1024    // sanity cap (upstream ini trains 1024)
#define VIEWER_ACT_MAX 25         // 26-way discrete action, high exclusive

// Upstream demo cadence (nmmo3.c:192-206): one sim tick per TICK_FRAMES
// == 36 render frames at FRAME_RATE 60 — i.e. 600 ms/tick at 1x. Kept
// as the interpolation-phase granularity and the meaning of one
// "60Hz-equivalent frame" (viewer_advance_frame).
#define VIEWER_FRAMES_PER_TICK 36
// Nominal playback advanced by WALL TIME, not render-callback count —
// rAF callbacks fire per display refresh, so counting them plays 2x too
// fast on a 120 Hz display and hitches on every dropped frame.
#define VIEWER_TICK_MS (1000.0 * VIEWER_FRAMES_PER_TICK / 60.0)  // 600 ms
// Per-callback dt clamp: a backgrounded tab can sit for seconds between
// callbacks; do not burst-run that gap on return.
#define VIEWER_MAX_DT_MS 100.0

static MMO env;
static int g_allocated = 0;

static const unsigned char* g_body = NULL;  // action log, inside JS-owned buf
static int g_num_agents = 0;   // replay body stride (bytes per tick)
static int g_total_ticks = 0;
static int g_tick = 0;         // ticks fed so far == current sim tick
static int g_playing = 0;
static int g_speed = 1;         // playback multiplier
static double g_time_acc = 0.0; // speed-scaled ms toward the next tick
                                // (invariant: < VIEWER_TICK_MS between
                                // viewer_advance calls)
static unsigned int g_seed = 0;
static int g_loaded = 0;

#ifdef NMMO3_RENDER
// Camera-follow seat (defined with the render half below); tentatively
// declared here so viewer_load can reset it on (re)load.
static int g_follow;
#endif

// Interpolation phase for the render half, in units of
// VIEWER_FRAMES_PER_TICK: how far the display sits through the current
// [last, cur] entity-position interpolation window. 0..N-1 mid-sweep;
// == N means "render exactly at-tick" (delta 1.0: entities drawn at
// their current tile, zero animation offset). Upstream's c_render
// interpolates from its OWN free-running client->frame counter
// (delta = frame/36), which assumes exactly one sim tick per 36
// phase-locked render calls; seeks and speed changes break that
// assumption and cause per-tick lurching, so frame() overwrites the
// client counter from this value before every c_render call.
static int g_phase = 0;

// Rebuild the sim exactly like a fresh wasm instance's nmmo_init(): the
// static struct starts zeroed, nmmo_configure sets the trained-on
// defaults + writes env.rng = seed (that write IS the seeding — the sim
// draws all randomness via rand_r(&env->rng)), allocate_mmo re-allocates
// buffers and generates the world, c_reset spawns. The renderer client
// survives across rebuilds (see the RNG-hazard note above).
static void sim_fresh(void) {
    void* client = env.client;  // Client*, owned by the render loop
    if (g_allocated)
        free_allocated_mmo(&env);
    memset(&env, 0, sizeof(env));
    env.client = client;
    nmmo_configure(&env, g_seed, g_num_agents);
    allocate_mmo(&env);
    c_reset(&env);
    g_allocated = 1;
    g_tick = 0;
}

static void feed_and_step(void) {
    // Replays are written post-clamp (0..25), but a hand-crafted body
    // byte >= 26 would drive upstream's unchecked `int action =
    // env->actions[pid]` decode out of range — clamp defensively.
    const unsigned char* a = g_body + (size_t)g_tick * (size_t)g_num_agents;
    for (int i = 0; i < g_num_agents; i++)
        env.actions[i] = (float)(a[i] > VIEWER_ACT_MAX ? VIEWER_ACT_MAX
                                                       : a[i]);
    c_step(&env);
    g_tick++;
}

// Parse replay bytes at ptr/len (JS-owned wasm heap memory that must stay
// alive while loaded) and start a fresh sim at tick 0, paused. The header
// JSON is parsed JS-side; JS passes the seed from header.config.seed and
// num_agents from the header config topology (players x heroes_per_seat).
// Returns total tick count, or -1 on malformed bytes/arguments.
int viewer_load(const unsigned char* data, int len, unsigned int seed,
                int num_agents) {
    if (data == NULL || len < VIEWER_MAGIC_LEN)
        return -1;
    if (num_agents < 1 || num_agents > VIEWER_MAX_AGENTS)
        return -1;
    if (memcmp(data, "NMMO", 4) != 0 || data[4] != 1)
        return -1;
    unsigned int header_len = (unsigned int)data[5]
        | ((unsigned int)data[6] << 8)
        | ((unsigned int)data[7] << 16)
        | ((unsigned int)data[8] << 24);
    // Non-wrappable on wasm32: len >= VIEWER_MAGIC_LEN is established
    // above, so the subtraction is safe; adding to header_len is not.
    if (header_len > (unsigned int)(len - VIEWER_MAGIC_LEN))
        return -1;
    size_t body_len = (size_t)len - VIEWER_MAGIC_LEN - header_len;
    if (body_len % (size_t)num_agents != 0)
        return -1;

    g_body = data + VIEWER_MAGIC_LEN + header_len;
    g_num_agents = num_agents;
    g_total_ticks = (int)(body_len / (size_t)num_agents);
    g_seed = seed;
    g_playing = 0;
    g_time_acc = 0.0;
    g_phase = VIEWER_FRAMES_PER_TICK;  // display tick 0 exactly
#ifdef NMMO3_RENDER
    // A follow pid kept from a previous, larger replay would index out
    // of the new roster (viewer_set_follow validates against
    // g_num_agents only at set time).
    g_follow = 0;
#endif
    sim_fresh();
    g_loaded = 1;
    return g_total_ticks;
}

// Re-sim from tick 0 to `tick` (clamped to 0..total), no rendering.
void viewer_seek(int tick) {
    if (!g_loaded)
        return;
    if (tick < 0) tick = 0;
    if (tick > g_total_ticks) tick = g_total_ticks;
    sim_fresh();
    while (g_tick < tick)
        feed_and_step();
    g_time_acc = 0.0;
    g_phase = VIEWER_FRAMES_PER_TICK;  // display the seek target exactly
    if (g_tick >= g_total_ticks)
        g_playing = 0;  // seek-to-end lands in the "ended" state
}

// Advance the simulation by dt_ms of wall time (clamped to
// VIEWER_MAX_DT_MS). At speed s, one sim tick per VIEWER_TICK_MS/s of
// wall time, regardless of display refresh rate. Pauses at end of
// replay instead of looping. Returns sim ticks stepped.
int viewer_advance(double dt_ms) {
    if (!g_loaded || !g_playing)
        return 0;
    if (dt_ms < 0.0)
        dt_ms = 0.0;
    if (dt_ms > VIEWER_MAX_DT_MS)
        dt_ms = VIEWER_MAX_DT_MS;  // tab-switch gap: no burst on return
    int stepped = 0;
    g_time_acc += dt_ms * (double)g_speed;
    // Epsilon: accumulated 60Hz dts can round a hair short of the tick
    // threshold; a sub-nanosecond shortfall must not defer the tick a
    // whole callback.
    while (g_time_acc >= VIEWER_TICK_MS - 1e-6) {
        g_time_acc -= VIEWER_TICK_MS;
        if (g_time_acc < 0.0)
            g_time_acc = 0.0;
        if (g_tick >= g_total_ticks) {
            g_playing = 0;   // ended: JS sees playing==0 && tick==total
            g_time_acc = 0.0;
            break;
        }
        feed_and_step();
        stepped++;
        if (g_tick >= g_total_ticks) {
            g_playing = 0;
            g_time_acc = 0.0;
            break;
        }
    }
    if (g_tick >= g_total_ticks || stepped > 1) {
        // Ended (show the final state exactly), or several ticks in one
        // callback: interpolating the last tick interval is
        // meaningless — render at-tick.
        g_phase = VIEWER_FRAMES_PER_TICK;
    } else if (stepped == 1 || g_phase != VIEWER_FRAMES_PER_TICK) {
        // Fresh interpolation window (a tick just stepped) or mid-sweep:
        // g_time_acc is wall-time progress toward the next tick, which
        // is exactly the progress through the current [last, cur]
        // window; quantize it to the renderer's 36ths. From an at-tick
        // display with no new tick (else-branch not taken) the phase
        // holds at-tick — sweeping backwards would lurch.
        g_phase = (int)(g_time_acc * VIEWER_FRAMES_PER_TICK / VIEWER_TICK_MS);
        if (g_phase < 0)
            g_phase = 0;
        if (g_phase >= VIEWER_FRAMES_PER_TICK)
            g_phase = VIEWER_FRAMES_PER_TICK - 1;
    }
    return stepped;
}

// One 60Hz-equivalent frame (fixed-dt API for the node harness):
// 36 calls == one tick at 1x.
int viewer_advance_frame(void) {
    return viewer_advance(1000.0 / 60.0);
}

// Current interpolation phase (see g_phase). Exported for the node
// harness so the phase-lock behavior is testable without pixels.
int viewer_render_phase(void) { return g_phase; }

int viewer_tick(void) { return g_tick; }

int viewer_total_ticks(void) { return g_total_ticks; }

void viewer_set_speed(int speed) {
    if (speed >= 1 && speed <= 1024)
        g_speed = speed;
}

int viewer_get_speed(void) { return g_speed; }

void viewer_set_playing(int playing) {
    if (!g_loaded)
        return;
    if (playing && g_tick >= g_total_ticks)
        return;  // ended: JS must seek first (no silent loop)
    g_playing = playing ? 1 : 0;
    // g_time_acc is deliberately kept: it is always < VIEWER_TICK_MS
    // here (the advance loop reduces it), so no tick burst is possible,
    // and preserving it resumes the interpolation sweep exactly where
    // the pause froze it (a reset would lurch the display backwards).
}

int viewer_playing(void) { return g_playing; }

// State digest at the current tick (FNV-1a over player r/c/hp/levels +
// env rng + tick; sim/shim_common.h). Must equal the recording host's
// state_digest() at the same tick — the headless verification
// (tests/test_viewer.py) and replay certification rely on this.
unsigned int viewer_state_digest(void) {
    return g_allocated ? nmmo_state_digest(&env) : 0;
}

#ifdef NMMO3_RENDER
// Seat the camera follows in centered mode (upstream client->my_player).
// Applied every frame so upstream's console "play" command can't
// silently clobber it. (Tentatively declared above; reset by
// viewer_load.)
static int g_follow = 0;

void viewer_set_follow(int pid) {
    if (g_loaded && pid >= 0 && pid < g_num_agents)
        g_follow = pid;
}

int viewer_get_follow(void) { return g_follow; }

// Render loop: one callback per browser animation frame. c_render
// (upstream, unchanged) interpolates entity positions between sim ticks
// from client->frame; its return value is the human keyboard action —
// discarded here, because the replay log is authoritative and upstream
// only uses the returned action for same-frame UI highlighting (nothing
// in the render half mutates sim state based on it).
static void frame(void) {
    if (!g_loaded)
        return;  // window/canvas appear on first frame after load
    // Advance by measured wall time (emscripten_get_now, ms): rAF fires
    // per display refresh, so callback-counting would play 2x too fast
    // on 120 Hz displays and hitch on dropped frames. The dt clamp in
    // viewer_advance absorbs tab-switch gaps; last_now keeps updating
    // even while paused so resume sees a normal dt.
    static double last_now = -1.0;
    double now = emscripten_get_now();
    double dt_ms = (last_now < 0.0) ? 0.0 : now - last_now;
    last_now = now;
    viewer_advance(dt_ms);
    if (env.client == NULL) {
        // Create the render client HERE, never inside c_render: upstream
        // make_client's render_conversion draws from env->rng, the same
        // stream the sim steps with, and the recording host never made a
        // client — consume it and every later tick diverges from the
        // recording. Save/restore makes client creation rng-neutral.
        unsigned int saved_rng = env.rng;
        env.client = make_client(&env);
        env.rng = saved_rng;
    }
    Client* client = env.client;
    // Phase-lock upstream's interpolation counter to the true inter-tick
    // progress (see g_phase): c_render computes delta = frame/36 for
    // entity/camera interpolation, sprite-frame selection, the water
    // animation and the shader time uniform, and assumes 1 sim tick per
    // 36 phase-aligned render calls — seeks, speeds and 120 Hz displays
    // all violate that. client->frame's only other use is the
    // end-of-render increment/wrap, which this per-frame overwrite
    // supersedes. g_phase == 36 yields delta 1.0: render exactly
    // at-tick (in-bounds: ANIMATIONS[].frames has 10 entries and
    // num_frames <= 8, so frames[num_frames] is defined).
    client->frame = g_phase;
    client->my_player = g_follow;
    (void)c_render(&env);  // returned keyboard action: replay is authoritative
}

int main(void) {
    // 0 fps == requestAnimationFrame; don't simulate an infinite loop —
    // main returns and the runtime stays alive (EXIT_RUNTIME=0) for the
    // viewer_* exports.
    emscripten_set_main_loop(frame, 0, 0);
    return 0;
}
#endif  // NMMO3_RENDER
