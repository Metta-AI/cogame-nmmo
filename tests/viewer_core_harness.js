#!/usr/bin/env node
// Headless verification harness for the viewer core wasm (no raylib).
//
// Usage: node viewer_core_harness.js <build/viewer_core.js> <replay.bin>
//
// Loads the replay exactly like viewer/index.html does (header JSON
// parsed JS-side, bytes copied into the wasm heap, seed and num_agents
// passed from the header), exercises the core API — including
// malformed-bytes rejection — and prints one JSON object for
// tests/test_viewer.py to assert on. Exits non-zero on any failure.
"use strict";

const fs = require("fs");
const path = require("path");

const [, , coreJsPath, replayPath] = process.argv;
if (!coreJsPath || !replayPath) {
  console.error("usage: viewer_core_harness.js <viewer_core.js> <replay.bin>");
  process.exit(2);
}

const bytes = fs.readFileSync(replayPath);
if (bytes.toString("latin1", 0, 4) !== "NMMO" || bytes[4] !== 1) {
  console.error("bad replay magic/version");
  process.exit(2);
}
const headerLen = bytes.readUInt32LE(5);
const header = JSON.parse(bytes.toString("utf-8", 9, 9 + headerLen));
if (!Number.isInteger(header.config.seed)) {
  console.error("header config.seed is not an integer:", header.config.seed);
  process.exit(2);
}
const numAgents =
  header.config.players.length * header.config.heroes_per_seat;

const createViewerCore = require(path.resolve(coreJsPath));

// The nmmo cadence: one sim tick per 36 60Hz-equivalent frames (600 ms)
// at 1x — upstream TICK_FRAMES (see sim/viewer_main.c).
const FRAMES_PER_TICK = 36;

// viewer_load must return -1 for each of these, before any real load.
function malformedResults(M, call) {
  const tryLoad = (buf, agents = 8) => {
    const p = M._malloc(buf.length);
    M.HEAPU8.set(buf, p);
    const r = call("viewer_load", "number",
      ["number", "number", "number", "number"], [p, buf.length, 1, agents]);
    M._free(p);
    return r;
  };
  const goodPrefix = Buffer.from("NMMO\x01", "latin1");
  const cases = {};
  cases.badMagic = tryLoad(Buffer.concat(
    [Buffer.from("MOBA\x01", "latin1"), Buffer.alloc(64)]));
  cases.badVersion = tryLoad(Buffer.concat(
    [Buffer.from("NMMO\x09", "latin1"), Buffer.alloc(64)]));
  cases.tooShort = tryLoad(Buffer.from("NMMO\x01\x00\x00", "latin1"));
  // header_len runs past end of buffer
  const truncated = Buffer.concat([goodPrefix, Buffer.alloc(4 + 8)]);
  truncated.writeUInt32LE(1000, 5);
  cases.truncatedHeader = tryLoad(truncated);
  // header_len near UINT32_MAX: 9 + header_len wraps on wasm32 — the
  // non-wrappable check must still reject it
  const wrapping = Buffer.concat([goodPrefix, Buffer.alloc(4 + 8)]);
  wrapping.writeUInt32LE(0xFFFFFFFF, 5);
  cases.wrappingHeaderLen = tryLoad(wrapping);
  // body not a multiple of num_agents (61 bytes, 8 agents)
  const raggedHeader = Buffer.from("{}", "utf-8");
  const ragged = Buffer.concat(
    [goodPrefix, Buffer.alloc(4), raggedHeader, Buffer.alloc(61)]);
  ragged.writeUInt32LE(raggedHeader.length, 5);
  cases.raggedBody = tryLoad(ragged, 8);
  // agent-count bounds: 0 and > the 1024 sanity cap must both reject
  const wellFormed = Buffer.concat(
    [goodPrefix, Buffer.alloc(4), raggedHeader, Buffer.alloc(64)]);
  wellFormed.writeUInt32LE(raggedHeader.length, 5);
  cases.zeroAgents = tryLoad(wellFormed, 0);
  cases.hugeAgents = tryLoad(wellFormed, 2000);
  return cases;
}

function run(M) {
  const call = (name, ret, args = [], vals = []) =>
    M.ccall(name, ret, args, vals);

  const malformed = malformedResults(M, call);

  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  const seed = header.config.seed >>> 0;  // & 0xFFFFFFFF, like the host
  const total = call("viewer_load", "number",
    ["number", "number", "number", "number"],
    [ptr, bytes.length, seed, numAgents]);

  // Frame cadence: at speed s, s ticks per 36 advance_frame calls.
  const ticksOver = (frames) => {
    let n = 0;
    for (let i = 0; i < frames; i++)
      n += call("viewer_advance_frame", "number");
    return n;
  };
  call("viewer_set_playing", null, ["number"], [1]);
  const cadence1 = ticksOver(FRAMES_PER_TICK);
  call("viewer_set_speed", null, ["number"], [4]);
  const cadence4 = ticksOver(FRAMES_PER_TICK);
  const pausedTicks = (() => {  // paused: advance_frame must be a no-op
    call("viewer_set_playing", null, ["number"], [0]);
    return ticksOver(2 * FRAMES_PER_TICK);
  })();

  const mid = Math.floor(total / 2);
  call("viewer_seek", null, ["number"], [mid]);
  const midTick = call("viewer_tick", "number");

  // Interpolation phase-lock (viewer_render_phase): at-tick (36) after
  // a seek; sweeps 0,1,2,... once ticks step at 1x; frozen across
  // pause/resume; pinned at-tick when one callback steps several ticks.
  const phase = () => call("viewer_render_phase", "number");
  const phaseAfterSeek = phase();
  call("viewer_set_speed", null, ["number"], [1]);
  call("viewer_set_playing", null, ["number"], [1]);
  let guard = 0;  // advance until the first tick fires (<= 36 frames)
  while (call("viewer_advance_frame", "number") === 0 && guard++ < 80) {}
  const phaseSweep = [phase()];
  call("viewer_advance_frame", "number");
  phaseSweep.push(phase());
  call("viewer_advance_frame", "number");
  phaseSweep.push(phase());
  call("viewer_set_playing", null, ["number"], [0]);
  ticksOver(5);  // paused: phase must freeze
  const phasePaused = phase();
  call("viewer_set_playing", null, ["number"], [1]);
  call("viewer_advance_frame", "number");
  const phaseResumed = phase();  // sweep continues, no backward reset
  // 64x with a 100ms callback: 6400 speed-scaled ms = 10 ticks in one
  // call — multi-tick frames must render pinned at-tick.
  call("viewer_set_speed", null, ["number"], [64]);
  const ticksMulti = call("viewer_advance", "number", ["number"], [100]);
  const phaseAtMulti = phase();

  // Time-based advance (viewer jitter fix): 1 tick per 600ms of
  // (speed-scaled) wall time, independent of callback count; a single
  // callback's dt clamps to 100ms so a backgrounded tab does not burst
  // on return.
  call("viewer_seek", null, ["number"], [mid]);
  call("viewer_set_speed", null, ["number"], [1]);
  call("viewer_set_playing", null, ["number"], [1]);
  const dtSteps = [];
  for (let i = 0; i < 6; i++)
    dtSteps.push(call("viewer_advance", "number", ["number"], [100]));
  // 5000ms in one callback clamps to 100ms: no burst
  const dtClamped = call("viewer_advance", "number", ["number"], [5000]);
  const dtAfterClamp = [];
  for (let i = 0; i < 5; i++)
    dtAfterClamp.push(call("viewer_advance", "number", ["number"], [100]));

  call("viewer_seek", null, ["number"], [total]);
  const endTick = call("viewer_tick", "number");
  const playingAtEnd = call("viewer_playing", "number");
  // set_playing(1) at end must refuse (no silent restart/loop)
  call("viewer_set_playing", null, ["number"], [1]);
  const playAtEndRefused = call("viewer_playing", "number") === 0 ? 1 : 0;

  console.log(JSON.stringify({
    malformed,
    total, cadence1, cadence4, pausedTicks, midTick, endTick,
    playingAtEnd, playAtEndRefused,
    phaseAfterSeek, phaseSweep, phasePaused, phaseResumed,
    ticksMulti, phaseAtMulti,
    dtSteps, dtClamped, dtAfterClamp,
    // u32 digest (ccall returns the i32 bit pattern; normalize)
    stateDigest: call("viewer_state_digest", "number") >>> 0,
    headerTickCount: header.tick_count,
    numAgents,
  }));
}

createViewerCore().then(run).catch((e) => {
  console.error("harness failed:", e);
  process.exit(1);
});
