"""Tests for the transport-free lockstep episode engine."""

import asyncio

import numpy as np
import pytest

from cogame_nmmo import defaults
from cogame_nmmo.config import GameConfig
from cogame_nmmo.engine import STAT_CODES, EpisodeResult, LockstepEngine

NOOP = list(defaults.NOOP_ACTION)


def make_config(num_seats=8, heroes_per_seat=1, **overrides):
    d = {
        "players": [{"name": f"p{i}"} for i in range(num_seats)],
        "tokens": [f"t{i}" for i in range(num_seats)],
        "heroes_per_seat": heroes_per_seat,
        "seed": 5,
        "max_ticks": 10,
        "tick_deadline_ms": 200,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class FakeSim:
    """Records actions fed per tick; obs row i is filled with byte value i.

    ``dones_at`` maps tick -> list of agent pids whose terminal fires on
    that tick's step (the engine must forward them as `resets` with the
    NEXT tick's obs). ``scores`` are returned by score(pid).
    """

    def __init__(self, num_agents=8, dones_at=None, scores=None):
        self.num_agents = num_agents
        self._tick = 0
        self.dones_at = dones_at or {}
        self.scores = list(scores) if scores is not None \
            else [0] * num_agents
        self.fed_actions = []  # list of (num_agents, 1) float32 arrays

    def observations(self):
        return np.tile(
            np.arange(self.num_agents, dtype=np.uint8).reshape(-1, 1),
            (1, 1707))

    def set_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self.num_agents, 1)
        self.fed_actions.append(actions.copy())

    def step(self):
        self._tick += 1

    def rewards(self):
        # agent pid p earns reward p+1 each tick
        return np.arange(1, self.num_agents + 1, dtype=np.float32)

    def dones(self):
        # step() already ran: tick T's flags live under key T = tick-1
        fired = self.dones_at.get(self._tick - 1, [])
        return [pid in fired for pid in range(self.num_agents)]

    def tick(self):
        return self._tick

    def agent_stat(self, pid, which):
        return pid * 100 + which

    def score(self, pid):
        return self.scores[pid]

    def state_digest(self):
        return 0xABCD1234


class ScriptedSource:
    """Returns a fixed per-agent action list every tick; records obs+resets."""

    def __init__(self, actions):
        self.actions = actions
        self.seen = []  # (tick, obs, resets) triples

    async def get_actions(self, tick, obs, resets):
        self.seen.append((tick, obs.copy(), list(resets)))
        return self.actions


class SlowSource:
    async def get_actions(self, tick, obs, resets):
        await asyncio.sleep(60)
        return [NOOP]


class NoneSource:
    async def get_actions(self, tick, obs, resets):
        return None


class RaisingSource:
    async def get_actions(self, tick, obs, resets):
        raise RuntimeError("player exploded")


class MalformedSource:
    def __init__(self, payload):
        self.payload = payload

    async def get_actions(self, tick, obs, resets):
        return self.payload


# -- action routing ----------------------------------------------------------

async def test_scripted_actions_reach_sim_rows():
    sim = FakeSim()
    sources = [ScriptedSource([[i % 26]]) for i in range(8)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 3
    assert len(sim.fed_actions) == 3
    for fed in sim.fed_actions:
        for pid in range(8):
            assert fed[pid].tolist() == [pid % 26]


async def test_multi_agent_seat_obs_slicing_and_row_mapping():
    sim = FakeSim()
    seat0 = ScriptedSource([[1]] * 4)
    seat1 = ScriptedSource([[2]] * 4)
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=2)
    await LockstepEngine(sim, cfg, [seat0, seat1]).run()
    # each seat saw exactly its agents' obs rows (row p is filled with p)
    for tick, obs, resets in seat0.seen:
        assert obs.shape == (4, 1707)
        assert obs[:, 0].tolist() == [0, 1, 2, 3]
        assert len(resets) == 4
    for tick, obs, resets in seat1.seen:
        assert obs[:, 0].tolist() == [4, 5, 6, 7]
    # and each seat's actions landed on its agents' rows
    fed = sim.fed_actions[0]
    for pid in range(4):
        assert fed[pid].tolist() == [1]
    for pid in range(4, 8):
        assert fed[pid].tolist() == [2]


async def test_solo_variant_obs_is_single_agent_row():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=1)
    await LockstepEngine(sim, cfg, sources).run()
    for seat, src in enumerate(sources):
        (tick, obs, resets), = src.seen
        assert tick == 0
        assert obs.shape == (1, 1707)
        assert obs[0, 0] == seat
        assert resets == [False]


# -- resets plumbing (protocol v2) -------------------------------------------

async def test_resets_forwarded_with_next_ticks_obs():
    """An agent whose terminal fires on tick T's step is flagged in the
    resets delivered with tick T+1's obs — and only that tick."""
    sim = FakeSim(dones_at={2: [3], 4: [0, 5]})
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=7)
    await LockstepEngine(sim, cfg, sources).run()
    for seat, src in enumerate(sources):
        for tick, obs, resets in src.seen:
            expect = (seat == 3 and tick == 3) or \
                (seat in (0, 5) and tick == 5)
            assert resets == [expect], (seat, tick, resets)


async def test_resets_all_false_at_tick_zero():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=1)
    await LockstepEngine(sim, cfg, sources).run()
    for src in sources:
        (tick, _obs, resets), = src.seen
        assert tick == 0 and resets == [False]


async def test_multi_agent_seat_resets_sliced_per_seat():
    sim = FakeSim(dones_at={0: [1, 6]})
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=2)
    seat0, seat1 = ScriptedSource([NOOP] * 4), ScriptedSource([NOOP] * 4)
    await LockstepEngine(sim, cfg, [seat0, seat1]).run()
    assert seat0.seen[1][2] == [False, True, False, False]
    assert seat1.seen[1][2] == [False, False, True, False]


# -- NOOP fallbacks ----------------------------------------------------------

@pytest.mark.parametrize("bad_source", [
    NoneSource(),
    RaisingSource(),
    MalformedSource("garbage"),
    MalformedSource([[1, 2]]),                       # wrong shape
    MalformedSource([[]]),                           # empty row
    MalformedSource([[float("nan")]]),               # non-finite
    MalformedSource([["a"]]),
])
async def test_bad_sources_get_noop(bad_source):
    sim = FakeSim()
    sources = [ScriptedSource([[1]]) for _ in range(7)]
    sources.append(bad_source)
    cfg = make_config(max_ticks=2)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 2  # episode never crashes
    for fed in sim.fed_actions:
        assert fed[7].tolist() == NOOP
        assert fed[0].tolist() == [1]


async def test_deadline_timeout_gets_noop():
    sim = FakeSim()
    sources = [ScriptedSource([[2]]) for _ in range(7)]
    sources.append(SlowSource())
    cfg = make_config(max_ticks=2, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 2
    for fed in sim.fed_actions:
        assert fed[7].tolist() == NOOP


async def test_out_of_range_actions_clamped():
    sim = FakeSim()
    sources = [ScriptedSource([[99]])] + \
        [ScriptedSource([NOOP]) for _ in range(7)]
    cfg = make_config(max_ticks=1)
    ticks = []
    engine = LockstepEngine(sim, cfg, sources,
                            on_tick=lambda t, a: ticks.append((t, a.copy())))
    await engine.run()
    assert sim.fed_actions[0][0].tolist() == [25]
    # replay hook sees the same post-clamp values, as uint8
    (t0, acts0), = ticks
    assert t0 == 0
    assert acts0.dtype == np.uint8
    assert acts0[0].tolist() == [25]
    assert acts0[5].tolist() == NOOP


async def test_noop_causes_classified_per_seat():
    """Every NOOP fallback is attributed to a cause (results
    observability: noop_causes)."""
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(4)] + [
        SlowSource(),                # -> timeout
        NoneSource(),                # -> disconnected (source had nothing)
        RaisingSource(),             # -> host_error
        MalformedSource("garbage"),  # -> malformed
    ]
    cfg = make_config(max_ticks=3, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    causes = result.seat_noop_causes
    assert causes[4]["timeout"] == 3
    assert causes[5]["disconnected"] == 3
    assert causes[6]["host_error"] == 3
    assert causes[7]["malformed"] == 3
    for s in range(4):
        assert sum(causes[s].values()) == 0, (s, causes[s])
    # every noop tick is attributed
    for s in range(8):
        assert sum(causes[s].values()) == result.seat_noop_ticks[s]


# -- termination + scoring ---------------------------------------------------

async def test_tick_cap_scores_from_sim():
    sim = FakeSim(scores=[5, 0, 12, 3, 3, 0, 7, 1])
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=5)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert isinstance(result, EpisodeResult)
    assert result.end_reason == "tick_cap"
    assert result.final_tick == 5
    assert list(result.seat_scores) == [5.0, 0.0, 12.0, 3.0, 3.0, 0.0,
                                        7.0, 1.0]
    assert result.state_digest == 0xABCD1234


async def test_multi_agent_seat_scores_summed():
    sim = FakeSim(scores=[1, 2, 3, 4, 10, 20, 30, 40])
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=2)
    sources = [ScriptedSource([NOOP] * 4) for _ in range(2)]
    result = await LockstepEngine(sim, cfg, sources).run()
    assert list(result.seat_scores) == [10.0, 100.0]


async def test_reward_sums_and_stats():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(sim, cfg, sources).run()
    # agent pid p earns p+1 per tick, 3 ticks, 1 agent per seat
    assert list(result.seat_reward_sums) == pytest.approx(
        [(p + 1) * 3 for p in range(8)])
    assert len(result.agent_stats) == 8
    assert set(result.agent_stats[0]) == set(STAT_CODES)
    assert result.agent_stats[2]["deaths"] == 2 * 100 + 1
    assert result.agent_stats[7]["cum_min_comb_prof"] == 7 * 100 + 0
    assert result.agent_stats[3]["gold"] == 3 * 100 + 5


def test_stat_codes_match_sim_module():
    """STAT_CODES duplicates sim/shim.c which codes (no wasmtime import
    in the engine); the sim module's constants are the reference."""
    from cogame_nmmo import sim
    assert STAT_CODES == {
        "cum_min_comb_prof": sim.STAT_CUM_MIN_COMB_PROF,
        "deaths": sim.STAT_DEATHS,
        "comb_lvl": sim.STAT_COMB_LVL,
        "prof_lvl": sim.STAT_PROF_LVL,
        "gold": sim.STAT_GOLD,
        "time_alive": sim.STAT_TIME_ALIVE,
    }


async def test_wall_clock_budget_ends_episode():
    """A slow episode ends at the wall-clock budget with
    end_reason="wall_clock", well before the platform's episode_timeout
    kill (which would lose the results and replay entirely)."""

    class Slowish:
        async def get_actions(self, tick, obs, resets):
            await asyncio.sleep(0.02)
            return [NOOP]

    sim = FakeSim(scores=[4, 0, 0, 0, 0, 0, 0, 9])
    sources = [Slowish() for _ in range(8)]
    cfg = make_config(max_ticks=10_000, tick_deadline_ms=1000,
                      wall_clock_budget_seconds=0.4)
    result = await asyncio.wait_for(
        LockstepEngine(sim, cfg, sources).run(), timeout=10)
    assert result.end_reason == "wall_clock"
    assert 0 < result.final_tick < 10_000
    # scores still read from the sim (partial episode, real standings)
    assert list(result.seat_scores) == [4.0, 0, 0, 0, 0, 0, 0, 9.0]


# -- sim fault containment (patch 0002) --------------------------------------

class FaultingSim(FakeSim):
    """fault() goes nonzero after `fault_at` steps (patch-0002 flag)."""

    def __init__(self, fault_at, **kw):
        super().__init__(**kw)
        self.fault_at = fault_at

    def fault(self):
        return 2 if self._tick >= self.fault_at else 0


class TrapDeath(Exception):
    """Stands in for a wasmtime trap (e.g. an ExitTrap on exit())."""


class TrappingSim(FakeSim):
    """step() traps after `trap_at` steps and every call raises after —
    like a wasm instance whose execution trapped."""

    def __init__(self, trap_at, **kw):
        super().__init__(**kw)
        self.trap_at = trap_at
        self.dead = False

    def _check(self):
        if self.dead:
            raise TrapDeath("wasm instance is dead")

    def step(self):
        self._check()
        if self._tick >= self.trap_at:
            self.dead = True
            raise TrapDeath("wasm trapped in step()")
        super().step()

    def observations(self):
        self._check()
        return super().observations()

    def rewards(self):
        self._check()
        return super().rewards()

    def dones(self):
        self._check()
        return super().dones()

    def tick(self):
        self._check()
        return super().tick()

    def agent_stat(self, pid, which):
        self._check()
        return super().agent_stat(pid, which)

    def score(self, pid):
        self._check()
        return super().score(pid)

    def state_digest(self):
        self._check()
        return super().state_digest()


async def test_sim_fault_flag_ends_episode_as_sim_fault():
    sim = FaultingSim(fault_at=3)
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=50)
    ticks = []
    result = await LockstepEngine(
        sim, cfg, sources, on_tick=lambda t, a: ticks.append(t)).run()
    assert result.end_reason == "sim_fault"
    assert result.final_tick == 3
    # equal scores: an infra fault is nobody's loss
    assert list(result.seat_scores) == [0.0] * 8
    # the faulting tick completed and is in the replay
    assert ticks == [0, 1, 2]


async def test_sim_trap_contained_as_sim_fault():
    """A hard wasm trap mid-step must not escape the engine: the episode
    ends as sim_fault with best-effort result fields, so the server can
    still write results and the partial replay."""
    sim = TrappingSim(trap_at=4)
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=50)
    ticks = []
    result = await LockstepEngine(
        sim, cfg, sources, on_tick=lambda t, a: ticks.append(t)).run()
    assert result.end_reason == "sim_fault"
    assert list(result.seat_scores) == [0.0] * 8
    # 4 ticks completed before the trap; the trapped tick is not recorded
    assert ticks == [0, 1, 2, 3]
    assert result.final_tick == 4
    # dead-instance reads fall back instead of raising
    assert result.state_digest == 0
    assert all(v == 0 for stats in result.agent_stats
               for v in stats.values())


async def test_real_sim_fault_export_is_zero():
    """The patched wasm exports nmmo_fault(); a normal episode never
    trips it (the fidelity gate relies on exactly that)."""
    from cogame_nmmo.sim import NmmoSim

    sim = NmmoSim(seed=11, num_agents=8)
    assert sim.fault() == 0
    sources = [ScriptedSource([NOOP]) for _ in range(8)]
    cfg = make_config(max_ticks=5, tick_deadline_ms=2000)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.end_reason == "tick_cap"
    assert sim.fault() == 0


# -- real wasm sim end-to-end ------------------------------------------------

async def test_real_sim_end_to_end():
    from cogame_nmmo.sim import NmmoSim

    class RngSource:
        def __init__(self, seat, heroes):
            self.rng = np.random.default_rng(seat)
            self.heroes = heroes

        async def get_actions(self, tick, obs, resets):
            assert obs.shape == (self.heroes, 1707)
            assert len(resets) == self.heroes
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(self.heroes, 1)).tolist()

    sim = NmmoSim(seed=11, num_agents=8)
    cfg = make_config(max_ticks=40, tick_deadline_ms=2000)
    sources = [RngSource(seat, 1) for seat in range(8)]
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 40
    assert result.end_reason == "tick_cap"
    assert sim.tick() == 40
    assert all(np.isfinite(result.seat_reward_sums))
    # everyone starts at comb=prof=1: scores are always >= 0, and any
    # agent alive contributes at least 1 (structure check, not magnitude)
    assert all(s >= 0 for s in result.seat_scores)
    assert sum(result.seat_scores) >= 1
    assert result.state_digest == sim.state_digest()
    assert all(s["comb_lvl"] >= 1 and s["prof_lvl"] >= 1
               for s in result.agent_stats)


async def test_real_sim_deaths_surface_as_resets():
    """Random 8-agent play sees deaths within a few hundred ticks (N1
    determinism test); the engine must forward them: some source observes
    resets=[True], and the flagged ticks match the sim's death counters."""
    from cogame_nmmo.sim import NmmoSim
    from cogame_nmmo.sim import STAT_DEATHS

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(seat)
            self.reset_ticks = []

        async def get_actions(self, tick, obs, resets):
            if resets[0]:
                self.reset_ticks.append(tick)
            return self.rng.integers(0, defaults.ACT_HIGH,
                                     size=(1, 1)).tolist()

    sim = NmmoSim(seed=7, num_agents=8)
    cfg = make_config(max_ticks=600, tick_deadline_ms=2000)
    sources = [RngSource(seat) for seat in range(8)]
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 600
    total_resets = sum(len(s.reset_ticks) for s in sources)
    assert total_resets > 0, \
        "no resets forwarded in 600 random ticks - seed regression?"
    # every death the sim counted was forwarded to exactly one source
    total_deaths = sum(sim.agent_stat(p, STAT_DEATHS) for p in range(8))
    assert total_resets == total_deaths


async def test_real_sim_multi_agent_seat_slicing():
    from cogame_nmmo.sim import NmmoSim

    seen = {}

    class CaptureSource:
        def __init__(self, seat):
            self.seat = seat

        async def get_actions(self, tick, obs, resets):
            if tick == 0:
                seen[self.seat] = obs.copy()
            return [NOOP] * 4

    sim = NmmoSim(seed=11, num_agents=8)
    full_obs = sim.observations()
    cfg = make_config(num_seats=2, heroes_per_seat=4, max_ticks=2,
                      tick_deadline_ms=2000)
    await LockstepEngine(sim, cfg, [CaptureSource(0), CaptureSource(1)]).run()
    np.testing.assert_array_equal(seen[0], full_obs[0:4])
    np.testing.assert_array_equal(seen[1], full_obs[4:8])


# -- silent-seat strike rule -------------------------------------------------

async def test_silent_seat_goes_dead_and_episode_races():
    """After 10 consecutive timeouts a seat is marked dead: its deadline is
    no longer waited on, so a silent seat costs at most ~strike_limit x
    deadline of wall clock for the whole episode."""
    import time

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [SlowSource()]
    cfg = make_config(max_ticks=40, tick_deadline_ms=100)
    t0 = time.monotonic()
    result = await LockstepEngine(sim, cfg, sources).run()
    elapsed = time.monotonic() - t0
    assert result.final_tick == 40
    # 10 strike ticks x 100ms deadline ~= 1s; the other 30 ticks are instant.
    # A non-dead silent seat would cost 40 x 100ms = 4s+.
    assert elapsed < 3.0
    assert result.seat_dead[7] is True
    assert result.seat_noop_ticks[7] == 40
    assert result.seat_dead[0] is False
    assert result.seat_noop_ticks[0] == 0


async def test_dead_seat_revives_on_valid_action():
    """A dead seat keeps getting revival probes; the first valid reply
    resets its strikes and resumes normal play (late reconnects work)."""

    class WakingSource:
        """Times out for the first `sleep_asks` asks, then answers instantly."""

        def __init__(self, sleep_asks):
            self.asks = 0
            self.sleep_asks = sleep_asks

        async def get_actions(self, tick, obs, resets):
            self.asks += 1
            if self.asks <= self.sleep_asks:
                await asyncio.sleep(60)
            return [[1]]

    sim = FakeSim()
    waking = WakingSource(sleep_asks=10)
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [waking]
    cfg = make_config(max_ticks=20, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 20
    # seat went dead (>= 10 strikes) but revived and played real actions
    assert result.seat_dead[7] is False
    assert 10 <= result.seat_noop_ticks[7] < 20
    assert sim.fed_actions[-1][7].tolist() == [1]


async def test_dead_revive_dead_revive_cycle():
    """A second dead spell must be revivable: the stale probe left over
    from the first revival (which may hang forever, e.g. a ws waiter
    clobbered by the next tick's send) must not block re-probing.

    Ask timeline for the flaky seat (strike limit 10, see phases):
      asks 1-10   hang    -> 10 strikes, seat dead
      ask  11     answer  -> revival probe reply
      ask  12     hang forever -> the leftover probe created on the
                   revival tick (with the fix it is cancelled and may
                   even be cancelled before it starts; either way later
                   asks shift by at most one, which the phases absorb)
      asks 13-14  answer  -> live play
      asks 15-24  hang    -> 10 strikes, seat dead again
      asks 25+    answer  -> second revival must happen
    """

    class DoubleWaker:
        def __init__(self):
            self.asks = 0

        async def get_actions(self, tick, obs, resets):
            self.asks += 1
            a = self.asks
            if a <= 10 or 15 <= a <= 24:
                await asyncio.sleep(60)   # deadline-cancelled: strike
            elif a == 12:
                await asyncio.sleep(600)  # leftover probe: hangs forever
            return [[1]]

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [DoubleWaker()]
    cfg = make_config(max_ticks=34, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 34
    # the seat died twice but ended the episode revived and playing
    assert result.seat_dead[7] is False
    assert result.seat_noop_ticks[7] >= 20
    assert sim.fed_actions[-1][7].tolist() == [1]


async def test_on_seat_dead_fires_once_per_death_transition():
    """The engine reports each strike-death transition exactly once (the
    server uses this to force-close the seat's stale websocket)."""

    class Phased:
        """Hangs for asks 1-10 (death), answers 11-13 (revive), hangs
        14-23 (second death), answers after."""

        def __init__(self):
            self.asks = 0

        async def get_actions(self, tick, obs, resets):
            self.asks += 1
            a = self.asks
            if a <= 10 or 14 <= a <= 23:
                await asyncio.sleep(60)
            return [[1]]

    deaths = []
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [Phased()]
    cfg = make_config(max_ticks=40, tick_deadline_ms=30)
    result = await LockstepEngine(
        sim, cfg, sources, on_seat_dead=deaths.append).run()
    assert result.final_tick == 40
    assert deaths.count(7) == 2, deaths
    assert set(deaths) == {7}


async def test_on_seat_dead_exception_never_crashes_episode():
    def boom(seat):
        raise RuntimeError("callback exploded")

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [NoneSource()]
    cfg = make_config(max_ticks=15, tick_deadline_ms=30)
    result = await LockstepEngine(
        sim, cfg, sources, on_seat_dead=boom).run()
    assert result.final_tick == 15
    assert result.seat_dead[7] is True


async def test_all_seats_dead_event_loop_keeps_yielding():
    """With EVERY seat dead the per-tick gather is empty and awaits
    nothing: without an explicit yield the while loop starves the event
    loop, stalling /healthz (liveness kill) and the revival probes
    themselves. The engine must yield each tick even with no live seats."""

    sim = FakeSim()
    seen_ticks = set()

    async def heartbeat():
        while True:
            seen_ticks.add(sim.tick())
            await asyncio.sleep(0)

    sources = [SlowSource() for _ in range(8)]
    cfg = make_config(max_ticks=100, tick_deadline_ms=10)
    hb = asyncio.create_task(heartbeat())
    try:
        result = await asyncio.wait_for(
            LockstepEngine(sim, cfg, sources).run(), timeout=30)
    finally:
        hb.cancel()
    assert result.final_tick == 100
    assert all(result.seat_dead)
    # The heartbeat must observe the all-dead stretch tick by tick. A
    # starved loop lets it see only the pre-dead ticks (and possibly the
    # final tick, at the engine's post-loop probe cleanup await).
    dead_stretch = {t for t in seen_ticks if 15 <= t < 100}
    assert len(dead_stretch) > 50, sorted(seen_ticks)


async def test_all_seats_dead_then_one_revives():
    """Revival must work even from the all-dead state: the probe tasks
    only run if the engine yields to the event loop each tick."""

    class WakingSource:
        def __init__(self, sleep_asks):
            self.asks = 0
            self.sleep_asks = sleep_asks

        async def get_actions(self, tick, obs, resets):
            self.asks += 1
            if self.asks <= self.sleep_asks:
                await asyncio.sleep(60)
            return [[1]]

    sim = FakeSim()
    sources = [SlowSource() for _ in range(7)] + [WakingSource(10)]
    cfg = make_config(max_ticks=60, tick_deadline_ms=10)
    result = await asyncio.wait_for(
        LockstepEngine(sim, cfg, sources).run(), timeout=30)
    assert result.final_tick == 60
    assert result.seat_dead[7] is False
    assert sim.fed_actions[-1][7].tolist() == [1]


async def test_progress_heartbeat_and_revival_logged(capsys):
    """Observability: a slow episode emits throttled progress lines, and
    a revival logs the seat + tick."""

    class WakingSource:
        def __init__(self, sleep_asks):
            self.asks = 0
            self.sleep_asks = sleep_asks

        async def get_actions(self, tick, obs, resets):
            self.asks += 1
            if self.asks <= self.sleep_asks:
                await asyncio.sleep(60)
            return [[1]]

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [WakingSource(10)]
    cfg = make_config(max_ticks=20, tick_deadline_ms=50)
    result = await LockstepEngine(
        sim, cfg, sources, progress_interval_seconds=0.05).run()
    assert result.final_tick == 20
    err = capsys.readouterr().err
    assert "progress: tick " in err
    assert "seat 7 revived at tick" in err


async def test_valid_action_resets_strike_counter():
    """Strikes are consecutive: a valid reply resets the count, so an
    intermittently slow seat is never marked dead."""

    class Intermittent:
        def __init__(self):
            self.asks = 0

        async def get_actions(self, tick, obs, resets):
            self.asks += 1
            if self.asks % 3 == 0:
                return [[2]]
            await asyncio.sleep(60)
            return None

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(7)] + [Intermittent()]
    cfg = make_config(max_ticks=12, tick_deadline_ms=30)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.seat_dead[7] is False
    assert result.final_tick == 12
