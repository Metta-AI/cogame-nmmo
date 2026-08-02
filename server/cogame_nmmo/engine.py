"""Transport-free lockstep episode engine.

Runs one NMMO3 episode against per-seat async action sources. Per tick:
slice per-seat observations, gather every seat's actions concurrently
under the config tick deadline, NOOP-fill anything missing/late/
malformed, feed the sim, step, accumulate rewards. A seat can never
crash the episode: any exception, timeout, or bad payload from a source
degrades to the no-op action for that seat's agents.

Each ``get_actions`` call also carries per-agent ``resets`` flags: True
when that agent's terminal fired on the previous sim step (attack death
or 500-tick stagnation reset, respawn in place) — i.e. the obs delivered
alongside is the first of a new life, and recurrent policies must zero
that agent's state (protocol v2 ``resets``; see docs/PROTOCOL.md).

The optional ``on_tick(tick, actions)`` callback receives the post-clamp
(num_agents, 1) uint8 action matrix exactly as fed to the sim — the
replay writer hooks here.

Scoring: NMMO3 has no env termination — episodes end at the tick cap
(or the wall-clock budget / a sim fault). Seats are ranked by score:
per-agent cumulative min(comb_lvl, prof_lvl) over ended lives plus the
current life's min (sim/shim.c ``nmmo_score``), summed over a seat's
agents. The shim's cumulative accessors are stateful, so the engine
reads scores ONCE at episode end (a live standings feed can read
``sim.score`` any time; the server's /global broadcast does).

Strike rule (bounds worst-case wall clock): a seat that fails to deliver
a valid action for ``STRIKE_LIMIT`` consecutive ticks is marked dead —
subsequent ticks apply NOOP for it instantly instead of waiting out the
tick deadline, so a silent seat costs at most ~strike_limit x deadline
of wall clock for the whole episode instead of deadline x max_ticks.
A dead seat is still probed each tick with a background (non-blocking)
``get_actions`` call carrying that tick's obs; the first valid reply is
applied, resets the strike counter, and revives the seat — so a late
reconnect resumes normal play. Per-seat NOOP-tick counts and end-of-
episode dead flags are reported on the EpisodeResult for observability.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import numpy as np

from . import defaults
from .config import GameConfig

# results agent_stats keys -> sim agent_stat() `which` codes. Mirrors
# sim/shim.c (duplicated so transport code needs no wasmtime import;
# tested equal to cogame_nmmo.sim.STAT_* in tests/test_engine.py).
STAT_CODES = {
    "cum_min_comb_prof": 0,   # sum of min(comb,prof) over ended lives
    "deaths": 1,              # ended lives (attack + stagnation)
    "comb_lvl": 2,            # final life's combat level
    "prof_lvl": 3,            # final life's profession level
    "gold": 5,
    "time_alive": 6,          # ticks since the final life spawned
}

# Closed enum (triple-sync rule): these values must match the manifest
# results_schema end_reason enum and any docker_smoke.sh expectations.
# NMMO3 has no env-side termination, so there is no "won" end state.
EndReason = Literal["tick_cap", "wall_clock", "sim_fault"]
END_REASON_TICK_CAP: EndReason = "tick_cap"
END_REASON_WALL_CLOCK: EndReason = "wall_clock"
END_REASON_SIM_FAULT: EndReason = "sim_fault"

# Consecutive invalid/missing ticks before a seat is marked dead (see the
# strike rule in the module docstring).
STRIKE_LIMIT = 10

# Per-seat NOOP-fallback cause taxonomy (results `noop_causes`):
#   timeout       deadline elapsed with no reply (incl. dead-seat ticks
#                 whose revival probe is still outstanding)
#   malformed     a reply arrived but failed shape/value sanitization
#   wrong_tick    messages answering a different tick (counted at the
#                 transport by WsSeat.deliver; such a tick itself also
#                 counts as timeout — wrong_tick is a message count)
#   disconnected  the source had nothing to offer (ws seat not connected)
#   host_error    the source raised (logged with type+message)
NOOP_CAUSES = ("timeout", "malformed", "wrong_tick", "disconnected",
               "host_error")

# Mid-episode stderr progress heartbeat: at most one line per interval
# (an operator tailing logs can tell a live slow episode from a hang).
PROGRESS_INTERVAL_SECONDS = 30.0


class ActionSource(Protocol):
    """Per-seat action provider (websocket seat, scripted bot, ...)."""

    async def get_actions(
            self, tick: int, obs: np.ndarray, resets: Sequence[bool]
    ) -> Sequence[Sequence[int]] | None:
        """Actions for this seat's agents at ``tick``.

        ``obs`` is the (heroes_per_seat, 1707) uint8 slice for this
        seat's agents in pid order; ``resets`` is the matching per-agent
        list of done flags from the previous step (all False at tick 0).
        Returns a (heroes_per_seat, 1)-shaped nested sequence of action
        values, or None to play NOOP this tick.
        """
        ...


@dataclass(frozen=True)
class EpisodeResult:
    end_reason: EndReason
    seat_scores: tuple[float, ...]        # final score per seat, higher=better
    seat_reward_sums: tuple[float, ...]   # sim reward sums per seat
    agent_stats: tuple[dict, ...]         # num_agents dicts keyed by STAT_CODES
    final_tick: int
    state_digest: int                     # sim state digest at episode end
    seat_noop_ticks: tuple[int, ...]      # ticks each seat played NOOP fallback
    seat_dead: tuple[bool, ...]           # strike-rule dead flag at episode end
    seat_noop_causes: tuple[dict, ...]    # per-seat counts keyed by NOOP_CAUSES


class LockstepEngine:
    def __init__(self, sim, config: GameConfig,
                 action_sources: Sequence[ActionSource],
                 on_tick: Callable[[int, np.ndarray], None] | None = None,
                 strike_limit: int = STRIKE_LIMIT,
                 on_seat_dead: Callable[[int], None] | None = None,
                 progress_interval_seconds: float = PROGRESS_INTERVAL_SECONDS):
        if len(action_sources) != config.num_seats:
            raise ValueError(
                f"need {config.num_seats} action sources, "
                f"got {len(action_sources)}")
        self._sim = sim
        self._config = config
        self._sources = list(action_sources)
        self._on_tick = on_tick
        self._strike_limit = strike_limit
        # Called with the seat index each time a seat transitions to
        # strike-dead (once per dead spell). The server force-closes the
        # seat's (possibly half-open) websocket here so the client sees
        # the close and reconnects instead of feeding a black hole.
        self._on_seat_dead = on_seat_dead
        # per-seat pid slices, bound once (obs, actions and rewards all
        # index agents by pid rows)
        self._seat_slices = [
            slice(pids.start, pids.stop)
            for pids in (defaults.seat_hero_pids(s, config.heroes_per_seat)
                         for s in range(config.num_seats))]
        self._strikes = [0] * config.num_seats
        self._noop_ticks = [0] * config.num_seats
        self._noop_causes = [dict.fromkeys(NOOP_CAUSES, 0)
                             for _ in range(config.num_seats)]
        self._host_error_logged = [False] * config.num_seats
        self._probes: list[asyncio.Task | None] = [None] * config.num_seats
        self._wall_clock_expired = False
        # Patch-0002 containment: set when the sim reports a fault flag
        # or a sim call raises (wasmtime trap). The episode then ends
        # with end_reason "sim_fault" and best-effort result fields, so
        # results and the partial replay still get written.
        self._sim_fault = False
        self._ticks_run = 0
        self._progress_interval = progress_interval_seconds
        # Previous step's per-agent done flags: the protocol v2 `resets`
        # delivered with the NEXT tick's obs (all False at tick 0).
        self._last_dones = [False] * config.num_agents

    async def run(self) -> EpisodeResult:
        sim = self._sim
        cfg = self._config
        h = cfg.heroes_per_seat
        deadline = cfg.tick_deadline_ms / 1000.0
        reward_sums = np.zeros(cfg.num_seats, dtype=np.float64)
        noop_row = np.asarray(defaults.NOOP_ACTION, dtype=np.uint8)

        start = time.monotonic()
        last_progress = start
        fault_fn = getattr(sim, "fault", None)
        try:
            while True:
                now = time.monotonic()
                if now - last_progress >= self._progress_interval:
                    last_progress = now
                    print(f"progress: tick {self._ticks_run}/"
                          f"{cfg.max_ticks}, elapsed {now - start:.0f}s, "
                          f"noops={self._noop_ticks}, "
                          f"strikes={self._strikes}", file=sys.stderr)
                if self._ticks_run >= cfg.max_ticks:
                    break
                if time.monotonic() - start >= cfg.wall_clock_budget_seconds:
                    # Hard stop under the platform's episode_timeout kill:
                    # end the episode now (end_reason="wall_clock") so
                    # results and the partial replay still get written.
                    self._wall_clock_expired = True
                    break
                # Sim calls are containment boundaries (patch 0002): a
                # wasmtime trap or a raised fault flag ends the episode
                # as "sim_fault" instead of crashing the process.
                try:
                    tick = sim.tick()
                    obs = sim.observations()
                except Exception:
                    self._sim_fault = True
                    break

                live = [s for s in range(cfg.num_seats)
                        if self._strikes[s] < self._strike_limit]
                if not live:
                    # With every seat dead the gather below is empty and
                    # awaits nothing: without this explicit yield the loop
                    # starves the event loop (stalled /healthz -> liveness
                    # kill; revival probes never even run).
                    await asyncio.sleep(0)
                gathered = await asyncio.gather(*(
                    self._seat_actions(
                        s, tick, obs[self._seat_slices[s]],
                        self._seat_resets(s), deadline)
                    for s in live))
                replies: list = [(None, "timeout")] * cfg.num_seats
                for s, reply_cause in zip(live, gathered):
                    replies[s] = reply_cause
                for s in range(cfg.num_seats):
                    if self._strikes[s] >= self._strike_limit:
                        replies[s] = self._poll_dead_seat(
                            s, tick, obs[self._seat_slices[s]],
                            self._seat_resets(s))

                actions = np.tile(noop_row, (cfg.num_agents, 1))
                for seat, (reply, cause) in enumerate(replies):
                    sanitized = _sanitize(reply, h)
                    if sanitized is not None:
                        actions[self._seat_slices[seat]] = sanitized
                        if self._strikes[seat] >= self._strike_limit:
                            print(f"seat {seat} revived at tick {tick} "
                                  f"(valid action after "
                                  f"{self._strikes[seat]} strikes)",
                                  file=sys.stderr)
                        self._strikes[seat] = 0  # valid action: reset/revive
                        # Drop any outstanding probe: a revival leaves the
                        # probe created on the harvest tick behind, and it
                        # can hang forever (e.g. a ws waiter clobbered by
                        # the next tick's send). Left in place it would
                        # block re-probing in a later dead spell.
                        probe = self._probes[seat]
                        if probe is not None:
                            probe.cancel()
                            self._probes[seat] = None
                    else:
                        self._strikes[seat] += 1
                        self._noop_ticks[seat] += 1
                        # reply arrived but failed sanitization
                        cause = "malformed" if reply is not None else \
                            (cause or "timeout")
                        self._noop_causes[seat][cause] += 1
                        if self._noop_causes[seat][cause] == 1:
                            print(f"seat {seat}: first '{cause}' NOOP "
                                  f"fallback at tick {tick}",
                                  file=sys.stderr)
                        if self._strikes[seat] == self._strike_limit \
                                and self._on_seat_dead is not None:
                            try:
                                self._on_seat_dead(seat)
                            except Exception:
                                pass  # observer hook: never crash the episode

                try:
                    sim.set_actions(actions.astype(np.float32))
                    sim.step()
                except Exception:
                    self._sim_fault = True
                    break  # tick did not complete: not recorded
                self._ticks_run += 1

                try:
                    rewards = np.asarray(sim.rewards(), dtype=np.float64)
                    self._last_dones = list(sim.dones())
                except Exception:
                    self._sim_fault = True
                    break
                for seat in range(cfg.num_seats):
                    reward_sums[seat] += float(
                        rewards[self._seat_slices[seat]].sum())

                if self._on_tick is not None:
                    self._on_tick(tick, actions)

                if fault_fn is not None:
                    # The faulting tick completed (the guard bailed out of
                    # a local operation, not the step), so it was recorded
                    # above; end the episode here.
                    try:
                        faulted = bool(fault_fn())
                    except Exception:
                        faulted = True
                    if faulted:
                        self._sim_fault = True
                        break
        finally:
            for probe in self._probes:
                if probe is not None and not probe.done():
                    probe.cancel()
            await asyncio.gather(
                *(p for p in self._probes if p is not None),
                return_exceptions=True)

        return self._build_result(reward_sums)

    def _seat_resets(self, seat: int) -> list[bool]:
        """One seat's slice of the previous step's done flags."""
        s = self._seat_slices[seat]
        return self._last_dones[s.start:s.stop]

    def _poll_dead_seat(self, seat: int, tick: int, seat_obs: np.ndarray,
                        seat_resets: list[bool]):
        """Non-blocking revival path for a dead seat.

        Harvests the previous background probe if it finished (its reply,
        if valid, revives the seat this tick — the reply was computed for
        the obs of the tick the probe was launched on, so a slow reply is
        harvested arbitrarily many ticks stale) and keeps at most one
        probe outstanding. Never awaits: dead seats cost no wall clock.
        """
        # Probe still outstanding: the seat has not answered in time.
        reply_cause = (None, "timeout")
        probe = self._probes[seat]
        if probe is not None and probe.done():
            self._probes[seat] = None
            if not probe.cancelled() and probe.exception() is None:
                reply_cause = probe.result()
            probe = None
        if probe is None:
            # On a revival harvest this new probe is created and then
            # immediately cancelled by run()'s valid-action handling —
            # intentional: creation here keeps this path branch-free.
            self._probes[seat] = asyncio.create_task(
                self._probe_seat(seat, tick, seat_obs, seat_resets))
        return reply_cause

    async def _probe_seat(self, seat: int, tick: int,
                          seat_obs: np.ndarray, seat_resets: list[bool]):
        """Un-deadlined get_actions for revival probes.

        Returns ``(reply, cause)`` like _seat_actions; never raises
        (except cancellation).
        """
        try:
            reply = await self._sources[seat].get_actions(
                tick, seat_obs, seat_resets)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_host_error(seat, exc)
            return None, "host_error"
        if reply is None:
            return None, "disconnected"
        return reply, None

    async def _seat_actions(self, seat: int, tick: int,
                            seat_obs: np.ndarray, seat_resets: list[bool],
                            deadline: float):
        """One seat's ``(reply, cause)``; degrade, never crash.

        ``cause`` (a NOOP_CAUSES key) is set when reply is None; the run
        loop overrides it with "malformed" when a present reply fails
        sanitization.
        """
        try:
            reply = await asyncio.wait_for(
                self._sources[seat].get_actions(tick, seat_obs, seat_resets),
                deadline)
        except asyncio.TimeoutError:
            return None, "timeout"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_host_error(seat, exc)
            return None, "host_error"
        if reply is None:
            # ws seats return None when no connection is up
            return None, "disconnected"
        return reply, None

    def _log_host_error(self, seat: int, exc: Exception) -> None:
        """Swallowed source exceptions are logged (once per seat)."""
        if self._host_error_logged[seat]:
            return
        self._host_error_logged[seat] = True
        print(f"seat {seat}: action source raised "
              f"{type(exc).__name__}: {exc} (degrading to NOOP; "
              f"further errors from this seat are counted, not logged)",
              file=sys.stderr)

    def _seat_score(self, seat: int) -> float:
        """A seat's final score: sim.score summed over its agents."""
        s = self._seat_slices[seat]
        return float(sum(self._sim.score(pid)
                         for pid in range(s.start, s.stop)))

    def _build_result(self, reward_sums: np.ndarray) -> EpisodeResult:
        sim = self._sim
        cfg = self._config

        if self._sim_fault:
            return self._build_fault_result(reward_sums)

        end_reason = END_REASON_WALL_CLOCK if self._wall_clock_expired \
            else END_REASON_TICK_CAP

        agent_stats = tuple(
            {name: int(sim.agent_stat(pid, which))
             for name, which in STAT_CODES.items()}
            for pid in range(cfg.num_agents))

        return EpisodeResult(
            end_reason=end_reason,
            seat_scores=tuple(self._seat_score(s)
                              for s in range(cfg.num_seats)),
            seat_reward_sums=tuple(float(r) for r in reward_sums),
            agent_stats=agent_stats,
            final_tick=int(sim.tick()),
            state_digest=int(sim.state_digest()),
            seat_noop_ticks=tuple(self._noop_ticks),
            seat_dead=tuple(
                s >= self._strike_limit for s in self._strikes),
            seat_noop_causes=self._final_noop_causes(),
        )

    def _final_noop_causes(self) -> tuple[dict, ...]:
        """Engine-attributed cause counts + the transport's wrong_tick
        message counts (WsSeat exposes wrong_tick_count; other sources
        have no transport layer and contribute 0)."""
        causes = []
        for seat, counts in enumerate(self._noop_causes):
            merged = dict(counts)
            merged["wrong_tick"] += int(getattr(
                self._sources[seat], "wrong_tick_count", 0))
            causes.append(merged)
        return tuple(causes)

    def _build_fault_result(self, reward_sums: np.ndarray) -> EpisodeResult:
        """Best-effort result for a sim fault: end_reason "sim_fault",
        all scores 0.0 (equal, i.e. a draw for ranking consumers — an
        infra fault is nobody's loss). Sim reads fall back to zeros — a
        trapped wasm instance raises on every access, but a fault-flag
        episode (patch 0002) usually has readable state.
        """
        sim = self._sim
        cfg = self._config

        def safe(fn, fallback):
            try:
                return fn()
            except Exception:
                return fallback

        agent_stats = tuple(
            {name: int(safe(lambda p=pid, w=which: sim.agent_stat(p, w), 0))
             for name, which in STAT_CODES.items()}
            for pid in range(cfg.num_agents))
        return EpisodeResult(
            end_reason=END_REASON_SIM_FAULT,
            seat_scores=tuple([0.0] * cfg.num_seats),
            seat_reward_sums=tuple(float(r) for r in reward_sums),
            agent_stats=agent_stats,
            final_tick=int(safe(sim.tick, self._ticks_run)),
            state_digest=int(safe(sim.state_digest, 0)),
            seat_noop_ticks=tuple(self._noop_ticks),
            seat_dead=tuple(
                s >= self._strike_limit for s in self._strikes),
            seat_noop_causes=self._final_noop_causes(),
        )


def _sanitize(reply, heroes_per_seat: int) -> np.ndarray | None:
    """Validate one seat's reply into (h, 1) uint8, or None if malformed."""
    if reply is None:
        return None
    try:
        arr = np.asarray(reply, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (heroes_per_seat, defaults.ACTIONS_PER_AGENT):
        return None
    if not np.isfinite(arr).all():
        return None
    return defaults.clamp_actions(arr)
