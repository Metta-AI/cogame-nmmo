"""Score-aligned PPO fine-tune of the pretrained MMONet.

Env: the natively-compiled LEAGUE sim (bit-exact vs the wasm the league
runs, musl rand_r). Reward is computed HERE, python-side, from shim
stats — no C patch, no obs drift between training and deployment:

    r_t = W_MIN * max(0, min(comb,prof) - best_min_this_life)
        + W_DEATH * terminal          (attack death AND stagnation bank)
        + W_SHAPE * upstream_reward   (item/level shaping as a prior)

This is the league seat-score gradient: banked min is the numerator,
every terminal grows the divisor, and the upstream shaping (which the
demo net was trained on) is kept at low weight as a behavioral prior.

Recurrent PPO, truncated BPTT over the rollout horizon, fine-tune from
nmmo3_weights.bin. Value head warms up alone first (the reward scale
and gamma changed), then joint updates.

Usage:
  train/.venv/bin/python train/ppo.py [--steps 1e9] [--out train/ckpt]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.mmonet import MMONet, load_puffer_bin, save_puffer_bin, \
    N_LAYERS, HIDDEN, OBS_SIZE  # noqa: E402
from train.native_env import NativeNmmo  # noqa: E402

WEIGHTS = "vendor/upstream/resources/nmmo3/nmmo3_weights.bin"

# reward
W_MIN = 1.0
W_DEATH = -3.0
W_SHAPE = 0.2

# ppo
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
GRAD_NORM = 0.6
LR = 1e-4
HORIZON = 64
EPOCHS = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=1e9)
    ap.add_argument("--agents", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="train/ckpt")
    ap.add_argument("--init", default=WEIGHTS)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30,
                    help="value-only updates before the policy moves")
    ap.add_argument("--mb", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cpu")
    torch.manual_seed(args.seed)

    B = args.agents
    env = NativeNmmo(seed=args.seed, num_agents=B)
    net = load_puffer_bin(args.init).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    state = net.initial_state(B, device=dev)
    # per-agent scoring shadow for the aligned reward
    best_min = np.ones(B, dtype=np.int64)

    def stats_now():
        """(comb, prof, hp) per agent."""
        s = np.empty((B, 3), dtype=np.int64)
        for i in range(B):
            s[i, 0] = env.stat(i, 2)
            s[i, 1] = env.stat(i, 3)
            s[i, 2] = env.stat(i, 7)
        return s

    total_steps = 0
    update = 0
    t_start = time.time()
    ep_deaths = 0
    ep_min_gain = 0.0
    log_path = out / "train.log"

    obs_buf = torch.zeros(HORIZON, B, OBS_SIZE, dtype=torch.uint8)
    act_buf = torch.zeros(HORIZON, B, dtype=torch.long)
    logp_buf = torch.zeros(HORIZON, B)
    val_buf = torch.zeros(HORIZON, B)
    rew_buf = torch.zeros(HORIZON, B)
    done_buf = torch.zeros(HORIZON, B)

    while total_steps < args.steps:
        roll_state0 = state.detach().clone()
        with torch.no_grad():
            for t in range(HORIZON):
                obs_np = env.obs.copy()
                obs = torch.from_numpy(obs_np)
                obs_buf[t] = obs
                o = obs.to(dev)
                logits, value, state = net(o, state)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp_buf[t] = dist.log_prob(act).cpu()
                val_buf[t] = value.cpu()
                act_buf[t] = act.cpu()

                env.step(act_buf[t].numpy().astype(np.float32))

                term = env.term.copy()          # 1.0 on life end
                done = torch.from_numpy((term > 0.5).astype(np.float32))
                done_buf[t] = done
                # aligned reward. Corpses keep their levels until the
                # respawn tick (hp==0) - freeze the shadow there or the
                # dead agent's old levels mint spurious gains against
                # the just-reset best_min.
                snow = stats_now()
                mn = np.minimum(snow[:, 0], snow[:, 1])
                alive = snow[:, 2] > 0
                gain = np.where(alive & (term <= 0.5),
                                np.maximum(0, mn - best_min), 0)
                best_min = np.where(
                    term > 0.5, 1,
                    np.where(alive, np.maximum(best_min, mn), best_min))
                r = (W_MIN * gain
                     + W_DEATH * (term > 0.5)
                     + W_SHAPE * env.rew.copy())
                rew_buf[t] = torch.from_numpy(r.astype(np.float32))
                ep_deaths += int((term > 0.5).sum())
                ep_min_gain += float(gain.sum())
                # reset recurrent state where the life ended
                if done.any():
                    idx = torch.nonzero(done).squeeze(1).to(dev)
                    state[:, idx, :] = 0.0

            # bootstrap value
            o = torch.from_numpy(env.obs.copy()).to(dev)
            _, last_val, _ = net(o, state)
            last_val = last_val.cpu()

        # GAE
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(B)
        for t in reversed(range(HORIZON)):
            nonterm = 1.0 - done_buf[t]
            next_val = last_val if t == HORIZON - 1 else val_buf[t + 1]
            delta = rew_buf[t] + GAMMA * next_val * nonterm - val_buf[t]
            last_gae = delta + GAMMA * LAMBDA * nonterm * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # PPO update (recurrent: replay sequences from rollout-start state)
        pg_on = update >= args.warmup
        mb_agents = min(args.mb, B)
        idx_all = torch.randperm(B)
        losses = []
        for e in range(EPOCHS):
            for mb0 in range(0, B, mb_agents):
                mb = idx_all[mb0:mb0 + mb_agents]
                st = roll_state0[:, mb, :].to(dev)
                adv = adv_buf[:, mb].to(dev)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ret = ret_buf[:, mb].to(dev)
                old_logp = logp_buf[:, mb].to(dev)
                pg_loss_t = torch.zeros((), device=dev)
                v_loss_t = torch.zeros((), device=dev)
                ent_t = torch.zeros((), device=dev)
                for t in range(HORIZON):
                    o = obs_buf[t, mb].to(dev)
                    logits, value, st = net(o, st)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = act_buf[t, mb].to(dev)
                    logp = dist.log_prob(a)
                    ratio = (logp - old_logp[t]).exp()
                    s1 = ratio * adv[t]
                    s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[t]
                    pg_loss_t = pg_loss_t - torch.min(s1, s2).mean()
                    v_loss_t = v_loss_t + F.mse_loss(value, ret[t])
                    ent_t = ent_t + dist.entropy().mean()
                    # cut BPTT at life ends, like the rollout did
                    d = done_buf[t, mb].to(dev)
                    if d.any():
                        st = st * (1.0 - d).view(1, -1, 1)
                pg_loss = pg_loss_t / HORIZON
                v_loss = v_loss_t / HORIZON
                ent = ent_t / HORIZON
                loss = VF_COEF * v_loss - ENT_COEF * ent
                if pg_on:
                    loss = loss + pg_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if not pg_on:
                    # warmup trains ONLY the value row: the shared trunk
                    # must not drift off the pretrained policy while the
                    # value head re-fits the new reward scale/gamma
                    for name, p in net.named_parameters():
                        if p.grad is None:
                            continue
                        if name == "decoder.weight":
                            p.grad[:26].zero_()
                        else:
                            p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               GRAD_NORM)
                opt.step()
                losses.append((float(pg_loss.detach()),
                               float(v_loss.detach()),
                               float(ent.detach())))

        state = state.detach()
        total_steps += HORIZON * B
        update += 1

        if update % 5 == 0:
            sps = total_steps / (time.time() - t_start)
            pl = np.mean([x[0] for x in losses])
            vl = np.mean([x[1] for x in losses])
            en = np.mean([x[2] for x in losses])
            score = np.mean([env.score(i) / (env.stat(i, 1) + 1)
                             for i in range(min(B, 64))])
            line = (f"upd {update} steps {total_steps/1e6:.1f}M "
                    f"sps {sps/1000:.0f}k pg {pl:.4f} v {vl:.3f} "
                    f"ent {en:.3f} deaths/1k "
                    f"{1000*ep_deaths/(5*HORIZON*B):.2f} "
                    f"mingain/1k {1000*ep_min_gain/(5*HORIZON*B):.3f} "
                    f"scoreproxy {score:.2f} "
                    f"{'PG' if pg_on else 'warmup'}")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            ep_deaths = 0
            ep_min_gain = 0.0

        if update % args.ckpt_every == 0:
            torch.save(net.state_dict(), out / f"net_{update:06d}.pt")
            save_puffer_bin(net, str(out / f"weights_{update:06d}.bin"))
            print(f"checkpoint {update} saved", flush=True)

    torch.save(net.state_dict(), out / "net_final.pt")
    save_puffer_bin(net, str(out / "weights_final.bin"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
