"""RLPD (Ball et al., 2023): SAC that learns from its own play and from other players' transitions at once, half of
every batch from each, with LayerNorm in the critics. Off-policy, so every transition is learned from many times,
and transitions the network never made itself count too: PPO only learns from its own play, and at the anvil's leap
the network has no landing of its own to learn from (its 52 "arrivals" at 543 were mid-air peaks), while Go-Explore
landed it 18 times. So: the offline half is the transitions of --offline demos (explore.py's paths, replayed) inside
the stretch; the online half is episodes as train.py's: from --demos' exported states in --demo-window, with the new
objective (falls end episodes, progress pays only where held, starts weighted by success).

The actor starts from a PPO specialist (same trunk; its tanh mean becomes SAC's tanh squashing); the critics start
fresh, so the actor is frozen for --critic-warmup steps while they catch up. Observation normalization is the
specialist's, fixed. Checkpoints are sac_<n>_steps.zip (train.Player plays them).

    python scripts/rlpd.py --warm runs/anvil-leap-m28-a/checkpoints/ppo_20003840_steps.zip --window 536 545 --demo-window 536 543 --demos runs/seeds/anvil-explore.npz --offline runs/demos/explore-anvil.npz --name rlpd-leap
"""
import argparse
import os
import pickle
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from goi_env import OBS_FIELDS, GoiEnv
from policy import LayerNormSACPolicy
from train import STALL_SECONDS, ProgressLog, demo_steps, make_env, vecnormalize_path

PROGRESS, LAKE, POT, HEAD = (OBS_FIELDS.index(f) for f in ("progress", "fell_in_lake", "pot_contact", "head_contact"))


class MixedBuffer(ReplayBuffer):
    """A replay buffer whose batches are half its own transitions, half `offline`'s (RLPD's symmetric sampling)."""
    offline = None

    def sample(self, batch_size, env=None):
        if self.offline is None:
            return super().sample(batch_size, env)
        own, path = super().sample(batch_size - batch_size // 2, env), self.offline.sample(batch_size // 2, env)
        return type(own)(*(None if a is None else torch.cat([a, b]) for a, b in zip(own, path)))


def path_transitions(game, path, lo, hi, buffer):
    """Every path's transitions inside [lo, hi) of route progress (explore.py's or a climb's paths, replayed), with the
    reward GoiEnv pays for them under the new objective (new ground where it is held, from the stretch's entry, minus
    the time penalty), into `buffer`. Returns how many."""
    z, n = np.load(path), 0
    for j in range(len(z["reached"])):
        held, prev = None, None
        for t, obs, a in demo_steps(game, z, j):
            if obs is None:
                continue
            p = float(obs[PROGRESS])
            touching = bool(obs[POT] or obs[HEAD])
            if prev is not None:
                o, act, best = prev
                reward = (max(0.0, p - best) if touching else 0.0) - game.time_penalty
                buffer.add(o[None], obs[None], act[None], np.array([reward]), np.array([obs[LAKE] > 0]), [{}])
                n += 1
            if a is None or p >= hi:
                break
            if p >= lo:
                held = p if held is None else max(held, p) if touching else held
                prev = (obs, np.asarray(a, np.float32), held)
            else:
                prev = None
    return n


def warm_actor(model, checkpoint):
    """The PPO specialist's actor trunk and mean layer as SAC's (both Tanh MLPs; PPO's mean is tanh(W h), SAC's
    deterministic action tanh(mu(h))); SAC's exploration starts at std 0.37 before the squashing."""
    ppo = PPO.load(checkpoint, device="cpu").policy
    model.actor.latent_pi.load_state_dict(ppo.mlp_extractor.policy_net.state_dict())
    mean = ppo.action_net[0] if isinstance(ppo.action_net, nn.Sequential) else ppo.action_net
    model.actor.mu.load_state_dict(mean.state_dict())
    with torch.no_grad():
        model.actor.log_std.weight.mul_(0.01)
        model.actor.log_std.bias.fill_(-1.0)


class FreezeActor(BaseCallback):
    """The actor's weights frozen for the first `steps` environment steps, while the fresh critics learn the values
    (SAC resets every optimizer's learning rate before each update, so a zero rate wouldn't hold)."""

    def __init__(self, steps):
        super().__init__()
        self.steps, self.frozen = steps, False

    def _on_training_start(self):
        self.model.actor.requires_grad_(False)
        self.frozen = True

    def _on_step(self):
        if self.frozen and self.num_timesteps >= self.steps:
            self.model.actor.requires_grad_(True)
            self.frozen = False
            print(f"actor unfrozen at {self.num_timesteps} steps", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", required=True, help="a PPO specialist checkpoint: the actor and the normalization")
    ap.add_argument("--window", type=float, nargs=2, required=True, metavar=("LO", "HI"), help="the stretch")
    ap.add_argument("--demo-window", type=float, nargs=2, metavar=("LO", "HI"),
                    help="the online starts' stretch and success (default: --window)")
    ap.add_argument("--demos", nargs="+", required=True, help="exported state files to start episodes from")
    ap.add_argument("--offline", nargs="+", required=True, help="demos files whose transitions are the offline half")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--utd", type=float, default=0.5, help="gradient steps per environment step")
    ap.add_argument("--critic-warmup", type=int, default=50_000, help="environment steps with the actor frozen")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--save-every", type=int, default=250_000)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    lo, hi = args.window
    demo_window = tuple(args.demo_window or args.window)
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    stats = os.path.join(run_dir, "demo-stats.npy")
    if not os.path.exists(stats):  # the games' shared success counts per start (train.py --demo-priority)
        np.lib.format.open_memmap(stats, mode="w+", dtype=np.float64, shape=(200_000, 3)).flush()

    stall = int(STALL_SECONDS * 30)
    env = partial(make_env, args.game_dir, 4, 1800, stall, 0.2, 1.0, traps=0.2, window=(lo - 6, hi + 5),
                  demos=args.demos, demo_share=0.6, demo_window=demo_window, demo_priority=True, falls_terminal=True,
                  demo_stats=stats, held_records=True)
    with open(vecnormalize_path(args.warm), "rb") as f:
        venv = pickle.load(f)
    venv.set_venv(VecMonitor(SubprocVecEnv([env] * args.envs), info_keywords=("best_progress", "from_bottom", "rewinds")))
    venv.training, venv.norm_reward = False, False  # the specialist's observation scale, fixed; raw rewards
    model = SAC(LayerNormSACPolicy, venv, gamma=args.gamma, buffer_size=1_000_000, batch_size=256, learning_starts=10_000,
                train_freq=1, gradient_steps=max(1, int(args.utd * args.envs)), replay_buffer_class=MixedBuffer,
                policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.Tanh), device="cuda",
                tensorboard_log=run_dir, verbose=1)
    warm_actor(model, args.warm)

    game = GoiEnv(args.game_dir, stall_steps=stall, stall_ends=False, snake_ends=True)
    try:
        offline = ReplayBuffer(500_000, venv.observation_space, venv.action_space, model.device)
        n = sum(path_transitions(game, f, lo, hi, offline) for f in args.offline)
        print(f"{n} offline transitions in {lo:.0f}-{hi:.0f}", flush=True)
    finally:
        game.close()
    model.replay_buffer.offline = offline

    save = CheckpointCallback(max(args.save_every // args.envs, 1), os.path.join(run_dir, "checkpoints"),
                              name_prefix="sac", save_vecnormalize=True)
    try:
        model.learn(args.steps, callback=[save, ProgressLog(), FreezeActor(args.critic_warmup)], tb_log_name="tb")
    except KeyboardInterrupt:
        print("interrupted: saving a final checkpoint")
    final = os.path.join(run_dir, "checkpoints", f"sac_{model.num_timesteps}_steps.zip")
    model.save(final)
    venv.save(vecnormalize_path(final))
    venv.close()
    print(f"saved {final}")


if __name__ == "__main__":
    main()
