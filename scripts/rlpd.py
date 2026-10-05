"""RLPD (Ball et al., 2023): SAC that learns from its own play and from a climb path's transitions at once, half of
every batch from each, with LayerNorm in the critics. Off-policy, so every transition is learned from many times
instead of once as in PPO: the bet is fewer game steps for the same skill, where PPO learns hard sections slowly.

Same episodes as train.py's path curriculum (starts on the path near a stretch's end, moving back as they succeed;
rewinds after falls; traps). The actor starts from a PPO specialist (same trunk; its tanh mean becomes SAC's tanh
squashing); the critics start fresh, so the actor is frozen for --critic-warmup steps while they catch up. Observation
normalization is the specialist's, fixed. Checkpoints are sac_<n>_steps.zip (train.Player plays them).

    python scripts/rlpd.py --warm runs/chx-rewind/checkpoints/ppo_18000000_steps.zip --window 415 465 --name rlpd-chapel
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

PROGRESS, LAKE = OBS_FIELDS.index("progress"), OBS_FIELDS.index("fell_in_lake")


class MixedBuffer(ReplayBuffer):
    """A replay buffer whose batches are half its own transitions, half `offline`'s (RLPD's symmetric sampling)."""
    offline = None

    def sample(self, batch_size, env=None):
        if self.offline is None:
            return super().sample(batch_size, env)
        own, path = super().sample(batch_size - batch_size // 2, env), self.offline.sample(batch_size // 2, env)
        return type(own)(*(None if a is None else torch.cat([a, b]) for a, b in zip(own, path)))


def path_transitions(game, path, lo, hi, buffer):
    """The path's transitions inside [lo, hi) of route progress, with the reward GoiEnv pays for them (new records
    from the stretch's entry, minus the time penalty), into `buffer`. Returns how many."""
    z, best, prev = np.load(path), None, None
    n = 0
    for t, obs, a in demo_steps(game, z, 0):
        p = float(obs[PROGRESS])
        if prev is not None:
            o, act, bp = prev
            reward = max(0.0, p - bp) - game.time_penalty
            buffer.add(o[None], obs[None], act[None], np.array([reward]), np.array([obs[LAKE] > 0]), [{}])
            n += 1
        if a is None or p >= hi:
            break
        if p >= lo:
            best = p if best is None else max(best, p)
            prev = (obs, np.asarray(a, np.float32), best)
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
    ap.add_argument("--path", default=os.path.join("runs", "climbs", "win-e4.npz"), help="a climb path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--utd", type=float, default=1.0, help="gradient steps per environment step")
    ap.add_argument("--critic-warmup", type=int, default=50_000, help="environment steps with the actor frozen")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--save-every", type=int, default=250_000)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    lo, hi = args.window
    run_dir = os.path.join("runs", args.name)

    stall = int(STALL_SECONDS * 30)
    env = partial(make_env, args.game_dir, 4, 1800, stall, 0.2, 1.0, traps=0.2, window=(lo - 6, hi + 5),
                  demos=args.path, demo_share=0.6, demo_window=(lo, hi), demo_curriculum=True)
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
        offline = ReplayBuffer(200_000, venv.observation_space, venv.action_space, model.device)
        print(f"{path_transitions(game, args.path, lo, hi, offline)} path transitions in {lo:.0f}-{hi:.0f}", flush=True)
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
