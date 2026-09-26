"""Train PPO on Getting Over It: parallel games, normalized observations, TensorBoard logs, resumable checkpoints."""
import argparse
import os
import re
import time
from contextlib import suppress
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from goi_env import OBS_FIELDS, GoiEnv

PROGRESS, HEAD, POT = (OBS_FIELDS.index(f) for f in ("progress", "head_contact", "pot_contact"))


class SavedStarts(gym.Wrapper):
    """Starts a share of episodes from states saved along the route instead of the bottom (Go-Explore style), so
    practice goes to the hard parts rather than to replaying the easy start.

    Each game keeps one state per bin_size of route progress, saved the first time an episode gets there with the pot or
    the hammer touching something (not mid-fall). Starts favor the states started from least (weight 1/sqrt(1 + starts)),
    so new ground near the frontier gets most of the practice. Infos carry from_bottom (1 or 0).
    """

    def __init__(self, env, share=0.5, bin_size=2.0):
        super().__init__(env)
        self.share, self.bin_size = share, bin_size
        self.states = {}  # progress bin -> [state id, starts]; ponytail: the first state per bin is kept for good
        self.rng = np.random.default_rng()
        self.from_bottom = 1

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self.from_bottom = int(not self.states or self.rng.random() >= self.share)
        if not self.from_bottom:
            bins = list(self.states)
            weights = np.array([1 / np.sqrt(1 + self.states[b][1]) for b in bins])
            b = bins[self.rng.choice(len(bins), p=weights / weights.sum())]
            self.states[b][1] += 1
            options["state"] = self.states[b][0]
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {**info, "from_bottom": self.from_bottom}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        b = int(obs[PROGRESS] // self.bin_size)
        if self.share and b not in self.states and not terminated and (obs[HEAD] or obs[POT]):
            self.states[b] = [self.env.unwrapped.save_state(), 0]
        return obs, reward, terminated, truncated, {**info, "from_bottom": self.from_bottom}


def make_env(game_dir, k, episode_steps, stall_steps, saved_starts):
    env = SavedStarts(GoiEnv(game_dir, k=k, stall_steps=stall_steps), saved_starts)
    return TimeLimit(env, episode_steps)


def vecnormalize_path(checkpoint):
    """The observation/reward normalization stats saved next to a checkpoint."""
    return re.sub(r"ppo_(\d+)_steps\.zip$", r"ppo_vecnormalize_\1_steps.pkl", checkpoint)


class ProgressLog(BaseCallback):
    """TensorBoard: best route progress per episode started from the bottom (mean over recent episodes), and the
    furthest any episode got."""

    def __init__(self):
        super().__init__()
        self.record = float("-inf")

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        episodes = self.model.ep_info_buffer
        if episodes:
            self.record = max(self.record, *(ep["best_progress"] for ep in episodes))
            self.logger.record("rollout/best_progress_record", self.record)
        bottom = [ep["best_progress"] for ep in episodes if ep["from_bottom"]]
        if bottom:
            self.logger.record("rollout/best_progress_mean", sum(bottom) / len(bottom))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--name", default=time.strftime("ppo-%Y%m%d-%H%M%S"), help="run folder under runs/")
    ap.add_argument("--steps", type=int, default=10_000_000, help="environment steps to train for")
    ap.add_argument("--envs", type=int, default=5, help="games in parallel (at most 5 per install)")
    ap.add_argument("--k", type=int, default=4, help="physics steps (1/120 s) per action")
    ap.add_argument("--episode-steps", type=int, default=3000, help="time limit in actions (3000 = 100 s at k=4)")
    ap.add_argument("--stall-seconds", type=float, default=10,
                    help="end an episode after this long without a new progress record (0: never)")
    ap.add_argument("--saved-starts", type=float, default=0.5,
                    help="share of episodes that start from states saved along the route instead of the bottom")
    ap.add_argument("--save-every", type=int, default=500_000, help="checkpoint interval in environment steps")
    ap.add_argument("--resume", help="a checkpoint .zip to continue from")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    run_dir = os.path.join("runs", args.name)
    checkpoints = os.path.join(run_dir, "checkpoints")
    stall_steps = int(args.stall_seconds * 120 / args.k) or None
    envs = [partial(make_env, args.game_dir, args.k, args.episode_steps, stall_steps, args.saved_starts)] * args.envs
    venv = VecMonitor(SubprocVecEnv(envs), info_keywords=("best_progress", "from_bottom"))
    try:
        if args.resume:
            venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
            model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=run_dir)
        else:
            venv = VecNormalize(venv, gamma=0.995)
            # target_kl: stop each update's epochs early once the policy has moved enough. Without it, action noise
            # collapsed (std 0.55 -> 0.03), updates grew (KL > 1) and performance regressed after ~1M steps.
            model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=512, gamma=0.995, target_kl=0.02,
                        policy_kwargs=dict(net_arch=[256, 256]), device="cpu", tensorboard_log=run_dir, verbose=1)
        save = CheckpointCallback(max(args.save_every // args.envs, 1), checkpoints, name_prefix="ppo",
                                  save_vecnormalize=True)
        try:
            model.learn(args.steps, callback=[save, ProgressLog()], tb_log_name="tb", reset_num_timesteps=not args.resume)
        except KeyboardInterrupt:
            print("interrupted: saving a final checkpoint")
        final = os.path.join(checkpoints, f"ppo_{model.num_timesteps}_steps.zip")
        model.save(final)
        venv.save(vecnormalize_path(final))
        print(f"saved {final}")
    finally:
        with suppress(EOFError, BrokenPipeError, ConnectionError):  # workers may already be gone after Ctrl+C
            venv.close()


if __name__ == "__main__":
    main()
