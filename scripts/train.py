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
GAP = -1  # the env's last observation field: how far the pot is behind the episode's progress record
GAMMA = 0.998  # rewards ~17 s ahead still count: a fall can cost far more time than 0.995's ~7 s horizon sees


class Starts(gym.Wrapper):
    """Where episodes start, from the game's exact save states, so practice goes to the hard parts.

    Saved starts (Go-Explore style): each game keeps one state per bin_size of route progress, saved the first time an
    episode gets there with the pot or the hammer touching something (not mid-fall). A share of episodes start from
    one, favoring those started from least (weight 1/sqrt(1 + starts)), so new ground gets most of the practice.

    Rewinds (DeepMimic-style early termination, plus a retry): in a share of episodes, falling more than `fall` below
    the record ends the episode, and the next one starts from a state saved 1-2 s before the fall began, so the agent
    retries that move instead of spending the time climbing back; at most max_retries in a row. The other episodes
    play on after falls, so recovering still gets practice. The cut is a truncation: the fall's cost is learned from
    the value of where the pot ended up, not as a death.

    Infos carry from_bottom and rewinds (1 or 0). Bottom starts without rewinds play the real game.
    """

    def __init__(self, env, saved_share=0.5, rewind_share=0.5, bin_size=2.0, fall=5.0, every=30, max_retries=3):
        super().__init__(env)
        self.saved_share, self.rewind_share, self.bin_size = saved_share, rewind_share, bin_size
        self.fall, self.every, self.max_retries = fall, every, max_retries
        self.bins = {}     # progress bin -> [state id, starts]; ponytail: the first state per bin is kept for good
        self.recent = []   # (state id, gap when saved): one per `every` steps in rewind episodes, the last 4 kept
        self.retry = None  # the state the next episode starts from, after a fall
        self.retries = self.t = 0
        self.rng = np.random.default_rng()
        self.from_bottom = self.rewinds = 1

    def reset(self, *, seed=None, options=None):
        game = self.env.unwrapped
        options = dict(options or {})
        for state, _ in self.recent:
            game.drop_state(state)
        self.recent = []
        self.from_bottom = 0
        if self.retry is not None:
            options["state"] = self.retry
            self.retries += 1
        else:
            self.retries = 0
            if self.bins and self.rng.random() < self.saved_share:
                bins = list(self.bins)
                weights = np.array([1 / np.sqrt(1 + self.bins[b][1]) for b in bins])
                b = bins[self.rng.choice(len(bins), p=weights / weights.sum())]
                self.bins[b][1] += 1
                options["state"] = self.bins[b][0]
            else:
                self.from_bottom = 1
        self.rewinds = int(self.retry is not None or self.rng.random() < self.rewind_share)
        obs, info = self.env.reset(seed=seed, options=options)
        if self.retry is not None:
            game.drop_state(self.retry)
            self.retry = None
        self.t = 0
        return obs, {**info, "from_bottom": self.from_bottom, "rewinds": self.rewinds}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped
        self.t += 1
        b = int(obs[PROGRESS] // self.bin_size)
        if self.saved_share and b not in self.bins and not terminated and (obs[HEAD] or obs[POT]):
            self.bins[b] = [game.save_state(), 0]
        if self.rewinds and not terminated:
            if obs[GAP] > self.fall:
                before = [state for state, gap in self.recent if gap < 1.0]  # saved near the record: before the fall
                if before and self.retries < self.max_retries:
                    self.retry = before[-2] if len(before) > 1 else before[-1]
                    self.recent = [r for r in self.recent if r[0] != self.retry]
                truncated = True
            elif self.t % self.every == 0:
                self.recent.append((game.save_state(), float(obs[GAP])))
                if len(self.recent) > 4:
                    game.drop_state(self.recent.pop(0)[0])
        return obs, reward, terminated, truncated, {**info, "from_bottom": self.from_bottom, "rewinds": self.rewinds}


def make_env(game_dir, k, episode_steps, stall_steps, saved_starts, rewinds):
    env = Starts(GoiEnv(game_dir, k=k, stall_steps=stall_steps), saved_starts, rewinds)
    return TimeLimit(env, episode_steps)


def vecnormalize_path(checkpoint):
    """The observation/reward normalization stats saved next to a checkpoint."""
    return re.sub(r"ppo_(\d+)_steps\.zip$", r"ppo_vecnormalize_\1_steps.pkl", checkpoint)


class ProgressLog(BaseCallback):
    """TensorBoard: best route progress per real-game episode (from the bottom, no rewinds; mean over recent episodes),
    and the furthest any episode got."""

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
        # A resumed checkpoint brings its earlier episodes, from before saved starts and rewinds: all real games.
        real = [ep["best_progress"] for ep in episodes if ep.get("from_bottom", 1) and not ep.get("rewinds", 0)]
        if real:
            self.logger.record("rollout/best_progress_mean", sum(real) / len(real))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--name", default=time.strftime("ppo-%Y%m%d-%H%M%S"), help="run folder under runs/")
    ap.add_argument("--steps", type=int, default=10_000_000, help="environment steps to train for")
    ap.add_argument("--envs", type=int, default=5, help="games in parallel (at most 5 per install)")
    ap.add_argument("--k", type=int, default=4, help="physics steps (1/120 s) per action")
    ap.add_argument("--episode-steps", type=int, default=3000, help="time limit in actions (3000 = 100 s at k=4)")
    ap.add_argument("--stall-seconds", type=float, default=10,
                    help="end an episode after this long without headway along the route (0: never)")
    ap.add_argument("--saved-starts", type=float, default=0.5,
                    help="share of episodes that start from states saved along the route instead of the bottom")
    ap.add_argument("--rewinds", type=float, default=0.5,
                    help="share of episodes where a fall ends the episode and the next retries from just before it")
    ap.add_argument("--save-every", type=int, default=500_000, help="checkpoint interval in environment steps")
    ap.add_argument("--resume", help="a checkpoint .zip to continue from")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    run_dir = os.path.join("runs", args.name)
    checkpoints = os.path.join(run_dir, "checkpoints")
    stall_steps = int(args.stall_seconds * 120 / args.k) or None
    envs = [partial(make_env, args.game_dir, args.k, args.episode_steps, stall_steps, args.saved_starts,
                    args.rewinds)] * args.envs
    venv = VecMonitor(SubprocVecEnv(envs), info_keywords=("best_progress", "from_bottom", "rewinds"))
    try:
        if args.resume:
            venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
            model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=run_dir)
        else:
            venv = VecNormalize(venv, gamma=GAMMA)
            # target_kl: stop each update's epochs early once the policy has moved enough. Without it, action noise
            # collapsed (std 0.55 -> 0.03), updates grew (KL > 1) and performance regressed after ~1M steps.
            # learning_rate: at 3e-4 every update hit target_kl, 45% within the first epoch, wasting rollout data.
            # gSDE: exploration noise that varies smoothly with the state (resampled every 4 steps) instead of per-step
            # jitter, which suits physical control; log_std_init as in the RL Zoo's gSDE settings.
            model = PPO("MlpPolicy", venv, learning_rate=1e-4, n_steps=1024, batch_size=512, gamma=GAMMA,
                        target_kl=0.02, use_sde=True, sde_sample_freq=4,
                        policy_kwargs=dict(net_arch=[256, 256], log_std_init=-2), device="cpu",
                        tensorboard_log=run_dir, verbose=1)
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
