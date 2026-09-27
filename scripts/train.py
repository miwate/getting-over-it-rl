"""Train PPO on Getting Over It: parallel games, normalized observations, TensorBoard logs, resumable checkpoints."""
import argparse
import os
import re
import time
from contextlib import suppress
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from goi_env import OBS_FIELDS, GoiEnv
from goi_env.env import MOUSE
from policy import BigPolicy, BoundedMeanPolicy, GrownPolicy

PROGRESS, HEAD, POT = (OBS_FIELDS.index(f) for f in ("progress", "head_contact", "pot_contact"))
GAP = -1  # the env's last observation field: how far the pot is behind the episode's progress record


class Starts(gym.Wrapper):
    """Where episodes start, from the game's exact save states, so practice goes to the hard parts.

    Saved starts (Go-Explore style): each game keeps one state per bin_size of route progress, saved the first time an
    episode gets there with the pot or the hammer touching something (not mid-fall). A share of episodes start from
    one, favoring those started from least (weight 1/sqrt(1 + starts)), so new ground gets most of the practice. With
    `refresh`, an episode entering a bin replaces its state with that chance, so the states follow the current policy:
    progress-6 kept its first, early states for 57M steps and never crossed the chimney top; a run with fresh pools
    crossed within 1.5M. With `frontier`, bin weights also fall off as exp(-(bins behind the furthest one) / frontier), so
    the furthest stretch of the route gets most of the practice.

    Rewinds (DeepMimic-style early termination, plus a retry): in a share of episodes, falling more than `fall` below
    the record ends the episode, and the next one starts from a state saved 1-2 s before the fall began, so the agent
    retries that move instead of spending the time climbing back; at most max_retries in a row. The other episodes
    play on after falls, so recovering still gets practice. The cut is a truncation: the fall's cost is learned from
    the value of where the pot ended up, not as a death.

    Traps: a state where an episode has made no headway for trap_after steps is saved into a pool (the last `pool`
    kept), and a share of episodes start from one, so the agent practices exactly where it gets stuck. Random mouse
    strokes escape abl-oldhp's traps at the steep climb 10-30% of the time where the policy itself escaped 0 of 10.

    With `window` (lo, hi), saved starts come only from that stretch of route progress, to focus practice on one part.

    Infos carry from_bottom and rewinds (1 or 0). Bottom starts without rewinds play the real game.
    """

    def __init__(self, env, saved_share=0.5, rewind_share=0.5, bin_size=2.0, fall=5.0, every=30, max_retries=3,
                 trap_share=0.0, trap_after=150, pool=30, refresh=0.0, frontier=0.0, window=None):
        super().__init__(env)
        self.refresh, self.frontier, self.window, self.bin = refresh, frontier, window, None
        self.saved_share, self.rewind_share, self.bin_size = saved_share, rewind_share, bin_size
        self.trap_share, self.trap_after, self.pool = trap_share, trap_after, pool
        self.traps = []    # state ids of recent traps, oldest first
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
        self.start = "bottom"  # what this episode starts from: bottom, bin, trap or retry
        if self.retry is not None:
            options["state"] = self.retry
            self.retries += 1
            self.start = "retry"
        else:
            self.retries = 0
            r = self.rng.random()
            lo, hi = self.window or (-np.inf, np.inf)
            bins = [b for b in self.bins if lo <= b * self.bin_size < hi]
            if self.traps and r < self.trap_share:
                options["state"] = self.traps[self.rng.integers(len(self.traps))]
                self.start = "trap"
            elif bins and r < self.trap_share + self.saved_share:
                weights = np.array([1 / np.sqrt(1 + self.bins[b][1]) for b in bins])
                if self.frontier:
                    weights *= np.exp((np.array(bins) - max(bins)) / self.frontier)
                b = bins[self.rng.choice(len(bins), p=weights / weights.sum())]
                self.bins[b][1] += 1
                options["state"] = self.bins[b][0]
                self.start = "bin"
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
        entering, self.bin = b != self.bin, b
        if self.saved_share and not terminated and (obs[HEAD] or obs[POT]):
            if b not in self.bins:
                self.bins[b] = [game.save_state(), 0]
            elif entering and self.rng.random() < self.refresh:
                game.drop_state(self.bins[b][0])
                self.bins[b] = [game.save_state(), 0]
        if self.trap_share and game.stalled == self.trap_after and not terminated:
            self.traps.append(game.save_state())
            if len(self.traps) > self.pool:
                game.drop_state(self.traps.pop(0))
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


def make_env(game_dir, k, episode_steps, stall_steps, saved_starts, rewinds, traps=0.0, refresh=0.0, frontier=0.0,
             window=None, **env_kwargs):
    """env_kwargs: GoiEnv's action_cost, mouse and reward weights."""
    env = Starts(GoiEnv(game_dir, k=k, stall_steps=stall_steps, **env_kwargs), saved_starts, rewinds, trap_share=traps,
                 trap_after=(stall_steps or 300) // 2, refresh=refresh, frontier=frontier, window=window)
    return TimeLimit(env, episode_steps)


def explore(model, step):
    """Re-draw gSDE exploration noise the way training does: every sde_sample_freq steps (or once per episode if
    that's -1). predict() alone never re-draws it, which turns a gSDE policy deterministic in a fixed, random way."""
    if model.use_sde and (step == 0 or model.sde_sample_freq > 0 and step % model.sde_sample_freq == 0):
        model.policy.reset_noise(1)


def act(model, obs, step, noise=1.0):
    """The action at an episode's `step` for one game's normalized observation: the mean action plus `noise` times the
    exploration noise (1: as in training, 0: the mean alone), clipped to [-1, 1]. rw-control plays best at 0.5: the
    noise it trained with blurs the moves it knows, and the mean alone freezes at the chimney top (docs/RECON.md)."""
    explore(model, step)
    with torch.no_grad():
        dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
        mean = dist.distribution.mean
        return np.clip((mean + noise * (dist.get_actions() - mean)).cpu().numpy(), -1, 1)


def warm_start(model, checkpoint):
    """Copy a checkpoint's weights into a new model, where they fit: a bounded mean nests the mean layer as action_net.0,
    and gSDE's noise matrix doesn't match a plain Gaussian's log_std (it starts fresh)."""
    new = model.policy.state_dict()
    nested = "action_net.0.weight" in new
    loaded = {}
    for k, v in PPO.load(checkpoint, device="cpu").policy.state_dict().items():
        if k.startswith("action_net."):
            k = "action_net." + ("0." if nested else "") + k.split(".")[-1]
        if k in new and new[k].shape == v.shape:
            loaded[k] = v
    model.policy.load_state_dict({**new, **loaded})
    print(f"warm start from {checkpoint}: {len(loaded)} of {len(new)} tensors; fresh: {sorted(set(new) - set(loaded))}")


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
    ap.add_argument("--envs", type=int, default=5,
                    help="games in parallel (at most 5 per install while BepInEx writes its disk log; 10 tested without)")
    ap.add_argument("--k", type=int, default=4, help="physics steps (1/120 s) per action")
    ap.add_argument("--episode-steps", type=int, default=3000, help="time limit in actions (3000 = 100 s at k=4)")
    ap.add_argument("--stall-seconds", type=float, default=10,
                    help="end an episode after this long without headway along the route (0: never)")
    ap.add_argument("--saved-starts", type=float, default=0.5,
                    help="share of episodes that start from states saved along the route instead of the bottom")
    ap.add_argument("--rewinds", type=float, default=0.0,
                    help="share of episodes where a fall ends the episode and the next retries from just before it")
    ap.add_argument("--frontier", type=float, default=0.0,
                    help="favor saved starts near the furthest point: weights fall off by e every this many 2-unit bins behind it (0: off)")
    ap.add_argument("--refresh", type=float, default=0.0,
                    help="chance that an episode entering a stretch of the route replaces its saved state")
    ap.add_argument("--traps", type=float, default=0.0,
                    help="share of episodes that start from states where an episode was stuck (no headway for half the stall time)")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                    help="saved starts only from this stretch of route progress (the chimney top: 170 200)")
    # Defaults: abl-oldhp's settings, the best of the 2M-step experiments (docs/RECON.md, "Settings experiments").
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    ap.add_argument("--lr-end", type=float, help="decay the learning rate linearly to this by the end of the run")
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="stop each update once the policy has moved this far (smaller: gentler fine-tuning)")
    ap.add_argument("--gamma", type=float, default=0.995, help="discount: 0.995 values rewards ~7 s ahead, 0.998 ~17 s")
    ap.add_argument("--gsde", action="store_true",
                    help="state-dependent exploration noise held for --sde-freq steps, instead of fresh Gaussian noise every step")
    ap.add_argument("--sde-freq", type=int, default=4, help="with --gsde: steps between noise draws")
    ap.add_argument("--bounded-mean", action="store_true", help="tanh on the mean action (policy.BoundedMeanPolicy)")
    ap.add_argument("--net", choices=("small", "big"), default="small",
                    help="small: 2x256 MLP; big: SimBa-style residual trunks with LayerNorm (policy.BigPolicy, bounded mean)")
    ap.add_argument("--width", type=int, default=512, help="with --net big: trunk width")
    ap.add_argument("--blocks", type=int, default=2, help="with --net big: residual blocks per trunk")
    ap.add_argument("--grow", type=int, default=0,
                    help="with --net small: this many residual blocks at the end of each trunk, adding nothing until trained, "
                         "so a warm start keeps its behavior with more room to learn (policy.GrownPolicy, bounded mean)")
    ap.add_argument("--device", default="cpu", help="where the network trains: cpu (fastest for the small net) or cuda")
    ap.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus")
    ap.add_argument("--action-cost", type=float, default=0.0, help="reward cost per step of the mean squared action")
    ap.add_argument("--mouse", type=float, default=MOUSE, help="mouse input at action 1 (evaluate with the same)")
    # Reward weights (goi_env.env.progress_reward); all 0: new progress records only.
    ap.add_argument("--recovery", type=float, default=0.0, help="reward per unit climbed back below the episode's record")
    ap.add_argument("--regress", type=float, default=0.0, help="cost per unit of progress lost (>= --recovery: falls never pay)")
    ap.add_argument("--air-bonus", type=float, default=0.0, help="extra reward share for new ground made with the pot touching nothing")
    ap.add_argument("--speed-bonus", type=float, default=0.0, help="extra reward share per 6 units/s of new ground (capped at 2)")
    ap.add_argument("--save-every", type=int, default=500_000, help="checkpoint interval in environment steps")
    ap.add_argument("--resume", help="a checkpoint .zip to continue from, with its own settings")
    ap.add_argument("--warm-start", help="a checkpoint .zip whose weights and normalization start this run's settings")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    run_dir = os.path.join("runs", args.name)
    checkpoints = os.path.join(run_dir, "checkpoints")
    stall_steps = int(args.stall_seconds * 120 / args.k) or None
    env_kwargs = dict(action_cost=args.action_cost, mouse=args.mouse, recovery=args.recovery, regress=args.regress,
                      air_bonus=args.air_bonus, speed_bonus=args.speed_bonus)
    envs = [partial(make_env, args.game_dir, args.k, args.episode_steps, stall_steps, args.saved_starts,
                    args.rewinds, args.traps, args.refresh, args.frontier, args.window, **env_kwargs)] * args.envs
    venv = VecMonitor(SubprocVecEnv(envs), info_keywords=("best_progress", "from_bottom", "rewinds"))
    try:
        if args.resume:
            venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
            model = PPO.load(args.resume, env=venv, device=args.device, tensorboard_log=run_dir)
        else:
            if args.warm_start:
                venv = VecNormalize.load(vecnormalize_path(args.warm_start), venv)
                venv.gamma = args.gamma
            else:
                venv = VecNormalize(venv, gamma=args.gamma)
            # target_kl: stop each update's epochs early once the policy has moved enough. Without it, action noise
            # collapsed (std 0.55 -> 0.03), updates grew (KL > 1) and performance regressed after ~1M steps.
            # learning_rate: at 3e-4 every update hit target_kl, 45% within the first epoch, wasting rollout data.
            # gSDE: exploration noise that varies smoothly with the state (resampled every 4 steps) instead of per-step
            # jitter, which suits physical control; log_std_init as in the RL Zoo's gSDE settings.
            # 5,120 steps per update whatever the number of games. Updates stop at target_kl after ~2 epochs anyway,
            # so learning is paced by the number of updates, not by samples per update.
            exploration = dict(use_sde=True, sde_sample_freq=args.sde_freq) if args.gsde else dict()
            policy_kwargs = dict(log_std_init=-2) if args.gsde else {}
            if args.net == "big":
                policy, policy_kwargs = BigPolicy, dict(policy_kwargs, width=args.width, blocks=args.blocks)
            elif args.grow:
                policy, policy_kwargs = GrownPolicy, dict(policy_kwargs, net_arch=[256, 256], grow=args.grow)
            else:
                policy = BoundedMeanPolicy if args.bounded_mean else "MlpPolicy"
                policy_kwargs["net_arch"] = [256, 256]
            lr = args.lr if args.lr_end is None else lambda left: args.lr_end + left * (args.lr - args.lr_end)
            model = PPO(policy, venv, learning_rate=lr, n_steps=5120 // args.envs, batch_size=512, gamma=args.gamma,
                        target_kl=args.target_kl, ent_coef=args.ent_coef, **exploration, policy_kwargs=policy_kwargs,
                        device=args.device, tensorboard_log=run_dir, verbose=1)
            if args.warm_start:
                warm_start(model, args.warm_start)
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
