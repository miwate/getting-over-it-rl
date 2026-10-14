"""Train PPO on Getting Over It: parallel games, normalized observations, TensorBoard logs, resumable checkpoints."""
import argparse
import os
import pickle
import re
import time
from contextlib import suppress
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from goi_env import OBS_FIELDS, GoiEnv
from goi_env.env import MOUSE, PREV_ACTIONS
from collector import Collector, Spaces
from policy import BigPolicy, BoundedMeanPolicy, GrownPolicy

PROGRESS, HEAD, POT = (OBS_FIELDS.index(f) for f in ("progress", "head_contact", "pot_contact"))
GAP = -1  # the env's last observation field: how far the pot is behind the episode's progress record
STALL_SECONDS = 10  # training's cutoff without headway; tests keep its stuck signal (GoiEnv stall_ends=False)


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
    the value of where the pot ended up, not as a death. But with every episode rewound, no episode plays on after a
    fall, so the critic never learns that value: below the chapel jump its value where a fall began was within 0-25%
    of its value before the try (falls looked free). With `falls_terminal` the cut ends the episode as a death.

    Traps: a state where an episode has made no headway for trap_after steps is saved into a pool (the last `pool`
    kept), and a share of episodes start from one, so the agent practices exactly where it gets stuck. Random mouse
    strokes escape abl-oldhp's traps at the steep climb 10-30% of the time where the policy itself escaped 0 of 10.

    With `window` (lo, hi), saved starts come only from that stretch of route progress, to focus practice on one part.
    With `place` (lo, hi), the saved starts are first seeded with the climber placed on the route every bin_size along
    that stretch (GoiEnv's place), so practice can begin where the agent has never been; `place_at` adds (distance, dx)
    spots beside the route, where it runs through a ledge the pot can only sit next to.

    Demos (backplay, Resnick et al. 2018): each game replays the demonstrations in `demos` (scripts/demos.py: a
    policy's own rare successes at a hard move, or explore.py's paths) and keeps a state every third of a second over
    the last `demo_back` steps before each success (300: 10 s); a share of episodes start from one, so the agent
    practices from where it nearly succeeds. With `demo_window` (lo, hi), the states kept are instead those along that
    stretch of route progress, every third of a second (a climb.py path from the bottom: a section of the route).
    With `demo_after` too, only states in that stretch after the path has been at that progress: where a game (e.g. a
    marathon.py game) landed after falling from above, which the networks didn't recover from.
    A demos file of exported states (marathon.py's <out>-<seed>-states.npz) is loaded as it is: its states in
    `demo_window` become starts, without replaying the moves that led there (with `demo_after`, those saved after the
    game had been at that progress).
    With `demo_curriculum` too (the backward algorithm, Salimans & Chen 2018): demo starts come only from the path's
    last `back_k` states before the stretch's end, 10 (3 s) at first: half from its frontier (its earliest 5 states),
    half from anywhere in it; 5 more each time 30% of the last 20 frontier-started episodes reached the end. Uniform
    starts over a whole section gave no reward signal in most sections; judging all starts moved the frontier back
    before the hard states were learned (chapel: 40 states back in training, 0/10 from 8 s before the end).
    With `demo_priority` (reverse curriculum by success rate: Florensa et al. 2017, Salimans & Chen 2018, prioritized
    replay of levels, Jiang et al. 2021): each demo state's tries and successes (reaching the window's end) are counted,
    decayed by 0.9 a try, and starts are drawn with weight (p(1 - p))^2 + 0.0001, p = (successes + 0.5) / (tries + 1):
    practice goes where the outcome is still open (~33x a decided start's share), which moves back along every path as
    the agent learns. With the prior of 1 and floor of 0.001 at first, a decided start still had a tenth of an open one's
    share, and at the anvil, where 250 of 270 starts were hopeless, 65% of practice went to them. Below the chapel jump the networks finished from 1 s before the end 84-88% of the time, from 2 s
    14-22%, from 3 s 3-4%. The counts are shared by the run's games through the `demo_stats` file (they load the same
    exported states in the same order, checked by progress): counted per game, each start had ~3 tries a game and the
    prior kept every p within 0.1-0.9.

    Infos carry from_bottom and rewinds (1 or 0). Bottom starts without rewinds play the real game.
    """

    def __init__(self, env, saved_share=0.5, rewind_share=0.5, bin_size=2.0, fall=5.0, every=30, max_retries=3,
                 trap_share=0.0, trap_after=150, pool=30, refresh=0.0, frontier=0.0, window=None, place=None, place_at=(), demos=None, demo_share=0.0, demo_back=300, demo_window=None, demo_curriculum=False, demo_after=None, demo_priority=False, falls_terminal=False, demo_stats=None,
                 airborne_falls=False):
        super().__init__(env)
        self.demo_priority, self.demo_stats, self.stats, self.demo_i, self.tried = demo_priority, demo_stats, None, None, 0
        self.demo_progress = []  # each demo state's progress (None once a replayed demo is loaded: games may differ)
        self.falls_terminal, self.airborne_falls = falls_terminal, airborne_falls
        self.demo_window, self.demo_after = demo_window, demo_after
        self.curriculum, self.back_k, self.outcomes, self.start = demo_curriculum, 10, [], None
        self.demo_path, self.demo_share, self.demo_back, self.demos = demos, demo_share, demo_back, []  # demo state ids
        self.refresh, self.frontier, self.window, self.bin = refresh, frontier, window, None
        self.place = [(a, 0.0) for a in np.arange(*place, bin_size)] if place else []
        self.place += list(place_at)
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
        if self.place:
            self._seed_bins(self.place)
            self.place = []
        if self.demo_path:
            for path in [self.demo_path] if isinstance(self.demo_path, str) else self.demo_path:
                self._seed_demos(path, self.demo_back)
            self.demo_path = None
            if self.demo_priority:
                self.stats = self._open_stats()
        if self.curriculum and self.start == "frontier":  # how the last episode from the frontier went
            self.outcomes.append(game.held_best >= self.demo_window[1])  # held there: a mid-air peak isn't
            if len(self.outcomes) >= 20:
                if np.mean(self.outcomes) >= 0.3 and self.back_k < len(self.demos):
                    self.back_k = min(self.back_k + 5, len(self.demos))
                    print(f"curriculum: starts from the last {self.back_k} of {len(self.demos)} path states", flush=True)
                self.outcomes = []
        if self.demo_priority and self.start == "demo":  # how the last episode from a demo start went
            n, wins = self.stats[self.demo_i, :2]
            self.stats[self.demo_i, :2] = (0.9 * n + 1, 0.9 * wins + (game.held_best >= self.demo_window[1]))
            self.tried += 1
            if self.tried % 1000 == 0:
                n = self.stats[:, 0]
                p, w, known = self._p(self.stats), self._weights(self.stats), n >= 3
                open_ = known & (p >= 0.1) & (p <= 0.9)
                print(f"demo starts: {np.mean(n > 0):.0%} of {len(n)} tried; with 3+ tries: {np.mean(p[known] < 0.1):.0%} "
                      f"hopeless, {np.mean(p[known] > 0.9):.0%} done, {np.mean(open_[known]):.0%} open (drawn "
                      f"{w[open_].sum() / w.sum():.0%} of the time)", flush=True)
        options = dict(options or {})
        for state, _ in self.recent:
            game.drop_state(state)
        self.recent = []
        self.from_bottom = 0
        self.start = "bottom"  # what this episode starts from: bottom, bin, trap, demo or retry
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
            elif self.demos and r < self.trap_share + self.demo_share:
                pool = self.demos[-self.back_k:] if self.curriculum else self.demos
                first = len(self.demos) - len(pool)
                frontier = self.curriculum and self.rng.random() < 0.5
                if frontier:
                    pool = pool[:5]
                if self.demo_priority:
                    weights = self._weights(self.stats[first:first + len(pool)])
                    self.demo_i = first + self.rng.choice(len(pool), p=weights / weights.sum())
                    options["state"] = self.demos[self.demo_i]
                else:
                    options["state"] = pool[self.rng.integers(len(pool))]
                self.start = "frontier" if frontier else "demo"
            elif bins and r < self.trap_share + self.demo_share + self.saved_share:
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

    def _seed_bins(self, spots):
        game = self.env.unwrapped
        for along, dx in spots:
            try:
                obs, _ = game.reset(options={"place": float(along), "dx": float(dx)})
            except ValueError:  # nowhere to fit, e.g. where the route passes through a ledge's end
                continue
            b = int(obs[PROGRESS] // self.bin_size)  # where it settled
            if b not in self.bins and (obs[HEAD] or obs[POT]):
                self.bins[b] = [game.save_state(), 0]

    @staticmethod
    def _p(stats):
        return (stats[:, 1] + 0.5) / (stats[:, 0] + 1)

    @staticmethod
    def _weights(stats):
        p = Starts._p(stats)
        return (p * (1 - p)) ** 2 + 1e-4

    def _open_stats(self):
        """(tries, successes, progress) per demo state: the rows of the shared demo_stats file if this game's demo
        states are the file's (same progress, same order), else this game's own."""
        n = len(self.demos)
        if self.demo_stats and self.demo_progress is not None and os.path.exists(self.demo_stats):
            shared = np.load(self.demo_stats, mmap_mode="r+")
            if n <= len(shared):
                rows = shared[:n]
                if not rows[:, 2].any():
                    rows[:, 2] = self.demo_progress
                if np.allclose(rows[:, 2], self.demo_progress):
                    return rows
            print("demo stats: this game's demo states differ from the shared file's; counting alone", flush=True)
        stats = np.zeros((n, 3))
        stats[:, 2] = self.demo_progress or 0.0
        return stats

    def _seed_demos(self, path, back=300, every=10):
        game = self.env.unwrapped
        z = np.load(path)
        if "exported" in z.files:  # states exported by other games (marathon.py): loaded as they are, no replay
            lo, hi = self.demo_window or (-np.inf, np.inf)
            p, step = z["progress"], z["step"]
            after = -1 if self.demo_after is None else min(step[p >= self.demo_after], default=np.inf)
            for i in range(len(p)):
                if lo <= p[i] < hi and step[i] > after:
                    self.demos.append(game.import_state(z[f"s{i}"].tobytes()))
                    if self.demo_progress is not None:
                        self.demo_progress.append(float(p[i]))
            return
        self.demo_progress = None
        for j, reached in enumerate(z["reached"]):
            kept, ok, above = [], False, False
            for t, obs, _ in demo_steps(game, z, j):
                if self.demo_window and self.demo_after is not None:  # where a game landed after falling from above
                    lo, hi = self.demo_window
                    above = above or obs[PROGRESS] >= self.demo_after
                    if above and lo <= obs[PROGRESS] < hi and t % every == 0:
                        kept.append(game.save_state())
                        ok = True
                    continue
                if self.demo_window:  # the path's states along this stretch (a climb path from the bottom)
                    lo, hi = self.demo_window
                    if obs[PROGRESS] >= hi:
                        ok = True
                        break
                    if obs[PROGRESS] >= lo and t % every == 0:
                        kept.append(game.save_state())
                    continue
                ok = ok or t == reached + 1 and obs[PROGRESS] >= z["goal"]
                if reached - back <= t <= reached and t % every == 0:
                    kept.append(game.save_state())
            if ok:
                self.demos += kept
            else:  # the replay went its own way (a multi-contact moment can resolve differently in another launch)
                for s in kept:
                    game.drop_state(s)

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
            # with airborne_falls, losing ground counts as a fall only in the air (nothing touched for 0.5 s) or past
            # 30: the way up goes back 9-25 below the record while holding on (swinging on the bucket, under the
            # anvil's boulder, the ice, the mast), and as deaths those moves could never be learned
            fell = obs[GAP] > self.fall and (not self.airborne_falls or game.air >= 15 or obs[GAP] > 30)
            if fell:
                before = [state for state, gap in self.recent if gap < 1.0]  # saved near the record: before the fall
                if before and self.retries < self.max_retries:
                    self.retry = before[-2] if len(before) > 1 else before[-1]
                    self.recent = [r for r in self.recent if r[0] != self.retry]
                if self.falls_terminal:
                    terminated = True
                else:
                    truncated = True
            elif self.t % self.every == 0:
                self.recent.append((game.save_state(), float(obs[GAP])))
                if len(self.recent) > 4:
                    game.drop_state(self.recent.pop(0)[0])
        return obs, reward, terminated, truncated, {**info, "from_bottom": self.from_bottom, "rewinds": self.rewinds}


def demo_steps(game, z, j):
    """Replay demo j of a demos file (demos.py: a finder's climb from reset, then a try from a loaded state; explore.py:
    a placement, then actions with loads where the explorer loaded saved states) in a GoiEnv. Yields (step,
    observation, action), the observation being the one the action was taken from; last (steps, final observation,
    None). A load rebuilds the physics world, so the replay loads at the same steps, which makes it exact."""
    if f"place{j}" in z:
        along, dx = z[f"place{j}"]
        game.reset(options={"place": float(along), "dx": float(dx)})
    else:
        game.reset()
        for a in z[f"climb{j}"]:
            game.step(a)
    cuts = set(z[f"cuts{j}"].tolist()) if f"cuts{j}" in z else {0}
    obs = None
    for t, a in enumerate(z[f"try{j}"]):
        if t in cuts:
            state = game.save_state()
            obs = game.load_state(state)
            game.drop_state(state)
        yield t, obs, a
        obs, *_ = game.step(a)
    yield len(z[f"try{j}"]), obs, None


def make_env(game_dir, k, episode_steps, stall_steps, saved_starts, rewinds, traps=0.0, refresh=0.0, frontier=0.0,
             window=None, place=None, place_at=(), demos=None, demo_share=0.0, demo_back=300, demo_window=None,
             demo_curriculum=False, demo_after=None, demo_priority=False, falls_terminal=False, demo_stats=None,
             airborne_falls=False, **env_kwargs):
    """env_kwargs: GoiEnv's action_cost, mouse and reward weights."""
    env = Starts(GoiEnv(game_dir, k=k, stall_steps=stall_steps, snake_ends=True, **env_kwargs), saved_starts, rewinds,
                 trap_share=traps,
                 trap_after=(stall_steps or 300) // 2, refresh=refresh, frontier=frontier, window=window, place=place,
                 place_at=place_at, demos=demos, demo_share=demo_share, demo_back=demo_back, demo_window=demo_window,
                 demo_curriculum=demo_curriculum, demo_after=demo_after, demo_priority=demo_priority,
                 falls_terminal=falls_terminal, demo_stats=demo_stats, airborne_falls=airborne_falls)
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
    if isinstance(model, SAC):  # rlpd.py: tanh of the actor's Gaussian, its noise scaled the same way
        with torch.no_grad():
            mu, log_std, _ = model.actor.get_action_dist_params(model.policy.obs_to_tensor(obs)[0])
            return torch.tanh(mu + noise * log_std.exp() * torch.randn_like(mu)).cpu().numpy()
    explore(model, step)
    with torch.no_grad():
        dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
        mean = dist.distribution.mean
        return np.clip((mean + noise * (dist.get_actions() - mean)).cpu().numpy(), -1, 1)


class Player:
    """A checkpoint with its observation normalization, acting at an exploration noise scale (act): one game's raw
    observation in, one action out."""

    def __init__(self, checkpoint, noise=0.5):
        self.model = (SAC if os.path.basename(checkpoint).startswith("sac_") else PPO).load(checkpoint, device="cpu")
        with open(vecnormalize_path(checkpoint), "rb") as f:
            self.norm = pickle.load(f)
        self.noise = noise

    def __call__(self, obs, step):
        return act(self.model, self.norm.normalize_obs(obs[None]), step, self.noise)[0]


def chain(base, then=(), noise=0.5, margin=2.0):
    """A player made of checkpoints that take over by route progress: base, then each (from, checkpoint) from that
    progress on. A specialist for a stretch hands back to the earlier player below it, e.g. after a fall, but only once
    progress is `margin` below its boundary, so progress wavering around one (standing there, or mid-air in a jump)
    doesn't swap players back and forth: games lost 7-8 min each at 250 that way. Handing over upward waits for
    nothing: waiting for `margin` past the boundary too left rw-port alone at the chimney's foot and cost 7-34 min a
    game at 140-180 (m19). At step 0 the player is picked by progress alone."""
    players = [(-np.inf, Player(base, noise))] + sorted(((t[0], Player(t[1], t[2] if len(t) > 2 else noise))
                                                         for t in then), key=lambda t: t[0])
    current = [0]

    def play(obs, step):
        p = obs[PROGRESS]
        up = max(i for i, (f, _) in enumerate(players) if p >= f)  # the player for this progress
        if step == 0:
            current[0] = up
        down = max(i for i, (f, _) in enumerate(players) if p >= f - margin)  # boundaries fallen `margin` below
        current[0] = min(max(current[0], up), down)
        return players[current[0]][1](obs, step)
    return play


def parse_then(specs):
    """--then FROM:CHECKPOINT[@NOISE] arguments as (from, checkpoint[, noise]): a specialist trained with the path
    curriculum plays best at its full noise (@1), older ones at 0.5."""
    out = []
    for s in specs:
        start, rest = s.split(":", 1)
        path, _, noise = rest.partition("@")
        out.append((float(start), path) + ((float(noise),) if noise else ()))
    return out


def widen(x, extra, axis):
    """Insert `extra` inputs (zeros) before GoiEnv's own last fields (past actions, stuck, gap): the bridge's new
    observation fields (head rays) come after its others."""
    at = x.shape[axis] - (2 * PREV_ACTIONS + 2)
    cut = [slice(None)] * x.ndim
    lib = torch if isinstance(x, torch.Tensor) else np
    shape = list(x.shape)
    shape[axis] = extra
    pad = lib.zeros(shape, dtype=x.dtype)
    cut[axis] = slice(None, at)
    head = x[tuple(cut)]
    cut[axis] = slice(at, None)
    return lib.cat([head, pad, x[tuple(cut)]], axis) if lib is torch else np.concatenate([head, pad, x[tuple(cut)]], axis)


def load_norm(checkpoint, venv):
    """A checkpoint's observation normalization for `venv`, widened (mean 0, var 1) if venv has new inputs."""
    with open(vecnormalize_path(checkpoint), "rb") as f:
        norm = pickle.load(f)
    extra = venv.observation_space.shape[0] - norm.observation_space.shape[0]
    if extra:
        rms = norm.obs_rms
        rms.mean, rms.var = widen(rms.mean, extra, 0), widen(rms.var - 1, extra, 0) + 1
        norm.observation_space = venv.observation_space
    norm.set_venv(venv)
    return norm


def warm_start(model, checkpoint, extra=0):
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
        elif k in new and new[k].shape[1:] == (v.shape[1] + extra,) and new[k].shape[0] == v.shape[0] and extra:
            loaded[k] = widen(v, extra, 1)  # a first layer: zero weights on the new inputs keep its behavior
    model.policy.load_state_dict({**new, **loaded})
    print(f"warm start from {checkpoint}: {len(loaded)} of {len(new)} tensors; fresh: {sorted(set(new) - set(loaded))}")


def vecnormalize_path(checkpoint):
    """The observation/reward normalization stats saved next to a checkpoint."""
    return re.sub(r"(ppo|sac)_(\d+)_steps\.zip$", r"\1_vecnormalize_\2_steps.pkl", checkpoint)


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
    ap.add_argument("--stall-seconds", type=float, default=STALL_SECONDS,
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
    ap.add_argument("--place", type=float, nargs=2, metavar=("LO", "HI"),
                    help="seed the saved starts with the climber placed along this stretch of the route")
    ap.add_argument("--demos", nargs="+", help="demonstration files to start episodes along (scripts/demos.py, "
                                               "climb.py, marathon arrival files)")
    ap.add_argument("--demo-share", type=float, default=0.3, help="with --demos: share of episodes that start along one")
    ap.add_argument("--demo-back", type=float, default=10, help="with --demos: seconds before each success to start from")
    ap.add_argument("--demo-window", type=float, nargs=2, metavar=("LO", "HI"),
                    help="with --demos: start from the demos' states along this stretch of route progress instead")
    ap.add_argument("--demo-after", type=float,
                    help="with --demo-window: only the demos' states in the window after they have been at this progress")
    ap.add_argument("--demo-curriculum", action="store_true",
                    help="with --demo-window: start near the stretch's end first, moving back as episodes succeed")
    ap.add_argument("--demo-priority", action="store_true",
                    help="with --demo-window: favor demo starts the agent crosses from only sometimes (weight p(1-p))")
    ap.add_argument("--falls-terminal", action="store_true",
                    help="with --rewinds: a fall's cut ends the episode as a death, not a truncation the critic fills in")
    ap.add_argument("--airborne-falls", action="store_true",
                    help="with --rewinds: a fall is losing ground in the air (or past 30), not a controlled move back")
    ap.add_argument("--held-records", action="store_true",
                    help="new ground pays only while the pot or hammer touches something, not at mid-air peaks")
    ap.add_argument("--place-at", nargs="+", default=[], metavar="ALONG:DX",
                    help="also seed placed starts at these route distances, moved sideways by DX (e.g. 226:1.2)")
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
    ap.add_argument("--collector", choices=("lockstep", "async"), default="lockstep",
                    help="async: each game steps at its own pace in its own worker (collector.py); several times faster")
    ap.add_argument("--device", default="cpu", help="where the network trains: cpu (fastest for the small net) or cuda")
    ap.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus")
    ap.add_argument("--action-cost", type=float, default=0.0, help="reward cost per step of the mean squared action")
    ap.add_argument("--head-rays", type=int, default=0,
                    help="observe this many distances to the terrain from the hammer head too (0: off; a warm start "
                         "from a checkpoint without them gets zero weights on them)")
    ap.add_argument("--mouse", type=float, default=MOUSE, help="mouse input at action 1 (evaluate with the same)")
    # Reward weights (goi_env.env.progress_reward); all 0: new progress records only.
    ap.add_argument("--recovery", type=float, default=0.0, help="reward per unit climbed back below the episode's record")
    ap.add_argument("--regress", type=float, default=0.0, help="cost per unit of progress lost (>= --recovery: falls never pay)")
    ap.add_argument("--air-bonus", type=float, default=0.0, help="extra reward share for new ground made with the pot touching nothing")
    ap.add_argument("--speed-bonus", type=float, default=0.0, help="extra reward share per 6 units/s of new ground (capped at 2)")
    ap.add_argument("--save-every", type=int, default=500_000, help="checkpoint interval in environment steps")
    ap.add_argument("--resume", help="a checkpoint .zip to continue from, with its own settings")
    ap.add_argument("--warm-start", help="a checkpoint .zip whose weights and normalization start this run's settings")
    ap.add_argument("--fresh-noise", action="store_true",
                    help="with --warm-start: exploration noise back to its initial size (gSDE's shrank ~10x in long runs)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    run_dir = os.path.join("runs", args.name)
    checkpoints = os.path.join(run_dir, "checkpoints")
    stall_steps = int(args.stall_seconds * 120 / args.k) or None
    demo_stats = os.path.join(run_dir, "demo-stats.npy") if args.demo_priority else None
    if demo_stats and not os.path.exists(demo_stats):  # the games' shared success counts per demo state
        os.makedirs(run_dir, exist_ok=True)
        np.lib.format.open_memmap(demo_stats, mode="w+", dtype=np.float64, shape=(200_000, 3)).flush()
    env_kwargs = dict(head_rays=args.head_rays, action_cost=args.action_cost, mouse=args.mouse, recovery=args.recovery, regress=args.regress,
                      air_bonus=args.air_bonus, speed_bonus=args.speed_bonus, held_records=args.held_records)
    envs = [partial(make_env, args.game_dir, args.k, args.episode_steps, stall_steps, args.saved_starts,
                    args.rewinds, args.traps, args.refresh, args.frontier, args.window, args.place,
                    [tuple(map(float, spot.split(":"))) for spot in args.place_at], args.demos, args.demo_share,
                    int(args.demo_back * 120 / args.k), args.demo_window, args.demo_curriculum, args.demo_after,
                    demo_priority=args.demo_priority, falls_terminal=args.falls_terminal, demo_stats=demo_stats,
                    airborne_falls=args.airborne_falls, **env_kwargs)] * args.envs
    collector = Collector(envs) if args.collector == "async" else None
    venv = (Spaces(args.envs, collector.observation_space, collector.action_space) if collector else
            VecMonitor(SubprocVecEnv(envs), info_keywords=("best_progress", "from_bottom", "rewinds")))
    try:
        if args.resume:
            venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
            model = PPO.load(args.resume, env=venv, device=args.device, tensorboard_log=run_dir)
        else:
            if args.warm_start:
                venv = load_norm(args.warm_start, venv)
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
                warm_start(model, args.warm_start, args.head_rays)
                if args.fresh_noise:
                    with torch.no_grad():
                        model.policy.log_std.fill_(policy_kwargs.get("log_std_init", 0.0))
        save = CheckpointCallback(max(args.save_every // args.envs, 1), checkpoints, name_prefix="ppo",
                                  save_vecnormalize=True)
        def save_checkpoint(path):
            model.save(path)
            venv.save(vecnormalize_path(path))

        try:
            if collector:
                collector.learn(model, venv, args.steps, args.save_every, checkpoints, save_checkpoint,
                                reset_num_timesteps=not args.resume)
            else:
                model.learn(args.steps, callback=[save, ProgressLog()], tb_log_name="tb",
                            reset_num_timesteps=not args.resume)
        except KeyboardInterrupt:
            print("interrupted: saving a final checkpoint")
        final = os.path.join(checkpoints, f"ppo_{model.num_timesteps}_steps.zip")
        model.save(final)
        venv.save(vecnormalize_path(final))
        print(f"saved {final}")
    finally:
        with suppress(EOFError, BrokenPipeError, ConnectionError):  # workers may already be gone after Ctrl+C
            venv.close()
            if collector:
                collector.close()


if __name__ == "__main__":
    main()
