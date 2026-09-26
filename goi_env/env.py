"""Gymnasium environment for Getting Over It, driven through the GOI RL Bridge plugin."""
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import OBS_FIELDS, GoiClient, free_port, launch

REACH = 3.5  # the game keeps its cursor within 3.5 units of the pot
Y, LAKE, WON = (OBS_FIELDS.index(f) for f in ("pot_y", "fell_in_lake", "won"))


class GoiEnv(gym.Env):
    """One game instance.

    Observation: the bridge's fields (OBS_FIELDS in client.py), then n_rays terrain distances.
    Action: a point in [-1, 1]^2, mapped onto the disk the hammer can reach around the pot.
    Reward: height gained above the episode's best so far (new records only), minus a time penalty per step.
    Episodes end when the pot falls in the lake or wins; wrap in TimeLimit for truncation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, game_dir=None, port=None, k=4, n_rays=16, ray_length=10.0, time_penalty=0.001,
                 render_mode=None, realtime=True, width=640, height=360):
        """Launches the game from game_dir on a free port, or with game_dir=None connects to one already running on
        `port` (default 5555).

        render_mode="human" shows the game window at real speed; realtime=False lets it run as fast as it can.
        """
        if port is None:
            port = free_port() if game_dir else 5555
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 120 / k}
        self._realtime = render_mode == "human" and realtime
        self._frame_due = 0.0
        self.time_penalty = time_penalty
        self._proc = launch(game_dir, port, width, height) if game_dir else None
        try:
            self.client = GoiClient(port=port, k=k, n_rays=n_rays, ray_length=ray_length,
                                    render=render_mode == "human", width=width, height=height)
        except BaseException:
            if self._proc:
                self._proc.kill()  # don't leak a game we launched but couldn't reach
            raise
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.client.obs_size,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.best_height = 0.0
        self._saved_best = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.asarray(self.client.reset(), np.float32)
        self.best_height = float(obs[Y])
        return obs, {"best_height": self.best_height}

    def step(self, action):
        x, y = np.clip(action, -1.0, 1.0)
        # Square to disk (elliptical grid mapping): every point of the action box reaches a distinct target.
        dx = x * math.sqrt(1 - y * y / 2) * REACH
        dy = y * math.sqrt(1 - x * x / 2) * REACH
        obs = np.asarray(self.client.step(dx, dy), np.float32)
        height = float(obs[Y])
        reward = max(0.0, height - self.best_height) - self.time_penalty
        self.best_height = max(self.best_height, height)
        terminated = bool(obs[LAKE] or obs[WON])
        if self._realtime:  # the bridge itself runs many times faster than real time
            self._frame_due = max(self._frame_due + 1 / self.metadata["render_fps"], time.perf_counter())
            time.sleep(max(0.0, self._frame_due - time.perf_counter()))
        return obs, reward, terminated, False, {"best_height": self.best_height}

    def render(self):
        pass  # "human": the game window draws itself

    def save_state(self):
        state_id = self.client.save_state()
        self._saved_best[state_id] = self.best_height
        return state_id

    def load_state(self, state_id):
        obs = np.asarray(self.client.load_state(state_id), np.float32)
        self.best_height = self._saved_best[state_id]
        return obs

    def close(self):
        self.client.close()
        if self._proc:
            self._proc.wait(timeout=30)
