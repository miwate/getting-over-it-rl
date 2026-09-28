"""Gymnasium environment for Getting Over It, driven through the GOI RL Bridge plugin."""
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import OBS_FIELDS, GoiClient, start

# Mouse input at action 1. The game's acceleration curve makes small inputs fine (below ~3: under 0.15 units of cursor
# travel per step) and large ones flicks (12: the whole 3.5 reach in one step).
MOUSE = 12.0
PROGRESS, LAKE, WON, POT_CONTACT = (OBS_FIELDS.index(f) for f in ("progress", "fell_in_lake", "won", "pot_contact"))
PREV_ACTIONS = 4  # as Linesight gives its Trackmania policy its last 5 actions
SPEED_REF = 0.2  # progress per step (6 units/s, the agent's "very fast" regime) worth one unit of speed bonus


def progress_reward(prev, now, best, air, recovery=0.0, regress=0.0, air_bonus=0.0, speed_bonus=0.0):
    """Reward for moving along the route from progress `prev` to `now`, `best` being the episode's record before.

    New ground (beyond the record) earns 1 per unit, times 1 + air_bonus if the pot touches nothing and + speed_bonus
    per SPEED_REF of new ground this step (capped at 2): bonuses only ever multiply new ground, which can't be earned
    twice. Climbing back below the record earns `recovery` per unit, losing ground costs `regress` per unit; with
    regress >= recovery a fall and its recovery never pay. All zero: new records only. recovery = regress = 1, no
    bonuses: the signed progress change (records plus a potential-based shaping term, Phi = progress - record).
    """
    new = max(0.0, now - best)
    forward, back = max(0.0, now - prev), max(0.0, prev - now)
    bonus = 1 + air_bonus * air + speed_bonus * min(new / SPEED_REF, 2.0)
    return new * bonus + recovery * (forward - new) - regress * back


class GoiEnv(gym.Env):
    """One game instance.

    Observation: the bridge's fields (OBS_FIELDS in client.py), n_rays terrain distances, the last PREV_ACTIONS actions
    (newest first), how long the pot has gone without headway (as a share of stall_steps; 0 without a cutoff), then how
    far it is behind the episode's progress record (the reward only pays beyond it).
    Action: mouse movement in [-1, 1]^2, as a player's mouse: the game's controller moves its cursor by it (with its own
    acceleration and smoothing) and drives the hammer head toward the cursor. A still mouse lets the cursor settle on
    the head: the hammer goes loose.
    Reward: progress_reward (new ground along the level route, by default only that), minus a time penalty per step.
    Raw height would pay for hopping in place; the route first goes right, over the lakeside rocks.
    Episodes end when the pot falls in the lake or wins, and are truncated after stall_steps without headway (if set):
    progress hasn't risen 1 above its lowest point since. Climbing back after a fall counts; perching or rocking in a
    hole doesn't. With stall_ends=False the count starts over instead, so a test episode goes on and the policy still
    gets the stuck signal it trained with. Wrap in TimeLimit for a time limit.
    reset(options={"state": id}) starts an episode from a saved state instead of the bottom; {"place": distance} puts the
    climber on the route there first (training starts beyond where the agent gets on its own).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, game_dir=None, port=None, k=4, n_rays=32, ray_length=15.0, time_penalty=0.001, stall_steps=None,
                 stall_ends=True, action_cost=0.0, mouse=MOUSE, recovery=0.0, regress=0.0, air_bonus=0.0, speed_bonus=0.0,
                 render_mode=None, realtime=True, width=640, height=360, game_args=()):
        """Launches the game from game_dir on a free port, or with game_dir=None connects to one already running on
        `port` (default 5555).

        render_mode="human" shows the game window at real speed; realtime=False lets it run as fast as it can.
        render_mode="rgb_array" makes render() return the frame the game drew after the last step.
        game_args: extra launch flags, e.g. ["-goi-colliders"] to draw the physics shapes over the game.
        action_cost: subtracted from the reward times the mean squared action, so big mouse moves have to pay off.
        mouse: mouse input at action 1.
        recovery, regress, air_bonus, speed_bonus: progress_reward's weights.
        """
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 120 / k}
        self._realtime = render_mode == "human" and realtime
        self._frame_due = 0.0
        self.time_penalty = time_penalty
        self.stall_steps, self.stall_ends = stall_steps, stall_ends
        self.action_cost = action_cost
        self.mouse = mouse
        self.weights = dict(recovery=recovery, regress=regress, air_bonus=air_bonus, speed_bonus=speed_bonus)
        config = dict(k=k, n_rays=n_rays, ray_length=ray_length, render=render_mode is not None,
                      width=width, height=height)
        if game_dir:
            self._proc, self.client = start(game_dir, port, args=game_args, **config)
        else:
            self._proc, self.client = None, GoiClient(port=port or 5555, **config)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.client.obs_size + 2 * PREV_ACTIONS + 2,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.best_progress = 0.0
        self._saved_best = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "place" in options:
            raw = self._place(options["place"])
        else:
            raw = self.client.reset() if options.get("state") is None else self.client.load_state(options["state"])
        self.best_progress = self._stall_mark = self._progress = raw[PROGRESS]
        self.stalled = 0
        self._actions = deque([(0.0, 0.0)] * PREV_ACTIONS, maxlen=PREV_ACTIONS)
        return self._observe(raw), {"best_progress": self.best_progress}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        self._actions.appendleft(tuple(a))
        raw = self.client.step(*(a * self.mouse))
        reward = (progress_reward(self._progress, raw[PROGRESS], self.best_progress, 1.0 - raw[POT_CONTACT], **self.weights)
                  - self.time_penalty - self.action_cost * float(np.mean(a * a)))
        self._progress = raw[PROGRESS]
        self.best_progress = max(self.best_progress, raw[PROGRESS])
        self.stalled += 1
        self._stall_mark = min(self._stall_mark, raw[PROGRESS])
        if raw[PROGRESS] >= self._stall_mark + 1.0:
            self._stall_mark, self.stalled = raw[PROGRESS], 0
        terminated = bool(raw[LAKE] or raw[WON])
        truncated = self.stall_steps is not None and self.stalled >= self.stall_steps
        if truncated and not self.stall_ends:  # start the count over and play on
            self._stall_mark, self.stalled, truncated = raw[PROGRESS], 0, False
        if self._realtime:  # the bridge itself runs many times faster than real time
            self._frame_due = max(self._frame_due + 1 / self.metadata["render_fps"], time.perf_counter())
            time.sleep(max(0.0, self._frame_due - time.perf_counter()))
        return self._observe(raw), reward, terminated, truncated, {"best_progress": self.best_progress}

    def render(self):
        if self.render_mode == "rgb_array":
            width, height, rgb = self.client.frame()
            return np.frombuffer(rgb, np.uint8).reshape(height, width, 3)
        # "human": the game window draws itself

    def save_state(self):
        state_id = self.client.save_state()
        self._saved_best[state_id] = self.best_progress
        return state_id

    def load_state(self, state_id):
        raw = self.client.load_state(state_id)
        self.best_progress = self._saved_best[state_id]
        self._progress = raw[PROGRESS]
        self._actions = deque([(0.0, 0.0)] * PREV_ACTIONS, maxlen=PREV_ACTIONS)
        return self._observe(raw)

    def _place(self, along, settle=30):
        """The climber put just above the route at `along` (hammer up where it fits), then left to settle with a still
        mouse, so it lands on whatever is below."""
        for lift in (0.5, 1.0, 0.0, -0.5, 2.0, 3.0, 4.0):  # the route runs close under some ledges and over others
            for angle in (90, 60, 120, 30, 150, 0, 180, 210, 330, 240, 300, 270):
                raw = self.client.place(along, lift, angle)
                if raw is not None:
                    for _ in range(settle):
                        raw = self.client.step(0.0, 0.0)
                    return raw
        self.client.reset()
        raise ValueError(f"the climber fits nowhere at route distance {along}")

    def drop_state(self, state_id):
        self.client.drop_state(state_id)
        self._saved_best.pop(state_id, None)

    def _observe(self, raw):
        stuck = self.stalled / self.stall_steps if self.stall_steps else 0.0
        return np.array([*raw, *(x for a in self._actions for x in a), stuck, self.best_progress - raw[PROGRESS]], np.float32)

    def close(self):
        self.client.close()
        if self._proc:
            self._proc.wait(timeout=30)


if __name__ == "__main__":  # python -m goi_env.env: progress_reward's guarantees
    rng = np.random.default_rng(0)
    for _ in range(10000):
        best = rng.uniform(0, 100)
        prev, now = best - rng.uniform(0, 10), best + rng.uniform(-10, 10)
        air = float(rng.integers(2))
        assert progress_reward(prev, now, best, air) == max(0.0, now - best)  # defaults: new records only
        signed = progress_reward(prev, now, best, air, recovery=1, regress=1)
        assert abs(signed - (now - prev)) < 1e-9  # the signed progress change
        # a fall below the record and the climb back never pay, whatever the bonuses, when regress >= recovery
        w = dict(recovery=rng.uniform(0, 1), air_bonus=rng.uniform(0, 1), speed_bonus=rng.uniform(0, 1))
        w["regress"] = w["recovery"] + rng.uniform(0, 1)
        low = prev - rng.uniform(0, 5)
        assert progress_reward(prev, low, best, air, **w) + progress_reward(low, prev, best, 1 - air, **w) <= 1e-9
        # bonuses only multiply new ground
        assert progress_reward(prev, min(now, best), best, 1.0, air_bonus=5, speed_bonus=5) == 0
    print("progress_reward: ok")
