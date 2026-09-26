"""Train PPO on Getting Over It: parallel games, normalized observations, TensorBoard logs, resumable checkpoints."""
import argparse
import os
import re
import time
from contextlib import suppress
from functools import partial

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from goi_env import GoiEnv


def make_env(game_dir, k, episode_steps):
    return TimeLimit(GoiEnv(game_dir, k=k), episode_steps)


def vecnormalize_path(checkpoint):
    """The observation/reward normalization stats saved next to a checkpoint."""
    return re.sub(r"ppo_(\d+)_steps\.zip$", r"ppo_vecnormalize_\1_steps.pkl", checkpoint)


class HeightLog(BaseCallback):
    """TensorBoard: best height per episode (mean over the last 100 episodes) and the best height reached so far."""

    def __init__(self):
        super().__init__()
        self.record = float("-inf")

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        heights = [ep["best_height"] for ep in self.model.ep_info_buffer]
        if heights:
            self.record = max(self.record, *heights)
            self.logger.record("rollout/best_height_mean", sum(heights) / len(heights))
            self.logger.record("rollout/best_height_record", self.record)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--name", default=time.strftime("ppo-%Y%m%d-%H%M%S"), help="run folder under runs/")
    ap.add_argument("--steps", type=int, default=10_000_000, help="environment steps to train for")
    ap.add_argument("--envs", type=int, default=5, help="games in parallel (at most 5 per install)")
    ap.add_argument("--k", type=int, default=4, help="physics steps (1/120 s) per action")
    ap.add_argument("--episode-steps", type=int, default=3000, help="time limit in actions (3000 = 100 s at k=4)")
    ap.add_argument("--save-every", type=int, default=500_000, help="checkpoint interval in environment steps")
    ap.add_argument("--resume", help="a checkpoint .zip to continue from")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    run_dir = os.path.join("runs", args.name)
    checkpoints = os.path.join(run_dir, "checkpoints")
    envs = [partial(make_env, args.game_dir, args.k, args.episode_steps)] * args.envs
    venv = VecMonitor(SubprocVecEnv(envs), info_keywords=("best_height",))
    try:
        if args.resume:
            venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
            model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=run_dir)
        else:
            venv = VecNormalize(venv, gamma=0.995)
            model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=512, gamma=0.995,
                        policy_kwargs=dict(net_arch=[256, 256]), device="cpu", tensorboard_log=run_dir, verbose=1)
        save = CheckpointCallback(max(args.save_every // args.envs, 1), checkpoints, name_prefix="ppo",
                                  save_vecnormalize=True)
        try:
            model.learn(args.steps, callback=[save, HeightLog()], tb_log_name="tb", reset_num_timesteps=not args.resume)
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
