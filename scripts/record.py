"""Record a video of one rollout of a trained policy, from the game's own frames.

Actions are the mean plus half the exploration noise (--noise, train.act): the mean alone can settle into a pose it
never leaves, and the full noise it trained with blurs the moves it knows.
"""
import argparse
import os

import imageio.v2 as imageio
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from goi_env import GoiEnv
from train import act, vecnormalize_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="a checkpoint .zip from runs/<name>/checkpoints")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--out", help="video file (default: the checkpoint's name with .mp4)")
    ap.add_argument("--seconds", type=float, default=60, help="game time to record")
    ap.add_argument("--k", type=int, default=4, help="physics steps per action; must match training")
    ap.add_argument("--colliders", action="store_true", help="draw the physics shapes over the game")
    ap.add_argument("--noise", type=float, default=0.5, help="exploration noise scale: 1 as in training, 0 the mean action")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    fps = 120 / args.k
    out = args.out or args.checkpoint[:-4] + ("-colliders" if args.colliders else "") + ".mp4"
    game = TimeLimit(GoiEnv(args.game_dir, k=args.k, render_mode="rgb_array", width=1280, height=720,
                            game_args=["-goi-colliders"] if args.colliders else []),
                     int(args.seconds * fps))
    env = VecNormalize.load(vecnormalize_path(args.checkpoint), DummyVecEnv([lambda: game]))
    env.training = env.norm_reward = False
    model = PPO.load(args.checkpoint, device="cpu")
    try:
        obs = env.reset()
        with imageio.get_writer(out, fps=fps) as video:
            video.append_data(env.render())
            step = 0
            while True:
                obs, _, done, infos = env.step(act(model, obs, step, args.noise))
                step += 1
                if done[0]:
                    break
                video.append_data(env.render())
    finally:
        env.close()
    print(f"wrote {out} ({infos[0]['best_progress']:.1f} best route progress)")


if __name__ == "__main__":
    main()
