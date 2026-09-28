"""Record a video of one rollout of a trained policy, from the game's own frames.

Actions are the mean plus half the exploration noise (--noise, train.act): the mean alone can settle into a pose it
never leaves, and the full noise it trained with blurs the moves it knows. The game is seeded like evaluate.py's, so
--seed N replays the first episode of its game N (rendering doesn't change the physics); --skip starts the video later.
"""
import argparse
import itertools
import os

import imageio.v2 as imageio
import torch
from gymnasium.wrappers import TimeLimit

from goi_env import GoiEnv
from train import STALL_SECONDS, chain, parse_then


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="a checkpoint .zip from runs/<name>/checkpoints")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--out", help="video file (default: the checkpoint's name with .mp4)")
    ap.add_argument("--seconds", type=float, default=60, help="game time to record")
    ap.add_argument("--skip", type=float, default=0, help="game time to play before recording")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="physics steps per action; must match training")
    ap.add_argument("--colliders", action="store_true", help="draw the physics shapes over the game")
    ap.add_argument("--noise", type=float, default=0.5, help="exploration noise scale: 1 as in training, 0 the mean action")
    ap.add_argument("--then", nargs="*", default=[], metavar="FROM:CHECKPOINT",
                    help="specialists that take over from a route progress on (as in evaluate.py)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    fps = 120 / args.k
    out = args.out or args.checkpoint[:-4] + ("-colliders" if args.colliders else "") + ".mp4"
    game = TimeLimit(GoiEnv(args.game_dir, k=args.k, stall_steps=int(STALL_SECONDS * 120 / args.k), stall_ends=False,
                            render_mode="rgb_array", width=1280, height=720,
                            game_args=["-goi-colliders"] if args.colliders else []),
                     int((args.skip + args.seconds) * fps))
    torch.set_num_threads(1)  # as evaluate.play, for the same numbers
    player = chain(args.checkpoint, parse_then(args.then), args.noise)
    torch.manual_seed(args.seed)
    try:
        obs, _ = game.reset()
        with imageio.get_writer(out, fps=fps) as video:
            for step in itertools.count():
                if step >= args.skip * fps:
                    video.append_data(game.render())
                obs, _, terminated, truncated, info = game.step(player(obs, step))
                if terminated or truncated:
                    break
    finally:
        game.close()
    print(f"wrote {out} ({info['best_progress']:.1f} best route progress)")


if __name__ == "__main__":
    main()
