"""Replay a found path (demos.py, explore.py or climb.py output), optionally as a video from the game's own frames.
Replays within a launch are exact, but across launches rare contact-heavy moments differ in the last digits and chaos
grows that (a climb's path parted from its record after 31.7 s in a fresh game). So a climb's path is checked against
its recorded progress (`trace`), and climb.py --video films its paths in the game that found them.

    python scripts/replay.py runs/climbs/b-3.npz --video runs/climbs/b-3.mp4
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np

from goi_env import OBS_FIELDS, GoiEnv
from train import demo_steps

PROGRESS = OBS_FIELDS.index("progress")


def replay(game, z, j=0, video=None, every=1):
    """Replay path j of demos file z in `game`, writing every n-th frame to `video` (an .mp4 path; the game must render).
    Returns (best progress, the step where it parted from the recorded trace or None, the largest gap, the last
    observation)."""
    trace = z["trace"] if "trace" in z else np.zeros(0)
    parted, worst, best = None, 0.0, -np.inf
    writer = imageio.get_writer(video, fps=30 / every if every > 1 else 30) if video else None
    try:
        for t, obs, a in demo_steps(game, z, j):
            p = float(obs[PROGRESS])
            best = max(best, p)
            if 0 < t <= len(trace):
                gap = abs(p - trace[t - 1])
                worst = max(worst, gap)
                if parted is None and gap > 1e-3:
                    parted = t
            if writer and t % every == 0:
                writer.append_data(game.render())
    finally:
        if writer:
            writer.close()
    return best, parted, worst, obs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", help=".npz of paths")
    ap.add_argument("--demo", type=int, default=0, help="which path in the file")
    ap.add_argument("--video", help="write an .mp4 of the replay")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame (a time-lapse)")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    z = np.load(args.demos)
    render = dict(render_mode="rgb_array", width=1280, height=720) if args.video else {}
    game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False, **render)
    try:
        best, parted, worst, _ = replay(game, z, args.demo, args.video, args.every)
    finally:
        game.close()
    print(f"best {best:.1f}" + ("" if "trace" not in z else f"; recorded best {z['trace'].max():.1f}, "
          + ("same path" if parted is None else f"parted at {parted / 30:.1f} s, largest gap {worst:.2f}")))


if __name__ == "__main__":
    main()
