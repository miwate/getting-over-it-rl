"""Film the networks playing alone, in the game that played: a game plays the chain (runs/chain.json) from the bottom as
marathon.py does, until it reaches --reach progress (then --after more seconds) or --minutes run out; if it got there,
rendering goes on in that same game and its own moves are replayed from the start and filmed (replays are exact
within a launch; across launches, long network games drift apart). Tries seeds until one gets there.

    python scripts/play_film.py --reach 415 --out runs/marathon/to-chapel.mp4
"""
import argparse
import json
import os

import numpy as np
import torch

from film import film
from goi_env import OBS_FIELDS, GoiEnv
from train import STALL_SECONDS, chain as make_chain

PROGRESS = OBS_FIELDS.index("progress")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default=os.path.join("runs", "chain.json"))
    ap.add_argument("--reach", type=float, required=True, help="progress a game must reach to be filmed")
    ap.add_argument("--after", type=float, default=10, help="seconds filmed after reaching it")
    ap.add_argument("--minutes", type=float, default=30, help="game time each try may take")
    ap.add_argument("--seeds", type=int, default=10, help="tries (seeds 0, 1, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Networks alone, real game, real time")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality: lower is better and bigger (16: ~4 MB/s)")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    links = json.load(open(args.chain))
    base, _, noise = links[0][1].partition("@")
    then = [(start, spec.partition("@")[0], float(spec.partition("@")[2] or 0.5)) for start, spec in links[1:]]
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        player = make_chain(base, then, float(noise or 0.5))
        game = GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False, record_reset=5)
        actions, trace, reached = [], [], None
        obs, _ = game.reset()
        for step in range(int(args.minutes * 60 * 30)):
            a = np.asarray(player(obs, step), np.float32)
            obs, *_ = game.step(a)
            actions.append(a)
            trace.append(obs[PROGRESS])
            if reached is None and obs[PROGRESS] >= args.reach:
                reached = step
            if reached is not None and step >= reached + args.after * 30:
                break
        if reached is None:
            print(f"seed {seed}: best {max(trace):.1f} in {args.minutes:.0f} min; next seed", flush=True)
            game.close()
            continue
        print(f"seed {seed}: reached {args.reach:.0f} at {reached / 1800:.1f} min; filming", flush=True)
        z = dict(try0=np.array(actions), trace=np.array(trace, np.float32), cuts0=np.array([0]),
                 climb0=np.zeros((0, 2), np.float32), reached=np.array([len(actions) - 1]))
        np.savez(args.out[:-4] + ".npz", goal=max(trace), **z)
        game.start_rendering(2560, 1440)
        film(game, z, args.out, args.title, args.chain, crf=args.crf)
        return
    raise SystemExit(f"no try reached {args.reach} in {args.seeds} seeds")


if __name__ == "__main__":
    main()
