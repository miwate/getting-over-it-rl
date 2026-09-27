"""Selection training: challengers fine-tuned from the current champion play the same fresh real-game episodes as the
champion, and one replaces it only if it scores clearly better, so training can't go backwards. At the chimney top,
further training kept losing the rare crossing skill that a lucky checkpoint had (docs/RECON.md, "Holding the chimney
crossing").

Each round, --challengers runs of train.py warm-start from the champion for --steps in parallel, one learning rate each
from --lrs, with TRAIN_ARGS. Then the champion and the challengers play the same --episodes episodes (new seeds every
round), scored by mean best progress. runs/<name>/log.jsonl records every round; runs/<name>/champion.txt the champion.
Ctrl+C stops the round's trainers too (they save a checkpoint) and exits. To continue, pass the champion.txt path as
--champion with the same --name: rounds are numbered on from the log, so earlier checkpoints are never overwritten.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from evaluate import play

# The settings of the best run so far (rw-control), plus starts favoring the frontier.
TRAIN_ARGS = ["--saved-starts", "0.35", "--frontier", "15", "--traps", "0.25", "--bounded-mean", "--gsde", "--sde-freq", "8"]


def latest(run_dir):
    zips = [z for z in os.listdir(os.path.join(run_dir, "checkpoints")) if z.startswith("ppo_") and z.endswith("_steps.zip")]
    return os.path.join(run_dir, "checkpoints", max(zips, key=lambda z: int(z.split("_")[1])))


def score(checkpoint, game_dir, seeds, per_game):
    """Best progress of each episode, per_game episodes on one game per seed."""
    with ThreadPoolExecutor(len(seeds)) as ex:
        runs = ex.map(lambda s: play(checkpoint, game_dir, 4, 3000, per_game, s), seeds)
    return [b for run in runs for b in run]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", required=True, help="the checkpoint .zip to start from")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--name", default="select", help="folder under runs/")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--challengers", type=int, default=3)
    ap.add_argument("--envs", type=int, default=4, help="games per challenger")
    ap.add_argument("--steps", type=int, default=1_000_000, help="training steps per challenger and round")
    ap.add_argument("--lrs", default="1e-4,5e-5,2e-4", help="learning rates, one per challenger (cycled)")
    ap.add_argument("--games", type=int, default=10, help="games in parallel for the test")
    ap.add_argument("--episodes", type=int, default=20, help="test episodes per checkpoint (a multiple of --games)")
    ap.add_argument("--margin", type=float, default=1.0, help="mean best progress a challenger must beat the champion by")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    torch.set_num_threads(1)  # the test runs --games policies side by side
    root = os.path.join("runs", args.name)
    os.makedirs(root, exist_ok=True)
    lrs = [float(x) for x in args.lrs.split(",")]
    champion = args.champion
    log_path = os.path.join(root, "log.jsonl")
    first = sum(1 for _ in open(log_path)) if os.path.exists(log_path) else 0
    procs = []
    try:
        for r in range(first, first + args.rounds):
            procs = []
            for i in range(args.challengers):
                name = f"{args.name}/r{r:03d}c{i}"
                cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"), "--name", name,
                       "--game-dir", args.game_dir, "--warm-start", champion, "--envs", str(args.envs), "--steps",
                       str(args.steps), "--save-every", str(args.steps), "--lr", str(lrs[i % len(lrs)]), *TRAIN_ARGS]
                with open(os.path.join("runs", name + ".log"), "w") as out:
                    procs.append((name, subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)))
            for _, p in procs:
                p.wait()
            challengers = [latest(os.path.join("runs", n)) for n, p in procs if p.returncode == 0]
            seeds = range(1000 + 100 * r, 1000 + 100 * r + args.games)
            results = {c: score(c, args.game_dir, seeds, args.episodes // args.games) for c in [champion, *challengers]}
            means = {c: float(np.mean(b)) for c, b in results.items()}
            best = max(challengers, key=means.get, default=None)
            promoted = best is not None and means[best] > means[champion] + args.margin
            with open(log_path, "a") as log:
                log.write(json.dumps({"round": r, "time": time.strftime("%Y-%m-%d %H:%M"), "champion": champion,
                                      "promoted": best if promoted else None,
                                      "scores": {c: {"mean": means[c], "past190": sum(b >= 190 for b in results[c]),
                                                     "bests": results[c]} for c in results}}) + "\n")
            print(f"round {r}: champion {means[champion]:.1f} ({sum(b >= 190 for b in results[champion])} past 190); "
                  + ", ".join(f"{c.split(os.sep)[-3]} {means[c]:.1f}" for c in challengers)
                  + (f" -> promoted {best}" if promoted else " -> champion stays"), flush=True)
            if promoted:
                champion = best
            with open(os.path.join(root, "champion.txt"), "w") as f:
                f.write(champion + "\n")
    except KeyboardInterrupt:
        for _, p in procs:
            p.wait()  # they got the Ctrl+C too and save a checkpoint
        print(f"stopped; champion: {champion}")


if __name__ == "__main__":
    main()
