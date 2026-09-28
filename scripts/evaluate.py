"""Evaluate a checkpoint the same way every time: episodes from the bottom with the mean action plus scaled exploration
noise (train.act), no stall cutoff, saved starts or rewinds, just the real game. Prints each episode's best route
progress. One process per game with seeded noise, so a rerun with the same --seed plays the same episodes. With
--then FROM:CHECKPOINT, a specialist takes over from that route progress on (train.chain)."""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

from goi_env import GoiEnv
from train import STALL_SECONDS, chain, parse_then


def play(checkpoint, game_dir, k, steps, episodes, seed, noise=0.5, then=()):
    """One game plays `episodes` episodes; returns each one's best progress."""
    torch.set_num_threads(1)
    player = chain(checkpoint, then, noise)
    env = TimeLimit(GoiEnv(game_dir, k=k, stall_steps=int(STALL_SECONDS * 120 / k), stall_ends=False), steps)
    torch.manual_seed(seed)
    bests = []
    try:
        while len(bests) < episodes:
            obs, _ = env.reset()
            for step in itertools.count():
                obs, _, terminated, truncated, info = env.step(player(obs, step))
                if terminated or truncated:
                    bests.append(info["best_progress"])
                    break
    finally:
        env.close()
    return bests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="a checkpoint .zip from runs/<name>/checkpoints")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--games", type=int, default=5, help="games in parallel (at most 5 per install)")
    ap.add_argument("--seconds", type=float, default=100, help="game time per episode")
    ap.add_argument("--k", type=int, default=4, help="physics steps per action; must match training")
    ap.add_argument("--noise", type=float, default=0.5, help="exploration noise scale: 1 as in training, 0 the mean action")
    ap.add_argument("--seed", type=int, default=0, help="first game's seed (games use seed, seed + 1, ...)")
    ap.add_argument("--then", nargs="*", default=[], metavar="FROM:CHECKPOINT",
                    help="specialists that take over from a route progress on")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    games = min(args.games, args.episodes)
    shares = [args.episodes // games + (i < args.episodes % games) for i in range(games)]
    steps = int(args.seconds * 120 / args.k)
    with ProcessPoolExecutor(games) as ex:
        runs = ex.map(play, [args.checkpoint] * games, [args.game_dir] * games, [args.k] * games, [steps] * games,
                      shares, range(args.seed, args.seed + games), [args.noise] * games,
                      [parse_then(args.then)] * games)
        bests = sorted((b for run in runs for b in run), reverse=True)
    print("best route progress per episode:", ", ".join(f"{b:.0f}" for b in bests))
    print(f"mean {np.mean(bests):.1f}, median {np.median(bests):.1f} over {len(bests)} episodes of {args.seconds:.0f} s")


if __name__ == "__main__":
    main()
