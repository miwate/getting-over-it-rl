"""Record demonstrations of a hard move from a policy's own rare successes, so training can start along them (backplay:
train.py --demos). As in gate.py, a finder plays from the bottom until progress >= --start (seeded); then the policy
tries from there. Each success is kept as the finder's actions from reset plus the try's actions from that state, which
replay exactly in any game (a load in between rebuilds the same physics world; train.py checks each replay).

    python scripts/demos.py runs/bar-explore/checkpoints/ppo_1500000_steps.zip --start 226 --goal 236 --out runs/demos/crux.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from train import STALL_SECONDS, Player

PROGRESS = OBS_FIELDS.index("progress")


def record(seed, args):
    """One game: the finder's climb for this seed, then --tries tries; returns [(climb, try, step of success)]."""
    torch.set_num_threads(1)
    finder, player = Player(args.finder, 0.5), Player(args.checkpoint, args.noise)
    game = GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False)
    demos = []
    try:
        torch.manual_seed(seed)
        for _ in range(args.finder_games):
            obs, _ = game.reset()
            climb = []
            for step in range(3000):
                climb.append(finder(obs, step))
                obs, _, terminated, _, _ = game.step(climb[-1])
                if obs[PROGRESS] >= args.start or terminated:
                    break
            if obs[PROGRESS] >= args.start:
                break
        else:
            return demos
        start = game.save_state()
        for i in range(args.tries):
            torch.manual_seed(seed * 1000 + i)
            obs = game.load_state(start)
            actions, reached = [], None
            for step in range(int(args.seconds * 30)):
                actions.append(player(obs, step))
                obs, _, terminated, _, _ = game.step(actions[-1])
                if reached is None and obs[PROGRESS] >= args.goal:
                    reached = step
                if terminated or reached is not None and step >= reached + 60:  # 2 s past the goal too
                    break
            if reached is not None:
                demos.append((np.array(climb, np.float32), np.array(actions, np.float32), reached))
    finally:
        game.close()
    return demos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the policy whose successes to keep")
    ap.add_argument("--out", required=True, help=".npz file to write")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--finder", default=os.path.join("runs", "rw-port", "checkpoints", "ppo_1500160_steps.zip"))
    ap.add_argument("--finder-games", type=int, default=10)
    ap.add_argument("--start", type=float, required=True, help="route progress the finder plays up to")
    ap.add_argument("--goal", type=float, required=True, help="route progress that counts as success")
    ap.add_argument("--states", type=int, default=48, help="finder seeds 0, 1, ...")
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=16)
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    with ProcessPoolExecutor(args.games) as ex:
        demos = [d for part in ex.map(record, range(args.states), repeat(args)) for d in part]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    arrays = {f"climb{j}": c for j, (c, _, _) in enumerate(demos)}
    arrays.update({f"try{j}": t for j, (_, t, _) in enumerate(demos)})
    np.savez(args.out, reached=np.array([r for _, _, r in demos]), goal=args.goal, **arrays)
    print(f"{len(demos)} successes from {args.states} states x {args.tries} tries -> {args.out}")


if __name__ == "__main__":
    main()
