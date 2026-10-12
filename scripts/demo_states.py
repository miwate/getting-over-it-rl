"""Export the states along a demos file's paths (explore.py's Go-Explore paths, demos.py's), as seeds for train.py
--demos and round.py --seeds: each path is replayed in a game (train.demo_steps: its placement, actions and the
explorer's loads) and kept if it still reaches the file's goal (the plugin has changed since some were found); its
states every --every steps up to there are exported. Random-stroke paths can't be imitated (garden: 1/320), but their
states are real positions all along a way through a section: starts for the reverse curriculum where the search finds
none (the anvil's cliff: 0 paths from 16 searches).

    python scripts/demo_states.py runs/demos/explore-anvil.npz --out runs/seeds/anvil-explore.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from train import demo_steps

PROGRESS = OBS_FIELDS.index("progress")
PERIODIC = 1
G = {}


def init(game_dir):
    torch.set_num_threads(1)
    G["game"] = GoiEnv(game_dir, stall_steps=300, stall_ends=False)


def replay(job):
    """One path: (whether its replay reached the goal, [(progress, step, exported state)])."""
    f, j, every = job
    z, game = np.load(f), G["game"]
    goal, reached = float(z["goal"]), int(z["reached"][j])
    states, best = [], -np.inf
    for t, obs, _ in demo_steps(game, z, j):
        if obs is None:
            continue
        best = max(best, float(obs[PROGRESS]))
        if t > reached + 1 or best >= goal:
            break
        if t % every == 0:
            sid = game.save_state()
            states.append((float(obs[PROGRESS]), t, game.export_state(sid)))
            game.drop_state(sid)
    return best >= goal, states


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", help="an explore.py or demos.py .npz")
    ap.add_argument("--out", required=True, help=".npz of the exported states along the paths that still work")
    ap.add_argument("--every", type=int, default=10, help="steps between exported states")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    paths = len(np.load(args.demos)["reached"])
    found, kept = [], 0
    with ProcessPoolExecutor(min(args.games, paths), initializer=init, initargs=(args.game_dir,)) as ex:
        for ok, states in ex.map(replay, [(args.demos, j, args.every) for j in range(paths)]):
            if ok:
                found += states
                kept += 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, exported=1, progress=np.array([s[0] for s in found], np.float32),
             step=np.array([s[1] for s in found]), kind=np.full(len(found), PERIODIC),
             **{f"s{i}": np.frombuffer(s[2], np.uint8) for i, s in enumerate(found)})
    p = np.array([s[0] for s in found]) if found else np.zeros(1)
    print(f"{kept} of {paths} paths still reach {float(np.load(args.demos)['goal']):.0f}; {len(found)} states, "
          f"progress {p.min():.0f} to {p.max():.0f} -> {args.out}")


if __name__ == "__main__":
    main()
