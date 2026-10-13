"""Self-imitation labels (Oh et al. 2018): a network plays from exported states (marathon arrivals, the states along
search successes) with its exploration noise, and each try that reaches --goal before falling more than --fall below
its record is kept: its (observation, action) pairs from the start, in plan.py's label format. distill.py then moves
the network's mean toward them, so moves it made by luck become moves it makes on purpose; the girder launch, decided
in its first third of a second, went from 4% to 27% that way (imitate.py, docs/RECON.md). The moves are the network's
own, in its own style, which it can learn; the search's knife-edge moves it couldn't.

With --stats (a run's shared demo-stats.npy, --states listed in the run's --demos order and --window its demo window),
only the starts that run crossed from less than --hard of the time: the hard part of the stretch.

    python scripts/successes.py runs/jump2-m18-b/checkpoints/ppo_20003840_steps.zip@1 --states runs/seeds/jump*.npz --window 420 445 --goal 445 --out runs/labels/jump-1.npz
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from train import STALL_SECONDS, Player

PROGRESS, POT, HEAD = (OBS_FIELDS.index(f) for f in ("progress", "pot_contact", "head_contact"))
G = {}


def init(args):
    torch.set_num_threads(1)
    path, _, noise = args.checkpoint.partition("@")
    G.update(args=args, player=Player(path, float(noise or 1.0)),
             game=GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False))


def tries(job):
    """From one start: the successful tries' (observations, actions), and how many tries succeeded."""
    (f, i), seed = job
    args, game, player = G["args"], G["game"], G["player"]
    sid = game.import_state(np.load(f)[f"s{i}"].tobytes())
    X, A, wins = [], [], 0
    try:
        for k in range(args.tries):
            torch.manual_seed(seed * 1000 + k)
            obs = game.load_state(sid)
            best, xs, acts = float(obs[PROGRESS]), [], []
            for t in range(int(args.seconds * 30)):
                a = np.asarray(player(obs, t), np.float32)
                xs.append(obs)
                acts.append(a)
                obs, *_ = game.step(a)
                if obs[POT] or obs[HEAD]:  # held there, not a mid-air peak
                    best = max(best, float(obs[PROGRESS]))
                if best >= args.goal:
                    X += xs
                    A += acts
                    wins += 1
                    break
                if best - obs[PROGRESS] > args.fall:
                    break
    finally:
        game.drop_state(sid)
    return X, A, wins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="checkpoint@NOISE (default noise 1: as it explores in training)")
    ap.add_argument("--states", nargs="+", required=True, help="exported state files (globs work)")
    ap.add_argument("--window", type=float, nargs=2, required=True, metavar=("LO", "HI"), help="starts' progress")
    ap.add_argument("--goal", type=float, required=True)
    ap.add_argument("--seconds", type=float, default=15, help="time a try may take")
    ap.add_argument("--fall", type=float, default=5, help="a try ends when it falls this far below its record")
    ap.add_argument("--tries", type=int, default=10, help="tries per start")
    ap.add_argument("--stats", help="a run's demo-stats.npy: keep only starts it crossed from less than --hard")
    ap.add_argument("--hard", type=float, default=0.3)
    ap.add_argument("--near", type=float, help="then only starts at this progress or more (the crux at a stretch's end)")
    ap.add_argument("--most", type=int, default=600, help="at most this many starts (spread out)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--out", required=True, help=".npz of labels (obs, actions) for distill.py")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    lo, hi = args.window
    starts, progress = [], []
    for f in [f for pattern in args.states for f in sorted(glob.glob(pattern))]:
        p = np.load(f)["progress"]
        keep = [i for i in range(len(p)) if lo <= p[i] < hi]
        starts += [(f, i) for i in keep]
        progress += [float(p[i]) for i in keep]
    if args.stats:
        stats = np.load(args.stats)[:len(starts)]
        if not np.allclose(stats[:, 2], progress):
            ap.error("--states aren't the run's demos in its order (progress differs)")
        p = (stats[:, 1] + 1) / (stats[:, 0] + 2)
        starts = [s for s, q, n, at in zip(starts, p, stats[:, 0], progress) if n >= 1 and q < args.hard
                  and (args.near is None or at >= args.near)]
    elif args.near is not None:
        starts = [s for s, at in zip(starts, progress) if at >= args.near]
    starts = [starts[int(k)] for k in np.linspace(0, len(starts) - 1, min(len(starts), args.most))] if starts else []
    print(f"{len(starts)} starts at {lo:.0f}-{hi:.0f}, {args.tries} tries each, to {args.goal:.0f}", flush=True)
    X, A, wins = [], [], []
    with ProcessPoolExecutor(args.games, initializer=init, initargs=(args,)) as ex:
        for x, a, w in ex.map(tries, [(s, n) for n, s in enumerate(starts)]):
            X += x
            A += a
            wins.append(w)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    width = len(X[0]) if X else 0
    np.savez(args.out, obs=np.array(X, np.float32).reshape(len(X), width), actions=np.array(A, np.float32).reshape(-1, 2))
    wins = np.array(wins)
    print(f"{wins.sum()} successes of {len(starts) * args.tries} tries ({np.mean(wins > 0):.0%} of starts crossed at "
          f"least once); {len(X)} labels -> {args.out}")


if __name__ == "__main__":
    main()
