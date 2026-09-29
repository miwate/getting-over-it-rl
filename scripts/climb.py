"""A search agent that climbs the real game from the bottom (expert iteration's expert, playing): before each half second
it tries futures with the game's own save/load (plan.decide: the prior policy's action plus held random offsets, a few
pure random strokes; --robust scores the best of them under action noise) and commits the best. Without headway for
--stall seconds, it backtracks 2-10 s and searches harder (more futures, more random ones); gaining ground relaxes it
again. The committed path is saved as a demo from the bottom (replayable: train.demo_steps; `trace` is the progress
after each of its steps, to check a replay) and every committed step is a label for distill.py.

    python scripts/climb.py runs/rw-port/checkpoints/ppo_1500160_steps.zip --then 224:runs/bar-sil/checkpoints/ppo_7_steps.zip --out runs/climbs/a
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from plan import decide
from replay import replay
from train import chain, parse_then

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")


def climb(seed, args):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prior = chain(args.prior, parse_then(args.then), 0.0)
    game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False, snake_ends=True)
    out = f"{args.out}-{seed}.npz"
    path, trace, cuts, X, A, timeline = [], [], [], [], [], []
    history = []  # (state id, path length, progress) per decision, the last `keep`
    keep = int(12 * 30 / args.commit)
    best, since, effort, won = -np.inf, 0, 0, False

    def save():
        np.savez(out, reached=np.array([len(path) - 1]), goal=best, climb0=np.zeros((0, 2), np.float32),
                 try0=np.array(path, np.float32).reshape(-1, 2), cuts0=np.array(cuts), obs=np.array(X, np.float32),
                 actions=np.array(A, np.float32), timeline=np.array(timeline, np.float32),
                 trace=np.array(trace, np.float32))

    try:
        obs, _ = game.reset()
        for decision in range(int(args.minutes * 60 * 30 / args.commit)):
            state = game.save_state()
            history.append((state, len(path), float(obs[PROGRESS])))
            if len(history) > keep:
                game.drop_state(history.pop(0)[0])
            harder = argparse.Namespace(**{**vars(args), "candidates": args.candidates * (1 + effort),
                                           "random": args.random * (1 + 2 * effort)})
            who, offsets, _ = decide(game, state, prior, rng, harder)
            obs = game.load_state(state)
            cuts.append(len(path))
            for t in range(args.commit):
                move = np.clip(offsets[t] if who is None else who(obs, t) + offsets[t], -1, 1).astype(np.float32)
                X.append(obs)
                A.append(move)
                a = np.clip(move + (args.sigma * rng.standard_normal(2) if args.robust else 0), -1, 1).astype(np.float32)
                path.append(a)
                obs, _, terminated, _, _ = game.step(a)
                trace.append(obs[PROGRESS])
                if terminated:
                    break
            p = float(obs[PROGRESS])
            timeline.append(p)
            if obs[WON] > 0.5:
                won = True
                break
            if p > best + 0.5:
                best, since, effort = p, 0, max(effort - 1, 0)
            else:
                since += 1
            near = [i for i, h in enumerate(history) if h[2] >= best - args.fall / 2]
            if (terminated or p < best - args.fall) and near:  # fell (or the lake, the Snake): back near the best
                i = near[-1]
            elif since * args.commit / 30 >= args.stall and len(history) > 4:  # stuck: back 2-10 s
                i = len(history) - int(rng.integers(4, min(len(history), int(10 * 30 / args.commit)) + 1))
            else:
                i = None
            if i is not None:  # and search harder
                back, length, _ = history[i]
                for st, *_ in history[i + 1:]:
                    game.drop_state(st)
                history = history[:i + 1]
                obs = game.load_state(back)
                del path[length:], trace[length:]
                cuts = [c for c in cuts if c < length]
                since, effort = 0, min(effort + 1, 3)
            if decision % 200 == 0:
                save()
    finally:
        save()
        if args.video:  # in this game: a fresh one can part from the path (replay.py)
            try:
                game.start_rendering()
                _, parted, _ = replay(game, np.load(out), video=out[:-4] + ".mp4")
                print(f"game {seed}: video {'matches its path' if parted is None else f'parted at {parted / 30:.1f} s'}")
            except Exception as e:
                print(f"game {seed}: no video ({e!r})")
        game.close()
    return seed, best, won, len(path) / 30, len(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prior", help="the policy the searches start from (the chain's base)")
    ap.add_argument("--then", nargs="*", default=[], metavar="FROM:CHECKPOINT", help="priors by route progress")
    ap.add_argument("--out", required=True, help="path prefix: each game writes <out>-<seed>.npz")
    ap.add_argument("--minutes", type=float, default=30, help="game time each game may take (counting retries)")
    ap.add_argument("--stall", type=float, default=10, help="seconds without headway before backtracking")
    ap.add_argument("--fall", type=float, default=5, help="progress lost below the best that backtracks at once")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--spread", type=float, default=1.5)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--commit", type=int, default=15)
    ap.add_argument("--robust", type=int, default=0, help="noisy plays that score each of the best futures (plan.py)")
    ap.add_argument("--robust-top", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--video", action="store_true", help="at the end, film each game's path in that game (<out>-<seed>.mp4)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with ProcessPoolExecutor(args.games) as ex:
        for seed, best, won, seconds, labels in ex.map(climb, range(args.games), [args] * args.games):
            print(f"game {seed}: best {best:.1f}{', WON' if won else ''}; path {seconds:.0f} s; {labels} labels")


if __name__ == "__main__":
    main()
