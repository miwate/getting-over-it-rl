"""A search agent that climbs the real game from the bottom (expert iteration's expert, playing): before each half second
it tries futures with the game's own save/load (plan.future: the prior policy's action plus held random offsets, a few
pure random strokes) and commits the best. Without headway for --stall seconds, it backtracks 2-10 s and searches
harder (more futures, more random ones); gaining ground relaxes it again. The committed path is saved as a demo from
the bottom (replayable: train.demo_steps) and every committed step is a label for distill.py.

    python scripts/climb.py runs/rw-port/checkpoints/ppo_1500160_steps.zip --then 224:runs/bar-sil/checkpoints/ppo_7_steps.zip --out runs/climbs/a
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from explore import strokes
from goi_env import OBS_FIELDS, GoiEnv
from plan import future
from train import chain, parse_then

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")


def decide(game, state, prior, rng, effort, args):
    """The best of the futures from `state` at this search effort (0 normal, up to 3)."""
    n, n_random = args.candidates * (1 + effort), args.random * (1 + 2 * effort)
    best, best_actions = -np.inf, None
    for k in range(n):
        if k < n_random:
            score, actions = future(game, state, None, strokes(rng, args.horizon), args.horizon)
        else:
            size = 0.0 if k == n_random else rng.uniform(0, args.spread)
            score, actions = future(game, state, prior, size * strokes(rng, args.horizon), args.horizon)
        if score > best:
            best, best_actions = score, actions
    return best_actions


def climb(seed, args):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prior = chain(args.prior, parse_then(args.then), 0.0)
    game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False, snake_ends=True)
    out = f"{args.out}-{seed}.npz"
    path, cuts, X, A, timeline = [], [], [], [], []
    history = []  # (state id, path length) per decision, the last `keep`
    keep = int(12 * 30 / args.commit)
    best, since, effort, won = -np.inf, 0, 0, False

    def save():
        np.savez(out, reached=np.array([len(path) - 1]), goal=best, climb0=np.zeros((0, 2), np.float32),
                 try0=np.array(path, np.float32).reshape(-1, 2), cuts0=np.array(cuts), obs=np.array(X, np.float32),
                 actions=np.array(A, np.float32), timeline=np.array(timeline, np.float32))

    try:
        obs, _ = game.reset()
        for decision in range(int(args.minutes * 60 * 30 / args.commit)):
            state = game.save_state()
            history.append((state, len(path)))
            if len(history) > keep:
                game.drop_state(history.pop(0)[0])
            actions = decide(game, state, prior, rng, effort, args) or [np.zeros(2, np.float32)]
            obs = game.load_state(state)
            cuts.append(len(path))
            for a in actions[:args.commit]:
                X.append(obs)
                A.append(a)
                path.append(a)
                obs, _, terminated, _, _ = game.step(a)
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
            if since * args.commit / 30 >= args.stall and len(history) > 4:  # stuck: back 2-10 s, search harder
                k = int(rng.integers(4, min(len(history), int(10 * 30 / args.commit)) + 1))
                back, length = history[-k]
                for st, _ in history[-k + 1:]:
                    game.drop_state(st)
                history = history[:-k + 1]
                obs = game.load_state(back)
                del path[length:]
                cuts = [c for c in cuts if c < length]
                since, effort = 0, min(effort + 1, 3)
            if decision % 200 == 0:
                save()
    finally:
        save()
        game.close()
    return seed, best, won, len(path) / 30, len(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prior", help="the policy the searches start from (the chain's base)")
    ap.add_argument("--then", nargs="*", default=[], metavar="FROM:CHECKPOINT", help="priors by route progress")
    ap.add_argument("--out", required=True, help="path prefix: each game writes <out>-<seed>.npz")
    ap.add_argument("--minutes", type=float, default=30, help="game time each game may take (counting retries)")
    ap.add_argument("--stall", type=float, default=10, help="seconds without headway before backtracking")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--spread", type=float, default=1.5)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--commit", type=int, default=15)
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
