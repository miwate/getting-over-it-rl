"""Search seeds for a hard move: from where marathon games stood before it (exported states), the search agent finds ways
through to --goal (plan.decide: the chain's networks plus nudges, the best of a batch of 2 s futures every half second,
backing up to its best state after a fall or when stuck); each success is replayed in its game (exact there) and the
states along it are exported as training starts (train.py --demos). PPO then practices from the brink of the move, in
situations the networks really reach, and learns its own robust moves from there (a found path is knife-edge to copy:
its moves, replayed with 1% noise, failed 12/12). Go-Explore's robustification (Ecoffet et al. 2021), started from the
agent's own states. One game per start state.

    python scripts/seeds.py --states "runs/marathon/m17-*-states.npz" --at 420 --to 430 --goal 445 --out runs/seeds/jump.npz
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from plan import decide
from train import chain

PROGRESS, POT, HEAD = (OBS_FIELDS.index(f) for f in ("progress", "pot_contact", "head_contact"))
PERIODIC = 1


def search(game, start, prior, rng, args):
    """The planner's way from saved state `start` to args.goal: (actions, cuts), or None within args.decisions."""
    obs = game.load_state(start)
    path, cuts, best, stuck, effort = [], [], float(obs[PROGRESS]), 0, 0
    peak = (game.save_state(), 0, 0)  # where it reached its best: state, path length, cuts length
    for _ in range(args.decisions):
        state = game.save_state()
        harder = argparse.Namespace(**{**vars(args), "candidates": args.candidates * (1 + effort),
                                       "random": args.random * (1 + 2 * effort)})
        who, offsets, _ = decide(game, state, prior, rng, harder)
        obs = game.load_state(state)
        game.drop_state(state)
        cuts.append(len(path))
        terminated = False
        for t in range(args.commit):
            move = np.clip(offsets[t] if who is None else who(obs, t) + offsets[t], -1, 1).astype(np.float32)
            path.append(move)
            obs, _, terminated, _, _ = game.step(move)
            if terminated:
                break
        p = float(obs[PROGRESS])
        if p >= args.goal and not terminated and (obs[POT] or obs[HEAD]):  # held there, not a mid-air peak
            game.drop_state(peak[0])
            return path, cuts
        if p > best + 0.5 and not terminated:
            game.drop_state(peak[0])
            best, stuck, effort = p, 0, max(effort - 1, 0)
            peak = (game.save_state(), len(path), len(cuts))
        else:
            stuck += 1
        if terminated or p < best - args.fall or stuck * args.commit / 30 >= args.stall:  # back to the best
            obs = game.load_state(peak[0])
            del path[peak[1]:], cuts[peak[2]:]
            stuck, effort = 0, min(effort + 1, 3)
    game.drop_state(peak[0])
    return None


def run(job, args):
    f, i, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    links = json.load(open(args.chain))
    prior = chain(links[0][1].partition("@")[0], [(s, spec.partition("@")[0]) for s, spec in links[1:]], 0.0)
    game = GoiEnv(args.game_dir, stall_steps=None, snake_ends=True)
    out = []
    try:
        start = game.import_state(np.load(f)[f"s{i}"].tobytes())
        found = search(game, start, prior, rng, args)
        if found:
            path, cuts = found
            obs = game.load_state(start)
            for t, a in enumerate(path):  # the success again, exactly (same game): its states
                if t in cuts:
                    s = game.save_state()
                    obs = game.load_state(s)
                    game.drop_state(s)
                if t % args.every == 0:
                    s = game.save_state()
                    out.append((float(obs[PROGRESS]), t, game.export_state(s)))
                    game.drop_state(s)
                obs, *_ = game.step(a)
            msg = f"through to {obs[PROGRESS]:.1f} in {len(path) / 30:.0f} s; {len(out)} states"
        else:
            msg = "no way through"
    finally:
        game.close()
    return f"{os.path.basename(f)} #{i}: {msg}", out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", required=True, help="exported state files (globs work) to start from")
    ap.add_argument("--at", type=float, required=True, help="start states from this progress...")
    ap.add_argument("--to", type=float, required=True, help="...up to this")
    ap.add_argument("--goal", type=float, required=True, help="progress a search must reach")
    ap.add_argument("--out", required=True, help=".npz of the exported states along the successes")
    ap.add_argument("--starts", type=int, default=16, help="at most this many start states (spread over the files)")
    ap.add_argument("--repeats", type=int, default=1, help="searches per start state (different seeds)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--chain", default=os.path.join("runs", "chain.json"), help="the networks the plans start from")
    ap.add_argument("--decisions", type=int, default=240, help="search budget per start: 0.5 s of play per decision")
    ap.add_argument("--stall", type=float, default=8, help="seconds without a new best before backing up")
    ap.add_argument("--fall", type=float, default=4, help="progress lost below the best that backs up at once")
    ap.add_argument("--every", type=int, default=10, help="steps between exported states along a success")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--spread", type=float, default=1.5)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--commit", type=int, default=15)
    ap.add_argument("--robust", type=int, default=0)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    pool = []
    for f in sorted(f for pattern in args.states for f in glob.glob(pattern)):
        p, kind = np.load(f)["progress"], np.load(f)["kind"]
        pool += [(f, i) for i in range(len(p)) if args.at <= p[i] < args.to]
    rng = np.random.default_rng(0)
    picked = [pool[i] for i in sorted(rng.choice(len(pool), min(args.starts, len(pool)), replace=False))]
    jobs = [(f, i, r) for r in range(args.repeats) for f, i in picked]
    print(f"{len(pool)} start states at {args.at:.0f}-{args.to:.0f}; searching from {len(picked)} x {args.repeats}", flush=True)
    with ProcessPoolExecutor(min(args.games, len(jobs))) as ex:
        found = []
        for msg, part in ex.map(run, jobs, repeat(args)):  # printed here: workers' own output is lost when detached
            print(msg, flush=True)
            found += part
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, exported=1, progress=np.array([s[0] for s in found], np.float32),
             step=np.array([s[1] for s in found]), kind=np.full(len(found), PERIODIC),
             **{f"s{i}": np.frombuffer(s[2], np.uint8) for i, s in enumerate(found)})
    print(f"wrote {args.out}: {len(found)} states")


if __name__ == "__main__":
    main()
