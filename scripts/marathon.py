"""The real test of the goal: chained networks play one continuous game each from the bottom for a long stretch of game
time, as a person would: no save/load, no search, falls allowed, the Snake played on. Per game: the best progress and
when each section was first reached, where the time went (the sections it was stuck in), and its big falls (from where
to where). Each game's actions and progress are saved (<out>-<seed>.npz, demo format: replayable, so states where it got
stuck can become training starts).

    python scripts/marathon.py runs/rw-port/checkpoints/ppo_1500160_steps.zip --then 224:runs/bar-sil/checkpoints/ppo_7_steps.zip 250:runs/garden-path/checkpoints/ppo_10000000_steps.zip --minutes 60 --out runs/marathon/a
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from sections import SECTIONS
from train import STALL_SECONDS, chain, parse_then

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")
FALL = 10  # progress lost from a local peak that counts as a fall


def section(p):
    return max(i for i, (start, _) in enumerate(SECTIONS) if p >= start or i == 0)


def play(seed, args):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    player = chain(args.checkpoint, parse_then(args.then), args.noise)
    # The agent's own record and stall count as in training episodes (GoiEnv record_reset; the stall count re-arms).
    game = GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False, record_reset=args.record_reset)
    actions, trace = [], []
    try:
        obs, _ = game.reset()
        for step in range(int(args.minutes * 60 * 30)):
            a = np.asarray(player(obs, step), np.float32)
            obs, *_ = game.step(a)
            actions.append(a)
            trace.append(obs[PROGRESS])
            if obs[WON] > 0.5:
                break
    finally:
        game.close()
    trace = np.array(trace, np.float32)
    np.savez(f"{args.out}-{seed}.npz", climb0=np.zeros((0, 2), np.float32), try0=np.array(actions),
             cuts0=np.array([0]), trace=trace, reached=np.array([len(actions) - 1]), goal=float(trace.max()))
    return trace, bool(obs[WON] > 0.5)


def falls(trace):
    """(from, to) of each drop of more than FALL below the running peak since the last fall."""
    out, peak, low = [], trace[0], trace[0]
    for p in trace:
        if p > peak:
            if peak - low > FALL:
                out.append((peak, low))
            peak = low = p
        low = min(low, p)
    if peak - low > FALL:
        out.append((peak, low))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the chain's base")
    ap.add_argument("--then", nargs="*", default=[], metavar="FROM:CHECKPOINT[@NOISE]", help="specialists")
    ap.add_argument("--noise", type=float, default=0.5, help="noise of the chain's members without their own")
    ap.add_argument("--minutes", type=float, default=60, help="game time per game")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--record-reset", type=float, default=5,
                    help="the agent's record restarts after a fall of more than this (training's rewind threshold)")
    ap.add_argument("--out", required=True, help="path prefix for each game's .npz")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with ProcessPoolExecutor(args.games) as ex:
        results = list(ex.map(play, range(args.games), repeat(args)))
    n = len(SECTIONS)
    time_in = np.zeros((args.games, n))
    for g, (trace, won) in enumerate(results):
        secs = np.array([section(p) for p in trace])
        time_in[g] = np.bincount(secs, minlength=n)[:n] / 30
        reach = []
        for i, (start, name) in enumerate(SECTIONS[1:], 1):
            hit = np.argmax(trace >= start) if (trace >= start).any() else None
            if hit is not None:
                reach.append(f"{name} {hit / 30 / 60:.1f}m")
        big = sorted(falls(trace), key=lambda f: f[1] - f[0])[:3]
        print(f"game {g}: best {trace.max():.1f}{' WON' if won else ''} in {len(trace) / 30 / 60:.0f} min; "
              f"falls {len(falls(trace))}, biggest " + ", ".join(f"{a:.0f}->{b:.0f}" for a, b in big))
        print(f"   first reached: {', '.join(reach[-4:])}")
    print(f"\nper section, over {args.games} games ({args.minutes:.0f} min each):")
    print(f"{'section':<20}{'reached':>9}{'mean min spent':>16}")
    for i, (start, name) in enumerate(SECTIONS[:-1]):
        reached = sum((tr >= start).any() for tr, _ in results)
        if reached:
            print(f"{name:<20}{reached:>5}/{args.games:<3}{time_in[:, i].mean() / 60:>14.1f}")
    print(f"won: {sum(w for _, w in results)}/{args.games}")


if __name__ == "__main__":
    main()
