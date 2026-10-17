"""Section map: where can a policy climb on its own? A winning climb path (climb.py) is replayed and the state where it
enters each section of the route is kept; from there the policy plays alone (no save/load) for --seconds, --tries
times. Per section: how often it reaches the next section (held: pot or hammer touching), how far it gets on
average, and how often it falls more than 5 below where it started.

    python scripts/sections.py runs/rw-port/checkpoints/ppo_1500160_steps.zip --then 224:runs/bar-sil/checkpoints/ppo_7_steps.zip
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from train import STALL_SECONDS, chain, demo_steps, parse_then

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")
# Route progress where each section starts (docs/RECON.md, the level route); the last one is the win.
SECTIONS = [(9, "lakeside"), (60, "rocks"), (120, "tree and slope"), (176, "chimney"), (204, "Construction Wall"),
            (228, "girder"), (245, "hut"), (256, "garden"), (290, "playground"), (310, "boxes"), (335, "furniture"),
            (385, "orange stretch"), (415, "chapel"), (465, "roofs"), (525, "anvil"), (560, "snow"), (630, "bucket"),
            (660, "Ice Mountain"), (735, "radio mast"), (760, "space"), (864, "win")]


def run(units, args):
    """One game: replay the path, keep the state where it first reaches each unit's start (units: (index, start,
    goal), by start), then the tries from each. Returns {index: [(best progress, final progress, won), ...]}."""
    torch.set_num_threads(1)
    player = chain(args.checkpoint, parse_then(args.then), args.noise)
    game = GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False)
    z = np.load(args.path)
    starts, todo, goals = {}, sorted(units, key=lambda u: u[1]), {i: g for i, _, g in units}
    try:
        for t, obs, a in demo_steps(game, z, 0):
            while todo and obs[PROGRESS] >= todo[0][1]:
                starts[todo.pop(0)[0]] = game.save_state()
            if not todo or a is None:
                break
        out = {}
        for i, state in starts.items():
            goal = goals[i]
            out[i] = []
            for k in range(args.tries):
                torch.manual_seed(1000 * i + k)
                obs = game.load_state(state)
                best = obs[PROGRESS]
                for step in range(int(args.seconds * 30)):
                    obs, _, terminated, _, info = game.step(player(obs, step))
                    best = max(best, info["held_best"])  # held (touching), not a mid-air peak
                    if best >= goal or obs[WON] > 0.5 or terminated:
                        break
                out[i].append((float(best), float(obs[PROGRESS]), bool(obs[WON] > 0.5)))
            game.drop_state(state)
    finally:
        game.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the policy (the chain's base)")
    ap.add_argument("--then", nargs="*", default=[], metavar="FROM:CHECKPOINT", help="specialists by route progress")
    ap.add_argument("--path", default=os.path.join("runs", "climbs", "win-e4.npz"), help="a winning climb.py path")
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--tries", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=90, help="game time per try")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--split", type=int, default=1, help="also start from this many points along each section")
    ap.add_argument("--bounds", nargs="*", metavar="FROM:NAME", help="other sections than SECTIONS, by where each "
                                                                      "starts, the last one the goal (e.g. 864:win)")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    units = []  # (index, start, goal, name)
    sections = [(float(a), b) for a, b in (s.split(":", 1) for s in args.bounds)] if args.bounds else SECTIONS
    for i in range(len(sections) - 1):
        (lo, name), hi = sections[i], sections[i + 1][0]
        for k in range(args.split):
            units.append((len(units), lo + (hi - lo) * k / args.split, hi, name if k == 0 else ""))
    n = len(units)
    chunks = [[u[:3] for u in units[g::args.games]] for g in range(args.games)]
    results = {}
    with ProcessPoolExecutor(args.games) as ex:
        for part in ex.map(run, chunks, repeat(args)):
            results.update(part)
    print(f"{args.tries} tries of {args.seconds:.0f} s from where the path reaches each start; noise {args.noise}")
    print(f"{'section':<20}{'from':>6}{'to':>6}{'crossed':>10}{'mean gain':>11}{'fell':>7}")
    for i in range(n):
        _, start, goal, name = units[i]
        if i not in results:
            print(f"{name:<20}{start:>6.0f}  (not reached)")
            continue
        tries = results[i]
        crossed = sum(b >= goal or w for b, _, w in tries)
        gain = np.mean([b - start for b, _, _ in tries])
        fell = sum(f < start - 5 for _, f, _ in tries)
        print(f"{name:<20}{start:>6.0f}{goal:>6.0f}{crossed:>5}/{len(tries):<4}{gain:>11.1f}{fell:>4}/{len(tries)}")


if __name__ == "__main__":
    main()
