"""Chimney-top benchmark, the judge for training at the frontier: from the same states just below the big rock's
top-left corner (route progress >= 176), how often does a policy get onto the rock (190) within 15 s?

A finder policy plays from the bottom with fixed seeds until it gets there, and that state is every candidate's start.
One process per game with seeded noise, so every run rebuilds the same states (the bridge is bit-exact, except that a
multi-contact moment can resolve differently in another game launch: judge candidates in one call). Keep a
candidate only if it beats rw-control (ported: scripts/port.py) here without losing ground in full games (evaluate.py).

    python scripts/gate.py runs/rw-port/checkpoints/ppo_1500160_steps.zip other.zip other.zip@1
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import torch

from goi_env import OBS_FIELDS, GoiEnv
from train import STALL_SECONDS, Player

PROGRESS = OBS_FIELDS.index("progress")


def find_gate(game, finder, start, steps, games=3):
    """Play from the bottom (up to `games` games of `steps`) until progress >= start; returns that state or None."""
    for _ in range(games):
        obs, _ = game.reset()
        for step in range(steps):
            obs, _, terminated, _, _ = game.step(finder(obs, step))
            if obs[PROGRESS] >= start:
                return game.save_state()
            if terminated:
                break
    return None


def try_from(game, player, gate, steps, goal):
    """One try's best progress; it stops at the goal."""
    obs = game.load_state(gate)
    best = obs[PROGRESS]
    for step in range(steps):
        obs, _, terminated, _, _ = game.step(player(obs, step))
        best = max(best, obs[PROGRESS])
        if best >= goal or terminated:
            break
    return best


def bench(seeds, candidates, args):
    """One game: a gate state per seed, then every candidate's tries from it. Returns {seed: (the state's progress,
    [each candidate's best progress per try])}."""
    torch.set_num_threads(1)
    finder = Player(args.finder, args.finder_noise)
    players = [Player(path, noise) for path, noise in candidates]
    game = GoiEnv(args.game_dir, k=args.k, stall_steps=int(STALL_SECONDS * 120 / args.k), stall_ends=False)
    steps = int(args.seconds * 120 / args.k)
    out = {}
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            if args.place:  # placed along the route instead: for stretches the finder can't reach
                try:
                    game.reset(options={"place": args.start + 0.5 * (seed - args.seed)})
                    gate = game.save_state()
                except ValueError:
                    gate = None
            else:
                gate = find_gate(game, finder, args.start, int(100 * 120 / args.k), args.finder_games)
            if gate is None:
                continue
            out[seed] = (game.load_state(gate)[PROGRESS], [])
            for player in players:
                bests = []
                for i in range(args.tries):
                    torch.manual_seed(seed * 1000 + i)
                    bests.append(try_from(game, player, gate, steps, args.goal))
                out[seed][1].append(bests)
            game.drop_state(gate)
    finally:
        game.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", nargs="+", help="checkpoint .zip files; append @<noise> to override --noise for one")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--finder", default=os.path.join("runs", "rw-port", "checkpoints", "ppo_1500160_steps.zip"),
                    help="the policy that plays up to the gate states (rw-control, ported)")
    ap.add_argument("--finder-noise", type=float, default=0.5)
    ap.add_argument("--finder-games", type=int, default=3, help="100 s games the finder gets per state (more for far states)")
    ap.add_argument("--place", action="store_true",
                    help="place the climber at --start, --start + 0.5, ... (one per state) instead of using the finder")
    ap.add_argument("--noise", type=float, default=0.5, help="candidates' exploration noise scale (train.act)")
    ap.add_argument("--states", type=int, default=16, help="gate states (seeds --seed, --seed + 1, ...)")
    ap.add_argument("--seed", type=int, default=0, help="first state's seed; another one gives independent states")
    ap.add_argument("--tries", type=int, default=20, help="tries per state and candidate")
    ap.add_argument("--seconds", type=float, default=15, help="game time per try")
    ap.add_argument("--start", type=float, default=176, help="route progress of the gate states")
    ap.add_argument("--goal", type=float, default=190, help="route progress that counts as crossing")
    ap.add_argument("--games", type=int, default=8, help="games in parallel")
    ap.add_argument("--k", type=int, default=4, help="physics steps per action; must match training")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    candidates = [(c.rpartition("@")[0], float(c.rpartition("@")[2])) if "@" in c else (c, args.noise)
                  for c in args.candidates]
    games = min(args.games, args.states)
    with ProcessPoolExecutor(games) as ex:
        results = {}
        chunks = [range(args.seed + i, args.seed + args.states, games) for i in range(games)]
        for part in ex.map(bench, chunks, repeat(candidates), repeat(args)):
            results.update(part)
    seeds = sorted(results)
    total = args.tries * len(seeds)
    print(f"{len(seeds)} of {args.states} gate states (progress >= {args.start:.0f}); {args.tries} tries of "
          f"{args.seconds:.0f} s each; success = reaching {args.goal:.0f}")
    for c, (path, noise) in enumerate(candidates):
        wins = [sum(b >= args.goal for b in results[s][1][c]) for s in seeds]
        gain = sum(b - results[s][0] for s in seeds for b in results[s][1][c]) / max(total, 1)
        print(f"  {sum(wins):4d}/{total} = {sum(wins) / max(total, 1):4.0%}  mean gain {gain:4.1f}  noise {noise:<4g} {path}")
        print(f"        per state: {' '.join(map(str, wins))}")


if __name__ == "__main__":
    main()
