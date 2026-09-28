"""Go-Explore (Ecoffet et al., 2021) for a stretch where no policy ever succeeds: no network, just random held mouse
strokes from an archive of the game's exact states, always returning to the most promising ones. Small lucky gains
(from the playground, 4% of random 15 s tries gain 5 where the trained specialist gains 0) chain up the stretch.

Each game explores on its own from a placed start (--start, --start + 0.5, ... per game). A cell is 1 unit of route
progress x 2 of height; it keeps the state first reached there (or reached in fewer steps). Cells are chosen by
1/sqrt(1 + times chosen) x exp((cell progress - furthest) / --focus). A path to the goal becomes a demo: the placement,
its actions, and where the explorer loaded a saved state (a load rebuilds the physics world, so a replay loads at the
same steps; train.demo_steps). Touching the Snake or the lake throws the segment away.

    python scripts/explore.py --start 290 --goal 335 --out runs/demos/playground.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

from goi_env import OBS_FIELDS, GoiEnv

PROGRESS, POT_Y = OBS_FIELDS.index("progress"), OBS_FIELDS.index("pot_y")


def cell(obs):
    return int(obs[PROGRESS]), int(obs[POT_Y] // 2)


def strokes(rng, n):
    """n steps of random mouse strokes, each held 8-24 steps; one in ten a still mouse (the hammer goes loose)."""
    out = []
    while len(out) < n:
        a = np.zeros(2) if rng.random() < 0.1 else rng.uniform(-1, 1, 2)
        a *= rng.uniform(0.3, 1.0) / max(np.abs(a).max(), 1e-6) if a.any() else 1
        out += [a] * int(rng.integers(8, 25))
    return np.array(out[:n], np.float32)


def explore(seed, args):
    """One game's Go-Explore; returns its demos [(place, actions, cuts, reached)] and the furthest progress."""
    rng = np.random.default_rng(seed)
    place = (args.start + 0.5 * seed, args.dx)
    game = GoiEnv(args.game_dir, snake_ends=True)
    archive = {}  # cell -> [state id, path (list of action arrays), steps, times chosen, progress]
    demos, furthest = [], 0.0
    try:
        obs, _ = game.reset(options={"place": place[0], "dx": place[1]})
        archive[cell(obs)] = [game.save_state(), [], 0, 0, float(obs[PROGRESS])]
        furthest, spent = float(obs[PROGRESS]), 0
        while spent < args.steps and len(demos) < args.demos:
            cells = list(archive)
            prog = np.array([archive[c][4] for c in cells])
            w = np.exp((prog - prog.max()) / args.focus) / np.sqrt(1 + np.array([archive[c][3] for c in cells]))
            c = cells[rng.choice(len(cells), p=w / w.sum())]
            state, path, steps, _, _ = archive[c]
            archive[c][3] += 1
            obs = game.load_state(state)
            segment = strokes(rng, args.segment)
            for t, a in enumerate(segment):
                obs, _, terminated, _, _ = game.step(a)
                spent += 1
                if terminated:  # the lake, the Snake (or the win: then the goal is past it anyway)
                    break
                p, k = float(obs[PROGRESS]), cell(obs)
                if k not in archive or steps + t + 1 < archive[k][2]:
                    if k in archive:
                        game.drop_state(archive[k][0])
                    archive[k] = [game.save_state(), path + [segment[:t + 1]], steps + t + 1, 0, p]
                furthest = max(furthest, p)
                if p >= args.goal:
                    new_path = path + [segment[:t + 1]]
                    actions = np.concatenate(new_path)
                    cuts = np.cumsum([0] + [len(s) for s in new_path[:-1]])
                    demos.append((np.array(place, np.float32), actions, cuts, len(actions) - 1))
                    break
    finally:
        game.close()
    return demos, furthest, len(archive)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--start", type=float, required=True, help="route distance to place the climber at (+0.5 per game)")
    ap.add_argument("--dx", type=float, default=0.0, help="placement's sideways offset from the route")
    ap.add_argument("--goal", type=float, required=True, help="route progress that makes a path a demo")
    ap.add_argument("--out", required=True, help=".npz of demos (train.py --demos, imitate.py)")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300_000, help="game steps each game may explore for")
    ap.add_argument("--demos", type=int, default=3, help="demos each game stops after")
    ap.add_argument("--segment", type=int, default=120, help="steps of random strokes per return (4 s)")
    ap.add_argument("--focus", type=float, default=4.0, help="how strongly cells near the furthest are chosen")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    with ProcessPoolExecutor(args.games) as ex:
        results = list(ex.map(explore, range(args.games), repeat(args)))
    demos = [d for part, _, _ in results for d in part]
    for g, (part, furthest, cells) in enumerate(results):
        print(f"game {g}: {cells} cells, furthest {furthest:.1f}, demos {len(part)}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    arrays = {}
    for j, (place, actions, cuts, reached) in enumerate(demos):
        arrays.update({f"place{j}": place, f"try{j}": actions, f"cuts{j}": cuts})
    np.savez(args.out, reached=np.array([r for *_, r in demos]), goal=args.goal, **arrays)
    print(f"{len(demos)} demos -> {args.out}")


if __name__ == "__main__":
    main()
