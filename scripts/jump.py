"""Seeds for one hard move by the cross-entropy method over open-loop mouse strokes, from where games stood before it
(exported states). seeds.py's planner only tries the networks' own moves plus nudges, and the networks never jumped at
the anvil (they sat in the trough below it), so its anvil searches found almost nothing. Here each start gets --iters
rounds of --pop stroke plans (--segments strokes of --seg steps each); every play adds N(0, --sigma) to each action, so
the method keeps plans that work despite small errors (a found path replayed with 1% noise failed 12/12: seeds.py).
A plan scores its held progress record plus a little for height (the jump's way up), and nothing if the game ends (the
lake, the Snake). The final plan is played --checks more times with noise; its successes are replayed exactly (same
game) and the states along them exported as training starts (train.py --demos), as seeds.py's are; their observations
with the plan's actions go to <out>-labels.npz (distill.py). One game per start.

    python scripts/jump.py --states runs/climbs/win-e4-states.npz "runs/marathon/m4*-states.npz" --at 532 --to 540 --goal 545 --out runs/seeds/anvil-jump.npz
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

from goi_env import OBS_FIELDS, GoiEnv

PROGRESS, POT_Y = OBS_FIELDS.index("progress"), OBS_FIELDS.index("pot_y")
PERIODIC = 1


def play(game, start, plan, rng, sigma, goal):
    """The plan with noise from saved state `start`: (score, success, the actions played, the observations before
    them)."""
    obs = game.load_state(start)
    top, played, seen = obs[POT_Y], [], []
    for a in plan:
        seen.append(obs)
        a = np.clip(a + rng.normal(0.0, sigma, 2), -1, 1).astype(np.float32)
        obs, _, ended, _, _ = game.step(a)
        played.append(a)
        top = max(top, obs[POT_Y])
        if ended or game.held_best >= goal:
            break
    won = not ended and game.held_best >= goal
    return (-1e9 if ended else game.held_best + 0.1 * top), won, played, seen


def run(job, args):
    f, i, seed = job
    rng = np.random.default_rng(seed)
    game = GoiEnv(args.game_dir, stall_steps=None, snake_ends=True)
    out, labels = [], []
    try:
        start = game.import_state(np.load(f)[f"s{i}"].tobytes())
        mu, sd = np.zeros((args.segments, 2)), np.full((args.segments, 2), 0.6)
        for _ in range(args.iters):
            plans = np.clip(mu + sd * rng.standard_normal((args.pop, args.segments, 2)), -1, 1)
            scores = [play(game, start, np.repeat(p, args.seg, 0), rng, args.sigma, args.goal)[0] for p in plans]
            elite = plans[np.argsort(scores)[-args.elites:]]
            mu, sd = elite.mean(0), elite.std(0) + 0.05
        plan = np.repeat(mu, args.seg, 0)
        checks = [play(game, start, plan, rng, args.sigma, args.goal) for _ in range(args.checks)]
        wins = [played for _, won, played, _ in checks if won]
        # labels as DART's (Laskey et al. 2017): the states of the noisy plays that worked, with the plan's own actions
        labels += [(np.array(seen), plan[:len(seen)]) for _, won, _, seen in checks if won]
        for played in wins[:args.keep]:  # each success again, exactly (same game): its states
            obs = game.load_state(start)
            for t, a in enumerate(played):
                if t % args.every == 0:
                    s = game.save_state()
                    out.append((float(obs[PROGRESS]), t, game.export_state(s)))
                    game.drop_state(s)
                obs, *_ = game.step(a)
        msg = f"{len(wins)}/{args.checks} noisy plays of the plan reached {args.goal}; {len(out)} states"
    finally:
        game.close()
    return f"{os.path.basename(f)} #{i}: {msg}", out, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", required=True, help="exported state files (globs work) to start from")
    ap.add_argument("--at", type=float, required=True, help="start states from this progress...")
    ap.add_argument("--to", type=float, required=True, help="...up to this")
    ap.add_argument("--goal", type=float, required=True, help="held progress a jump must reach")
    ap.add_argument("--out", required=True, help=".npz of the exported states along the successes")
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--segments", type=int, default=12, help="strokes per plan")
    ap.add_argument("--seg", type=int, default=10, help="steps per stroke (a third of a second)")
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elites", type=int, default=6)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.05, help="action noise in every play")
    ap.add_argument("--checks", type=int, default=8)
    ap.add_argument("--keep", type=int, default=3, help="successful plays exported per start")
    ap.add_argument("--every", type=int, default=5, help="steps between exported states along a success")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    pool = []
    for f in sorted(f for pattern in args.states for f in glob.glob(pattern)):
        p = np.load(f)["progress"]
        pool += [(f, i) for i in range(len(p)) if args.at <= p[i] < args.to]
    rng = np.random.default_rng(0)
    picked = [pool[i] for i in sorted(rng.choice(len(pool), min(args.starts, len(pool)), replace=False))]
    print(f"{len(pool)} start states at {args.at:.0f}-{args.to:.0f}; jumping from {len(picked)}", flush=True)
    found, labels = [], []
    with ProcessPoolExecutor(min(args.games, len(picked))) as ex:
        for msg, part, lab in ex.map(run, [(f, i, k) for k, (f, i) in enumerate(picked)], repeat(args)):
            print(msg, flush=True)
            found += part
            labels += lab
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, exported=1, progress=np.array([s[0] for s in found], np.float32),
             step=np.array([s[1] for s in found]), kind=np.full(len(found), PERIODIC),
             **{f"s{i}": np.frombuffer(s[2], np.uint8) for i, s in enumerate(found)})
    if labels:  # distill.py's format
        np.savez(args.out.replace(".npz", "-labels.npz"), obs=np.concatenate([o for o, _ in labels]),
                 actions=np.concatenate([a for _, a in labels]).astype(np.float32))
    print(f"wrote {args.out}: {len(found)} states, {sum(len(o) for o, _ in labels)} labels")


if __name__ == "__main__":
    main()
