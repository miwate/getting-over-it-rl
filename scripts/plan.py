"""A receding-horizon planner that uses the game itself as its simulator (exact save/load makes failed tries free), the
expert for expert iteration (Anthony et al., 2017; POLO, Lowrey et al., 2019): it finds moves, a network learns them.

From the current state it plays --candidates futures of --horizon steps and scores each by the route progress it holds
at the end (the lake or the Snake rules a future out). Each future is the policy's own mean action plus a held random
offset of a random size (0 for one of them), and a few are pure random mouse strokes: the plans stay close to how the
policy moves, so it can learn them (random-stroke paths, like explore.py's, proved unlearnable). The best future's first
--commit steps are played for real, then it plans again. The committed steps (observation, action) are the labels.

Robust moves (--robust, --sigma): the game is chaotic, and the best of many futures is mostly the luckiest one. Paths the
explorer found through the garden all reached the goal replayed exactly, and none did with N(0, 0.01) added to each
action: no network can repeat moves that precisely. So the --robust-top best futures are played --robust more times with
N(0, --sigma) added to each action (the policy reacting to it), and a future scores the mean of its plays (the lake or
the Snake in any of them rules it out). The committed steps are played with that noise too (DART, Laskey et al., 2017:
the labels, without noise, then cover the states a slightly-off policy reaches, and show how to recover from them).

DAgger (Ross et al., 2011): with --beta below 1, a decision plays the planner's move only with that chance, else the
policy's own mean actions; then only the decision state is labeled (with the planner's first action). A policy
trained on the planner's states alone drifts into states the planner never visited and can't recover there.

    python scripts/plan.py --policy runs/x/checkpoints/ppo_1_steps.zip --place 256 --goal 300 --out runs/plans/garden.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from explore import strokes
from goi_env import OBS_FIELDS, GoiEnv
from train import Player

PROGRESS = OBS_FIELDS.index("progress")


def future(game, state, policy, offsets, horizon):
    """Play one candidate from `state`: returns (score, actions). Offsets None: pure random strokes. The score is the
    progress held at the end (the pot or hammer touching something); a future that ends in the air scores the lowest
    progress it passed: scoring the end progress, mid-air included, preferred flings that were high at that moment
    and fell (at the anvil's leap the search found ~2 paths in 40 tries, where it found the jump in 20 of 24)."""
    obs, actions, low = game.load_state(state), [], np.inf
    for t in range(horizon):
        a = offsets[t] if policy is None else np.clip(policy(obs, t) + offsets[t], -1, 1)
        actions.append(np.asarray(a, np.float32))
        obs, _, terminated, _, _ = game.step(actions[-1])
        if terminated and (obs[OBS_FIELDS.index("won")] < 0.5):
            return -np.inf, actions  # the lake or the Snake
        low = min(low, float(obs[PROGRESS]))
    held = obs[OBS_FIELDS.index("pot_contact")] or obs[OBS_FIELDS.index("head_contact")]
    return (float(obs[PROGRESS]) if held else low), actions


def decide(game, state, policy, rng, args):
    """The best future from `state`: (its policy or None, its offsets, its actions); zeros if every future fails."""
    tries = []
    for k in range(args.candidates):
        who = None if k < args.random else policy
        size = 1.0 if who is None else 0.0 if k == args.random else rng.uniform(0, args.spread)
        offsets = size * strokes(rng, args.horizon)
        tries.append([*future(game, state, who, offsets, args.horizon), who, offsets])
    tries.sort(key=lambda t: -t[0])
    if args.robust:
        tries = tries[:args.robust_top]
        for t in tries:
            plays = [future(game, state, t[2], t[3] + args.sigma * rng.standard_normal(t[3].shape), args.horizon)[0]
                     for _ in range(args.robust)]
            t[0] = float(np.mean([t[0], *plays]))
        tries.sort(key=lambda t: -t[0])
    if tries[0][0] == -np.inf:
        return None, np.zeros((args.horizon, 2), np.float32), None
    return tries[0][2], tries[0][3], tries[0][1]


def plan_run(seed, args):
    """One game: plan from a placed start until the goal or the time budget. Returns the labels and the timeline."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = Player(args.policy, 0.0)
    game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False, snake_ends=True)
    X, A, timeline = [], [], []
    try:
        obs, _ = game.reset(options={"place": args.place + 0.5 * seed, "dx": args.dx})
        for decision in range(int(args.seconds * 30 / args.commit)):
            state = game.save_state()
            who, offsets, _ = decide(game, state, policy, rng, args)
            obs = game.load_state(state)
            game.drop_state(state)
            planner = rng.random() < args.beta
            for t in range(args.commit):
                move = np.clip(offsets[t] if who is None else who(obs, t) + offsets[t], -1, 1).astype(np.float32)
                if planner or t == 0:
                    X.append(obs)
                    A.append(move)
                a = move if planner else np.asarray(policy(obs, t), np.float32)
                noise = args.sigma * rng.standard_normal(2) if args.robust else 0.0
                obs, _, terminated, _, _ = game.step(np.clip(a + noise, -1, 1).astype(np.float32))
                if terminated:
                    break
            timeline.append(float(obs[PROGRESS]))
            if obs[PROGRESS] >= args.goal or terminated:
                break
    finally:
        game.close()
    return np.array(X, np.float32), np.array(A, np.float32), np.array(timeline, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="the checkpoint the plans stay close to")
    ap.add_argument("--place", type=float, required=True, help="route distance to start at (+0.5 per game)")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--goal", type=float, required=True)
    ap.add_argument("--seconds", type=float, default=60, help="game time each game may take")
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--random", type=int, default=4, help="candidates that are pure random strokes")
    ap.add_argument("--spread", type=float, default=1.5, help="largest held offset added to the policy's action")
    ap.add_argument("--horizon", type=int, default=60, help="steps each candidate plays (60: 2 s)")
    ap.add_argument("--commit", type=int, default=15, help="steps of the best candidate played before planning again")
    ap.add_argument("--robust", type=int, default=0, help="noisy plays that score each of the best futures (0: off)")
    ap.add_argument("--robust-top", type=int, default=8, help="how many of the best futures get the noisy plays")
    ap.add_argument("--sigma", type=float, default=0.05, help="action noise of the noisy plays and the committed steps")
    ap.add_argument("--beta", type=float, default=1.0, help="chance a decision plays the planner's move (DAgger)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--out", help=".npz of the labels (observations, actions) and each game's progress timeline")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    with ProcessPoolExecutor(args.games) as ex:
        runs = list(ex.map(plan_run, range(args.games), [args] * args.games))
    for g, (_, _, tl) in enumerate(runs):
        reached = np.argmax(tl >= args.goal) if (tl >= args.goal).any() else None
        print(f"game {g}: start {tl[0]:.1f}, best {tl.max():.1f}, end {tl[-1]:.1f}; goal "
              + (f"at {(reached + 1) * args.commit / 30:.1f} s" if reached is not None else "not reached"))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, obs=np.concatenate([r[0] for r in runs]), actions=np.concatenate([r[1] for r in runs]),
                 **{f"timeline{g}": r[2] for g, r in enumerate(runs)})


if __name__ == "__main__":
    main()
