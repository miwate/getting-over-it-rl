"""From where marathon games first arrived at a stretch of the route (marathon.py's exported states, loaded exactly: no
replay), how often does each candidate cross it (reach GOAL within SECONDS) and how far does it get, playing as in a
marathon (episode-like record and stall count)? Candidates are judged on the same states and seeds. One game per state.

    python scripts/arrival_test.py 415 465 90 runs/chapel-a/checkpoints/ppo_10004480_steps.zip@1 --states runs/marathon/m8-*-states.npz
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from train import STALL_SECONDS, Player, demo_steps

PROGRESS, POT, HEAD = (OBS_FIELDS.index(f) for f in ("progress", "pot_contact", "head_contact"))
ARRIVAL = 0  # marathon.py's kind of a state saved where a game first reached a section


def arrivals(files, at, goal, after=None):
    """(file, index) of each game's state where it first reached `at` (an arrival export at a section start), else its
    first state in [at, goal). With `after`: all its states in [at, goal) saved after it had been at that progress
    (climbing back after a fall from above)."""
    out = []
    for f in files:
        z = np.load(f)
        p, kind = z["progress"], z["kind"]
        if after is not None:
            first = min(z["step"][p >= after], default=np.inf)
            out += [(f, i) for i in range(len(p)) if at <= p[i] < goal and z["step"][i] > first]
            continue
        first = [i for i in range(len(p)) if kind[i] == ARRIVAL and at <= p[i] < at + 3] or \
            [i for i in range(len(p)) if at <= p[i] < goal]
        if first:
            out.append((f, first[0]))
    return out


def run(state, args, candidates):
    f, i = state
    torch.set_num_threads(1)
    game = GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False, record_reset=5)
    try:
        if i is None:  # no marathon got here: where the path first reaches `at`
            for _, obs, _ in demo_steps(game, np.load(f), 0):
                if obs[PROGRESS] >= args.at:
                    sid = game.save_state()
                    break
            else:
                raise RuntimeError(f"{f} never reaches {args.at}")
        else:
            sid = game.import_state(np.load(f)[f"s{i}"].tobytes())
        out = []
        for spec in candidates:
            path, _, noise = spec.partition("@")
            player = Player(path, float(noise or 0.5))
            wins, gain = 0, 0.0
            for k in range(args.tries):
                torch.manual_seed(k)
                obs = game.load_state(sid)
                start = best = obs[PROGRESS]
                for step in range(int(args.seconds * 30)):
                    obs, *_ = game.step(player(obs, step))
                    if obs[POT] or obs[HEAD]:  # held there: the anvil's leap peaked at 543 in the air 52 times
                        best = max(best, obs[PROGRESS])
                    if best >= args.goal:
                        wins += 1
                        break
                gain += min(best, args.goal) - start
            out.append((wins, gain / args.tries))
        return out
    finally:
        game.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("at", type=float, help="the stretch's start (a section start: marathon.py exports arrivals there)")
    ap.add_argument("goal", type=float)
    ap.add_argument("seconds", type=float)
    ap.add_argument("candidates", nargs="+", help="checkpoints, each with @NOISE (default 0.5)")
    ap.add_argument("--states", nargs="+", required=True, help="marathon.py's exported state files (globs work)")
    ap.add_argument("--path", default=os.path.join("runs", "climbs", "win-e4-states.npz"),
                    help="with no exported state at the stretch: start where this path reaches it (its exported "
                         "states, path_states.py, or the path itself, replayed)")
    ap.add_argument("--after", type=float, help="test re-climbs: from the states saved in the stretch after a game "
                                                 "had been at this progress (at most --most of them, spread out)")
    ap.add_argument("--fill", nargs="*", default=[], help="with fewer than --least marathon states: also the states of "
                                                          "these files (seeds, paths) at at..at+3, all of them")
    ap.add_argument("--least", type=int, default=8)
    ap.add_argument("--most", type=int, default=24)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--games", type=int, default=8, help="games at once (one per state; ~2.3 GB of RAM each)")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    states = arrivals(sorted(f for pattern in args.states for f in glob.glob(pattern)), args.at, args.goal, args.after)
    if len(states) < args.least and args.after is None:  # few games got here (the anvil's cliff: 1): seeds' states too
        for f in args.fill:
            p = np.load(f)["progress"]
            states += [(f, i) for i in range(len(p)) if args.at <= p[i] < args.at + 3]
    states = [states[int(k)] for k in np.linspace(0, len(states) - 1, min(len(states), args.most))] if states else []
    if not states:  # no marathon game got here yet: the path's state, tried 4 times as often
        print(f"no exported state at {args.at:.0f}-{args.goal:.0f}: from {args.path}")
        exported = "exported" in np.load(args.path).files  # path_states.py's: loaded, not replayed
        states = arrivals([args.path], args.at, args.goal) if exported else [(args.path, None)]
        args.tries *= 4
    with ProcessPoolExecutor(min(len(states), args.games)) as ex:
        res = list(ex.map(run, states, repeat(args), repeat(args.candidates)))
    what = "arrivals" if args.after is None else f"states (after {args.after:.0f})"
    print(f"reach {args.goal:.0f} within {args.seconds:.0f} s from {len(states)} marathon {what} at {args.at:.0f} "
          f"({args.tries} tries each):")
    for c, spec in enumerate(args.candidates):
        wins = sum(r[c][0] for r in res)
        gain = sum(r[c][1] for r in res) / len(res)
        print(f"  {wins:3d}/{args.tries * len(res)} gain {gain:5.1f} {spec}   per state: {[r[c][0] for r in res]}")


if __name__ == "__main__":
    main()
