"""The real test of the goal: chained networks play one continuous game each from the bottom for a long stretch of game
time, as a person would: no save/load, no search, falls allowed, the Snake played on. Per game: the best progress and
when each section was first reached, where the time went (the sections it was stuck in), and its big falls (from where
to where). Each game's actions and progress are saved (<out>-<seed>.npz, demo format), and game states exported as it
plays (<out>-<seed>-states.npz: where it first reached each section start and each multiple of 5, and where it was
every minute), which any game can load (train.py --demos, arrival_test.py): replaying a long game in another launch
drifts.

Regional (--start-states, --region): each game starts from a state in those files at that stretch of progress instead
of the bottom (e.g. the winning path's states just below the roofs), and starts over from another one when it falls
--drop below the stretch: the chain's own arrivals at upper sections, before games from the bottom get there.

    python scripts/marathon.py runs/rw-port/checkpoints/ppo_1500160_steps.zip --then 224:runs/bar-sil/checkpoints/ppo_7_steps.zip 250:runs/garden-path/checkpoints/ppo_10000000_steps.zip --minutes 60 --out runs/marathon/a
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from goi_env import OBS_FIELDS, GoiEnv
from sections import SECTIONS
from train import STALL_SECONDS, chain, parse_then

PROGRESS, WON, POT, HEAD = (OBS_FIELDS.index(f) for f in ("progress", "won", "pot_contact", "head_contact"))
FALL = 10  # progress lost from a local peak that counts as a fall
ARRIVAL, PERIODIC, EVERY = 0, 1, 60 * 30  # kinds of exported states; a periodic one every minute of game time
MARK = 5  # an arrival state the first time a game reaches each multiple of this (and each section start)


def section(p):
    return max(i for i, (start, _) in enumerate(SECTIONS) if p >= start or i == 0)


def play(seed, args):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    player = chain(args.checkpoint, parse_then(args.then), args.noise)
    # The agent's own record and stall count as in training episodes (GoiEnv record_reset; the stall count re-arms).
    game = GoiEnv(args.game_dir, stall_steps=int(STALL_SECONDS * 30), stall_ends=False, record_reset=args.record_reset)
    actions, trace, held, states = [], [], [], []  # states: (progress, step, kind, exported bytes)
    marks = sorted(set(start for start, _ in SECTIONS[1:]) | set(range(MARK, int(SECTIONS[-1][0]) + 1, MARK)))
    pool = [(f, i) for f in sorted(f for pattern in args.start_states for f in glob.glob(pattern))
            for i, p in enumerate(np.load(f)["progress"]) if args.region[0] <= p < args.region[1]]
    rng = np.random.default_rng(seed)
    obs = None

    def start():  # a fresh start: the bottom, or (regional) a state at the stretch
        if not pool:
            return game.reset()[0]
        f, i = pool[rng.integers(len(pool))]
        sid = game.import_state(np.load(f)[f"s{i}"].tobytes())
        try:
            return game.load_state(sid)
        finally:
            game.drop_state(sid)

    def export(kind):
        sid = game.save_state()
        states.append((float(obs[PROGRESS]), step, kind, game.export_state(sid)))
        game.drop_state(sid)

    try:
        obs, t = start(), 0
        todo = [m for m in marks if m > obs[PROGRESS]]
        for step in range(int(args.minutes * 60 * 30)):
            if pool and obs[PROGRESS] < args.region[0] - args.drop:  # fell below the stretch: start over there
                obs, t = start(), 0
                todo = [m for m in marks if m > obs[PROGRESS]]
            a = np.asarray(player(obs, t), np.float32)
            t += 1
            obs, *_ = game.step(a)
            actions.append(a)
            trace.append(obs[PROGRESS])
            held.append(bool(obs[POT] or obs[HEAD]))
            if obs[WON] > 0.5:
                break
            while todo and obs[PROGRESS] >= todo[0]:  # first arrival at a mark (stretches start anywhere, e.g. 430)
                todo.pop(0)
                export(ARRIVAL)
            if step % EVERY == EVERY - 1:  # where it is every minute: where it gets stuck, mostly
                export(PERIODIC)
    except (ConnectionError, OSError, EOFError) as e:  # the game died (rare): keep what it played
        print(f"game {seed}: lost after {len(trace) / 1800:.1f} min ({e!r})", flush=True)
    finally:
        try:
            game.close()
        except (ConnectionError, OSError):
            pass
    np.savez(f"{args.out}-{seed}-states.npz", exported=1, progress=np.array([s[0] for s in states], np.float32),
             step=np.array([s[1] for s in states]), kind=np.array([s[2] for s in states]),
             **{f"s{i}": np.frombuffer(s[3], np.uint8) for i, s in enumerate(states)})
    trace = np.array(trace, np.float32)
    np.savez(f"{args.out}-{seed}.npz", climb0=np.zeros((0, 2), np.float32), try0=np.array(actions),
             cuts0=np.array([0]), trace=trace, reached=np.array([len(actions) - 1]), goal=float(trace.max()))
    held = np.array(held, bool)
    return trace, obs is not None and bool(obs[WON] > 0.5), float(trace[held].max()) if held.any() else float("nan")


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
    ap.add_argument("--start-states", nargs="*", default=[], help="regional: exported state files (globs) to start from")
    ap.add_argument("--region", type=float, nargs=2, default=(-np.inf, np.inf), metavar=("LO", "HI"),
                    help="regional: start from the states at this stretch of progress")
    ap.add_argument("--drop", type=float, default=20, help="regional: start over after falling this far below LO")
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
    for g, (trace, won, held_best) in enumerate(results):
        secs = np.array([section(p) for p in trace])
        time_in[g] = np.bincount(secs, minlength=n)[:n] / 30
        reach = []
        for i, (start, name) in enumerate(SECTIONS[1:], 1):
            hit = np.argmax(trace >= start) if (trace >= start).any() else None
            if hit is not None:
                reach.append(f"{name} {hit / 30 / 60:.1f}m")
        big = sorted(falls(trace), key=lambda f: f[1] - f[0])[:3]
        # held: the best progress with the pot or hammer touching something (a mid-air peak falls back)
        print(f"game {g}: best {trace.max():.1f} (held {held_best:.1f}){' WON' if won else ''} in "
              f"{len(trace) / 30 / 60:.0f} min; "
              f"falls {len(falls(trace))}, biggest " + ", ".join(f"{a:.0f}->{b:.0f}" for a, b in big))
        print(f"   first reached: {', '.join(reach[-4:])}")
    print(f"\nper section, over {args.games} games ({args.minutes:.0f} min each):")
    print(f"{'section':<20}{'reached':>9}{'mean min spent':>16}")
    for i, (start, name) in enumerate(SECTIONS[:-1]):
        reached = sum((tr >= start).any() for tr, _, _ in results)
        if reached:
            print(f"{name:<20}{reached:>5}/{args.games:<3}{time_in[:, i].mean() / 60:>14.1f}")
    print(f"won: {sum(w for _, w, _ in results)}/{args.games}")


if __name__ == "__main__":
    main()
