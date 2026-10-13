"""One round of the loop that pushes the frontier of the networks playing alone: train a stretch twice from the states
where the latest marathon's games were in it (exported states, loaded exactly) and from win-e4's states, keep the
better run if it beats the chain's current network there on the arrival test, update runs/chain.json, run the next
marathon. A first marathon is run if the latest one has no exported states. Resumes stopped training runs. Each
stretch is trained twice because a run of this recipe either learns the stretch or doesn't (garden: 69%, 69%, 84% vs
4%, 0%, 0%). Log: runs/rounds.txt.

    python scripts/round.py chapel 415 465
    python scripts/round.py reclimb-garden 250 290 --after 300   (re-climbs: where games were after falling from 300+)
    python scripts/round.py roofs 465 525 --regional 455 465   (above the frontier: games started just below the roofs)
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG = os.path.join("runs", "rounds.txt")
PATH_STATES = os.path.join("runs", "climbs", "win-e4-states.npz")  # the search agent's winning path (path_states.py)


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{msg}  {time.strftime('%m-%d %H:%M')}\n")


def last_checkpoint(name):
    files = glob.glob(f"runs/{name}/checkpoints/ppo_*_steps.zip")
    return max(files, key=steps_of) if files else None


def steps_of(path):
    return int(re.search(r"ppo_(\d+)_steps", path).group(1))


def latest_marathon():
    found = [int(m.group(1)) for f in os.listdir("runs/marathon") if (m := re.fullmatch(r"m(\d+)\.txt", f))]
    return max(found) if found else 0


def marathon(name, chain, extra=("--minutes", "60")):
    """A marathon with the chain, 8 games (runs/marathon/<name>*), logging each game's best."""
    then = [f"{start}:{spec}" for start, spec in chain[1:]]
    with open(f"runs/marathon/{name}.txt", "w") as f:
        subprocess.run([PY, "scripts/marathon.py", chain[0][1].partition("@")[0], "--then", *then, *extra,
                        "--games", "8", "--out", f"runs/marathon/{name}"], stdout=f, stderr=subprocess.DEVNULL)
    lines = open(f"runs/marathon/{name}.txt").read().splitlines()
    log(f"   marathon {name}: " + "; ".join(l.split(" in ")[0] for l in lines if l.startswith("game")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the stretch's name (runs are <name>-m<marathon>-a/b)")
    ap.add_argument("lo", type=float, help="the stretch's start: a section start (sections.SECTIONS)")
    ap.add_argument("hi", type=float)
    ap.add_argument("--seconds", type=float, default=90, help="arrival test: game time to reach hi")
    ap.add_argument("--runs", type=int, default=2, help="training runs (a run of this recipe learns a stretch or not)")
    ap.add_argument("--steps", type=int, default=10_000_000, help="steps per run")
    ap.add_argument("--marathons", type=int, default=1, help="exported states from this many latest marathons")
    ap.add_argument("--seeds", nargs="*", default=[], help="more exported starts, e.g. seeds.py's for a hard move")
    ap.add_argument("--regional", type=float, nargs=2, metavar=("LO", "HI"),
                    help="first a regional marathon (once per round name): chain games started from the winning path's "
                         "and the marathons' states at this stretch, below the round's: the chain's own arrivals there")
    ap.add_argument("--after", type=float, help="re-climbs: train and test from the states saved in the stretch after "
                                                 "a game had been at this progress (it fell back into the stretch)")
    ap.add_argument("--from", dest="sources", nargs="+", help="warm-start the runs from these checkpoints in turn "
                                                                "(default: the chain's network at the stretch's start)")
    ap.add_argument("--tries", type=int, default=10, help="arrival test: tries per state")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more train.py arguments (e.g. --regress 1)")
    args = ap.parse_args()
    if not os.environ.get("GOI_GAME_DIR"):
        ap.error("set GOI_GAME_DIR to the game's install folder")
    os.chdir(ROOT)
    chain = json.load(open("runs/chain.json"))
    n = latest_marathon()
    if not glob.glob(f"runs/marathon/m{n}-*-states.npz"):
        n += 1
        log(f"== marathon m{n} for exported states")
        marathon(f"m{n}", chain)
    states = sorted(f for k in range(n - args.marathons + 1, n + 1) for f in glob.glob(f"runs/marathon/m{k}-*-states.npz"))
    if args.regional:
        name = f"r-{args.name}"
        if not glob.glob(f"runs/marathon/{name}-*-states.npz"):
            log(f"== regional marathon {name}: from {args.regional[0]:.0f}-{args.regional[1]:.0f}")
            marathon(name, chain, ["--minutes", "30", "--start-states", PATH_STATES, *states, "--region",
                                   *map(str, args.regional)])
        states += sorted(glob.glob(f"runs/marathon/{name}-*-states.npz"))
    lo, hi = args.lo, args.hi
    after = lambda flag: [] if args.after is None else [flag, str(args.after)]  # noqa: E731
    current = [spec for start, spec in chain if start <= lo][-1]
    log(f"== round {args.name} {lo:.0f}-{hi:.0f}: states from m{n}; current {current}")
    common = ["--collector", "async", "--gsde", "--bounded-mean", "--envs", "10", "--demo-share", "0.6",
              "--saved-starts", "0.2", "--traps", "0.2", "--rewinds", "1.0", "--episode-steps", "1800",
              "--save-every", "2000000", "--demos", *states, *args.seeds, PATH_STATES, "--demo-window", str(lo), str(hi),
              "--window", str(lo - 6), str(hi + 5), *after("--demo-after"), *args.extra]
    runs = [f"{args.name}-m{n}-{chr(ord('a') + k)}" for k in range(args.runs)]
    for k, run in enumerate(runs):
        source = args.sources[k % len(args.sources)] if args.sources else current.partition("@")[0]
        ckpt = last_checkpoint(run)
        if ckpt and steps_of(ckpt) >= args.steps:
            continue
        with open(f"runs/{run}.log", "a") as f:
            if ckpt:
                log(f"   {run} resumed at {steps_of(ckpt)}")
                start = ["--resume", ckpt, "--steps", str(args.steps - steps_of(ckpt))]
            else:
                log(f"   {run} from {source}")
                start = ["--warm-start", source, "--steps", str(args.steps)]
            subprocess.run([PY, "scripts/train.py", "--name", run, *start, *common], stdout=f, stderr=subprocess.STDOUT)
    candidates = [f"{last_checkpoint(r)}@{noise}".replace("\\", "/") for r in runs for noise in (1, 0.5)] + [current]
    test = subprocess.run([PY, "scripts/arrival_test.py", str(lo), str(hi), str(args.seconds), *candidates,
                           "--states", *states, "--path", PATH_STATES, "--tries", str(args.tries), *after("--after"),
                           "--fill", *args.seeds, PATH_STATES],
                          capture_output=True,
                          text=True).stdout
    scores = {}
    for line in test.splitlines():
        m = re.match(r"\s+(\d+)/(\d+) gain\s+(-?[\d.]+) (\S+)", line)
        if m:  # successes first, then mean progress gained (long stretches are rarely crossed in one try)
            scores[m.group(4)] = (int(m.group(1)), float(m.group(3)))
    for spec, (wins, gain) in scores.items():
        log(f"   {wins:3d} gain {gain:5.1f} {spec}")
    best = max(scores, key=scores.get) if scores else current
    if best != current and scores[best] > scores.get(current, (-1, 0.0)):
        # the stretch's network owns all of it: entries inside it go (a 430 entry would take over mid-jump); whatever
        # played from hi on before goes on doing so (a narrow stretch inside a section keeps the section's network after)
        after = [spec for start, spec in chain if start <= hi][-1]
        chain = [c for c in chain if not lo <= c[0] < hi] + [[lo, best]]
        if not any(start == hi for start, _ in chain):
            chain.append([hi, after])
        chain = sorted(chain, key=lambda c: c[0])
        json.dump(chain, open("runs/chain.json", "w"), indent=2)
        log(f"   chain: {lo:.0f} -> {best}")
    else:
        log("   chain unchanged")
    marathon(f"m{n + 1}", chain)
    log(f"== round {args.name} done")


if __name__ == "__main__":
    main()
