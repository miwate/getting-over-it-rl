"""Print the JAX-vs-Box2D error table for the parity scenes (tests/box2d/box2d_scenes.py).

    JAX_PLATFORMS=cpu .venv/bin/python tests/box2d/report.py [scene ...] [--steps 120]

Each scene: max |error| and max ulp distance per quantity after 1 step, after `steps` steps, and over all steps,
plus structural mismatches (contact list, manifold ids, awake flags, limit states) and timing. Last line DONE.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from box2d_harness import QUANTITIES, initial_mismatches, run, summarize  # noqa: E402
from box2d_scenes import SCENES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    for name in args.scenes:
        twin = SCENES[name]()
        if twin is None:
            print(f"== {name}: skipped (no dump)")
            continue
        bad = initial_mismatches(twin)
        res = run(twin, args.steps)
        summ = summarize(res, (1, args.steps))
        print(f"== {name}: compile {res['compile_s']:.1f} s, {1 / res['step_s']:.0f} steps/s, "
              f"max touching contacts {res['max_touching']}, initial mismatches {len(bad)}")
        for b in bad[:5]:
            print("   init:", b)
        print(f"   {'quantity':16s} {'1 step abs':>11s} {'ulp':>5s} {f'{args.steps} steps abs':>14s} {'ulp':>6s}"
              f" {'max abs':>10s} {'ulp':>6s}")
        for q in QUANTITIES:
            r = summ[q]
            a1, u1 = r["1"]
            an, un = r[str(args.steps)]
            am, um = r["max"]
            print(f"   {q:16s} {a1:11.3g} {u1:5d} {an:14.3g} {un:6d} {am:10.3g} {um:6d}")
        div = next((k + 1 for k, e in enumerate(res["errors"]) if any(e[q][1] for q in QUANTITIES)), None)
        print(f"   first step with any non-zero error: {div if div else 'none in ' + str(args.steps)}")
        first = res["issues"][:6]
        print(f"   structural issues: {len(res['issues'])}" + ("" if not first else ", first:"))
        for k, msg in first:
            print(f"     step {k}: {msg}")
    print("DONE")


if __name__ == "__main__":
    main()
