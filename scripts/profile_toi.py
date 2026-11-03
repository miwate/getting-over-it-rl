"""Time the parts of b2World::SolveTOI (goi_jax/box2d/toi.py) on the game scene, batched, on the default device.

    scripts/cuda_run.sh python scripts/profile_toi.py --worlds 2048 8192

Warm-up: `--warm` full steps from the recorded state (velocities perturbed per world, sleeping off, as bench_gpu.py).
Then on the state after one more step without TOI (sweeps c0 -> c of that step), times: the whole solve_toi, one
search pass (_search), the search with the TOI root finder stubbed out (the sequential alignment pass alone), one
event (_event), and the step without TOI; and reports TOI events per world. Last line DONE.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from goi_jax import box2d as jb  # noqa: E402
from goi_jax.box2d import toi as T  # noqa: E402
from goi_jax.box2d.world import empty_toi  # noqa: E402
from bench_gpu import H, batch, load_scene  # noqa: E402


def timed(f, *a, reps=5):
    out = jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return out, round(1e3 * float(np.median(ts)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, nargs="+", default=[2048])
    ap.add_argument("--capacity", type=int, default=96)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--solver", default="pallas")
    ap.add_argument("--parts", default="toi,search,align,event,notoi")
    args = ap.parse_args()
    print(jax.devices(), flush=True)
    model, state, settings = load_scene("game", args.capacity)
    s = replace(settings, allow_sleep=False, solver=args.solver)
    s_off = replace(s, continuous_physics=False)
    parts = args.parts.split(",")
    vm = lambda fn: jax.jit(jax.vmap(fn, in_axes=(None, 0), axis_name="w"))  # noqa: E731
    full = vm(lambda md, st: jb.step(md, st, s, H, axis_name="w"))
    notoi = vm(lambda md, st: jb.step(md, st, s_off, H, axis_name="w"))
    for n in args.worlds:
        out = dict(worlds=n, solver=args.solver)
        st = batch(state, n)
        t0 = time.perf_counter()
        for _ in range(args.warm):
            st = full(model, st)
        jax.block_until_ready(st)
        out["warm_s"] = round(time.perf_counter() - t0, 1)
        pre, out["step_notoi_ms"] = timed(notoi, model, st)
        if "toi" in parts:
            f = vm(lambda md, x: T.solve_toi(md, x._replace(toi=empty_toi(*_sizes(x))), s, H, "w"))
            post, out["solve_toi_ms"] = timed(f, model, pre)
            ev = np.asarray(post.toi.events)
            out["events"] = dict(max=int(ev.max()), mean=round(float(ev.mean()), 3),
                                 hist=np.bincount(np.minimum(ev, 10), minlength=11).tolist())
            out["live_contacts_max"] = int(np.max(np.sum(np.asarray(pre.contacts.pair) >= 0, -1)))
        go = jnp.asarray(True)
        f_search, f_event = T._search, T._event
        if args.solver == "pallas" and hasattr(__import__("goi_jax.box2d.gpu", fromlist=["x"]), "__path__"):
            try:
                from goi_jax.box2d.gpu.toi_search import search as f_search  # noqa: F811
                from goi_jax.box2d.gpu.toi_event import event as f_event  # noqa: F811
            except ImportError:
                pass

        def search(md, x):
            B, C = _sizes(x)
            return f_search(md, x, empty_toi(B, C), s, go, "w")
        if "search" in parts:
            sr, out["search_ms"] = timed(vm(search), model, pre)
            out["found_worlds"] = int(np.sum(np.asarray(sr[2])))
        if "align" in parts and f_search is T._search:
            orig = T.time_of_impact
            T.time_of_impact = lambda pA, swA, pB, swB, slop, active: (jnp.full(active.shape, T.E_SEPARATED, jnp.int32),
                                                                       jnp.ones(active.shape, jnp.float32))
            try:
                _, out["search_align_only_ms"] = timed(vm(search), model, pre)
            finally:
                T.time_of_impact = orig
        if "event" in parts:
            def event(md, x):
                x2, tst, found, ms, ma = search(md, x)
                return f_event(md, x2, tst, s, H, found, ms, ma, "w")
            _, ev_ms = timed(vm(event), model, pre)
            out["search_plus_event_ms"] = ev_ms
            if "search_ms" in out:
                out["event_ms"] = round(ev_ms - out["search_ms"], 2)
        print(json.dumps(out), flush=True)
    print("DONE")


def _sizes(x):
    return x.bodies.type.shape[0], x.contacts.pair.shape[0]


if __name__ == "__main__":
    main()
