"""Compile time and parity of the TOI kernels (gpu/toi_search.py, gpu/toi_event.py) on the default device, one at
a time, on a tests/box2d TOI scene. Last line DONE.

    scripts/cuda_run.sh python scripts/compile_toi.py --scene island_walk --worlds 64
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "box2d"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from goi_jax import box2d as jb  # noqa: E402
from goi_jax.box2d import toi as T  # noqa: E402
from goi_jax.box2d.world import empty_toi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="island_walk")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--parts", default="search,event")
    ap.add_argument("--ablate", nargs="*", default=[""],
                    help="variants to compile, each a '+'-joined list of stubs (compile-time bisection): "
                         "evaluate, prepare, velocity, position, distance, findmin; '' = the full kernels")
    args = ap.parse_args()
    print(jax.devices(), flush=True)
    import types
    try:
        import pytest  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pytest")
        stub.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
        sys.modules["pytest"] = stub
    if args.scene == "game":  # the level (scripts/bench_gpu.py load_scene), contact capacity 96
        sys.path.insert(0, str(ROOT / "scripts"))
        from bench_gpu import load_scene
        model, state, settings = load_scene("game", 96)
    else:
        from test_toi import SCENES
        twin = SCENES[args.scene]()
        model, state = twin.defn.build(twin.capacity)
        settings = twin.settings
    s = replace(settings, solver="pallas")
    rng = np.random.default_rng(0)
    st = jax.tree.map(lambda x: jnp.broadcast_to(x, (args.worlds,) + jnp.shape(x)), state)
    dv = jnp.asarray(rng.normal(0, 0.05, st.bodies.v.shape).astype(np.float32))
    st = st._replace(bodies=st.bodies._replace(v=st.bodies.v + dv))
    B, C = state.bodies.type.shape[0], state.contacts.pair.shape[0]
    vm = lambda fn: jax.jit(jax.vmap(fn, in_axes=(None, 0), axis_name="w"))  # noqa: E731
    go = jnp.asarray(True)
    for variant in args.ablate:
        stubs = [v for v in variant.split("+") if v]
        restore = _stub(stubs)
        try:
            _compile(args, model, st, s, B, C, vm, go, variant or "full")
        finally:
            restore()
    print("DONE")


def _compile(args, model, st, s, B, C, vm, go, tag):
    if "search" in args.parts:
        from goi_jax.box2d.gpu.toi_search import search
        f = vm(lambda md, x: search(md, x, empty_toi(B, C), s, go, "w")[2:])
        t0 = time.perf_counter()
        c = f.lower(model, st).compile()
        print(f"{tag}: search compile {time.perf_counter() - t0:.1f} s", flush=True)
        t0 = time.perf_counter()
        jax.block_until_ready(c(model, st))
        print(f"search run {1e3 * (time.perf_counter() - t0):.1f} ms", flush=True)
    if "event" in args.parts:
        from goi_jax.box2d.gpu.toi_event import event
        f = vm(lambda md, x: event(md, x, empty_toi(B, C), s, jb.GAME_DT, go, jnp.int32(0), jnp.float32(0.5), "w")[0].bodies.c)
        t0 = time.perf_counter()
        c = f.lower(model, st).compile()
        print(f"{tag}: event compile {time.perf_counter() - t0:.1f} s", flush=True)
        t0 = time.perf_counter()
        jax.block_until_ready(c(model, st))
        print(f"event run {1e3 * (time.perf_counter() - t0):.1f} ms", flush=True)


def _stub(stubs):
    """Replace kernel pieces by trivial functions of their inputs (compile-time bisection only)."""
    from goi_jax.box2d import contact_solver as cs
    from goi_jax.box2d.gpu import toi_event as E
    from goi_jax.box2d.gpu import toi_search as S
    saved = []

    def put(mod, name, fn):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, fn)
    z = lambda x: jnp.zeros_like(x)  # noqa: E731
    if "evaluate" in stubs:
        def ev(geo, fa, fb, xfa, xfb, slop):
            zf, zi = z(xfa[0]), z(fa)
            return dict(mtype=zi, count=(fa > fb).astype(jnp.int32), local_normal=(zf, zf), local_point=(zf, zf),
                        point_local=((zf, zf), (zf, zf)), point_id=(zi, zi))
        put(E, "evaluate", ev)
    if "prepare" in stubs:
        def prep(cfg, bf, ci, cf_, gbody, slot, fa, fb, ia, ib, pos, vel, radius):
            zf = z(radius(fa))
            return dict(ia=ia, ib=ib, mA=zf, mB=zf, iA=zf, iB=zf, nx=zf, ny=zf, rAx=(zf, zf), rAy=(zf, zf),
                        rBx=(zf, zf), rBy=(zf, zf), count=z(ia), tangent_speed=zf, tmass=(zf, zf), friction=zf,
                        nmass=(zf, zf), bias=(zf, zf), K=(zf,) * 4, NM=(zf,) * 4)
        put(E, "_prepare_row", prep)
    if "velocity" in stubs:
        def so(c, vel, ni, ti):
            g = lambda q, i: vel[q].get(i)  # noqa: E731
            return ((g(0, c["ia"]), g(1, c["ia"])), g(2, c["ia"]), (g(0, c["ib"]), g(1, c["ib"])), g(2, c["ib"])), ni, ti
        put(cs, "_solve_one", so)
    if "position" in stubs:
        def sp(c, pos, settings, baumgarte):
            g = lambda q, i: pos[q].get(i)  # noqa: E731
            return (g(0, c["ia"]), g(1, c["ia"])), g(2, c["ia"]), (g(0, c["ib"]), g(1, c["ib"])), g(2, c["ib"]), z(g(0, c["ia"]))
        put(cs, "solve_position_one", sp)
    if "distance" in stubs:
        put(S, "distance", lambda cache, pA, xfA, pB, xfB, active: (z(xfA[0]) + 1, cache))
    if "findmin" in stubs:
        put(S, "_find_min", lambda f, pA, xfA, pB, xfB: (z(xfA[0]), z(f["type"]), z(f["type"])))
        put(S, "_sep_points", lambda f, pA, xfA, iA, pB, xfB, iB: z(xfA[0]))

    def restore():
        for mod, name, fn in reversed(saved):
            setattr(mod, name, fn)
    return restore


if __name__ == "__main__":
    main()
