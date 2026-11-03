"""GPU benchmark and parity check of the Box2D step: XLA island solver vs the Pallas kernel (goi_jax/box2d/gpu).

    scripts/rocm_run.sh python scripts/bench_gpu.py bench --scene player_rig --worlds 1024 16384 --solver pallas
    scripts/rocm_run.sh python scripts/bench_gpu.py bench --scene game --worlds 1024 16384 65536
    scripts/rocm_run.sh python scripts/bench_gpu.py check --scene player_rig --worlds 512 --steps 60
    scripts/cuda_run.sh python scripts/bench_gpu.py bench --scene game --worlds 16384 --compact   # static fixtures shared

Scenes: any tests/box2d scene (player_rig, pile, ...) or `game` = the real level (goi_jax/game/level.py, ~1500
static fixtures) from a recorded pre-Simulate state of dumps/parity, contact capacity --capacity.
bench: sleeping off, each world's velocities perturbed; reports compile s, world-steps/s, device memory and the
split collide+islands+setup (step with velocity_iterations=0, position_iterations=0, no TOI) vs solver.
check: steps a CPU XLA trajectory and, from each of its states, one GPU step; reports per-step ulp distance of the
GPU result from the CPU next state (bodies, impulses). Last line DONE.
"""
import argparse
import json
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
import goi_jax.box2d.gpu.solve as gsolve  # noqa: E402

H = jb.GAME_DT


def load_scene(name, capacity):
    """(model, state, settings) for one world; model may carry per-step joint fields (game)."""
    if name == "game":
        from goi_jax.game import level as L
        from goi_jax.parity import states as S
        lv = L.load_level(ROOT / "dumps" / "parity")
        e = S.load_index(ROOT / "dumps" / "parity")["chains"][0]
        chain = S.load_chain(ROOT / "dumps" / "parity" / e["file"])
        jm, st, _ = lv.native_states(chain, [10], 0, capacity)
        settings = jb.Settings.game()
        st = lv.set_limit_states(jm, st, settings)
        jm, st = L.jax_tree_index(jm, 0), L.jax_tree_index(st, 0)
        model = lv.model_for(jm)
        st = jax.tree.map(jnp.asarray, st)
        return model, st, settings
    if name.startswith("toi_"):  # tests/box2d/test_toi.py scenes (continuous bodies, TOI events)
        try:
            import pytest  # noqa: F401
        except ImportError:  # .venv-rocm has no pytest; the scene builders only need its decorators to import
            import types
            stub = types.ModuleType("pytest")
            stub.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
            sys.modules["pytest"] = stub
        from test_toi import SCENES
        name = name[4:]
    else:
        from box2d_scenes import SCENES
    twin = SCENES[name]()
    model, state = twin.defn.build(twin.capacity)
    return model, state, twin.settings


def batch(state, n, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    dv = jnp.asarray(rng.normal(0, scale, (n,) + state.bodies.v.shape).astype(np.float32))
    st = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), state)
    dyn = (state.bodies.type == jb.DYNAMIC)[None, :, None]
    return st._replace(bodies=st.bodies._replace(v=jnp.where(dyn, st.bodies.v + dv, st.bodies.v)))


def stepper(model, settings, device=None):
    """jit(vmap(step)) with the Model as an argument, not a closed-over constant: the game Model's 31k candidate
    pairs and 1.5k fixtures as HLO constants make XLA constant-fold gathers over them for minutes."""
    tb = model.toi_bodies
    if settings.unity_toi and tb is not None and not np.any(np.asarray(tb)):
        settings = replace(settings, continuous_physics=False)  # what toi.py decides at trace for a constant Model
    f = jax.vmap(lambda md, st: jb.step(md, st, settings, H, axis_name="w"), in_axes=(None, 0), axis_name="w")
    f = jax.jit(f, device=device) if device is not None else jax.jit(f)
    return lambda st: f(model, st)


def mem_mb():
    try:
        s = jax.devices()[0].memory_stats() or {}
        return s.get("peak_bytes_in_use", 0) / 2**20, s.get("bytes_limit", 0) / 2**20
    except Exception:
        return 0.0, 0.0


def compact(model, state, on):
    """--compact: the compact fixture layout (goi_jax/box2d/world.py compact_model; same floats)."""
    if not on:
        return model, state
    from goi_jax.box2d import world as W
    model = W.compact_model(model, state)
    return model, W.compact_state(model, state)


def bench(args):
    model, state, settings = load_scene(args.scene, args.capacity)
    model, state = compact(model, state, args.compact)
    settings = replace(settings, allow_sleep=False, solver=args.solver)
    if args.no_toi:
        settings = replace(settings, continuous_physics=False)
    if args.no_solver:
        settings = replace(settings, velocity_iterations=0, position_iterations=0)
    gsolve.BLOCK, gsolve.NUM_WARPS = args.block, args.warps
    rows = []
    for n in args.worlds:
        out = dict(scene=args.scene, solver=args.solver, worlds=n, block=args.block, warps=args.warps,
                   compact=bool(args.compact))
        for tag, s in (("full", settings), ("no_solver", replace(settings, velocity_iterations=0,
                                                                   position_iterations=0, continuous_physics=False)),
                       ("no_toi", replace(settings, continuous_physics=False))):
            if tag != "full" and not args.split:
                continue
            f = stepper(model, s)
            st = batch(state, n)
            t0 = time.perf_counter()
            st = jax.block_until_ready(f(st))
            comp = time.perf_counter() - t0
            times = []
            for _ in range(args.steps):  # per-step times: the shared GPU (it drives the display) is noisy
                t0 = time.perf_counter()
                st = jax.block_until_ready(f(st))
                times.append(time.perf_counter() - t0)
            dt = float(np.median(times))
            out[tag] = dict(compile_s=round(comp, 1), ms_per_step=round(dt * 1e3, 3), ms_min=round(min(times) * 1e3, 3),
                            ms_max=round(max(times) * 1e3, 3), world_steps_s=round(n / dt))
            bad = int(np.sum(~np.isfinite(np.asarray(st.bodies.c))))
            out[tag]["nonfinite"] = bad
            del st
        out["peak_mb"], out["limit_mb"] = mem_mb()
        print(json.dumps(out), flush=True)
        rows.append(out)
    return rows


def ulp(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    oa = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ob = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(oa - ob)


def check(args):
    """Per-step parity of the GPU Pallas step against the CPU XLA step."""
    model, state, settings = load_scene(args.scene, args.capacity)
    model, state = compact(model, state, args.compact)
    settings = replace(settings, solver="xla")
    if args.no_toi:
        settings = replace(settings, continuous_physics=False)
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices()[0]
    gsolve.BLOCK, gsolve.NUM_WARPS = args.block, args.warps
    to = lambda t, d: jax.tree.map(lambda x: jax.device_put(x, d), t)
    f_cpu = stepper(to(model, cpu), settings, cpu)
    f_gpu = stepper(to(model, gpu), replace(settings, solver=args.solver), gpu)
    st = to(batch(state, args.worlds[0]), cpu)
    fields = dict(c=lambda s: s.bodies.c, a=lambda s: s.bodies.a, v=lambda s: s.bodies.v, w=lambda s: s.bodies.w,
                  ni=lambda s: s.contacts.normal_impulse, ti=lambda s: s.contacts.tangent_impulse,
                  jimp=lambda s: s.joints.impulse)
    events = 0
    worst = {k: 0 for k in fields}
    worst_n = {k: 0 for k in fields}  # ulp where either value is a normal float (subnormals: XLA CPU flushes them)
    absd = {k: 0.0 for k in fields}
    hist = {k: np.zeros(6, np.int64) for k in fields}  # 0, 1, 2-4, 5-16, 17-256, >256 ulp
    edges = np.array([0, 1, 2, 5, 17, 257])
    for t in range(args.steps):
        nxt = jax.block_until_ready(f_cpu(st))
        g = jax.tree.map(np.asarray, f_gpu(to(st, gpu)))
        n = jax.tree.map(np.asarray, nxt)
        if not np.array_equal(g.contacts.pair, n.contacts.pair):
            print(f"step {t + 1}: contact slots differ in {int(np.sum(g.contacts.pair != n.contacts.pair))} entries")
        for k, fn in fields.items():
            d = ulp(fn(g), fn(n)).ravel()
            if args.verbose and d.max(initial=0) > 0:
                idx = np.unravel_index(np.argmax(ulp(fn(g), fn(n))), fn(n).shape)
                print(f"step {t + 1} {k}: {int((d > 0).sum())} differ; worst at {idx}: gpu {fn(g)[idx]!r} cpu {fn(n)[idx]!r}")
            worst[k] = max(worst[k], int(d.max(initial=0)))
            a_, b_ = np.asarray(fn(g), np.float64).ravel(), np.asarray(fn(n), np.float64).ravel()
            normal = (np.abs(a_) >= np.finfo(np.float32).tiny) | (np.abs(b_) >= np.finfo(np.float32).tiny)
            dn = d[normal]
            worst_n[k] = max(worst_n[k], int(dn.max(initial=0)))
            absd[k] = max(absd[k], float(np.max(np.abs(a_ - b_), initial=0)))
            hist[k] += np.bincount(np.searchsorted(edges, d, side="right") - 1, minlength=6)
        if n.toi is not None:
            events += int(np.sum(n.toi.events))
        st = nxt
    print(json.dumps(dict(scene=args.scene, worlds=args.worlds[0], steps=args.steps, toi_events=events, max_ulp=worst, max_ulp_normal=worst_n, max_abs=absd,
                          ulp_hist_0_1_2to4_5to16_17to256_over={k: v.tolist() for k, v in hist.items()})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("bench", "check"))
    ap.add_argument("--scene", default="player_rig")
    ap.add_argument("--worlds", type=int, nargs="*", default=[1024, 16384])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--solver", default="pallas", choices=("pallas", "xla"))
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--split", action="store_true", help="also time the step without solver and without TOI")
    ap.add_argument("--no-toi", action="store_true")
    ap.add_argument("--compact", action="store_true", help="compact fixture layout (static fixtures in the Model)")
    ap.add_argument("--verbose", action="store_true", help="check: print the worst differing value per step")
    ap.add_argument("--no-solver", action="store_true", help="0 velocity / position iterations (time the rest)")
    args = ap.parse_args()
    print(jax.devices(), flush=True)
    (bench if args.mode == "bench" else check)(args)
    print("DONE")


if __name__ == "__main__":
    main()
