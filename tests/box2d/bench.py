"""CPU timing of the JAX step: compile time, single-world steps/s, vmapped world-steps/s.

    JAX_PLATFORMS=cpu .venv/bin/python tests/box2d/bench.py [scene] [--worlds 256] [--steps 50]

Sleeping is disabled (Settings.allow_sleep=False) so every step does full work; each world gets a slightly
different initial velocity so the batch does not stay identical. Last line DONE.
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from goi_jax import box2d as jb  # noqa: E402
from box2d_scenes import SCENES  # noqa: E402
from box2d_harness import H  # noqa: E402


def batch(state, n, seed=0):
    rng = np.random.default_rng(seed)
    dv = jnp.asarray(rng.normal(0, 0.05, (n,) + state.bodies.v.shape).astype(np.float32))
    st = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), state)
    dyn = (state.bodies.type == jb.DYNAMIC)[None, :, None]
    return st._replace(bodies=st.bodies._replace(v=jnp.where(dyn, st.bodies.v + dv, st.bodies.v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="player_rig")
    ap.add_argument("--worlds", type=int, nargs="*", default=[1, 256])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    twin = SCENES[args.scene]()
    twin.settings = replace(twin.settings, allow_sleep=False)
    twin.defn.settings = twin.settings
    model, state = twin.defn.build(twin.capacity)
    s = twin.settings
    for n in args.worlds:
        if n == 1:
            f = jax.jit(lambda st: jb.step(model, st, s, H))
            st = state
        else:
            f = jax.jit(jax.vmap(lambda st: jb.step(model, st, s, H, axis_name="w"), axis_name="w"))
            st = batch(state, n)
        t0 = time.perf_counter()
        st = f(st)
        jax.block_until_ready(st)
        comp = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(args.steps):
            st = f(st)
        jax.block_until_ready(st)
        dt = (time.perf_counter() - t0) / args.steps
        print(f"{args.scene}: worlds {n}: compile+first {comp:.1f} s, {dt * 1e3:.2f} ms/step, "
              f"{n / dt:.0f} world-steps/s")
    print("DONE")


if __name__ == "__main__":
    main()
