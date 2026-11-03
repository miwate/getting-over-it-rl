"""The Pallas island solver (goi_jax/box2d/gpu) against the XLA step, bit for bit.

On CPU the kernel runs in Pallas interpret mode and must reproduce the XLA step exactly (every State leaf, every
step) on the tests/box2d scenes, batched (a few worlds with perturbed velocities, so lanes differ) and unbatched.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/box2d/test_gpu.py -q
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goi_jax import box2d as jb  # noqa: E402
import goi_jax.box2d.gpu.solve as gsolve  # noqa: E402
from box2d_scenes import SCENE_STEPS, SCENES as _BASE  # noqa: E402
from test_toi import SCENES as TOI_SCENES  # noqa: E402  (continuous bodies: TOI events every few steps)

SCENES = dict(_BASE, **{"toi_" + k: v for k, v in TOI_SCENES.items()})

H = jb.GAME_DT
STEPS = 40
WORLDS = 3
# Interpret mode compiles each scene's kernel for ~30 s, so the default run covers the scenes that reach every
# solver path (2-point block solver, all joint types, soft weld, limits, polygon-circle manifolds, sleeping);
# GPU_TEST_ALL=1 runs every scene.
DEFAULT = ("box_stack", "pile", "player_rig", "limits_mid_run", "weld", "polygon_circle", "late_wake",
           "toi_hammer_chain", "toi_multi_event", "toi_island_walk", "toi_stock_bullets", "toi_ball_wall",
           "toi_poly_angles")
# With Settings.solver == "pallas" the whole TOI event loop runs in kernels (gpu/toi_search.py: candidate sweeps,
# b2TimeOfImpact, min-TOI selection; gpu/toi_event.py: advance, manifold updates, island walk, TOI island solve), so
# the toi_ scenes check those bit for bit too; test_toi_kernels_blocks adds several kernel programs and padded lanes.
NAMES = sorted(SCENES) if os.environ.get("GPU_TEST_ALL") else [n for n in DEFAULT if n in SCENES]


def batch(state, n, seed=0):
    rng = np.random.default_rng(seed)
    dv = jnp.asarray(rng.normal(0, 0.05, (n,) + state.bodies.v.shape).astype(np.float32))
    st = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), state)
    dyn = (state.bodies.type == jb.DYNAMIC)[None, :, None]
    dv = dv.at[0].set(0)
    return st._replace(bodies=st.bodies._replace(v=jnp.where(dyn, st.bodies.v + dv, st.bodies.v)))


def runners(model, settings, batched):
    sp = replace(settings, solver="pallas")
    if batched:
        mk = lambda s: jax.jit(jax.vmap(lambda st: jb.step(model, st, s, H, axis_name="w"), axis_name="w"))
    else:
        mk = lambda s: jax.jit(lambda st: jb.step(model, st, s, H))
    return mk(settings), mk(sp)


def mismatches(a, b):
    out = []
    for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(a), jax.tree.leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype == np.float32:
            same = (x.view(np.int32) == y.view(np.int32)) | (np.isnan(x) & np.isnan(y))
        else:
            same = x == y
        if not np.all(same):
            out.append((jax.tree_util.keystr(path), float(np.max(np.abs(x.astype(np.float64) - y)))))
    return out


def compare_scene(name, steps=STEPS, worlds=WORLDS, batched=True, block=8):
    twin = SCENES[name]()
    if twin is None:
        pytest.skip("scene needs dumps/engine-dump-start.jsonl")
    model, state = twin.defn.build(twin.capacity)
    f_xla, f_pal = runners(model, twin.settings, batched)
    st = batch(state, worlds) if batched else state
    # strong-typed, committed arrays, like the step's outputs: one compile per runner
    st = jax.device_put(jax.tree.map(np.asarray, st), jax.devices()[0])
    old = gsolve.BLOCK
    gsolve.BLOCK = block
    try:
        a = b = st
        for k in range(steps):
            a, b = f_xla(a), f_pal(b)
            bad = mismatches(a, b)
            if bad:
                return k + 1, bad
    finally:
        gsolve.BLOCK = old
    return None, []


@pytest.mark.parametrize("name", NAMES)
def test_pallas_interpret_bit_exact(name):
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    step, bad = compare_scene(name, steps=max(STEPS, SCENE_STEPS.get(name, 0)))
    assert not bad, f"{name}: first mismatch at step {step}: {bad[:6]}"


def test_toi_kernels_blocks():
    """TOI scene over 10 worlds in blocks of 4 (3 programs, 2 padded lanes): lanes with different event counts."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    step, bad = compare_scene("toi_island_walk", steps=30, worlds=10, block=4)
    assert not bad, f"first mismatch at step {step}: {bad[:6]}"


@pytest.mark.parametrize("name", ["toi_island_walk", "toi_multi_event"])
def test_toi_compacted_loop(name):
    """gpu/toi_loop.py with fewer lanes than worlds (10 worlds, 4 lanes per iteration): worlds with pending TOI
    events are processed a few at a time, in different iterations than the XLA loop does them; same bits."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    from goi_jax.box2d.gpu import toi_loop
    old = toi_loop.MIN_LANES, toi_loop.DIVISOR
    toi_loop.MIN_LANES, toi_loop.DIVISOR = 1, 4
    try:
        step, bad = compare_scene(name, steps=30, worlds=10, block=4)
    finally:
        toi_loop.MIN_LANES, toi_loop.DIVISOR = old
    assert not bad, f"first mismatch at step {step}: {bad[:6]}"


def test_pallas_unbatched():
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    step, bad = compare_scene("limits_mid_run", steps=20, batched=False)
    assert not bad, f"first mismatch at step {step}: {bad[:6]}"


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(SCENES)
    for n in names:
        step, bad = compare_scene(n)
        print(n, "OK" if not bad else f"MISMATCH at step {step}: {bad[:6]}", flush=True)
    print("DONE")
