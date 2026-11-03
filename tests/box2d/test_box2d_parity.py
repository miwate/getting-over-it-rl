"""goi_jax.box2d against the Box2D 2.3.1 oracle (goi_jax/reference): constants, trig, every parity scene step by
step, batching. CPU, about 3 minutes (one XLA compile of 6-15 s per scene).

The port reproduces Box2D's float32 arithmetic operation for operation, so the expected error is zero: these tests
demand bit-identical bodies, contact impulses, manifold ids, contact list order, joint impulses, limit states and
awake flags at every step (the parity target is 1e-5; anything non-zero means an ordering or rounding slip).
"""
import ast
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goi_jax import box2d as jb
from goi_jax.box2d import math2d
from goi_jax.reference import box2d as b2

from box2d_harness import H, QUANTITIES, initial_mismatches, run, summarize
from box2d_scenes import SCENE_STEPS, SCENES

F32 = np.float32
PKG = Path(jb.__file__).parent


def test_stock_settings_match_the_library():
    """Stock defaults are the b2Settings.h values the oracle library holds, bit for bit."""
    s = jb.Settings()
    for name in b2._PARAMS:
        assert F32(getattr(s, name)).tobytes() == b2._stock(b2._PARAMS[name]).tobytes(), name


def test_game_settings_match_the_reference():
    """Settings.game() is the oracle's Settings.game() (the twin tests build the oracle from ours)."""
    ours, ref = jb.Settings.game(), b2.Settings.game()
    for name in b2._PARAMS:
        assert F32(getattr(ours, name)).tobytes() == ref.resolved(name).tobytes(), name
    assert (ours.velocity_iterations, ours.position_iterations) == (ref.velocity_iterations, ref.position_iterations)
    assert tuple(map(float, ours.gravity)) == tuple(map(float, ref.gravity))


def test_sin_cos_correctly_rounded():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(-8, 8, 20000), rng.uniform(-300, 300, 5000), [0.0, -0.0, np.pi]]).astype(F32)
    s, c = jax.jit(math2d.sin_cos)(x)
    assert np.array_equal(np.asarray(s), np.array([math.sin(float(v)) for v in x], F32))
    assert np.array_equal(np.asarray(c), np.array([math.cos(float(v)) for v in x], F32))
    assert np.signbit(np.asarray(s)[-2])  # sin(-0) = -0, as C


@pytest.mark.parametrize("name", list(SCENES))
def test_scene_parity(name):
    twin = SCENES[name]()
    if twin is None:
        pytest.skip("no dumps/engine-dump-start.jsonl")
    assert initial_mismatches(twin) == []
    res = run(twin, SCENE_STEPS.get(name, 120))
    assert res["issues"] == []
    summ = summarize(res)
    for q in QUANTITIES:
        assert summ[q]["max"] == (0.0, 0), (q, summ[q])


def _batched(twin, n, seed=1):
    model, state = twin.model, twin.state
    rng = np.random.default_rng(seed)
    dyn = np.asarray(state.bodies.type == jb.DYNAMIC)[None, :, None]
    dv = jnp.asarray(np.where(dyn, rng.normal(0, 0.2, (n,) + state.bodies.v.shape), 0).astype(F32))
    st = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), state)
    return st._replace(bodies=st.bodies._replace(v=st.bodies.v + dv))


@pytest.mark.parametrize("axis_name", [None, "w"])
def test_vmap_256_worlds_match_single_world(axis_name):
    """256 perturbed copies of the pile scene stepped as one batch equal each world stepped alone."""
    twin = SCENES["pile"]().build()
    model, s = twin.model, twin.settings
    batch = _batched(twin, 256)
    f = jax.jit(jax.vmap(lambda st: jb.step(model, st, s, H, axis_name=axis_name), axis_name=axis_name))
    one = jax.jit(lambda st: jb.step(model, st, s, H))
    out = batch
    for _ in range(30):
        out = f(out)
    for i in (0, 77, 255):
        st = jax.tree.map(lambda x: x[i], batch)
        for _ in range(30):
            st = one(st)
        got = jax.tree.map(lambda x: x[i], out)
        for a, b in zip(jax.tree.leaves(st), jax.tree.leaves(got)):
            assert np.array_equal(np.asarray(a), np.asarray(b))
    # the perturbations really differ between worlds
    assert not np.array_equal(np.asarray(out.bodies.xf_p[0]), np.asarray(out.bodies.xf_p[77]))


def _fma_hazards(tree):
    """`a * b` written directly inside `+` / `-`: XLA would fuse it into one rounding (see math2d)."""
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            for side in (node.left, node.right):
                if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Mult):
                    bad.append(node.lineno)
    return bad


def test_no_raw_multiply_add():
    """Engine code forms products with math2d.mul; only math2d's float-float helpers (whose products are exact or
    deliberately fused) may write `x + a * b`."""
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text())
        if path.name == "math2d.py":
            fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
                   and n.name not in ("sin_cos",)]
            bad = [ln for f in fns for ln in _fma_hazards(f)]
        else:
            bad = _fma_hazards(tree)
        assert bad == [], f"{path.name}: lines {bad}"
