"""The compact fixture layout (goi_jax/box2d/world.py compact_model) against the full layout, bit for bit.

Static bodies' fixtures move into the shared Model and each world keeps only the non-static fixtures' rows; the step
must give the same State (expanded back to the full layout) on every step, with the XLA solver and with the Pallas
kernels (interpret mode on CPU), and the game Sim must give the same worlds with a compact Model.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/box2d/test_compact.py -q
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
from goi_jax.box2d import world as W  # noqa: E402
import goi_jax.box2d.gpu.solve as gsolve  # noqa: E402
from test_gpu import SCENES, batch, mismatches  # noqa: E402

H = jb.GAME_DT
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
XLA_SCENES = ("pile", "player_rig", "polygon_circle", "late_wake", "toi_hammer_chain", "toi_multi_event")
PALLAS_SCENES = ("player_rig", "toi_hammer_chain")


def _run(name, solver, steps, worlds=3):
    twin = SCENES[name]()
    if twin is None:
        pytest.skip("scene needs dumps/engine-dump-start.jsonl")
    model, state = twin.defn.build(twin.capacity)
    cmodel = W.compact_model(model, state)
    if cmodel.fixture_slot is None:
        pytest.skip("scene has no static fixtures")
    s = replace(twin.settings, solver=solver)
    mk = lambda md: jax.jit(jax.vmap(lambda st: jb.step(md, st, s, H, axis_name="w"), axis_name="w"))  # noqa: E731
    f_full, f_comp = mk(model), mk(cmodel)
    st = jax.device_put(jax.tree.map(np.asarray, batch(state, worlds)), jax.devices()[0])
    a, b = st, W.compact_state(cmodel, st)
    assert b.fixtures.fat.shape[-2] == int(np.sum(np.asarray(cmodel.fixture_slot) >= 0))
    old = gsolve.BLOCK
    gsolve.BLOCK = 8
    try:
        for k in range(steps):
            a, b = f_full(a), f_comp(b)
            bad = mismatches(a, W.expand_state(cmodel, b))
            assert not bad, f"{name} ({solver}): first mismatch at step {k + 1}: {bad[:6]}"
    finally:
        gsolve.BLOCK = old


@pytest.mark.parametrize("name", XLA_SCENES)
def test_compact_xla_bit_exact(name):
    _run(name, "xla", 40)


@pytest.mark.parametrize("name", PALLAS_SCENES)
def test_compact_pallas_interpret_bit_exact(name):
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    _run(name, "pallas", 25)


def test_round_trip():
    twin = SCENES["pile"]()
    model, state = twin.defn.build(twin.capacity)
    cmodel = W.compact_model(model, state)
    st = batch(state, 2)
    c = W.compact_state(cmodel, st)
    assert W.compact_state(cmodel, c) is c  # idempotent
    assert not mismatches(st, W.expand_state(cmodel, c))
    bad = st._replace(fixtures=st.fixtures._replace(friction=st.fixtures.friction + 1))
    with pytest.raises(ValueError):
        W.compact_state(cmodel, bad)


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no recordings in {DATA}")
def test_game_sim_compact_bit_exact():
    """The game's lockstep Sim: K=4 actions from recorded states with the compact Model equal the full layout's."""
    from goi_jax.game import sim as SM
    from goi_jax.parity import states as S
    full = SM.Sim.from_recordings(DATA)
    comp = SM.Sim.from_recordings(DATA, compact_fixtures=True)
    assert comp.model.fixture_slot is not None
    chain = S.load_chain(DATA / "coffee" / "c01-random-62.npz")
    t0 = SM.canonical_step(chain)
    ts = [t0, t0 + 20]
    sf = jax.tree.map(lambda *a: np.stack(a), *[full.state_from_recording(chain, t) for t in ts])
    sc = jax.tree.map(lambda *a: np.stack(a), *[comp.state_from_recording(chain, t) for t in ts])
    assert sc.world.fixtures.fat.shape[-2] < sf.world.fixtures.fat.shape[-2]
    act = lambda sim: jax.jit(jax.vmap(lambda md, p, s, a: sim.act(md, p, s, a, 4, "w")[0],  # noqa: E731
                                       (None, None, 0, 0), axis_name="w"))
    f_full, f_comp = act(full), act(comp)
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = jnp.asarray(rng.uniform(-40, 40, (len(ts), 2)).astype(np.float32))
        sf, sc = f_full(full.model, full.params, sf, a), f_comp(comp.model, comp.params, sc, a)
        bad = mismatches(sf.world, W.expand_state(comp.model, sc.world))
        bad += mismatches(sf._replace(world=None), sc._replace(world=None))
        assert not bad, bad[:6]


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no recordings in {DATA}")
def test_culled_scan_game():
    """The GPU FindNewContacts on the culled scan order (Model.scan_*, gpu/pairs.py _kernel_cull; interpret mode on
    CPU) equals the XLA pairs.find_new_contacts on the real level, from recorded states as they are, with every
    contact removed (each overlapping candidate is then new: dozens per world), and with that at capacity 8 (more
    new pairs than slots: the first ones in candidate order are kept)."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    from goi_jax.box2d import pairs as XP
    from goi_jax.box2d.gpu import pairs as GP
    from goi_jax.game import sim as SM
    from goi_jax.parity import states as S
    comp = SM.Sim.from_recordings(DATA, compact_fixtures=True)
    md = comp.model
    assert md.scan_cand is not None
    sc = np.asarray(md.scan_cand)
    assert sorted(sc[sc >= 0].tolist()) == list(range(md.pair_a.shape[0]))  # a permutation of the candidates
    chain = S.load_chain(DATA / "coffee" / "c01-random-62.npz")
    t0 = SM.canonical_step(chain)
    sts = [comp.state_from_recording(chain, t).world for t in (t0, t0 + 20, t0 + 40)]
    st = jax.tree.map(lambda *a: np.stack(a), *sts)
    ct = st.contacts
    empty = st._replace(contacts=ct._replace(pair=np.full_like(ct.pair, -1)))
    small = empty._replace(contacts=jax.tree.map(lambda a: a[:, :8], empty.contacts))
    for name, case in (("recorded", st), ("empty", empty), ("capacity 8", small)):
        ref = jax.jit(jax.vmap(lambda s: XP.find_new_contacts(md, s)))(case)
        got = jax.jit(jax.vmap(lambda s: GP.find_new_contacts(md, s, block=8)))(case)
        bad = mismatches(ref, got)
        assert not bad, (name, bad[:6])
        if name == "capacity 8":
            assert int(np.min(np.asarray(ref.overflow))) > 0  # more new pairs than the 8 slots
