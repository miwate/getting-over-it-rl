"""The speed options of goi_jax/box2d/gpu, bit for bit against the XLA step (interpret mode on CPU):
- gpu/islands.py (the island search as a Pallas kernel) against island.find_islands, every field;
- gpu/toi_loop.SMALL_DIVISOR (a smaller lane count for TOI iterations with few active worlds), whole steps.

States come from the tests/box2d scenes stepped with the XLA step (a few worlds with perturbed velocities, so lanes
differ), plus states with random awake flags / touching contacts; blocks of 4 lanes so several programs and padded
lanes run. On CPU the kernel runs in Pallas interpret mode.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/box2d/test_gpu_speed.py -q
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goi_jax import box2d as jb  # noqa: E402
from goi_jax.box2d.gpu import islands as GI  # noqa: E402
from goi_jax.box2d.island import find_islands  # noqa: E402
from test_gpu import SCENES, batch, compare_scene  # noqa: E402

NAMES = ("box_stack", "pile", "player_rig", "limits_mid_run", "weld", "late_wake", "toi_island_walk",
         "toi_multi_event", "toi_hammer_chain")


def _cmp(a, b):
    bad = []
    for f, x, y in zip(a._fields, a, b):
        x, y = np.asarray(x), np.asarray(y)
        same = (x.view(np.int32) == y.view(np.int32)) if x.dtype == np.float32 else (x == y)
        if not np.all(same):
            bad.append(f)
    return bad


def _scramble(state, seed):
    """Random awake flags and touching contacts (more islands, seeds and sleeping bodies than the scenes reach)."""
    rng = np.random.default_rng(seed)
    bd, ct = state.bodies, state.contacts
    aw = jnp.asarray(rng.random(bd.awake.shape) < 0.6) & bd.awake
    tc = jnp.asarray(rng.random(ct.touching.shape) < 0.7) & ct.touching
    st = jnp.asarray(rng.random(bd.sleep_time.shape).astype(np.float32))
    return state._replace(bodies=bd._replace(awake=aw, sleep_time=st), contacts=ct._replace(touching=tc))


@pytest.mark.parametrize("bits", [True, False])
@pytest.mark.parametrize("name", [n for n in NAMES if n in SCENES])
def test_islands_kernel_bit_exact(name, bits, monkeypatch):
    """Both kernel forms: bit masks in registers (_kernel_bits, scenes with <= 16 bodies) and the memory form."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    twin = SCENES[name]()
    if twin is None:
        pytest.skip("scene needs dumps/engine-dump-start.jsonl")
    model, state = twin.defn.build(twin.capacity)
    if bits and state.bodies.type.shape[0] > 16:
        pytest.skip("more than 16 bodies: the memory form runs")
    monkeypatch.setattr(GI, "BITS", bits)
    st = batch(state, 6)
    step = jax.jit(jax.vmap(lambda s: jb.step(model, s, twin.settings, jb.GAME_DT, axis_name="w"), axis_name="w"))
    ref = jax.jit(jax.vmap(lambda s: find_islands(model, s)))
    ker = jax.jit(jax.vmap(lambda s: GI.find_islands(model, s, block=4)))
    for k in range(40):
        for s in (st, _scramble(st, k)):
            bad = _cmp(ref(s), ker(s))
            assert not bad, f"{name}: step {k}: {bad}"
        st = step(st)
    one = jax.tree.map(lambda x: x[1], st)  # unbatched call
    assert not _cmp(find_islands(model, one), GI.find_islands(model, one, block=4))


@pytest.mark.parametrize("name", ["toi_island_walk", "toi_multi_event"])
def test_toi_small_lanes(name):
    """gpu/toi_loop.py with the small lane count: 10 worlds, blocks of 2, M = 4 lanes, 2 lanes once at most 2 worlds
    still have an event; and the island kernel on in the same steps."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    from goi_jax.box2d.gpu import toi_loop
    old = toi_loop.MIN_LANES, toi_loop.DIVISOR, toi_loop.SMALL_DIVISOR, GI.ENABLED
    toi_loop.MIN_LANES, toi_loop.DIVISOR, toi_loop.SMALL_DIVISOR, GI.ENABLED = 1, 4, 10, True
    try:
        step, bad = compare_scene(name, steps=30, worlds=10, block=2)
    finally:
        toi_loop.MIN_LANES, toi_loop.DIVISOR, toi_loop.SMALL_DIVISOR, GI.ENABLED = old
    assert not bad, f"first mismatch at step {step}: {bad[:6]}"


@pytest.mark.parametrize("name", ["toi_island_walk", "player_rig"])
def test_sorted_lanes(name):
    """gpu/kernel.SORT_LANES (worlds regrouped by loop counts before the island solver, TOI search and TOI event
    kernels): 10 worlds in blocks of 2."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    from goi_jax.box2d.gpu import kernel as K
    old = K.SORT_LANES
    K.SORT_LANES = True
    try:
        step, bad = compare_scene(name, steps=25, worlds=10, block=2)
    finally:
        K.SORT_LANES = old
    assert not bad, f"first mismatch at step {step}: {bad[:6]}"


def test_sorted_lanes_culled_scan_game():
    """gpu/pairs.py's culled FindNewContacts with SORT_LANES (worlds grouped by position) and SLOT_PLACE on recorded
    game states (several places of the level, contacts removed so there are many new pairs), blocks of 2."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    root = Path(__file__).resolve().parents[2] / "dumps" / "parity"
    if not (root / "index.json").exists():
        pytest.skip("no recordings")
    from goi_jax.box2d import pairs as XP
    from goi_jax.box2d.gpu import kernel as K
    from goi_jax.box2d.gpu import pairs as GP
    from goi_jax.game import sim as SM
    from goi_jax.parity import states as S
    from test_gpu import mismatches
    comp = SM.Sim.from_recordings(root, compact_fixtures=True)
    md = comp.model
    sts = []
    for e in S.load_index(root)["chains"][:6]:
        chain = S.load_chain(root / e["file"])
        t0 = SM.canonical_step(chain)
        sts += [comp.state_from_recording(chain, t).world for t in (t0, t0 + 10)]
    st = jax.tree.map(lambda *a: np.stack(a), *sts)
    ct = st.contacts
    keep = np.arange(ct.pair.shape[-1])[None, :] < np.arange(len(sts))[:, None] % 3 * 4  # some contacts left
    st = st._replace(contacts=ct._replace(pair=np.where(keep, ct.pair, -1)))
    small = st._replace(contacts=jax.tree.map(lambda a: a[:, :8], st.contacts))  # more new pairs than slots
    old = K.SORT_LANES, GP.SLOT_PLACE
    K.SORT_LANES, GP.SLOT_PLACE = True, True
    try:
        for case in (st, small):
            ref = jax.jit(jax.vmap(lambda s: XP.find_new_contacts(md, s)))(case)
            got = jax.jit(jax.vmap(lambda s: GP.find_new_contacts(md, s, block=2)))(case)
            bad = mismatches(ref, got)
            assert not bad, bad[:6]
    finally:
        K.SORT_LANES, GP.SLOT_PLACE = old
    assert int(np.max(np.asarray(ref.overflow))) > 0


@pytest.mark.parametrize("name", ["late_wake", "toi_hammer_chain", "player_rig", "limits_mid_run", "weld"])
def test_all_speed_options(name):
    """Every GPU speed option at once (islands, sort, small, idle: the start-of-step FindNewContacts skipped when no
    world has a new fixture, place: new pairs placed per slot, joints: one joint type per block where the lanes
    agree), 10 worlds in blocks of 2; the scenes start with new fixtures (the first step runs it); player_rig,
    limits_mid_run and weld have every joint type."""
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    from goi_jax.box2d.gpu import kernel as K
    from goi_jax.box2d.gpu import pairs as P
    from goi_jax.box2d.gpu import toi_loop
    from goi_jax.box2d.gpu import toi_search as TS
    mods = ((GI, "ENABLED", True), (K, "SORT_LANES", True), (P, "SKIP_IDLE_START", True), (P, "SLOT_PLACE", True),
            (K, "BRANCH_SPLIT", True), (P, "SYNC_ROT", True), (TS, "SHARED_PROXIES", True),
            (toi_loop, "SMALL_DIVISOR", 10), (toi_loop, "MIN_LANES", 1))
    old = [getattr(mod, a) for mod, a, _ in mods]
    for mod, a, v in mods:
        setattr(mod, a, v)
    try:
        step, bad = compare_scene(name, steps=30, worlds=10, block=2)
    finally:
        for (mod, a, _), v in zip(mods, old):
            setattr(mod, a, v)
    assert not bad, f"first mismatch at step {step}: {bad[:6]}"
