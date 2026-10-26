"""island.find_islands (b2World::Solve's depth-first island search) as one Pallas kernel, one world per lane.

The XLA search is a while loop with one iteration per pop or seed (~20 per physics step on the game level), each
iteration ~40 small kernels over the whole batch plus a device-to-host copy of the loop condition: ~1.1 s of a
16k-env training update on the 3090. Here every lane runs its own world's search sequentially: pop a body, walk
its contact edges in list order (the sorted contact order of find_islands, valid ones first), then its joint edges
newest first, pushing unvisited non-static bodies the first time they are met; when the stack is empty, seed the
highest-index awake, active, non-static unvisited body. This is the same sequence of states the vectorized search
goes through (its prefix sums and first-occurrence rule are exactly the sequential pushes), with the same cap of
2*B iterations, so the result is identical bit for bit (integers only; tests/box2d/test_gpu.py).

Kernel results per lane are kept by sorted contact position / joint edge position (no per-lane scatter by slot);
the slot-indexed tables (contact_order, contact_island, joint_order, joint_island) are rebuilt after the call.

ENABLED (default False; GOI_GPU_SPEED=islands or all sets it) switches gpu/solve.py from island.find_islands to this.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from ..island import Islands, _neg
from ..world import STATIC
from . import speed_option

ENABLED = speed_option("islands")
BITS = True  # the register (bit mask) form of the kernel where B <= 16 and J <= 31 (_kernel_bits); same results
BLOCK = 128
NUM_WARPS = 4


class _Static:
    def __init__(self, v):
        self.v = v


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.v), lambda v, _: _Static(v))


def _kernel(B, C, J, ja_ref, jb_ref, ce_ref, bf_ref, st_ref,
            cord_ref, cisl_ref, jord_ref, jisl_ref, bisl_ref, awk_ref, sto_ref, cnt_ref, flag_ref, stk_ref):
    """ce: (3, C, blk) s_ok, s_a, s_b in edge order; bf: (B, blk) bits 0 seedable, 1 static, 2 active, 3 awake;
    st: (B, blk) sleep_time. Outputs as described in the module docstring; flag / stk are scratch."""
    blk = bf_ref.shape[1]
    lane = jax.lax.broadcasted_iota(jnp.int32, (blk,), 0)
    zero = jnp.zeros((blk,), jnp.int32)
    for i in range(C):
        cord_ref[i, :] = zero - 1
        cisl_ref[i, :] = zero - 1
    for q in range(J):
        jord_ref[q, :] = zero - 1
        jisl_ref[q, :] = zero - 1
    for b in range(B):
        bisl_ref[b, :] = zero - 1
        awk_ref[b, :] = (bf_ref[b, :] >> 3) & 1
        sto_ref[b, :] = st_ref[b, :]
        flag_ref[b, :] = zero
        stk_ref[b, :] = zero
    # the valid contacts come first in edge order (find_islands' sort), so the walk stops at the block's max count
    n_ok = zero
    for i in range(C):
        n_ok = n_ok + ce_ref[0, i, :]
    nc_max = jnp.max(n_ok)
    bits = lambda i, k: (bf_ref[jnp.clip(i, 0, B - 1), lane] >> k) & 1  # noqa: E731  (per-lane gather)

    def get(ref, i):
        return ref[jnp.clip(i, 0, B - 1), lane]

    def put(ref, i, v, on):
        i = jnp.clip(i, 0, B - 1)
        ref[i, lane] = jnp.where(on, v, ref[i, lane])

    def step(carry):
        k, done, sp, nc, nj, nisl = carry
        empty = sp == 0
        seed = zero - 1
        for b in range(B):  # highest index wins
            f = bf_ref[b, :]
            cand = ((f & 1) != 0) & (awk_ref[b, :] != 0) & (flag_ref[b, :] == 0)
            seed = jnp.where(cand, b, seed)
        start = ~done & empty & (seed >= 0)
        done = done | (empty & (seed < 0))
        pop = ~done & ~empty
        b = get(stk_ref, sp - 1)
        isl = nisl - 1
        sp = sp - pop.astype(jnp.int32)

        def contact(i, c):
            sp, nc = c
            ok = ce_ref[0, i, :] != 0
            sa, sb = ce_ref[1, i, :], ce_ref[2, i, :]
            el = pop & ok & ((sa == b) | (sb == b)) & (cord_ref[i, :] < 0)
            cord_ref[i, :] = jnp.where(el, nc, cord_ref[i, :])
            cisl_ref[i, :] = jnp.where(el, isl, cisl_ref[i, :])
            other = jnp.where(sa == b, sb, sa)
            pu = el & (get(flag_ref, other) == 0) & (bits(other, 1) == 0)
            put(stk_ref, sp, other, pu)
            put(flag_ref, other, jnp.int32(1), pu)
            return sp + pu.astype(jnp.int32), nc + el.astype(jnp.int32)

        sp, nc = jax.lax.fori_loop(0, nc_max, contact, (sp, nc))
        for q in range(J):  # joint edges newest first: joint J-1-q
            a_, b_ = ja_ref[J - 1 - q], jb_ref[J - 1 - q]
            jo = jnp.where(a_ == b, b_, a_)
            el = pop & ((a_ == b) | (b_ == b)) & (jord_ref[q, :] < 0) & (bits(jo, 2) != 0)
            jord_ref[q, :] = jnp.where(el, nj, jord_ref[q, :])
            jisl_ref[q, :] = jnp.where(el, isl, jisl_ref[q, :])
            nj = nj + el.astype(jnp.int32)
            pu = el & (get(flag_ref, jo) == 0) & (bits(jo, 1) == 0)
            put(stk_ref, sp, jo, pu)
            put(flag_ref, jo, jnp.int32(1), pu)
            sp = sp + pu.astype(jnp.int32)
        # island membership + SetAwake(true) of the popped body
        put(bisl_ref, b, isl, pop)
        was = get(awk_ref, b) != 0
        put(awk_ref, b, jnp.int32(1), pop)
        put(sto_ref, b, jnp.float32(0), pop & ~was)
        # seed branch (stack empty)
        put(stk_ref, zero, seed, start)
        put(flag_ref, seed, jnp.int32(1), start)
        sp = jnp.where(start, 1, sp)
        nisl = nisl + start.astype(jnp.int32)
        return k + 1, done, sp, nc, nj, nisl

    def more(carry):
        return (carry[0] < 2 * B) & (jnp.min(carry[1].astype(jnp.int32)) == 0)

    f = jnp.zeros((blk,), bool)
    _, _, _, nc, nj, nisl = jax.lax.while_loop(more, step, (jnp.int32(0), f, zero, zero, zero, zero))
    cnt_ref[0, :] = nc
    cnt_ref[1, :] = nj
    cnt_ref[2, :] = nisl


def _kernel_bits(B, C, J, ja_ref, jb_ref, ce_ref, bf_ref, st_ref,
                 cord_ref, cisl_ref, jord_ref, jisl_ref, bisl_ref, awk_ref, sto_ref, cnt_ref, flag_ref, stk_ref):
    """_kernel for B <= 16 bodies and J <= 31 joints with the per-body flags, the joint flags and the stack in
    registers: bit masks (visited, static, active, seedable, awake) and the stack as 4-bit entries in two int32
    words. The contact walk then reads only its own rows (no per-lane gathers or scatters per contact). Same
    sequence of pops, pushes and seeds as _kernel."""
    del flag_ref, stk_ref
    blk = bf_ref.shape[1]
    lane = jax.lax.broadcasted_iota(jnp.int32, (blk,), 0)
    zero = jnp.zeros((blk,), jnp.int32)
    one = zero + 1
    for i in range(C):
        cord_ref[i, :] = zero - 1
        cisl_ref[i, :] = zero - 1
    seedm = staticm = activem = awake0 = zero
    for b in range(B):
        f = bf_ref[b, :]
        seedm = seedm | ((f & 1) << b)
        staticm = staticm | (((f >> 1) & 1) << b)
        activem = activem | (((f >> 2) & 1) << b)
        awake0 = awake0 | (((f >> 3) & 1) << b)
        bisl_ref[b, :] = zero - 1
        sto_ref[b, :] = st_ref[b, :]
    n_ok = zero
    for i in range(C):
        n_ok = n_ok + ce_ref[0, i, :]
    nc_max = jnp.max(n_ok)
    bit = lambda m_, i: (m_ >> i) & 1  # noqa: E731

    def push(st, other, pu):
        """Write `other` at stack entry sp where pu (4 bits per entry, 8 per word); returns (s0, s1, sp)."""
        s0, s1, sp = st
        sh = (sp & 7) * 4
        put = lambda w, word: jnp.where(pu & ((sp >> 3) == word), (w & ~(15 << sh)) | (other << sh), w)  # noqa
        return put(s0, 0), put(s1, 1), sp + pu.astype(jnp.int32)

    def step(carry):
        k, done, s0, s1, sp, nc, nj, nisl, flagm, awakem, jflagm = carry
        empty = sp == 0
        cand = seedm & awakem & ~flagm
        seed = zero - 1
        for b in range(B):  # highest index wins
            seed = jnp.where(bit(cand, b) != 0, b, seed)
        start = ~done & empty & (seed >= 0)
        done = done | (empty & (seed < 0))
        pop = ~done & ~empty
        top = sp - 1
        tsh = (top & 7) * 4
        b = (jnp.where((top >> 3) == 0, s0, s1) >> tsh) & 15
        isl = nisl - 1
        sp = sp - pop.astype(jnp.int32)

        def contact(i, c):
            s0, s1, sp, nc, flagm = c
            ok = ce_ref[0, i, :] != 0
            sa, sb = ce_ref[1, i, :], ce_ref[2, i, :]
            co = cord_ref[i, :]
            el = pop & ok & ((sa == b) | (sb == b)) & (co < 0)
            cord_ref[i, :] = jnp.where(el, nc, co)
            cisl_ref[i, :] = jnp.where(el, isl, cisl_ref[i, :])
            other = jnp.where(sa == b, sb, sa)
            pu = el & (bit(flagm, other) == 0) & (bit(staticm, other) == 0)
            s0, s1, sp = push((s0, s1, sp), other, pu)
            flagm = flagm | jnp.where(pu, one << other, 0)
            return s0, s1, sp, nc + el.astype(jnp.int32), flagm

        s0, s1, sp, nc, flagm = jax.lax.fori_loop(0, nc_max, contact, (s0, s1, sp, nc, flagm))
        for q in range(J):  # joint edges newest first: joint J-1-q
            a_, b_ = ja_ref[J - 1 - q], jb_ref[J - 1 - q]
            jo = jnp.where(a_ == b, b_, a_)
            el = pop & ((a_ == b) | (b_ == b)) & (bit(jflagm, q) == 0) & (bit(activem, jo) != 0)
            jord_ref[q, :] = jnp.where(el, nj, jord_ref[q, :])
            jisl_ref[q, :] = jnp.where(el, isl, jisl_ref[q, :])
            jflagm = jflagm | jnp.where(el, 1 << q, 0)
            nj = nj + el.astype(jnp.int32)
            pu = el & (bit(flagm, jo) == 0) & (bit(staticm, jo) == 0)
            s0, s1, sp = push((s0, s1, sp), jo, pu)
            flagm = flagm | jnp.where(pu, one << jo, 0)
        # island membership + SetAwake(true) of the popped body (one per-lane write each)
        bb = jnp.clip(b, 0, B - 1)
        bisl_ref[bb, lane] = jnp.where(pop, isl, bisl_ref[bb, lane])
        was = bit(awakem, b) != 0
        sto_ref[bb, lane] = jnp.where(pop & ~was, jnp.float32(0), sto_ref[bb, lane])
        awakem = awakem | jnp.where(pop, one << b, 0)
        # seed branch (stack empty)
        s0 = jnp.where(start, (s0 & ~15) | jnp.maximum(seed, 0), s0)
        flagm = flagm | jnp.where(start, one << jnp.maximum(seed, 0), 0)
        sp = jnp.where(start, 1, sp)
        nisl = nisl + start.astype(jnp.int32)
        return k + 1, done, s0, s1, sp, nc, nj, nisl, flagm, awakem, jflagm

    def more(carry):
        return (carry[0] < 2 * B) & (jnp.min(carry[1].astype(jnp.int32)) == 0)

    for q in range(J):
        jord_ref[q, :] = zero - 1
        jisl_ref[q, :] = zero - 1
    f = jnp.zeros((blk,), bool)
    out = jax.lax.while_loop(more, step, (jnp.int32(0), f, zero, zero, zero, zero, zero, zero, zero, awake0, zero))
    nc, nj, nisl, awakem = out[5], out[6], out[7], out[9]
    for b in range(B):
        awk_ref[b, :] = bit(awakem, b)
    cnt_ref[0, :] = nc
    cnt_ref[1, :] = nj
    cnt_ref[2, :] = nisl


def _call(cfg, ja, jb, ce, bf, st):
    """Batched: ce (N, 3, C), bf (N, B), st (N, B) -> per-lane tables with the world axis first."""
    B, C, J, blk, interpret = cfg
    N = bf.shape[0]
    Np = -(-N // blk) * blk

    def minor(x):
        x = jnp.moveaxis(x, 0, -1)
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, Np - N)])

    ce, bf, st = jax.lax.optimization_barrier((ce, bf, st))
    ce, bf, st = minor(ce), minor(bf), minor(st)
    full = lambda a: pl.BlockSpec(a.shape, lambda i: (0,) * a.ndim)  # noqa: E731
    s2 = lambda r: pl.BlockSpec((r, blk), lambda i: (0, i))  # noqa: E731
    i32 = lambda r: jax.ShapeDtypeStruct((r, Np), jnp.int32)  # noqa: E731
    kw = {}
    if not interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=1)
    outs = pl.pallas_call(
        partial(_kernel_bits if (BITS and B <= 16 and J <= 31) else _kernel, B, C, J), grid=(Np // blk,),
        in_specs=[full(ja), full(jb), pl.BlockSpec((3, C, blk), lambda i: (0, 0, i)), s2(B), s2(B)],
        out_specs=[s2(C), s2(C), s2(J), s2(J), s2(B), s2(B), s2(B), s2(3), s2(B), s2(B)],
        out_shape=[i32(C), i32(C), i32(J), i32(J), i32(B), i32(B), jax.ShapeDtypeStruct((B, Np), jnp.float32),
                   i32(3), i32(B), i32(B)],
        interpret=interpret, name="box2d_find_islands", **kw)(ja, jb, ce, bf, st)
    return tuple(jnp.moveaxis(o[..., :N], -1, 0) for o in outs[:8])


@jax.custom_batching.custom_vmap
def _search(cfg_box, ja, jb, ce, bf, st):
    out = _call(cfg_box.v, ja, jb, ce[None], bf[None], st[None])
    return tuple(o[0] for o in out)


@_search.def_vmap
def _search_rule(axis_size, in_batched, cfg_box, ja, jb, ce, bf, st):
    if in_batched[1] or in_batched[2]:
        raise NotImplementedError("gpu find_islands: the joint tables must be shared by all worlds")
    ce, bf, st = (a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((ce, bf, st),
                                                                                        in_batched[3:]))
    out = _call(cfg_box.v, ja, jb, ce, bf, st)
    return out, tuple(True for _ in out)


def find_islands(model, state, block=None, interpret=None):
    """island.find_islands with the search in a kernel; same Islands, bit for bit."""
    bd, ct = state.bodies, state.contacts
    B, C, J = bd.type.shape[0], ct.pair.shape[0], model.joint_type.shape[0]
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    if block is None:
        from . import solve as G
        block = G.BLOCK
    fbody = model.fixture_body
    ca, cb = fbody[ct.fa], fbody[ct.fb]
    c_ok = (ct.pair >= 0) & ct.touching
    sort_key = jnp.where(c_ok, ct.key, jnp.iinfo(jnp.int32).min)
    perm = jnp.argsort(_neg(sort_key), stable=True).astype(jnp.int32)
    ce = jnp.stack([c_ok[perm].astype(jnp.int32), ca[perm].astype(jnp.int32), cb[perm].astype(jnp.int32)])
    static = bd.type == STATIC
    seedable = bd.active & ~static
    i = lambda x: x.astype(jnp.int32)  # noqa: E731
    bf = i(seedable) | (i(static) << 1) | (i(bd.active) << 2) | (i(bd.awake) << 3)
    Jk = max(J, 1)
    ja = model.joint_a.astype(jnp.int32) if J else jnp.full((1,), -1, jnp.int32)
    jb = model.joint_b.astype(jnp.int32) if J else jnp.full((1,), -1, jnp.int32)
    cord, cisl, jord, jisl, bisl, awk, sto, cnt = _search(
        _Static((B, C, Jk, block, interpret)), ja, jb, ce, bf, bd.sleep_time.astype(jnp.float32))
    # back to slot-indexed tables (find_islands' layout)
    corder = jnp.full(C, -1, jnp.int32).at[jnp.where(cord >= 0, cord, C)].set(perm, mode="drop")
    contact_island = jnp.full(C, -1, jnp.int32).at[perm].set(cisl)
    if J:
        jperm = jnp.arange(J - 1, -1, -1, dtype=jnp.int32)
        jorder = jnp.full(J, -1, jnp.int32).at[jnp.where(jord >= 0, jord, J)].set(jperm, mode="drop")
        joint_island = jnp.full(J, -1, jnp.int32).at[jperm].set(jisl)
    else:
        jorder = joint_island = jnp.zeros((0,), jnp.int32)
    return Islands(contact_order=corder, n_contacts=cnt[0], joint_order=jorder, n_joints=cnt[1],
                   body_island=bisl, contact_island=contact_island, joint_island=joint_island,
                   n_islands=cnt[2], awake=awk != 0, sleep_time=sto)
