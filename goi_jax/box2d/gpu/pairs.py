"""b2ContactManager::FindNewContacts and b2Body::SynchronizeFixtures (pairs.py) as Pallas kernels.

FindNewContacts: the candidate scan runs in a kernel.

The XLA version tests all P candidate pairs of the Model in every world at once: for the real level (~1500 static
fixtures, P ~ 31k) that is P-sized temporaries per world (~0.8 MB) and most of the step's time. Here one kernel
walks the candidates in their order, one world per lane, and returns per world only the new pairs in creation
order (at most C, the contact capacity; any more could not be placed anyway) and how many there were. Candidates
go in chunks of 32: a lane's hits (fat AABBs overlap, b2Body::ShouldCollide's one-dynamic rule, both bodies
active) are collected as a bit mask, and only a chunk where some lane of the block has a hit takes the slow path
that checks the lane's existing contacts and appends the new pairs. The candidate list is split into up to
SEGMENTS segments scanned by separate programs (grid axis 1) and their new-pair lists are merged in candidate order
after the call: one program walking all ~31k candidates took ~15-19 ms per call on the 3090 whatever the batch size,
and the call runs once per step plus once per TOI event (most of a training rollout after the TOI kernels). Placement into free slots, keys, mixing of
friction / restitution and the wakes then run in XLA on the C-sized list, with the same formulas as pairs.py.

Culled scan (compact layout with Model.scan_*, world.scan_tables; CULL): each segment is regrouped by the
non-static fixture its candidates pair with, spatially sorted, in chunks of 16 with the union of their static fat
AABBs; a lane tests a chunk only where that fixture's fat AABB overlaps the union (~9x fewer tests on the game
level: ~1.4k candidate and ~2k chunk tests instead of 31k candidates per world and call), and keeps its new pairs
sorted by candidate index, so each segment still yields its first C new pairs in candidate order.

`find_new_contacts(model, state, enable=None)` equals pairs.find_new_contacts bit for bit (interpret mode on CPU;
tests/box2d/test_gpu.py). `enable` (bool per world) skips the scan where false (the start-of-step call only matters
for worlds with State.new_fixture); a disabled world gets no new contacts.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from .. import math2d as m
from .. import shapes
from ..pairs import _wake
from ..world import DYNAMIC, fixture_friction, fixture_restitution, fixture_rows, is_compact
from . import speed_option

F32 = np.float32
CHUNK = 32


CULL = True  # use the Model's culled scan order (Model.scan_*, compact layout) when it has one
SEGMENTS = 16  # candidate segments scanned by separate programs (latency: one program per segment, merged after)
SKIP_IDLE_START = speed_option("idle")  # step.start_new_contacts: skip the call when no world has a new fixture
SLOT_PLACE = speed_option("place")  # find_new_contacts' placement per slot (_place_by_slot), same results
SYNC_ROT = speed_option("sync")  # gpu/solve.py: the fixture sync reuses the solver's body rotations (one sin/cos less)


@dataclass(frozen=True)
class Cfg:
    P: int  # candidates (the tables are padded to a multiple of CHUNK)
    C: int
    block: int
    interpret: bool
    segments: int = 1
    chunk: int = CHUNK      # culled scan: candidates per chunk
    seg_chunks: int = 0     # culled scan: chunks per segment


def _kernel(cfg, ra_ref, rb_ref, sbox_ref, ba_ref, bb_ref, fat_ref, flag_ref, pair_ref, en_ref, lst_ref, n_ref):
    C = cfg.C
    en = en_ref[0, :] != 0
    any_en = jnp.max(en.astype(jnp.int32)) > 0
    zero = jnp.zeros_like(en, dtype=jnp.int32)

    def chunk(ci, carry):
        base = ci * CHUNK
        bits = zero
        for j in range(CHUNK):
            k = base + j
            ra, rb, ba, bb = ra_ref[k], rb_ref[k], ba_ref[k], bb_ref[k]
            # a side with row -1 is a static fixture: its fat AABB is the candidate's shared sbox (compact layout)
            A = tuple(jnp.where(ra >= 0, fat_ref[jnp.maximum(ra, 0), q, :], sbox_ref[k, q]) for q in range(4))
            Bx = tuple(jnp.where(rb >= 0, fat_ref[jnp.maximum(rb, 0), q, :], sbox_ref[k, q]) for q in range(4))
            over = shapes.overlap(A, Bx)
            fa_b, fb_b = flag_ref[ba, :], flag_ref[bb, :]  # bit 0 dynamic, bit 1 active
            can = (((fa_b & 1) | (fb_b & 1)) != 0) & ((fa_b & fb_b & 2) != 0)
            hit = over & can & en & (k < cfg.P)
            bits = bits | (hit.astype(jnp.int32) << j)

        def slow(carry):
            def one(j, carry):
                n, lst = carry
                k = base + j
                hit = ((bits >> j) & 1) != 0

                def ex(c, e):
                    return e | (pair_ref[c, :] == k)
                exists = jax.lax.fori_loop(0, C, ex, jnp.zeros_like(hit))
                new = hit & ~exists
                lst = [jnp.where(new & (n == q), k, x) for q, x in enumerate(lst)]
                return n + new.astype(jnp.int32), lst
            return jax.lax.fori_loop(0, CHUNK, one, carry)

        return jax.lax.cond(jnp.max((bits != 0).astype(jnp.int32)) > 0, slow, lambda c: c, carry)

    nch = -(-cfg.P // CHUNK)
    per = -(-nch // cfg.segments)
    lo = pl.program_id(1) * per  # this program's segment of the candidate chunks (grid axis 1)
    hi = jnp.minimum(lo + per, nch)

    def scan(carry):
        return jax.lax.fori_loop(lo, hi, chunk, carry)

    carry = (zero, [zero - 1 for _ in range(C)])
    n, lst = jax.lax.cond(any_en, scan, lambda c: c, carry)
    for q in range(C):
        lst_ref[q, :] = lst[q]
    n_ref[0, :] = n


def _kernel_cull(cfg, cand_ref, ra_ref, rb_ref, sbox_ref, ba_ref, bb_ref, crow_ref, cbox_ref, fat_ref, flag_ref,
                 pair_ref, en_ref, lst_ref, n_ref):
    """_kernel on the culled scan order (Model.scan_*): this program's segment is a contiguous range of the candidate
    order, walked chunk by chunk in the regrouped order; a chunk is tested only in lanes where its non-static
    fixture's fat AABB overlaps the union of the chunk's static fat AABBs (a candidate that overlaps overlaps the
    union, so nothing is lost). The lane's new pairs are kept sorted by candidate index (the first C of them, as
    the in-order scan keeps), and n counts all of them: the same (list, count) per segment as _kernel."""
    C, CH = cfg.C, cfg.chunk
    en = en_ref[0, :] != 0
    any_en = jnp.max(en.astype(jnp.int32)) > 0
    zero = jnp.zeros_like(en, dtype=jnp.int32)
    BIG = jnp.int32(2 ** 30)

    def chunk(ci, carry):
        r = crow_ref[ci]
        bx = tuple(cbox_ref[ci, q] for q in range(4))
        F = tuple(fat_ref[jnp.maximum(r, 0), q, :] for q in range(4))
        pre = en & ((r == -1) | ((r >= 0) & shapes.overlap(F, bx)))

        def test(carry):
            base = ci * CH
            bits = zero
            for j in range(CH):
                k = base + j
                kk, ra, rb, ba, bb = cand_ref[k], ra_ref[k], rb_ref[k], ba_ref[k], bb_ref[k]
                A = tuple(jnp.where(ra >= 0, fat_ref[jnp.maximum(ra, 0), q, :], sbox_ref[k, q]) for q in range(4))
                Bx = tuple(jnp.where(rb >= 0, fat_ref[jnp.maximum(rb, 0), q, :], sbox_ref[k, q]) for q in range(4))
                over = shapes.overlap(A, Bx)
                fa_b, fb_b = flag_ref[ba, :], flag_ref[bb, :]  # bit 0 dynamic, bit 1 active
                can = (((fa_b & 1) | (fb_b & 1)) != 0) & ((fa_b & fb_b & 2) != 0)
                hit = over & can & pre & (kk >= 0)
                bits = bits | (hit.astype(jnp.int32) << j)

            def slow(carry):
                def one(j, carry):
                    kk = cand_ref[base + j]
                    hit = ((bits >> j) & 1) != 0

                    def check(carry):  # only for a candidate that some lane of the block hits
                        n, lst = carry

                        def ex(c, e):
                            return e | (pair_ref[c, :] == kk)
                        exists = jax.lax.fori_loop(0, C, ex, jnp.zeros_like(hit))
                        new = hit & ~exists

                        def insert(lst):  # sorted insert of kk (BIG past the list's end); a full list drops the max
                            pos = sum((x < kk).astype(jnp.int32) for x in lst)
                            return [jnp.where(new & (q == pos), kk, jnp.where(new & (q > pos), lst[q - 1], x)) if q
                                    else jnp.where(new & (pos == 0), kk, x) for q, x in enumerate(lst)]
                        lst = jax.lax.cond(jnp.max(new.astype(jnp.int32)) > 0, insert, lambda l_: l_, lst)
                        return n + new.astype(jnp.int32), lst
                    return jax.lax.cond(jnp.max(hit.astype(jnp.int32)) > 0, check, lambda c: c, carry)
                return jax.lax.fori_loop(0, CH, one, carry)

            return jax.lax.cond(jnp.max((bits != 0).astype(jnp.int32)) > 0, slow, lambda c: c, carry)

        return jax.lax.cond(jnp.max(pre.astype(jnp.int32)) > 0, test, lambda c: c, carry)

    per = cfg.seg_chunks
    lo = pl.program_id(1) * per

    def scan(carry):
        return jax.lax.fori_loop(lo, lo + per, chunk, carry)

    carry = (zero, [zero + BIG for _ in range(C)])
    n, lst = jax.lax.cond(any_en, scan, lambda c: c, carry)
    for q in range(C):
        lst_ref[q, :] = jnp.where(lst[q] == BIG, -1, lst[q])
    n_ref[0, :] = n


def _call(cfg, ra, rb, sbox, ba, bb, fat, flag, pair, en, cull=None):
    """Batched: fat (N, rows, 4), flag (N, B), pair (N, C), en (N,) -> new list (N, C), count (N,)."""
    N = fat.shape[0]
    blk = cfg.block
    Np = -(-N // blk) * blk

    # Materialize the inputs first: otherwise XLA fuses their producers (e.g. synchronize_fixtures' AABBs, sin/cos
    # included) into the transposes below and recomputes them per element (~0.7 s/step at 16k worlds).
    fat, flag, pair, en = jax.lax.optimization_barrier((fat, flag, pair, en))
    from .kernel import lane_order, to_minor, from_minor
    # K.SORT_LANES: worlds grouped by where they are (8 m cells of the first moving fixture's fat AABB, x then y), so
    # a program's lanes pass the culled scan's chunk tests together; disabled worlds last
    q = lambda v: jnp.clip(jnp.floor(jnp.nan_to_num(v) / 8) + 512, 0, 1023)  # noqa: E731
    key = jnp.where(en != 0, q(fat[:, 0, 0]) * 1024 + q(fat[:, 0, 1]), 1 << 21).astype(jnp.int32)
    order = lane_order(key, N, blk) if cull is not None else None
    fat, flag, pair, en = (to_minor(x, N, Np, order) for x in (fat, flag, pair, en[:, None]))
    full1 = lambda a: pl.BlockSpec(a.shape, lambda i, s: (0,) * a.ndim)
    kw = {}
    if not cfg.interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=4, num_stages=1)
    S, C = cfg.segments, cfg.C
    lane_specs = [pl.BlockSpec(fat.shape[:2] + (blk,), lambda i, s: (0, 0, i)),
                  pl.BlockSpec(flag.shape[:1] + (blk,), lambda i, s: (0, i)),
                  pl.BlockSpec(pair.shape[:1] + (blk,), lambda i, s: (0, i)),
                  pl.BlockSpec((1, blk), lambda i, s: (0, i))]
    out = dict(out_specs=[pl.BlockSpec((C, blk), lambda i, s: (s, i)), pl.BlockSpec((1, blk), lambda i, s: (s, i))],
               out_shape=[jax.ShapeDtypeStruct((S * C, Np), jnp.int32), jax.ShapeDtypeStruct((S, Np), jnp.int32)],
               interpret=cfg.interpret, name="box2d_find_new_contacts", **kw)
    if cull is None:
        lst, n = pl.pallas_call(
            partial(_kernel, cfg), grid=(Np // blk, S),
            in_specs=[full1(ra), full1(rb), full1(sbox), full1(ba), full1(bb)] + lane_specs,
            **out)(ra, rb, sbox, ba, bb, fat, flag, pair, en)
    else:
        cand, crow, cbox = cull
        lst, n = pl.pallas_call(
            partial(_kernel_cull, cfg), grid=(Np // blk, S),
            in_specs=[full1(cand), full1(ra), full1(rb), full1(sbox), full1(ba), full1(bb), full1(crow),
                      full1(cbox)] + lane_specs,
            **out)(cand, ra, rb, sbox, ba, bb, crow, cbox, fat, flag, pair, en)
    lst = from_minor(lst, N, order).reshape(N, S, C)
    n = from_minor(n, N, order)
    if S == 1:
        return lst[:, 0], n[:, 0]
    # Merge the segments in candidate order: segment s's q-th new pair has rank (pairs of segments < s) + q; the
    # first C ranks are kept (as one scan would), and the count is the total.
    off = jnp.cumsum(n, axis=1) - n
    rank = off[:, :, None] + jnp.arange(C, dtype=jnp.int32)[None, None, :]
    keep = (jnp.arange(C)[None, None, :] < n[:, :, None]) & (rank < C)
    rows = jnp.arange(N)[:, None, None]
    out = jnp.full((N, C + 1), -1, jnp.int32).at[rows, jnp.where(keep, rank, C)].set(lst)
    return out[:, :C], jnp.sum(n, axis=1)


@jax.custom_batching.custom_vmap
def _scan(cfg_box, ra, rb, sbox, ba, bb, cull, fat, flag, pair, en):
    lst, n = _call(cfg_box.cfg, ra, rb, sbox, ba, bb, fat[None], flag[None], pair[None], en[None], cull)
    return lst[0], n[0]


@_scan.def_vmap
def _scan_rule(axis_size, in_batched, cfg_box, ra, rb, sbox, ba, bb, cull, fat, flag, pair, en):
    if any(jax.tree.leaves(in_batched[1:7])):
        raise NotImplementedError("gpu find_new_contacts: candidate pairs must be shared by all worlds")
    args = [a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((fat, flag, pair, en),
                                                                                   in_batched[7:])]
    C = cfg_box.cfg.C
    none = lambda: (jnp.full((axis_size, C), -1, jnp.int32), jnp.zeros((axis_size,), jnp.int32))
    # a batch with no enabled world (the start-of-step call, normally) skips the kernel and its transposes
    out = jax.lax.cond(jnp.any(args[3] != 0), lambda: _call(cfg_box.cfg, ra, rb, sbox, ba, bb, *args, cull=cull),
                       none)
    return out, (True, True)


class _Static:
    def __init__(self, cfg):
        self.cfg = cfg


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.cfg), lambda cfg, _: _Static(cfg))


def find_new_contacts(model, state, enable=None, block=256, interpret=None):
    """pairs.find_new_contacts with the candidate scan as a kernel (see the module docstring)."""
    P = model.pair_a.shape[0]
    if P == 0:
        return state
    ct, bd, fx = state.contacts, state.bodies, state.fixtures
    C = ct.pair.shape[0]
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    Pp = -(-P // CHUNK) * CHUNK
    pad = lambda a: jnp.pad(a.astype(jnp.int32), (0, Pp - P))
    pa, pb = model.pair_a, model.pair_b
    ba, bb = model.fixture_body[pa], model.fixture_body[pb]
    flag = (bd.type == DYNAMIC).astype(jnp.int32) | (bd.active.astype(jnp.int32) << 1)
    en = jnp.asarray(True if enable is None else enable)
    cfg = Cfg(P=P, C=C, block=block, interpret=interpret, segments=max(1, min(SEGMENTS, -(-P // CHUNK))))
    compact = is_compact(model, fx)
    if compact and CULL and getattr(model, "scan_cand", None) is not None:
        # culled scan (Model.scan_*): the candidate tables in scan order, segments of the Model's tables
        from ..world import SCAN_CHUNK, SCAN_SEGMENTS
        sc = model.scan_cand
        k = jnp.maximum(sc, 0)
        ra, rb = model.fixture_slot[pa[k]], model.fixture_slot[pb[k]]
        sbox = jnp.where((ra < 0)[:, None], model.static_fat[pa[k]], model.static_fat[pb[k]])
        nch = model.scan_row.shape[0]
        cfg = Cfg(P=P, C=C, block=block, interpret=interpret, segments=SCAN_SEGMENTS, chunk=SCAN_CHUNK,
                  seg_chunks=nch // SCAN_SEGMENTS)
        lst, n_new = _scan(_Static(cfg), ra.astype(jnp.int32), rb.astype(jnp.int32), sbox,
                           ba[k].astype(jnp.int32), bb[k].astype(jnp.int32),
                           (sc.astype(jnp.int32), model.scan_row.astype(jnp.int32), model.scan_box),
                           fx.fat, flag, ct.pair, en.astype(jnp.int32))
    else:
        if compact:  # rows of the per-world table; a static side (row -1) reads the shared box
            ra, rb = model.fixture_slot[pa], model.fixture_slot[pb]
            sbox = jnp.where((ra < 0)[:, None], model.static_fat[pa], model.static_fat[pb])
        else:
            ra, rb, sbox = pa, pb, jnp.zeros((P, 4), jnp.float32)
        sbox = jnp.pad(sbox, ((0, Pp - P), (0, 0)))
        lst, n_new = _scan(_Static(cfg), pad(ra), pad(rb), sbox, pad(ba), pad(bb), None, fx.fat, flag, ct.pair,
                           en.astype(jnp.int32))

    # Placement (pairs.find_new_contacts on the compact list: rank r = list position).
    free = ct.pair < 0
    free_rank = jnp.cumsum(free) - 1
    n_free = jnp.sum(free)
    if SLOT_PLACE:
        return _place_by_slot(model, state, lst, n_new, free, free_rank, n_free)
    slot_of_rank = jnp.full(C, C, jnp.int32).at[jnp.where(free, free_rank, C)].set(jnp.arange(C, dtype=jnp.int32),
                                                                                mode="drop")
    rank = jnp.arange(C, dtype=jnp.int32)
    placed = (rank < n_new) & (rank < n_free)
    target = jnp.where(placed, slot_of_rank, C)
    cand = jnp.where(placed, lst, 0)
    fa, fb = pa[cand], pb[cand]
    fr_a, fr_b = fixture_friction(model, fx, fa), fixture_friction(model, fx, fb)
    rs_a, rs_b = fixture_restitution(model, fx, fa), fixture_restitution(model, fx, fb)
    fr = jnp.sqrt(m.mul(fr_a, fr_b))  # b2MixFriction
    rs = jnp.where(rs_a > rs_b, rs_a, rs_b)
    put = lambda arr, val: arr.at[target].set(val, mode="drop")
    zero2 = jnp.zeros((C, 2), F32)
    ct = ct._replace(
        pair=put(ct.pair, cand), fa=put(ct.fa, fa), fb=put(ct.fb, fb),
        key=put(ct.key, state.next_key + rank), touching=put(ct.touching, jnp.zeros(C, bool)),
        count=put(ct.count, jnp.zeros(C, jnp.int32)), normal_impulse=put(ct.normal_impulse, zero2),
        tangent_impulse=put(ct.tangent_impulse, zero2), point_id=put(ct.point_id, jnp.zeros((C, 2), jnp.int32)),
        friction=put(ct.friction, fr), restitution=put(ct.restitution, rs),
        tangent_speed=put(ct.tangent_speed, jnp.zeros(C, F32)))
    bodies = _wake(bd, placed, model.fixture_body[fa], model.fixture_body[fb])
    n_placed = jnp.minimum(n_new, n_free)
    return state._replace(contacts=ct, bodies=bodies, next_key=state.next_key + n_new.astype(jnp.int32),
                          overflow=state.overflow + (n_new - n_placed).astype(jnp.int32))


def _place_by_slot(model, state, lst, n_new, free, free_rank, n_free):
    """The placement above computed per slot (slot c, free, takes new pair number free_rank[c] if that is < n_new):
    a gather and selects instead of a dozen batched scatters; the same contacts, keys, wakes and counters."""
    ct, bd, fx = state.contacts, state.bodies, state.fixtures
    C = ct.pair.shape[0]
    pa, pb = model.pair_a, model.pair_b
    take = free & (free_rank < n_new)
    r = jnp.clip(free_rank, 0, C - 1).astype(jnp.int32)
    cand = jnp.where(take, lst[r], 0)
    fa, fb = pa[cand], pb[cand]
    fr_a, fr_b = fixture_friction(model, fx, fa), fixture_friction(model, fx, fb)
    rs_a, rs_b = fixture_restitution(model, fx, fa), fixture_restitution(model, fx, fb)
    fr = jnp.sqrt(m.mul(fr_a, fr_b))  # b2MixFriction
    rs = jnp.where(rs_a > rs_b, rs_a, rs_b)
    t2 = take[:, None]
    put = lambda old, val: jnp.where(take, val, old)  # noqa: E731
    ct = ct._replace(
        pair=put(ct.pair, cand), fa=put(ct.fa, fa), fb=put(ct.fb, fb), key=put(ct.key, state.next_key + r),
        touching=ct.touching & ~take, count=put(ct.count, 0),
        normal_impulse=jnp.where(t2, F32(0), ct.normal_impulse),
        tangent_impulse=jnp.where(t2, F32(0), ct.tangent_impulse), point_id=jnp.where(t2, 0, ct.point_id),
        friction=put(ct.friction, fr), restitution=put(ct.restitution, rs), tangent_speed=put(ct.tangent_speed, F32(0)))
    bodies = _wake(bd, take, model.fixture_body[fa], model.fixture_body[fb])
    n_placed = jnp.minimum(n_new, n_free)
    return state._replace(contacts=ct, bodies=bodies, next_key=state.next_key + n_new.astype(jnp.int32),
                          overflow=state.overflow + (n_new - n_placed).astype(jnp.int32))


# ---------------------------------------------------------------- b2Body::SynchronizeFixtures

def _sync_kernel(F, B, fb_ref, geo_ref, body_ref, fat_in, fat_ref):
    """For each fixture whose body moved in some lane of the block: the swept AABB of its shape between the two
    body transforms, re-fattened where it left the fat AABB (pairs.synchronize_fixtures, one fixture at a time)."""
    del fat_in  # aliased with fat_ref
    anyb = [jnp.max(body_ref[8, b, :]) > 0 for b in range(B)]
    inf = F32(np.inf)

    def aabb(f, px, py, c, s):
        r = geo_ref[f, 16]  # columns: 8 (x, y) vertices, radius, vertex count, shape type
        n = geo_ref[f, 17].astype(jnp.int32)
        circ = geo_ref[f, 18] == F32(shapes.CIRCLE)
        lx = ly = inf
        ux = uy = -inf
        for i in range(8):
            vx, vy = geo_ref[f, 2 * i], geo_ref[f, 2 * i + 1]
            wx, wy = m.xf_mul(px, py, c, s, vx, vy)
            ok = i < n
            lx = jnp.minimum(lx, jnp.where(ok, wx, inf))
            ly = jnp.minimum(ly, jnp.where(ok, wy, inf))
            ux = jnp.maximum(ux, jnp.where(ok, wx, -inf))
            uy = jnp.maximum(uy, jnp.where(ok, wy, -inf))
        rx, ry = m.rot(c, s, geo_ref[f, 0], geo_ref[f, 1])
        cx, cy = px + rx, py + ry
        return (jnp.where(circ, cx - r, lx - r), jnp.where(circ, cy - r, ly - r),
                jnp.where(circ, cx + r, ux + r), jnp.where(circ, cy + r, uy + r))

    def one(f, _):
        b = fb_ref[f]
        hit = anyb[0]
        for q in range(1, B):
            hit = jnp.where(b == q, anyb[q], hit)

        def upd(_):
            row = lambda q: body_ref[q, b, :]
            a1 = aabb(f, row(0), row(1), row(2), row(3))
            a2 = aabb(f, row(4), row(5), row(6), row(7))
            lx, ly = m.bmin(a1[0], a2[0]), m.bmin(a1[1], a2[1])
            ux, uy = m.bmax(a1[2], a2[2]), m.bmax(a1[3], a2[3])
            dx, dy = row(4) - row(0), row(5) - row(1)
            old = [fat_ref[f, q, :] for q in range(4)]
            contains = (old[0] <= lx) & (old[1] <= ly) & (ux <= old[2]) & (uy <= old[3])
            nfat = shapes.fatten(lx, ly, ux, uy, dx, dy)
            u = (row(8) != 0) & ~contains
            for q in range(4):
                fat_ref[f, q, :] = jnp.where(u, nfat[q], old[q])
            return 0
        return jax.lax.cond(hit, upd, lambda _: 0, 0)

    jax.lax.fori_loop(0, F, one, 0)


def _sync_call(cfg, fb, geo, body, fat):
    """Batched: body (N, 9, B), fat (N, F, 4) -> new fat (N, F, 4)."""
    N, F = fat.shape[0], fat.shape[1]
    B = body.shape[2]
    blk = cfg.block
    Np = -(-N // blk) * blk

    def minor(x):
        x = jnp.moveaxis(x, 0, -1)
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, Np - N)])

    body, fat = jax.lax.optimization_barrier((body, fat))  # see _call
    body, fat = minor(body), minor(fat)
    full = lambda a: pl.BlockSpec(a.shape, lambda i: (0,) * a.ndim)
    fspec = pl.BlockSpec((F, 4, blk), lambda i: (0, 0, i))
    kw = {}
    if not cfg.interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=4, num_stages=1)
    out = pl.pallas_call(
        partial(_sync_kernel, F, B), grid=(Np // blk,),
        in_specs=[full(fb), full(geo), pl.BlockSpec((9, B, blk), lambda i: (0, 0, i)), fspec],
        out_specs=fspec, out_shape=jax.ShapeDtypeStruct(fat.shape, jnp.float32),
        input_output_aliases={3: 0}, interpret=cfg.interpret, name="box2d_synchronize_fixtures", **kw)(
            fb, geo, body, fat)
    return jnp.moveaxis(out[..., :N], -1, 0)


@jax.custom_batching.custom_vmap
def _sync(cfg_box, fb, geo, body, fat):
    return _sync_call(cfg_box.cfg, fb, geo, body[None], fat[None])[0]


@_sync.def_vmap
def _sync_rule(axis_size, in_batched, cfg_box, fb, geo, body, fat):
    if in_batched[1] or in_batched[2]:
        raise NotImplementedError("gpu synchronize_fixtures: fixture geometry must be shared by all worlds")
    body, fat = (a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((body, fat), in_batched[3:]))
    return _sync_call(cfg_box.cfg, fb, geo, body, fat), True


def synchronize_fixtures(model, state, moved, block=128, interpret=None, rot0=None):
    """pairs.synchronize_fixtures with the per-fixture work in a kernel that skips fixtures of bodies no world of
    the block moved (the level's ~1500 static fixtures): no (worlds, fixtures, vertices) temporaries. Compact
    layout (world.compact_model): only State.fixtures' rows (the non-static fixtures) are walked.
    rot0: m.rot_of(bodies.a0) where `moved` (other bodies' rows are computed but never kept), if the caller has it."""
    bd, fx = state.bodies, state.fixtures
    F = fx.fat.shape[0]
    if F == 0:
        return state
    rows = fixture_rows(model, fx)
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    c1, s1 = m.rot_of(bd.a0) if rot0 is None else rot0
    lcx, lcy = bd.local_center[:, 0], bd.local_center[:, 1]
    rx, ry = m.rot(c1, s1, lcx, lcy)
    p1x, p1y = bd.c0[:, 0] - rx, bd.c0[:, 1] - ry
    body = jnp.stack([p1x, p1y, c1, s1, bd.xf_p[:, 0], bd.xf_p[:, 1], bd.xf_q[:, 0], bd.xf_q[:, 1],
                      moved.astype(jnp.float32)])
    geo = jnp.concatenate([model.verts[rows].reshape(F, 16), model.radius[rows][:, None],
                           model.count[rows][:, None].astype(jnp.float32),
                           model.shape_type[rows][:, None].astype(jnp.float32)], 1)
    cfg = Cfg(P=0, C=0, block=block, interpret=interpret)
    fat = _sync(_Static(cfg), model.fixture_body[rows].astype(jnp.int32), geo, body, fx.fat)
    return state._replace(fixtures=fx._replace(fat=fat))
