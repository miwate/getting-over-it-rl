"""The island solver of b2World::Solve as one Pallas kernel, one world per lane.

What runs inside (b2Island::Solve after the velocity integration and the constraint set-up, which stay in XLA):
contact warm start, joint warm start, `velocity_iterations` passes (all joints, then all contacts, in island solve
order), position integration with the per-step clamps, and the position passes with each island's early exit.

Layout: structure of arrays with worlds as the minor axis. A table of R rows (contacts or joints in solve order)
is a (R, fields, N) float32 array; a lane (one world) reads row k of its own world, so the per-world solve order
(island.py) is kept exactly while every lane walks the same row index k. Integer and boolean fields travel as
float32 values (all < 2**24). Body state (vx, vy, w, cx, cy, a per body) lives in the kernel's output ref; a
constraint reads and writes its two bodies with a per-lane gather / scatter (RefBodies; math2d.pick is redirected
to it while the kernel is traced). Holding the bodies in registers as select chains over B instead spilled
massively (B = 16 in the game). The impulse accumulators also live in the output refs.

The arithmetic is not re-implemented: the kernel calls contact_solver._solve_one / solve_position_one and the
joints module's per-type velocity and position functions on (block,) vectors, so it performs the same float32
operations in the same order as the XLA step. Two things differ on the Triton backend only (interpret mode runs
the unmodified code):
- math2d.sin_cos (float-float series, correctly rounded) becomes (float)sin((double)x) on the GPU's f64 units
  (_sin_cos_f64, a compact double-precision kernel): the same value except with probability ~2**-25 per call, and
  it compiles in seconds instead of ~minutes per instance (the position pass has ~10 instances; with float-float
  the kernel took >10 min to compile).
- jnp.stack along a leading axis is not supported; contact_solver's two-point stacks become tuples.
FMA contraction: Triton on this ROCm build does not contract a * b + c (measured: raw x * y + z in a kernel matches
the separately rounded result on 100% of samples), and math2d.mul's select keeps it from fusing anyway.

`island_solve(cf, jf, wf, cimp, jimp, cfg)` is a custom_vmap function: called once per world (inside
`jax.vmap(step)`) its vmap rule receives the whole batch and runs one pallas_call over it.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser as _fuser  # noqa: F401 (defines a jit-key config at import; importing it
#   here, not on the first pallas_call, keeps step jits traced before and after that from missing the cache)

from .. import contact_solver as cs
from .. import joints as jn
from .. import math2d as m
from . import speed_option

F32 = np.float32
# Group worlds by their loop counts before each kernel call (lane_order; island solver, TOI search and TOI event).
# Same results bit for bit. Default off; GOI_GPU_SPEED=sort (or all) sets it (read at trace time).
SORT_LANES = speed_option("sort")
# The island solver's joint rows: only the block's joint type computed when every live lane has the same one (else
# all three, selected per lane). Same results. Default off; GOI_GPU_SPEED=joints (or all) sets it.
BRANCH_SPLIT = speed_option("joints")


# ---------------------------------------------------------------- row tables

@dataclass(frozen=True)
class Layout:
    """Static description of a packed row table: (name, offset, width, kind, vector) per field.
    kind: 'f' float32, 'i' int32, 'b' bool; vector: the per-row value is a tuple of `width` values."""
    fields: tuple
    width: int


def layout_of(rows, skip=()):
    """The Layout for a dict of per-row arrays (R,) or (R, w)."""
    out, off = [], 0
    for name in sorted(rows):
        if name in skip:
            continue
        a = rows[name]
        w = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
        kind = "b" if a.dtype == jnp.bool_ else ("i" if jnp.issubdtype(a.dtype, jnp.integer) else "f")
        out.append((name, off, w, kind, a.ndim > 1))
        off += w
    return Layout(tuple(out), off)


def pack(rows, layout):
    """(R, layout.width) float32 for one world."""
    cols = []
    for name, _, w, kind, _ in layout.fields:
        a = rows[name]
        a = a.reshape(a.shape[0], w)
        cols.append(a.astype(jnp.float32))
    return jnp.concatenate(cols, axis=1)


def _cast(v, kind):
    if kind == "f":
        return v
    if kind == "b":
        return v != 0
    return v.astype(jnp.int32)


def load_row(ref, k, layout, extra=None):
    """Row k of a packed table ref (R, fields, block) as a dict of (block,) vectors / tuples of them."""
    out = dict(extra or {})
    for name, off, w, kind, vec in layout.fields:
        vals = tuple(_cast(ref[k, off + q, :], kind) for q in range(w))
        out[name] = vals if vec else vals[0]
    return out


# ---------------------------------------------------------------- Triton-only substitutions

# fdlibm's pi/2 split (33-bit parts) and __kernel_sin / __kernel_cos coefficients (|r| <= pi/4, < 1 ulp in double)
_PIO2 = (1.57079632673412561417e+00, 6.07710050630396597660e-11, 2.02226624871116645580e-21)
_S = (-1.66666666666666324348e-01, 8.33333333332248946124e-03, -1.98412698298579493134e-04,
      2.75573137070700676789e-06, -2.50507602534068634195e-08, 1.58969099521155010221e-10)
_C = (4.16666666666666019037e-02, -1.38888888888741095749e-03, 2.48015872894767294178e-05,
      -2.75573143513906633035e-07, 2.08757232129817482790e-09, -1.13596475577881948265e-11)


def _sin_cos_f64(x):
    """(float)sin((double)x), (float)cos((double)x): the value math2d.sin_cos computes with float-float arithmetic,
    here evaluated in double precision with a compact branch-free fdlibm-style kernel (Cody-Waite reduction by pi/2
    in three parts, degree-13/12 polynomials), then rounded to float32. The float-float version costs minutes of
    Triton compile per instance and the GPU's libdevice double sin/cos (inlined, with its large-argument path) most
    of the TOI kernels' compile time; this one is ~40 double operations. It can differ from the correctly rounded
    float32 only when the double result lies within ~1 double ulp of a float32 rounding midpoint: 4 of 2e8 sin/cos
    values (|x| up to 1e4) in runs/toi_kernel/sincos_check.py."""
    with _x64():
        d = jnp.asarray(x).astype(jnp.float64)
        k = jnp.floor(d * 6.36619772367581382433e-01 + 0.5)
        r = ((d - k * _PIO2[0]) - k * _PIO2[1]) - k * _PIO2[2]
        z = r * r
        ps = jnp.full_like(z, _S[5])
        for c in _S[4::-1]:
            ps = c + z * ps
        sn = r + (z * r) * ps
        pc = jnp.full_like(z, _C[5])
        for c in _C[4::-1]:
            pc = c + z * pc
        hz = 0.5 * z
        w = 1.0 - hz
        cs = w + (((1.0 - w) - hz) + (z * z) * pc)
        q = k.astype(jnp.int32) & 3
        sin_x = jnp.where(q == 0, sn, jnp.where(q == 1, cs, jnp.where(q == 2, -sn, -cs)))
        cos_x = jnp.where(q == 0, cs, jnp.where(q == 1, -sn, jnp.where(q == 2, -cs, sn)))
        sin_x = jnp.where(d == 0, d, sin_x)
        return sin_x.astype(jnp.float32), cos_x.astype(jnp.float32)


def _x64():
    try:
        from jax.experimental import enable_x64
    except ImportError:  # newer jax
        enable_x64 = jax.enable_x64
    return enable_x64(True)


class _NumpyProxy:
    """jax.numpy with stack along a leading axis (unsupported by the Triton lowering) returning a tuple."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, k):
        return getattr(self._base, k)

    @staticmethod
    def stack(xs, axis=0):
        assert axis == 0
        return tuple(xs)


@contextmanager
def triton_mode(enabled):
    """While tracing the kernel for Triton: math2d.sin_cos in double precision, contact_solver's stacks as tuples."""
    if not enabled:
        yield
        return
    saved = (m.sin_cos, cs.jnp)
    m.sin_cos, cs.jnp = _sin_cos_f64, _NumpyProxy(jnp)
    try:
        yield
    finally:
        m.sin_cos, cs.jnp = saved


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class Config:
    settings: object  # box2d Settings (hashable)
    clay: Layout  # contact rows
    jlay: Layout  # joint rows
    B: int
    C: int
    J: int
    W: int  # joint impulse columns: 5 (revolute v2.4: x, y, z, lower, upper) or 3
    block: int = 256
    num_warps: int = 4
    interpret: bool = False
    parts: tuple = ("warm", "jvel", "cvel", "pos")  # kernel sections (gpu/toi.py runs subsets)
    okay_factor: F32 = F32(-3)  # an island is solved at min separation >= okay_factor * linear_slop


def world_width(B):
    return 7 * B + 6  # vx, vy, w, cx, cy, a, in_isl per body; n_c, n_j, n_isl, h, max_t, max_r


# ---------------------------------------------------------------- the kernel

def _joint_warm(r, v, imp, mi, v24):
    """jn.warm_start's per-type velocity updates, selected by branch (vmap of its lax.switch)."""
    (vA, wA, vB, wB) = v
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    ix, iy, iz = imp[0], imp[1], imp[2]

    def lin(Px, Py):
        return (vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py)), (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py))
    # revolute
    Px, Py = ix, iy
    rvA, rvB = lin(Px, Py)
    if v24:
        axial = (mi + imp[3]) - imp[4]
        rwA = wA - m.mul(iA, m.cross(r["rAx"], r["rAy"], Px, Py) + axial)
        rwB = wB + m.mul(iB, m.cross(r["rBx"], r["rBy"], Px, Py) + axial)
    else:
        rwA = wA - m.mul(iA, (m.cross(r["rAx"], r["rAy"], Px, Py) + mi) + iz)
        rwB = wB + m.mul(iB, (m.cross(r["rBx"], r["rBy"], Px, Py) + mi) + iz)
    # prismatic
    mz = mi + iz
    Px = m.mul(ix, r["pex"]) + m.mul(mz, r["axx"])
    Py = m.mul(ix, r["pey"]) + m.mul(mz, r["axy"])
    LA = (m.mul(ix, r["s1"]) + iy) + m.mul(mz, r["a1"])
    LB = (m.mul(ix, r["s2"]) + iy) + m.mul(mz, r["a2"])
    pvA, pvB = lin(Px, Py)
    pwA, pwB = wA - m.mul(iA, LA), wB + m.mul(iB, LB)
    # weld
    Px, Py = ix, iy
    wvA, wvB = lin(Px, Py)
    wwA = wA - m.mul(iA, m.cross(r["rAx"], r["rAy"], Px, Py) + iz)
    wwB = wB + m.mul(iB, m.cross(r["rBx"], r["rBy"], Px, Py) + iz)
    br = r["branch"]
    sel = lambda a, b, c: jnp.where(br == 0, a, jnp.where(br == 1, b, c))
    return ((sel(rvA[0], pvA[0], wvA[0]), sel(rvA[1], pvA[1], wvA[1])), sel(rwA, pwA, wwA),
            (sel(rvB[0], pvB[0], wvB[0]), sel(rvB[1], pvB[1], wvB[1])), sel(rwB, pwB, wwB))


def _contact_warm(c, v, ni, ti):
    """contact_solver.warm_start's per-contact update."""
    (vAx, vAy), wA, (vBx, vBy), wB = v
    nx, ny = c["nx"], c["ny"]
    tx, ty = m.cross_vs(nx, ny, F32(1.0))
    for j in range(2):
        on = j < c["count"]
        Px = m.mul(ni[j], nx) + m.mul(ti[j], tx)
        Py = m.mul(ni[j], ny) + m.mul(ti[j], ty)
        wA = jnp.where(on, wA - m.mul(c["iA"], m.cross(c["rAx"][j], c["rAy"][j], Px, Py)), wA)
        vAx = jnp.where(on, vAx - m.mul(c["mA"], Px), vAx)
        vAy = jnp.where(on, vAy - m.mul(c["mA"], Py), vAy)
        wB = jnp.where(on, wB + m.mul(c["iB"], m.cross(c["rBx"][j], c["rBy"][j], Px, Py)), wB)
        vBx = jnp.where(on, vBx + m.mul(c["mB"], Px), vBx)
        vBy = jnp.where(on, vBy + m.mul(c["mB"], Py), vBy)
    return ((vAx, vAy), wA), ((vBx, vBy), wB)


class RefBodies:
    """One body quantity (rows row0 .. row0 + B - 1 of a (rows, block) ref): math2d.pick on it is a per-lane gather
    (each lane reads its own world's body i), put a per-lane scatter. Replaces the select chain over B bodies."""

    def __init__(self, ref, row0, B, lane):
        self.ref, self.row0, self.B, self.lane = ref, row0, B, lane

    def get(self, i):
        return self.ref[self.row0 + jnp.clip(i, 0, self.B - 1), self.lane]

    def set(self, i, v):
        self.ref[self.row0 + jnp.clip(i, 0, self.B - 1), self.lane] = v


@contextmanager
def ref_pick():
    """math2d.pick reads a RefBodies by gather (other arguments as before) while tracing the kernel."""
    orig = m.pick
    m.pick = lambda arr, i: arr.get(i) if isinstance(arr, RefBodies) else orig(arr, i)
    try:
        yield
    finally:
        m.pick = orig


def _get2(q3, i):
    return (q3[0].get(i), q3[1].get(i)), q3[2].get(i)


def _set2(q3, ia, ib, A, Bv, live):
    """Store A then B (Box2D's order), unchanged where not live (the values were read at ia / ib)."""
    (ax, ay), aa = A
    (bx, by), ba = Bv
    for q, va in zip(q3, (ax, ay, aa)):
        q.set(ia, va)
    for q, vb in zip(q3, (bx, by, ba)):
        q.set(ib, vb)


def _masked(new, old, live):
    """where(live, new, old) over the ((x, y), angle) pairs of _get2."""
    (nx, ny), na = new
    (ox, oy), oa = old
    return (jnp.where(live, nx, ox), jnp.where(live, ny, oy)), jnp.where(live, na, oa)


def make_kernel(cfg):
    s = cfg.settings
    B, W = cfg.B, cfg.W
    v24 = W == 5
    if v24:
        jvel = (jn._rev24_velocity, jn._pass_limits(jn._pri_velocity), jn._pass_limits(jn._weld_velocity))
    else:
        jvel = (jn._rev_velocity, jn._pri_velocity, jn._weld_velocity)
    jpos = (jn._rev24_position if s.revolute_v24 else jn._rev_position, jn._pri_position, jn._weld_position)

    def kernel(cf_ref, jf_ref, wf_ref, ci_ref, ji_ref, wo_ref, co_ref, jo_ref, sc_ref):
        blk = cfg.block
        lane = jax.lax.broadcasted_iota(jnp.int32, (blk,), 0)
        rd = lambda q: wf_ref[q, :]
        # body state lives in the output ref: rows f * B + b for f = vx, vy, w, cx, cy, a; 6B + i: island i done
        for q in range(6 * B):
            wo_ref[q, :] = wf_ref[q, :]
        vel = tuple(RefBodies(wo_ref, f * B, B, lane) for f in range(3))
        pos = tuple(RefBodies(wo_ref, (3 + f) * B, B, lane) for f in range(3))
        done = RefBodies(wo_ref, 6 * B, B, lane)
        min_sep = RefBodies(sc_ref, 0, B, lane)
        jok = RefBodies(sc_ref, B, B, lane)
        n_c = rd(7 * B).astype(jnp.int32)
        n_j = rd(7 * B + 1).astype(jnp.int32)
        n_isl = rd(7 * B + 2).astype(jnp.int32)
        h, max_t, max_r = rd(7 * B + 3), rd(7 * B + 4), rd(7 * B + 5)
        nc_max = jnp.max(n_c)
        nj_max = jnp.max(n_j)
        slop3 = m.mul(cfg.okay_factor, s.linear_slop)

        # impulse accumulators: the output refs, initialised from the warm-start values
        def copy_c(k, _):
            for q in range(4):
                co_ref[k, q, :] = ci_ref[k, q, :]
            return 0
        jax.lax.fori_loop(0, cfg.C, copy_c, 0)

        def copy_j(k, _):
            for q in range(W + 1):
                jo_ref[k, q, :] = ji_ref[k, q, :]
            return 0
        jax.lax.fori_loop(0, cfg.J, copy_j, 0)

        crow = lambda k: load_row(cf_ref, k, cfg.clay)
        jrow = lambda k: load_row(jf_ref, k, cfg.jlay)

        def sel3(br, outs):
            sel = lambda a, b, c: jnp.where(br == 0, a, jnp.where(br == 1, b, c))
            flat = [jax.tree.leaves(o) for o in outs]
            return jax.tree.unflatten(jax.tree.structure(outs[0]), [sel(a, b, c) for a, b, c in zip(*flat)])

        def by_branch(br, live, fns):
            """sel3 over the three joint types' results (fns[t]() loads its row and solves it as type t); with
            BRANCH_SPLIT, when every live lane of the block has the same type (the game: one joint-type sequence
            in every world) only that type is computed. Lanes that are not live are masked by the caller."""
            if not BRANCH_SPLIT:
                return sel3(br, [f() for f in fns])
            lo = jnp.min(jnp.where(live, br, 99))
            hi = jnp.max(jnp.where(live, br, -1))
            # every branch of a cond must give the same types: (block,) float32 (bool kept)
            norm = lambda o: jax.tree.map(  # noqa: E731
                lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.bool_ if jnp.asarray(x).dtype == jnp.bool_
                                                       else jnp.float32), br.shape), o)
            fn = [partial(lambda f: norm(f()), f) for f in fns]

            def one():
                return jax.lax.cond(hi == 0, fn[0], lambda: jax.lax.cond(hi == 1, fn[1], fn[2]))
            return jax.lax.cond(lo >= hi, one, lambda: norm(sel3(br, [f() for f in fns])))

        # ---- warm starting: contacts, then joints (b2Island::Solve order)
        def cwarm(k, _):
            c = crow(k)
            live = k < n_c
            ni = (co_ref[k, 0, :], co_ref[k, 1, :])
            ti = (co_ref[k, 2, :], co_ref[k, 3, :])
            A0, B0 = _get2(vel, c["ia"]), _get2(vel, c["ib"])
            A, Bv = _contact_warm(c, (A0[0], A0[1], B0[0], B0[1]), ni, ti)
            _set2(vel, c["ia"], c["ib"], _masked(A, A0, live), _masked(Bv, B0, live), live)
            return 0

        def jwarm(k, _):
            r = jrow(k)
            live = k < n_j
            imp = tuple(jo_ref[k, q, :] for q in range(W))
            mi = jo_ref[k, W, :]
            A0, B0 = _get2(vel, r["ia"]), _get2(vel, r["ib"])
            vA, wA, vB, wB = _joint_warm(r, (A0[0], A0[1], B0[0], B0[1]), imp, mi, v24)
            _set2(vel, r["ia"], r["ib"], _masked((vA, wA), A0, live), _masked((vB, wB), B0, live), live)
            return 0

        if s.warm_starting and "warm" in cfg.parts:
            jax.lax.fori_loop(0, nc_max, cwarm, 0)
        if "warm" in cfg.parts:
            jax.lax.fori_loop(0, nj_max, jwarm, 0)

        # ---- velocity iterations
        def jvel_one(k, _):
            r = jrow(k)
            live = k < n_j
            imp = tuple(jo_ref[k, q, :] for q in range(W))
            mi = jo_ref[k, W, :]
            A0, B0 = _get2(vel, r["ia"]), _get2(vel, r["ib"])
            fns = [partial(lambda f: f(jrow(k), A0[0], A0[1], B0[0], B0[1], imp, mi), f) for f in jvel]
            nvA, nwA, nvB, nwB, nimp, nmi = by_branch(r["branch"], live, fns)
            for q in range(W):
                jo_ref[k, q, :] = jnp.where(live, nimp[q], imp[q])
            jo_ref[k, W, :] = jnp.where(live, nmi, mi)
            _set2(vel, r["ia"], r["ib"], _masked((nvA, nwA), A0, live), _masked((nvB, nwB), B0, live), live)
            return 0

        def cvel_one(k, _):
            c = crow(k)
            live = k < n_c
            ni = (co_ref[k, 0, :], co_ref[k, 1, :])
            ti = (co_ref[k, 2, :], co_ref[k, 3, :])
            A0, B0 = _get2(vel, c["ia"]), _get2(vel, c["ib"])
            (vA, wA, vB, wB), nik, tik = cs._solve_one(c, vel, ni, ti)
            co_ref[k, 0, :] = jnp.where(live, nik[0], ni[0])
            co_ref[k, 1, :] = jnp.where(live, nik[1], ni[1])
            co_ref[k, 2, :] = jnp.where(live, tik[0], ti[0])
            co_ref[k, 3, :] = jnp.where(live, tik[1], ti[1])
            _set2(vel, c["ia"], c["ib"], _masked((vA, wA), A0, live), _masked((vB, wB), B0, live), live)
            return 0

        def vel_iter(_, x):
            if "jvel" in cfg.parts:
                jax.lax.fori_loop(0, nj_max, jvel_one, 0)
            if "cvel" in cfg.parts:
                jax.lax.fori_loop(0, nc_max, cvel_one, 0)
            return x
        jax.lax.fori_loop(0, s.velocity_iterations, vel_iter, 0)

        # ---- integrate positions (island bodies) with the per-step clamps
        for b in range(B):
            q = lambda f: f * B + b
            vx0, vy0, w0 = wo_ref[q(0), :], wo_ref[q(1), :], wo_ref[q(2), :]
            cx, cy, a = wo_ref[q(3), :], wo_ref[q(4), :], wo_ref[q(5), :]
            vx, vy, w = vx0, vy0, w0
            tx, ty = m.mul(h, vx), m.mul(h, vy)
            big_t = m.dot(tx, ty, tx, ty) > m.mul(max_t, max_t)
            ratio = max_t / jnp.where(big_t, m.length(tx, ty), F32(1))
            vx = jnp.where(big_t, m.mul(vx, ratio), vx)
            vy = jnp.where(big_t, m.mul(vy, ratio), vy)
            rot = m.mul(h, w)
            big_r = m.mul(rot, rot) > m.mul(max_r, max_r)
            rr = max_r / jnp.where(big_r, m.babs(rot), F32(1))
            w = jnp.where(big_r, m.mul(w, rr), w)
            ncx, ncy, na = cx + m.mul(h, vx), cy + m.mul(h, vy), a + m.mul(h, w)
            on = rd(6 * B + b) != 0
            wo_ref[q(3), :] = jnp.where(on, ncx, cx)
            wo_ref[q(4), :] = jnp.where(on, ncy, cy)
            wo_ref[q(5), :] = jnp.where(on, na, a)
            wo_ref[q(0), :] = jnp.where(on, vx, vx0)
            wo_ref[q(1), :] = jnp.where(on, vy, vy0)
            wo_ref[q(2), :] = jnp.where(on, w, w0)

        # ---- position iterations, per-island early exit (step._solve_positions); done as 0/1 floats
        for b in range(B):
            wo_ref[6 * B + b, :] = (b >= n_isl).astype(jnp.float32)

        def cpos(k, _):
            c = crow(k)
            i = c["isl"]
            live = (k < n_c) & (done.get(i) == 0)
            A0, B0 = _get2(pos, c["ia"]), _get2(pos, c["ib"])
            cA, aA, cB, aB, ms = cs.solve_position_one(c, pos, s, s.baumgarte)
            _set2(pos, c["ia"], c["ib"], _masked((cA, aA), A0, live), _masked((cB, aB), B0, live), live)
            old = min_sep.get(i)
            min_sep.set(i, jnp.where(live, m.bmin(old, ms), old))
            return 0

        def jpos_one(k, _):
            r = jrow(k)
            i = r["isl"]
            live = (k < n_j) & (done.get(i) == 0)
            A0, B0 = _get2(pos, r["ia"]), _get2(pos, r["ib"])
            fns = [partial(lambda f: f(jrow(k), A0[0], A0[1], B0[0], B0[1], s), f) for f in jpos]
            ncA, naA, ncB, naB, ok = by_branch(r["branch"], live, fns)
            _set2(pos, r["ia"], r["ib"], _masked((ncA, naA), A0, live), _masked((ncB, naB), B0, live), live)
            old = jok.get(i)
            jok.set(i, jnp.where(live & ~ok, F32(0), old))
            return 0

        def cond(it):
            notdone = jnp.zeros_like(n_c)
            for b in range(B):
                notdone = notdone | (wo_ref[6 * B + b, :] == 0).astype(jnp.int32)
            return (it < s.position_iterations) & (jnp.max(notdone) > 0)

        def body(it):
            for b in range(B):
                sc_ref[b, :] = jnp.zeros_like(h)
                sc_ref[B + b, :] = jnp.ones_like(h)
            if "nocpos" not in cfg.parts:
                jax.lax.fori_loop(0, nc_max, cpos, 0)
            if "nojpos" not in cfg.parts:
                jax.lax.fori_loop(0, nj_max, jpos_one, 0)
            for b in range(B):
                okay = (sc_ref[b, :] >= slop3) & (sc_ref[B + b, :] != 0)
                d = wo_ref[6 * B + b, :]
                wo_ref[6 * B + b, :] = jnp.where(okay, F32(1), d)
            return it + 1

        if "pos" in cfg.parts:
            jax.lax.while_loop(cond, body, jnp.int32(0))

    return kernel


def lane_order(key, N, blk):
    """SORT_LANES: the world order that puts worlds with similar loop counts (key) into the same kernel program
    (a block walks its constraint rows up to the block's largest count; every world is independent, so which
    lanes share a program changes no bit). Returns (order, inverse) or None when off or a single program."""
    if not SORT_LANES or N <= blk:
        return None
    order = jnp.argsort(key, stable=True).astype(jnp.int32)
    inv = jnp.zeros((N,), jnp.int32).at[order].set(jnp.arange(N, dtype=jnp.int32))
    return order, inv


def to_minor(x, N, Np, order=None):
    """(N, ...) world-major -> (..., Np) worlds minor, padded, in `order` (lane_order) if given."""
    if order is not None:
        x = x[order[0]]
    x = jnp.moveaxis(x, 0, -1)
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, Np - N)])


def from_minor(x, N, order=None):
    """to_minor's inverse."""
    x = jnp.moveaxis(x[..., :N], -1, 0)
    return x if order is None else x[order[1]]


def _call(cfg, cf, jf, wf, ci, ji):
    """Batched pallas_call: inputs with the world axis first, as built by vmap; returns the same layout."""
    N = wf.shape[0]
    blk = cfg.block
    Np = -(-N // blk) * blk
    nb = Np // blk

    cf, jf, wf, ci, ji = jax.lax.optimization_barrier((cf, jf, wf, ci, ji))  # see pairs._call
    # sort key: the contact rows a world walks (n_contacts, row 7B of wf), then its joint rows
    order = lane_order(wf[:, 7 * cfg.B] * 4096 + wf[:, 7 * cfg.B + 1], N, blk)
    cf, jf, wf, ci, ji = (to_minor(x, N, Np, order) for x in (cf, jf, wf, ci, ji))
    spec3 = lambda a: pl.BlockSpec(a.shape[:2] + (blk,), lambda i: (0, 0, i))
    spec2 = lambda a: pl.BlockSpec(a.shape[:1] + (blk,), lambda i: (0, i))
    wout = jax.ShapeDtypeStruct((7 * cfg.B, Np), jnp.float32)
    kw = {}
    if not cfg.interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=cfg.num_warps, num_stages=1)
    scratch = jax.ShapeDtypeStruct((2 * cfg.B, Np), jnp.float32)  # per island: min separation, joints ok
    with triton_mode(not cfg.interpret), ref_pick():
        wo, co, jo, _ = pl.pallas_call(
            make_kernel(cfg), grid=(nb,),
            in_specs=[spec3(cf), spec3(jf), spec2(wf), spec3(ci), spec3(ji)],
            out_specs=[pl.BlockSpec((7 * cfg.B, blk), lambda i: (0, i)), spec3(ci), spec3(ji),
                       pl.BlockSpec((2 * cfg.B, blk), lambda i: (0, i))],
            out_shape=[wout, jax.ShapeDtypeStruct(ci.shape, jnp.float32), jax.ShapeDtypeStruct(ji.shape, jnp.float32),
                       scratch],
            interpret=cfg.interpret, name="box2d_island_solve", **kw)(cf, jf, wf, ci, ji)
    return tuple(from_minor(x, N, order) for x in (wo, co, jo))


@partial(jax.custom_batching.custom_vmap)
def _island_solve(cfg_box, cf, jf, wf, ci, ji):
    wo, co, jo = _call(cfg_box.cfg, cf[None], jf[None], wf[None], ci[None], ji[None])
    return wo[0], co[0], jo[0]


@_island_solve.def_vmap
def _island_solve_rule(axis_size, in_batched, cfg_box, cf, jf, wf, ci, ji):
    args = [a if b else jnp.broadcast_to(a, (axis_size,) + a.shape)
            for a, b in zip((cf, jf, wf, ci, ji), in_batched[1:])]
    out = _call(cfg_box.cfg, *args)
    return out, (True, True, True)


class _Static:
    """Wraps the static Config as a pytree with no leaves (custom_vmap arguments must be pytrees)."""

    def __init__(self, cfg):
        self.cfg = cfg


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.cfg), lambda cfg, _: _Static(cfg))


def island_solve(cfg, cf, jf, wf, ci, ji):
    """One world's packed tables -> (world out (7B,), contact impulses (C, 4), joint impulses (J, W + 1))."""
    return _island_solve(_Static(cfg), cf, jf, wf, ci, ji)
