"""b2World::SolveTOI's min-TOI search (toi._search) as one Pallas kernel, one world per lane.

toi._search in XLA is launch bound on a GPU: the sequential sweep alignment is a loop over list positions, and
b2TimeOfImpact is a flat while loop over every contact lane of the batch with b2Distance's while loop nested in it,
so one search costs thousands of small kernels whatever the batch size. Here one pallas_call does all of it per world:

1. the alignment pass (b2Sweep::Advance of the body that is behind), newest contact first, recording the sweeps each
   contact saw and queueing the contacts that need a TOI (toi._search's `need`);
2. b2TimeOfImpact for the queued contacts, each lane working down its own queue (a lane that finishes a contact
   starts its next one in the same loop iteration), with toi.time_of_impact's flat phase machine (DIST: b2Distance
   and a new separating axis; FIND: the deepest points; ROOT: one root finder step) and b2Distance's loop nested;
3. alpha, the contacts' cached TOI and flag, and the minimum (ties: first in list order, or the last with
   Settings.toi_tie_oldest).

Same floats as toi._search (the same math2d operations in the same order; interpret mode on CPU is bit-identical,
tests/box2d/test_gpu.py); sin/cos in double precision on the Triton backend as in kernel.py. Per-contact convex
proxies are gathered per world in XLA before the call (rows of the lane-local tables the kernel reads by index).
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from .. import math2d as m
from .. import toi as T
from ..settings import EPSILON, MAX_POLYGON_VERTICES
from . import kernel as K
from . import speed_option

F32 = np.float32
I32 = jnp.int32
NV = MAX_POLYGON_VERTICES
PROXY = 2 * NV + 2  # rows per proxy: x[8], y[8], count, radius
_REC = 7  # recorded per contact: c0xA, c0yA, a0A, c0xB, c0yB, a0B, alpha0
EPS2 = T.EPS2
# The contacts' convex proxies read from one shared per-fixture table in the kernel instead of a per-world copy
# gathered in XLA (C x 2 x 18 floats per world and call). Same values. GOI_GPU_SPEED=proxy (or all) sets it.
SHARED_PROXIES = speed_option("proxy")


@dataclass(frozen=True)
class Cfg:
    B: int
    C: int
    linear_slop: F32
    tie_oldest: bool
    block: int
    num_warps: int
    interpret: bool
    shared_px: int = 0  # > 0: px is the shared per-fixture proxy table of that many fixtures (SHARED_PROXIES)


class Proxy:
    """A b2DistanceProxy held in lane-local rows base .. base + PROXY - 1 of a (rows, block) ref."""

    def __init__(self, ref, base, lane):
        self.ref, self.base, self.lane = ref, base, lane

    def x(self, i):
        return self.ref[self.base + i, :]

    def y(self, i):
        return self.ref[self.base + NV + i, :]

    def count(self):
        return self.ref[self.base + 2 * NV, :].astype(I32)

    def radius(self):
        return self.ref[self.base + 2 * NV + 1, :]

    def vertex(self, i):
        """GetVertex(i) for a per-lane index (clipped as toi._vertex does)."""
        i = jnp.clip(i, 0, NV - 1)
        return self.ref[self.base + i, self.lane], self.ref[self.base + NV + i, self.lane]

    def support(self, dx, dy):
        """GetSupport: first index of the largest b2Dot(v_i, d) over the proxy's vertices."""
        cnt = self.count()
        best_i = jnp.zeros_like(cnt)
        best_v = m.mul(self.x(0), dx) + m.mul(self.y(0), dy)
        best_v = jnp.where(0 < cnt, best_v, -jnp.inf)
        for i in range(1, NV):
            v = m.mul(self.x(i), dx) + m.mul(self.y(i), dy)
            v = jnp.where(i < cnt, v, -jnp.inf)
            take = v > best_v
            best_i = jnp.where(take, i, best_i)
            best_v = jnp.where(take, v, best_v)
        return best_i


# ---------------------------------------------------------------- b2Distance (toi.distance), one lane per contact

_SF = ("wAx", "wAy", "wBx", "wBy", "wx", "wy", "a", "iA", "iB")


def _make_vertex(pA, xfA, iA, pB, xfB, iB):
    wAx, wAy = m.xf_mul(*xfA, *pA.vertex(iA))
    wBx, wBy = m.xf_mul(*xfB, *pB.vertex(iB))
    return dict(wAx=wAx, wAy=wAy, wBx=wBx, wBy=wBy, wx=wBx - wAx, wy=wBy - wAy, a=jnp.zeros_like(wAx), iA=iA, iB=iB)


def _col(s, j):
    return {k: s[k][j] for k in _SF}


def _set(s, j, v, mask):
    return {k: tuple(jnp.where(mask, v[k], x) if q == j else x for q, x in enumerate(s[k])) for k in _SF}


def _set_a(s, j, val, mask):
    return dict(s, a=tuple(jnp.where(mask, val, x) if q == j else x for q, x in enumerate(s["a"])))


def _metric(s, count):
    (w1x, w2x, w3x), (w1y, w2y, w3y) = s["wx"], s["wy"]
    two = m.length(w1x - w2x, w1y - w2y)
    three = m.cross(w2x - w1x, w2y - w1y, w3x - w1x, w3y - w1y)
    return jnp.where(count == 2, two, jnp.where(count == 3, three, F32(0.0)))


def _solve2(s, count, mask):
    w1x, w1y, w2x, w2y = s["wx"][0], s["wy"][0], s["wx"][1], s["wy"][1]
    ex, ey = w2x - w1x, w2y - w1y
    d12_2 = -m.dot(w1x, w1y, ex, ey)
    d12_1 = m.dot(w2x, w2y, ex, ey)
    r1 = mask & (d12_2 <= 0)
    r2 = mask & ~r1 & (d12_1 <= 0)
    r3 = mask & ~r1 & ~r2
    inv = F32(1.0) / (d12_1 + d12_2)
    s = _set_a(s, 0, F32(1.0), r1)
    s = _set_a(s, 1, F32(1.0), r2)
    s = _set(s, 0, _col(s, 1), r2)
    s = _set_a(s, 0, m.mul(d12_1, inv), r3)
    s = _set_a(s, 1, m.mul(d12_2, inv), r3)
    count = jnp.where(r1 | r2, 1, jnp.where(r3, 2, count))
    return s, count


def _solve3(s, count, mask):
    (w1x, w2x, w3x), (w1y, w2y, w3y) = s["wx"], s["wy"]
    e12x, e12y = w2x - w1x, w2y - w1y
    d12_1, d12_2 = m.dot(w2x, w2y, e12x, e12y), -m.dot(w1x, w1y, e12x, e12y)
    e13x, e13y = w3x - w1x, w3y - w1y
    d13_1, d13_2 = m.dot(w3x, w3y, e13x, e13y), -m.dot(w1x, w1y, e13x, e13y)
    e23x, e23y = w3x - w2x, w3y - w2y
    d23_1, d23_2 = m.dot(w3x, w3y, e23x, e23y), -m.dot(w2x, w2y, e23x, e23y)
    n123 = m.cross(e12x, e12y, e13x, e13y)
    d123_1 = m.mul(n123, m.cross(w2x, w2y, w3x, w3y))
    d123_2 = m.mul(n123, m.cross(w3x, w3y, w1x, w1y))
    d123_3 = m.mul(n123, m.cross(w1x, w1y, w2x, w2y))
    left = mask
    r_w1 = left & (d12_2 <= 0) & (d13_2 <= 0)
    left = left & ~r_w1
    r_e12 = left & (d12_1 > 0) & (d12_2 > 0) & (d123_3 <= 0)
    left = left & ~r_e12
    r_e13 = left & (d13_1 > 0) & (d13_2 > 0) & (d123_2 <= 0)
    left = left & ~r_e13
    r_w2 = left & (d12_1 <= 0) & (d23_2 <= 0)
    left = left & ~r_w2
    r_w3 = left & (d13_1 <= 0) & (d23_1 <= 0)
    left = left & ~r_w3
    r_e23 = left & (d23_1 > 0) & (d23_2 > 0) & (d123_1 <= 0)
    r_tri = left & ~r_e23
    one = F32(1.0)
    i12 = one / (d12_1 + d12_2)
    i13 = one / (d13_1 + d13_2)
    i23 = one / (d23_1 + d23_2)
    i123 = one / ((d123_1 + d123_2) + d123_3)
    a1 = jnp.where(r_w1, one, jnp.where(r_e12, m.mul(d12_1, i12), jnp.where(r_e13, m.mul(d13_1, i13),
                                                                             m.mul(d123_1, i123))))
    s = _set_a(s, 0, a1, r_w1 | r_e12 | r_e13 | r_tri)
    s = _set_a(s, 1, jnp.where(r_e12, m.mul(d12_2, i12), jnp.where(r_w2, one, jnp.where(r_e23, m.mul(d23_1, i23),
                                                                                         m.mul(d123_2, i123)))),
               r_e12 | r_w2 | r_e23 | r_tri)
    s = _set_a(s, 2, jnp.where(r_e13, m.mul(d13_2, i13), jnp.where(r_w3, one, jnp.where(r_e23, m.mul(d23_2, i23),
                                                                                         m.mul(d123_3, i123)))),
               r_e13 | r_w3 | r_e23 | r_tri)
    c2, c3 = _col(s, 1), _col(s, 2)
    s = _set(s, 1, c3, r_e13)  # m_v2 = m_v3
    s = _set(s, 0, c2, r_w2)  # m_v1 = m_v2
    s = _set(s, 0, c3, r_w3 | r_e23)  # m_v1 = m_v3
    count = jnp.where(r_w1 | r_w2 | r_w3, 1, jnp.where(r_e12 | r_e13 | r_e23, 2, jnp.where(r_tri, 3, count)))
    return s, count


def distance(cache, pA, xfA, pB, xfB, active):
    """toi.distance for one contact per lane. cache = (count, (iA0, iA1, iA2), (iB0, iB1, iB2), metric)."""
    ccount, ciA, ciB, cmetric = cache
    zf = jnp.zeros_like(cmetric)
    zi = jnp.zeros_like(ccount)
    s = {k: (zf, zf, zf) for k in _SF[:7]}
    s.update(iA=(zi, zi, zi), iB=(zi, zi, zi))
    for j in range(3):  # ReadCache
        s = _set(s, j, _make_vertex(pA, xfA, ciA[j], pB, xfB, ciB[j]), j < ccount)
    count = ccount
    metric2 = _metric(s, count)
    flush = (count > 1) & ((metric2 < m.mul(F32(0.5), cmetric)) | (m.mul(F32(2.0), cmetric) < metric2)
                           | (metric2 < EPSILON))
    count = jnp.where(flush, 0, count)
    empty = count == 0
    v0 = _make_vertex(pA, xfA, zi, pB, xfB, zi)
    v0["a"] = jnp.ones_like(zf)
    s = _set(s, 0, v0, empty)
    count = jnp.where(empty, 1, count)

    def body(c):
        s, count, it, run = c
        save_count, save_a, save_b = count, s["iA"], s["iB"]
        s, count = _solve2(s, count, run & (count == 2))
        s, count = _solve3(s, count, run & (count == 3))
        run = run & (count != 3)
        w1x, w1y, w2x, w2y = s["wx"][0], s["wy"][0], s["wx"][1], s["wy"][1]
        ex, ey = w2x - w1x, w2y - w1y
        sgn = m.cross(ex, ey, -w1x, -w1y)
        lx, ly = m.cross_sv(F32(1.0), ex, ey)
        rx, ry = m.cross_vs(ex, ey, F32(1.0))
        dx = jnp.where(count == 1, -w1x, jnp.where(sgn > 0, lx, rx))
        dy = jnp.where(count == 1, -w1y, jnp.where(sgn > 0, ly, ry))
        run = run & ~(m.dot(dx, dy, dx, dy) < EPS2)
        iA = pA.support(*m.rot_t(xfA[2], xfA[3], -dx, -dy))
        iB = pB.support(*m.rot_t(xfB[2], xfB[3], dx, dy))
        v = _make_vertex(pA, xfA, iA, pB, xfB, iB)
        for j in (1, 2):
            keep_a = s["a"][j]
            s = _set(s, j, v, run & (count == j))
            s = _set_a(s, j, keep_a, run & (count == j))
        it = it + run.astype(I32)
        dup = jnp.zeros_like(run)
        for j in range(3):
            dup = dup | ((j < save_count) & (save_a[j] == iA) & (save_b[j] == iB))
        run = run & ~dup
        count = count + run.astype(I32)
        run = run & (it < T.GJK_ITERS)
        return s, count, it, run

    s, count, _, _ = jax.lax.while_loop(lambda c: jnp.max(c[3].astype(I32)) > 0, body, (s, count, zi, active))
    a1, a2, a3 = s["a"]
    wAx, wAy, wBx, wBy = s["wAx"], s["wAy"], s["wBx"], s["wBy"]
    pa2 = (m.mul(a1, wAx[0]) + m.mul(a2, wAx[1]), m.mul(a1, wAy[0]) + m.mul(a2, wAy[1]))
    pb2 = (m.mul(a1, wBx[0]) + m.mul(a2, wBx[1]), m.mul(a1, wBy[0]) + m.mul(a2, wBy[1]))
    pa3 = (pa2[0] + m.mul(a3, wAx[2]), pa2[1] + m.mul(a3, wAy[2]))
    sel = lambda c1, c2, c3: jnp.where(count == 1, c1, jnp.where(count == 2, c2, c3))  # noqa: E731
    pAx, pAy = sel(wAx[0], pa2[0], pa3[0]), sel(wAy[0], pa2[1], pa3[1])
    pBx, pBy = sel(wBx[0], pb2[0], pa3[0]), sel(wBy[0], pb2[1], pa3[1])
    dist = m.length(pAx - pBx, pAy - pBy)
    keep = lambda new, old: jnp.where(active, new, old)  # noqa: E731
    cache = (keep(count, ccount), tuple(keep(x, y) for x, y in zip(s["iA"], ciA)),
             tuple(keep(x, y) for x, y in zip(s["iB"], ciB)), keep(_metric(s, count), cmetric))
    return dist, cache


# ---------------------------------------------------------------- b2SeparationFunction

def _sep_init(cache, pA, xfA, pB, xfB):
    count, ciA, ciB, _ = cache
    lpA, lpB = pA.vertex(ciA[0]), pB.vertex(ciB[0])
    pax, pay = m.xf_mul(*xfA, *lpA)
    pbx, pby = m.xf_mul(*xfB, *lpB)
    ptx, pty, _ = m.normalize(pbx - pax, pby - pay)
    b1, b2 = pB.vertex(ciB[0]), pB.vertex(ciB[1])
    fbx, fby, _ = m.normalize(*m.cross_vs(b2[0] - b1[0], b2[1] - b1[1], F32(1.0)))
    nbx, nby = m.rot(xfB[2], xfB[3], fbx, fby)
    lbx, lby = m.mul(F32(0.5), b1[0] + b2[0]), m.mul(F32(0.5), b1[1] + b2[1])
    qbx, qby = m.xf_mul(*xfB, lbx, lby)
    sb = m.dot(pax - qbx, pay - qby, nbx, nby)
    fbx, fby = jnp.where(sb < 0, -fbx, fbx), jnp.where(sb < 0, -fby, fby)
    a1, a2 = pA.vertex(ciA[0]), pA.vertex(ciA[1])
    fax, fay, _ = m.normalize(*m.cross_vs(a2[0] - a1[0], a2[1] - a1[1], F32(1.0)))
    nax, nay = m.rot(xfA[2], xfA[3], fax, fay)
    lax_, lay = m.mul(F32(0.5), a1[0] + a2[0]), m.mul(F32(0.5), a1[1] + a2[1])
    qax, qay = m.xf_mul(*xfA, lax_, lay)
    sa = m.dot(pbx - qax, pby - qay, nax, nay)
    fax, fay = jnp.where(sa < 0, -fax, fax), jnp.where(sa < 0, -fay, fay)
    typ = jnp.where(count == 1, T.E_POINTS, jnp.where(ciA[0] == ciA[1], T.E_FACE_B, T.E_FACE_A))
    pick = lambda p, a, b: jnp.where(typ == T.E_POINTS, p, jnp.where(typ == T.E_FACE_A, a, b))  # noqa: E731
    zero = jnp.zeros_like(ptx)
    return dict(type=typ, axx=pick(ptx, fax, fbx), axy=pick(pty, fay, fby), lpx=pick(zero, lax_, lbx),
                lpy=pick(zero, lay, lby))


def _sep_points(f, pA, xfA, iA, pB, xfB, iB):
    axx, axy = f["axx"], f["axy"]
    pax, pay = m.xf_mul(*xfA, *pA.vertex(iA))
    pbx, pby = m.xf_mul(*xfB, *pB.vertex(iB))
    s_pts = m.dot(pbx - pax, pby - pay, axx, axy)
    nax, nay = m.rot(xfA[2], xfA[3], axx, axy)
    qax, qay = m.xf_mul(*xfA, f["lpx"], f["lpy"])
    s_fa = m.dot(pbx - qax, pby - qay, nax, nay)
    nbx, nby = m.rot(xfB[2], xfB[3], axx, axy)
    qbx, qby = m.xf_mul(*xfB, f["lpx"], f["lpy"])
    s_fb = m.dot(pax - qbx, pay - qby, nbx, nby)
    t = f["type"]
    return jnp.where(t == T.E_POINTS, s_pts, jnp.where(t == T.E_FACE_A, s_fa, s_fb))


def _find_min(f, pA, xfA, pB, xfB):
    axx, axy = f["axx"], f["axy"]
    typ = f["type"]
    pa_i = pA.support(*m.rot_t(xfA[2], xfA[3], axx, axy))
    pb_i = pB.support(*m.rot_t(xfB[2], xfB[3], -axx, -axy))
    nax, nay = m.rot(xfA[2], xfA[3], axx, axy)
    fa_i = pB.support(*m.rot_t(xfB[2], xfB[3], -nax, -nay))
    nbx, nby = m.rot(xfB[2], xfB[3], axx, axy)
    fb_i = pA.support(*m.rot_t(xfA[2], xfA[3], -nbx, -nby))
    iA = jnp.where(typ == T.E_POINTS, pa_i, jnp.where(typ == T.E_FACE_A, -1, fb_i))
    iB = jnp.where(typ == T.E_POINTS, pb_i, jnp.where(typ == T.E_FACE_A, fa_i, -1))
    return _sep_points(f, pA, xfA, iA, pB, xfB, iB), iA, iB


# ---------------------------------------------------------------- the kernel

def _kernel(cfg, bi_ref, ci_ref, cf_ref, px_ref, k_ref, bo_ref, co_ref, so_ref, rec_ref, lp_ref):
    """bi: body rows c0x, c0y, a0, alpha0, cx, cy, a, lcx, lcy (B each); ci: contact rows ia, ib, perm, pos, need,
    candidate base (go & base), flag (C each), then n_live; cf: toi (C); px: proxies A, B per contact (C * 2 *
    PROXY rows); k: constants (2 pi). Outputs: bo: c0x, c0y, a0, alpha0 (B each); co: toi, flag (C each); so: found,
    min slot, min alpha. Scratch: rec (7 C rows recorded sweeps, then C rows of the queue), lp (lane-local sweeps
    and proxies of the contact a lane is working on)."""
    B, C = cfg.B, cfg.C
    blk = bo_ref.shape[-1]
    lane = jax.lax.broadcasted_iota(I32, (blk,), 0)
    slop = cfg.linear_slop
    two_pi = k_ref[0, :]
    ci = lambda f, k: ci_ref[f * C + k, :].astype(I32)  # noqa: E731
    cig = lambda f, kv: ci_ref[f * C + kv, lane].astype(I32)  # noqa: E731
    n_live = ci_ref[7 * C, :].astype(I32)

    # ---- 1. alignment pass, newest contact first (toi._search's ordered loop)
    def copy_b(q, _):
        bo_ref[q, :] = bi_ref[q, :]
        return 0
    jax.lax.fori_loop(0, 4 * B, copy_b, 0)
    c0x, c0y, a0, al0 = (K.RefBodies(bo_ref, f * B, B, lane) for f in range(4))
    cx, cy, a = (K.RefBodies(bi_ref, (4 + f) * B, B, lane) for f in range(3))

    def align(k, nq):
        slot = ci(2, k)
        n = (cig(4, slot) != 0) & (k < n_live)
        A, Bb = cig(0, slot), cig(1, slot)
        alA, alB = al0.get(A), al0.get(Bb)
        sA = (c0x.get(A), c0y.get(A), a0.get(A))
        sB = (c0x.get(Bb), c0y.get(Bb), a0.get(Bb))
        advA = n & (alA < alB)
        advB = n & ~(alA < alB) & (alB < alA)
        nA = T._advance(*sA, alA, cx.get(A), cy.get(A), a.get(A), alB)
        nB = T._advance(*sB, alB, cx.get(Bb), cy.get(Bb), a.get(Bb), alA)
        sA = tuple(jnp.where(advA, x, y) for x, y in zip(nA[:3], sA))
        sB = tuple(jnp.where(advB, x, y) for x, y in zip(nB[:3], sB))
        common = jnp.where(advA, alB, alA)
        for arr, va, vb in zip((c0x, c0y, a0), sA, sB):
            arr.set(A, jnp.where(advA, va, arr.get(A)))
            arr.set(Bb, jnp.where(advB, vb, arr.get(Bb)))
        al0.set(A, jnp.where(advA, alB, al0.get(A)))
        al0.set(Bb, jnp.where(advB, alA, al0.get(Bb)))
        for f, v in enumerate((*sA, *sB, common)):
            rec_ref[f * C + slot, lane] = v
        rec_ref[_REC * C + jnp.minimum(nq, C - 1), lane] = slot.astype(jnp.float32)
        return nq + n.astype(I32)

    nq = jax.lax.fori_loop(0, jnp.max(n_live), align, jnp.zeros((blk,), I32))

    # ---- 2. b2TimeOfImpact per queued contact (toi.time_of_impact's phase machine, one contact per lane at a time)
    pA, pB = Proxy(lp_ref, 16, lane), Proxy(lp_ref, 16 + PROXY, lane)
    XF = 16 + 2 * PROXY  # lane-local rows of the four transforms
    total_radius = lambda: pA.radius() + pB.radius()  # noqa: E731
    tol = F32(F32(0.25) * slop)
    DIST, FIND, ROOT = 0, 1, 2
    zf, zi = jnp.zeros((blk,), jnp.float32), jnp.zeros((blk,), I32)
    one = jnp.ones((blk,), jnp.float32)

    def start(c, go):
        """Load the next queued contact into the lanes where `go` (lane-local sweeps and proxies)."""
        qi = c["qi"]
        # clipped: a lane with nothing queued (not `go`) reads a queue row never written in this call (scratch
        # memory, any bits), and its slot must still address its own rows (it is never used)
        slot = jnp.clip(rec_ref[_REC * C + jnp.minimum(qi, C - 1), lane].astype(I32), 0, C - 1)
        recv = [rec_ref[f * C + slot, lane] for f in range(_REC)]
        ia, ib = cig(0, slot), cig(1, slot)
        lc = lambda bfield, body: bi_ref[bfield * B + jnp.clip(body, 0, B - 1), lane]  # noqa: E731
        swA = (lc(7, ia), lc(8, ia), recv[0], recv[1], lc(4, ia), lc(5, ia), recv[2], lc(6, ia))
        swB = (lc(7, ib), lc(8, ib), recv[3], recv[4], lc(4, ib), lc(5, ib), recv[5], lc(6, ib))
        for base, sw in ((0, swA), (8, swB)):  # b2Sweep::Normalize
            d = m.mul(two_pi, jnp.floor(sw[6] / two_pi))
            sw = sw[:6] + (sw[6] - d, sw[7] - d)
            for q in range(8):
                lp_ref[base + q, :] = jnp.where(go, sw[q], lp_ref[base + q, :])
        if cfg.shared_px:  # the contact's fixtures (ci rows after n_live) index the shared proxy table
            nf = cfg.shared_px
            fA = jnp.clip(ci_ref[7 * C + 1 + slot, lane].astype(I32), 0, nf - 1)
            fB = jnp.clip(ci_ref[8 * C + 1 + slot, lane].astype(I32), 0, nf - 1)
            for q in range(2 * PROXY):
                src = px_ref[fA * PROXY + q] if q < PROXY else px_ref[fB * PROXY + q - PROXY]
                lp_ref[16 + q, :] = jnp.where(go, src, lp_ref[16 + q, :])
        else:
            for q in range(2 * PROXY):
                lp_ref[16 + q, :] = jnp.where(go, px_ref[slot * (2 * PROXY) + q, lane], lp_ref[16 + q, :])
        new = dict(slot=slot, phase=zi + DIST, state=zi + T.E_UNKNOWN, out_t=one, t1=zf, t2=one, it=zi, push=zi,
                   cache=(zi, (zi, zi, zi), (zi, zi, zi), zf), f=dict(type=zi, axx=zf, axy=zf, lpx=zf, lpy=zf),
                   iA=zi, iB=zi, a1=zf, a2=zf, rs1=zf, rs2=zf, cnt=zi, run=go | c["run"], qi=qi + go.astype(I32))
        keep = {k: v for k, v in c.items() if k not in new}
        sel = jax.tree.map(lambda x, y: jnp.where(go, x, y), {k: new[k] for k in new if k not in ("run", "qi")},
                           {k: c[k] for k in new if k not in ("run", "qi")})
        return dict(keep, **sel, run=new["run"], qi=new["qi"])

    def sweep(base):
        return tuple(lp_ref[base + q, :] for q in range(8))

    def finish(c, done):
        """Write alpha / toi / flag of the contacts that finished this iteration."""
        slot = c["slot"]
        al = rec_ref[6 * C + slot, lane]
        alpha = jnp.where(c["state"] == T.E_TOUCHING, m.bmin(al + m.mul(F32(1.0) - al, c["out_t"]), F32(1.0)),
                          F32(1.0))
        old = co_ref[slot, lane]
        co_ref[slot, lane] = jnp.where(done, alpha, old)

    def body(c):
        idle = ~c["run"] & (c["qi"] < nq)
        c = start(c, idle)
        run, ph = c["run"], c["phase"]
        dist_l, find_l, root_l = run & (ph == DIST), run & (ph == FIND), run & (ph == ROOT)
        t1, t2, f = c["t1"], c["t2"], c["f"]
        target = m.bmax(slop, total_radius() - F32(F32(3.0) * slop))
        a1, a2, rs1, rs2, cnt = c["a1"], c["a2"], c["rs1"], c["rs2"], c["cnt"]
        secant = a1 + (m.mul(target - rs1, a2 - a1) / (rs2 - rs1))
        troot = jnp.where((cnt & 1) == 1, secant, m.mul(F32(0.5), a1 + a2))
        tq1 = jnp.where(dist_l, t1, jnp.where(find_l, t2, troot))
        swA, swB = sweep(0), sweep(8)

        def xf_one(i, _):  # the four transforms (A, B at tq1; A, B at t1) through one copy of the code (sin/cos)
            useB = (i & 1) == 1
            sw = tuple(jnp.where(useB, b, a) for a, b in zip(swA, swB))
            x = T.sweep_xf(sw, jnp.where(i >= 2, t1, tq1))
            for q in range(4):
                lp_ref[XF + 4 * i + q, :] = x[q]
            return 0
        jax.lax.fori_loop(0, 4, xf_one, 0)
        xA1, xB1, xA2, xB2 = (tuple(lp_ref[XF + 4 * i + q, :] for q in range(4)) for i in range(4))

        # ---- DIST
        dist, cache = distance(c["cache"], pA, xA1, pB, xB1, dist_l)
        over = dist_l & (dist <= 0)
        touch = dist_l & ~over & (dist < target + tol)
        cont = dist_l & ~over & ~touch
        fn = _sep_init(cache, pA, xA1, pB, xB1)
        f = {k: jnp.where(cont, fn[k], f[k]) for k in f}
        state = jnp.where(over, T.E_OVERLAPPED, jnp.where(touch, T.E_TOUCHING, c["state"]))
        out_t = jnp.where(over, F32(0.0), jnp.where(touch, t1, c["out_t"]))
        run = run & ~over & ~touch
        phase = jnp.where(cont, FIND, ph)
        t2 = jnp.where(cont, one, t2)
        push = jnp.where(cont, 0, c["push"])

        # ---- FIND
        s2, fiA, fiB = _find_min(f, pA, xA1, pB, xB1)
        s1 = _sep_points(f, pA, xA2, fiA, pB, xB2, fiB)
        sep = find_l & (s2 > target + tol)
        adv = find_l & ~sep & (s2 > target - tol)
        rest = find_l & ~sep & ~adv
        fail = rest & (s1 < target - tol)
        tch = rest & ~fail & (s1 <= target + tol)
        root = rest & ~fail & ~tch
        state = jnp.where(sep, T.E_SEPARATED, jnp.where(fail, T.E_FAILED, jnp.where(tch, T.E_TOUCHING, state)))
        out_t = jnp.where(sep, one, jnp.where(fail | tch, t1, out_t))
        run = run & ~(sep | fail | tch)
        t1n = jnp.where(adv, t2, t1)
        iA, iB = jnp.where(root, fiA, c["iA"]), jnp.where(root, fiB, c["iB"])
        a1, a2 = jnp.where(root, t1, a1), jnp.where(root, t2, a2)
        rs1, rs2 = jnp.where(root, s1, rs1), jnp.where(root, s2, rs2)
        cnt = jnp.where(root, 0, cnt)
        phase = jnp.where(root, ROOT, phase)

        # ---- ROOT
        sr = _sep_points(f, pA, xA1, c["iA"], pB, xB1, c["iB"])
        cnt = jnp.where(root_l, cnt + 1, cnt)
        conv = root_l & (m.babs(sr - target) < tol)
        t2 = jnp.where(conv, troot, t2)
        go = root_l & ~conv
        up, lo = go & (sr > target), go & ~(sr > target)
        a1, rs1 = jnp.where(up, troot, a1), jnp.where(up, sr, rs1)
        a2, rs2 = jnp.where(lo, troot, a2), jnp.where(lo, sr, rs2)
        root_end = conv | (go & (cnt == T.ROOT_ITERS))
        push = jnp.where(root_end, push + 1, push)
        back = root_end & (push != NV)
        phase = jnp.where(back, FIND, phase)

        # ---- end of a separating-axis iteration
        axis_end = adv | (root_end & ~back)
        it = jnp.where(axis_end, c["it"] + 1, c["it"])
        stuck = axis_end & (it == T.TOI_ITERS)
        state = jnp.where(stuck, T.E_FAILED, state)
        out_t = jnp.where(stuck, t1n, out_t)
        run = run & ~stuck
        phase = jnp.where(axis_end, DIST, phase)
        cache = jax.tree.map(lambda x, y: jnp.where(dist_l, x, y), cache, c["cache"])
        out = dict(c, phase=phase, state=state, out_t=out_t, t1=t1n, t2=t2, it=it, push=push, cache=cache, f=f,
                   iA=iA, iB=iB, a1=a1, a2=a2, rs1=rs1, rs2=rs2, cnt=cnt, run=run)
        finish(out, c["run"] & ~run)
        return out

    def copy_c(k, _):  # results default to the cached toi (contacts not needing a TOI keep theirs)
        co_ref[k, :] = cf_ref[k, :]
        return 0
    jax.lax.fori_loop(0, C, copy_c, 0)
    c0 = dict(slot=zi, phase=zi, state=zi, out_t=one, t1=zf, t2=one, it=zi, push=zi,
              cache=(zi, (zi, zi, zi), (zi, zi, zi), zf), f=dict(type=zi, axx=zf, axy=zf, lpx=zf, lpy=zf), iA=zi,
              iB=zi, a1=zf, a2=zf, rs1=zf, rs2=zf, cnt=zi, run=jnp.zeros((blk,), bool), qi=zi)
    more = lambda c: jnp.max((c["run"] | (c["qi"] < nq)).astype(I32)) > 0  # noqa: E731
    jax.lax.while_loop(more, body, c0)

    # ---- 3. flags, candidates and the minimum (toi._search's tail)
    def tail(k, c):
        best_t, best_p, best_s, anyc = c
        need = ci(4, k) != 0
        flag = (ci(6, k) != 0) | need
        toi = co_ref[k, :]
        co_ref[C + k, :] = flag.astype(jnp.float32)
        cand = (ci(5, k) != 0) & flag & (toi < F32(1.0))
        pos = ci(3, k)
        if cfg.tie_oldest:
            take = cand & ((toi < best_t) | ((toi == best_t) & (pos > best_p)))
        else:
            take = cand & ((toi < best_t) | ((toi == best_t) & (pos < best_p)))
        return (jnp.where(take, toi, best_t), jnp.where(take, pos, best_p), jnp.where(take, k, best_s),
                anyc | cand)
    best_t, _, best_s, anyc = jax.lax.fori_loop(
        0, C, tail, (one, zi + (-1 if cfg.tie_oldest else C), zi, jnp.zeros((blk,), bool)))
    so_ref[0, :] = (anyc & ~(T.DONE_ALPHA < best_t)).astype(jnp.float32)
    so_ref[1, :] = best_s.astype(jnp.float32)
    so_ref[2, :] = best_t


def _call(cfg, bi, ci, cf, px, kc):
    """Batched: inputs (N, rows) world-major; returns bo (N, 4B), co (N, 2C), so (N, 3)."""
    N = bi.shape[0]
    blk = cfg.block
    Np = -(-N // blk) * blk

    bi, ci, cf, px, kc = jax.lax.optimization_barrier((bi, ci, cf, px, kc))
    C = cfg.C
    # K.SORT_LANES: worlds grouped by queued TOI contacts (need, rows 4C..5C of ci), then live contacts (row 7C)
    order = K.lane_order(jnp.sum(ci[:, 4 * C:5 * C], axis=1) * 256 + ci[:, 7 * C], N, blk)
    bi, ci, cf, kc = (K.to_minor(x, N, Np, order) for x in (bi, ci, cf, kc))
    px_spec = pl.BlockSpec(px.shape, lambda i: (0,)) if cfg.shared_px else None
    if not cfg.shared_px:
        px = K.to_minor(px, N, Np, order)
    spec = lambda rows: pl.BlockSpec((rows, blk), lambda i: (0, i))  # noqa: E731
    B, C = cfg.B, cfg.C
    shapes = [(4 * B, Np), (2 * C, Np), (3, Np), ((_REC + 1) * C, Np), (32 + 2 * PROXY, Np)]
    kw = {}
    if not cfg.interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=cfg.num_warps, num_stages=1)
    with K.triton_mode(not cfg.interpret):
        bo, co, so, _, _ = pl.pallas_call(
            partial(_kernel, cfg), grid=(Np // blk,),
            in_specs=[spec(bi.shape[0]), spec(ci.shape[0]), spec(cf.shape[0]), px_spec or spec(px.shape[0]),
                      spec(kc.shape[0])],
            out_specs=[spec(s[0]) for s in shapes],
            out_shape=[jax.ShapeDtypeStruct(s, jnp.float32) for s in shapes],
            interpret=cfg.interpret, name="box2d_toi_search", **kw)(bi, ci, cf, px, kc)
    return tuple(K.from_minor(x, N, order) for x in (bo, co, so))


@jax.custom_batching.custom_vmap
def _search_call(cfg_box, bi, ci, cf, px, kc):
    shared = cfg_box.cfg.shared_px
    bo, co, so = _call(cfg_box.cfg, bi[None], ci[None], cf[None], px if shared else px[None], kc[None])
    return bo[0], co[0], so[0]


@_search_call.def_vmap
def _search_rule(axis_size, in_batched, cfg_box, bi, ci, cf, px, kc):
    shared = cfg_box.cfg.shared_px
    if shared and in_batched[4]:
        raise NotImplementedError("gpu toi search: the shared proxy table must not be batched")
    args = [a if (b or (shared and k == 3)) else jnp.broadcast_to(a, (axis_size,) + a.shape)
            for k, (a, b) in enumerate(zip((bi, ci, cf, px, kc), in_batched[1:]))]
    return _call(cfg_box.cfg, *args), (True, True, True)


class _Static:
    def __init__(self, cfg):
        self.cfg = cfg


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.cfg), lambda cfg, _: _Static(cfg))


def search(model, state, tst, settings, go, axis_name=None, block=None, num_warps=None):
    """toi._search with the kernel: same arguments and results."""
    from . import solve as G
    ct, bd = state.contacts, state.bodies
    B, C = bd.type.shape[0], ct.pair.shape[0]
    live = ct.pair >= 0
    fb = model.fixture_body
    ia, ib = fb[ct.fa], fb[ct.fb]
    base = live & tst.enabled & (tst.count <= T.MAX_SUB_STEPS)
    act = bd.awake & (bd.type != T.STATIC)
    need = go & base & ~tst.flag & (act[ia] | act[ib]) & T._pair_rule(bd, ia, ib, settings.unity_toi)
    perm, pos, n_live = T._list_order(ct)
    f32 = lambda x: jnp.asarray(x).astype(jnp.float32)  # noqa: E731
    bi = jnp.concatenate([bd.c0[:, 0], bd.c0[:, 1], bd.a0, tst.alpha0, bd.c[:, 0], bd.c[:, 1], bd.a,
                          bd.local_center[:, 0], bd.local_center[:, 1]])
    ci = [f32(ia), f32(ib), f32(perm), f32(pos), f32(need), f32(go & base), f32(tst.flag), f32(n_live)[None]]

    def proxy(f):
        v = model.verts[f]
        return jnp.concatenate([v[..., 0], v[..., 1], f32(model.count[f])[:, None], model.radius[f][:, None]], 1)
    nf = model.fixture_body.shape[0] if SHARED_PROXIES else 0
    if nf:  # one table of every fixture's proxy, shared by the worlds; the contacts' fixtures go with ci
        px = proxy(jnp.arange(nf)).reshape(-1)
        ci += [f32(ct.fa), f32(ct.fb)]
    else:
        px = jnp.concatenate([proxy(ct.fa), proxy(ct.fb)], 1).reshape(-1)
    ci = jnp.concatenate(ci)
    kc = jnp.asarray([T.TWO_PI], jnp.float32)
    cfg = Cfg(B=B, C=C, linear_slop=F32(settings.linear_slop), tie_oldest=bool(settings.toi_tie_oldest),
              block=block or G.BLOCK, num_warps=num_warps or G.NUM_WARPS, interpret=G._interpret(), shared_px=nf)
    bo, co, so = _search_call(_Static(cfg), bi, ci, tst.toi, px, kc)
    c0 = jnp.stack([bo[:B], bo[B:2 * B]], -1)
    bd = bd._replace(c0=c0, a0=bo[2 * B:3 * B])
    tst = tst._replace(alpha0=bo[3 * B:4 * B], toi=co[:C], flag=co[C:] != 0)
    state = state._replace(bodies=bd)
    return state, tst, so[0] != 0, so[1].astype(I32), so[2]
