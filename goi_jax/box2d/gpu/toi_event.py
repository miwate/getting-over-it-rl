"""One b2World::SolveTOI event (toi._event) as one Pallas kernel, one world per lane, plus the XLA tail.

toi._event in XLA is a long straight-line graph (every body advanced, a manifold for every contact slot, prefix sums
for the island walk, contact updates, the TOI island solve) that costs a few hundred small kernels per event
whatever the batch size. The kernel does the same per world, sequentially where Box2D is sequential:

1. every body advanced to minAlpha (b2Body::Advance) and its transform, and the restored transform;
2. minContact->Update (b2Contact::Evaluate on the advanced poses, impulses carried over by feature key);
3. the island walk over bodyA's then bodyB's contact edges, newest first, with toi._event's eligibility rules and its
   32-contact / 64-body break, evaluating and updating each eligible contact (toi._event evaluates every slot at the
   same poses; each Update in the walk sees every body "advanced to minAlpha", see toi.py);
4. the body sweeps after the walk (advanced, restored or unchanged), the wakes, the island order;
5. b2Island::SolveTOI (toi._solve_island): up to 20 position passes with the TOI masses, the velocity constraints
   (contact_solver.prepare per row), `velocity_iterations` velocity passes, integration, b2Island::Report.

The narrow phase (collide.py) is re-expressed per lane with explicit 8-vertex loops (same float operations, argmax /
argmin first-index ties). SynchronizeFixtures, FindNewContacts and the TOI bookkeeping that follows them stay as in
toi._event (gpu/pairs.py kernels and a few XLA ops). Bit-identical to toi._event in interpret mode on CPU
(tests/box2d/test_gpu.py); sin/cos in double precision on the Triton backend as in kernel.py.

Integer state travels in an int32 table (feature keys exceed float32's exact range), floats in a float32 table, both
aliased with the outputs and updated in place.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from .. import collide as nc
from .. import contact_solver as cs
from .. import math2d as m
from .. import toi as T
from ..settings import K_TOL, MAX_FLOAT, MAX_POLYGON_VERTICES, MAX_TOI_CONTACTS
from ..shapes import CIRCLE, POLYGON
from ..world import CIRCLES, DYNAMIC, FACE_A, FACE_B, STATIC
from . import kernel as K

F32 = np.float32
I32 = jnp.int32
NV = MAX_POLYGON_VERTICES
NG = 4 * NV + 4  # geometry per fixture: vx[8], vy[8], nx[8], ny[8], count, radius, shape type, body
G_CNT, G_RAD, G_TYPE, G_BODY = 4 * NV, 4 * NV + 1, 4 * NV + 2, 4 * NV + 3
NI = MAX_TOI_CONTACTS  # island contact rows

BF = ("c0x", "c0y", "a0", "cx", "cy", "a", "vx", "vy", "w", "pxx", "pxy", "qc", "qs", "lcx", "lcy", "st", "im", "ii",
      "al0")
CF = ("lnx", "lny", "lpx", "lpy", "p0x", "p0y", "p1x", "p1y", "ni0", "ni1", "ti0", "ti1", "fr", "rs", "ts")
BI = ("type", "awake", "bullet", "cont")
CI = ("pair", "fa", "fb", "touch", "mtype", "count", "pid0", "pid1", "tcount", "flag", "enabled", "perm")
# float scratch rows (per body unless noted)
SB = ("ac0x", "ac0y", "aa0", "aal0", "axpx", "axpy", "axqc", "axqs", "oxpx", "oxpy", "oxqc", "oxqs", "flip", "ino",
      "tou", "newb", "seen", "pcx", "pcy", "pa", "pvx", "pvy", "pw")
# velocity constraint row layout (contact_solver.prepare's fields read by _solve_one)
_VROW = (("ia", 1, "i"), ("ib", 1, "i"), ("mA", 1, "f"), ("mB", 1, "f"), ("iA", 1, "f"), ("iB", 1, "f"),
         ("nx", 1, "f"), ("ny", 1, "f"), ("rAx", 2, "f"), ("rAy", 2, "f"), ("rBx", 2, "f"), ("rBy", 2, "f"),
         ("count", 1, "i"), ("tangent_speed", 1, "f"), ("tmass", 2, "f"), ("friction", 1, "f"), ("nmass", 2, "f"),
         ("bias", 2, "f"), ("K", 4, "f"), ("NM", 4, "f"))


def _layout():
    out, off = [], 0
    for name, w, kind in _VROW:
        out.append((name, off, w, kind, w > 1))
        off += w
    return K.Layout(tuple(out), off)


VLAY = _layout()


@dataclass(frozen=True)
class Cfg:
    settings: object
    B: int
    C: int
    block: int
    num_warps: int
    interpret: bool


class Tab:
    """Named per-world tables in (rows, block) refs: field f of item i is row off(f) + i."""

    def __init__(self, ref, names, n, lane, base=0):
        self.ref, self.n, self.lane = ref, n, lane
        self.off = {k: base + q * n for q, k in enumerate(names)}

    def get(self, f, i):
        return self.ref[self.off[f] + jnp.clip(i, 0, self.n - 1), self.lane]

    def set(self, f, i, v):
        self.ref[self.off[f] + jnp.clip(i, 0, self.n - 1), self.lane] = v

    def row(self, f, k):
        return self.ref[self.off[f] + k, :]

    def set_row(self, f, k, v):
        self.ref[self.off[f] + k, :] = v

    def bodies(self, f):
        return K.RefBodies(self.ref, self.off[f], self.n, self.lane)


# ---------------------------------------------------------------- narrow phase, one contact per lane

class Shape:
    """A fixture's shape for one lane: count, radius and type loaded, vertices and normals gathered on demand from
    the shared geometry table (a per-lane index is one gather, not a chain of selects over the 8 vertices)."""

    def __init__(self, geo, f):
        self.geo, self.f = geo, f
        self.count = geo[f * NG + G_CNT].astype(I32)
        self.radius = geo[f * NG + G_RAD]
        self.type = geo[f * NG + G_TYPE].astype(I32)

    def g(self, q, i=0):
        return self.geo[self.f * NG + q + i]

    def v(self, i):
        return self.g(0, i), self.g(NV, i)

    def n(self, i):
        return self.g(2 * NV, i), self.g(3 * NV, i)

    @staticmethod
    def select(flag, p, q):
        out = object.__new__(Shape)
        out.geo = p.geo
        for k in ("f", "count", "radius", "type"):
            setattr(out, k, jnp.where(flag, getattr(p, k), getattr(q, k)))
        return out


def _argmax(vals):
    bi, bv = jnp.zeros(vals[0].shape, I32), vals[0]
    for q in range(1, len(vals)):
        t = vals[q] > bv
        bi, bv = jnp.where(t, q, bi), jnp.where(t, vals[q], bv)
    return bi, bv


def _argmin(vals):
    bi, bv = jnp.zeros(vals[0].shape, I32), vals[0]
    for q in range(1, len(vals)):
        t = vals[q] < bv
        bi, bv = jnp.where(t, q, bi), jnp.where(t, vals[q], bv)
    return bi, bv


def _find_max_separations(pa, xfa, pb, xfb):
    """b2FindMaxSeparation(A, B) and (B, A) as one loop body (two iterations; poly1's vertex loop is dynamic), to
    keep the kernel small: returns (edge_a, sep_a, edge_b, sep_b)."""
    zi = jnp.zeros(pa.count.shape, I32)
    zf = jnp.zeros(pa.radius.shape, jnp.float32)

    def one(it, carry):
        sw = it == 1
        p1, p2 = Shape.select(sw, pb, pa), Shape.select(sw, pa, pb)
        xf1 = tuple(jnp.where(sw, b, a) for a, b in zip(xfa, xfb))
        xf2 = tuple(jnp.where(sw, a, b) for a, b in zip(xfa, xfb))
        px, py, qc, qs = m.xf_mul_t_xf(*xf2, *xf1)
        v2 = [p2.v(j) for j in range(NV)]

        def vertex(i, c):
            bi, bv = c
            nx, ny = m.rot(qc, qs, *p1.n(i))
            wx, wy = m.xf_mul(px, py, qc, qs, *p1.v(i))
            best = None
            for j in range(NV):
                sj = m.dot(nx, ny, v2[j][0] - wx, v2[j][1] - wy)
                sj = jnp.where(j < p2.count, sj, MAX_FLOAT)
                best = sj if best is None else jnp.minimum(best, sj)
            si = jnp.where(i < p1.count, best, -MAX_FLOAT)
            take = (i == 0) | (si > bv)  # first index of the maximum
            return jnp.where(take, i, bi), jnp.where(take, si, bv)
        e, v = jax.lax.fori_loop(0, NV, vertex, (zi, zf))
        ea, sa, eb, sb = carry
        return jnp.where(sw, ea, e), jnp.where(sw, sa, v), jnp.where(sw, e, eb), jnp.where(sw, v, sb)
    return jax.lax.fori_loop(0, 2, one, (zi, zf, zi, zf))


def collide_polygons(pa, xfa, pb, xfb, linear_slop):
    total_radius = pa.radius + pb.radius
    edge_a, sep_a, edge_b, sep_b = _find_max_separations(pa, xfa, pb, xfb)
    k_tol = m.mul(K_TOL, linear_slop)
    flip = sep_b > sep_a + k_tol
    p1, p2 = Shape.select(flip, pb, pa), Shape.select(flip, pa, pb)
    xf1 = tuple(jnp.where(flip, b, a) for a, b in zip(xfa, xfb))
    xf2 = tuple(jnp.where(flip, a, b) for a, b in zip(xfa, xfb))
    edge1 = jnp.where(flip, edge_b, edge_a)
    c1, c2 = p1.count, p2.count

    rnx, rny = m.rot(xf1[2], xf1[3], *p1.n(edge1))
    rnx, rny = m.rot_t(xf2[2], xf2[3], rnx, rny)
    dots = [jnp.where(j < c2, m.dot(rnx, rny, *p2.n(j)), MAX_FLOAT) for j in range(NV)]
    i1, _ = _argmin(dots)
    i2 = jnp.where(i1 + 1 < c2, i1 + 1, 0)
    inc0 = m.xf_mul(*xf2, *p2.v(i1))
    inc1 = m.xf_mul(*xf2, *p2.v(i2))
    face, vert = jnp.full_like(i1, nc.E_FACE), jnp.full_like(i1, nc.E_VERTEX)
    id0, id1 = (edge1, i1, face, vert), (edge1, i2, face, vert)

    iv1 = edge1
    iv2 = jnp.where(edge1 + 1 < c1, edge1 + 1, 0)
    v11 = p1.v(iv1)
    v12 = p1.v(iv2)
    tx, ty, _ = m.normalize(v12[0] - v11[0], v12[1] - v11[1])
    lnx, lny = m.cross_vs(tx, ty, F32(1.0))
    ppx, ppy = m.mul(F32(0.5), v11[0] + v12[0]), m.mul(F32(0.5), v11[1] + v12[1])
    tgx, tgy = m.rot(xf1[2], xf1[3], tx, ty)
    nx, ny = m.cross_vs(tgx, tgy, F32(1.0))
    w11 = m.xf_mul(*xf1, *v11)
    w12 = m.xf_mul(*xf1, *v12)
    front = m.dot(nx, ny, *w11)
    side1 = -m.dot(tgx, tgy, *w11) + total_radius
    side2 = m.dot(tgx, tgy, *w12) + total_radius

    c10, c11, cid10, cid11, np1 = nc._clip(inc0, inc1, id0, id1, -tgx, -tgy, side1, iv1)
    c20, c21, cid20, cid21, np2 = nc._clip(c10, c11, cid10, cid11, tgx, tgy, side2, iv2)
    ok = (np1 >= 2) & (np2 >= 2)
    pts, ids, keep = [], [], []
    for p, pid in ((c20, cid20), (c21, cid21)):
        sep = m.dot(nx, ny, *p) - front
        keep.append(sep <= total_radius)
        pts.append(m.xf_mul_t(*xf2, *p))
        ia, ib, ta, tb = pid
        ids.append(jnp.where(flip, nc.feature_key(ib, ia, tb, ta), nc.feature_key(ia, ib, ta, tb)))
    k0, k1 = keep
    count = jnp.where(ok, k0.astype(I32) + k1.astype(I32), 0)
    p0 = tuple(jnp.where(k0, a, b) for a, b in zip(pts[0], pts[1]))
    id_0 = jnp.where(k0, ids[0], ids[1])
    count = jnp.where((sep_a > total_radius) | (sep_b > total_radius), 0, count)
    return dict(mtype=jnp.where(flip, FACE_B, FACE_A).astype(I32), count=count, local_normal=(lnx, lny),
                local_point=(ppx, ppy), point_local=(p0, pts[1]), point_id=(id_0, ids[1]))


def collide_polygon_circle(pa, xfa, pb, xfb):
    cx, cy = m.xf_mul(*xfb, *pb.v(0))
    clx, cly = m.xf_mul_t(*xfa, cx, cy)
    radius = pa.radius + pb.radius
    s = []
    early = jnp.zeros(cx.shape, bool)
    for i in range(NV):
        (vx, vy), (nx, ny) = pa.v(i), pa.n(i)
        v = m.dot(nx, ny, clx - vx, cly - vy)
        valid = i < pa.count
        early = early | (valid & (v > radius))
        s.append(jnp.where(valid, v, -MAX_FLOAT))
    ni, separation = _argmax(s)
    count = pa.count
    i2 = jnp.where(ni + 1 < count, ni + 1, 0)
    v1 = pa.v(ni)
    v2 = pa.v(i2)
    nfx, nfy = pa.n(ni)
    rr = m.mul(radius, radius)
    face_c = (m.mul(F32(0.5), v1[0] + v2[0]), m.mul(F32(0.5), v1[1] + v2[1]))
    inside = separation < m.EPS
    u1 = m.dot(clx - v1[0], cly - v1[1], v2[0] - v1[0], v2[1] - v1[1])
    u2 = m.dot(clx - v2[0], cly - v2[1], v1[0] - v2[0], v1[1] - v2[1])
    d1x, d1y = clx - v1[0], cly - v1[1]
    d2x, d2y = clx - v2[0], cly - v2[1]
    far1 = m.dot(d1x, d1y, d1x, d1y) > rr
    far2 = m.dot(d2x, d2y, d2x, d2y) > rr
    n1x, n1y, _ = m.normalize(d1x, d1y)
    n2x, n2y, _ = m.normalize(d2x, d2y)
    sep_face = m.dot(clx - face_c[0], cly - face_c[1], nfx, nfy)
    c1, c2 = (u1 <= 0), (~(u1 <= 0)) & (u2 <= 0)
    c3 = ~(u1 <= 0) & ~(u2 <= 0)
    none = early | (~inside & ((c1 & far1) | (c2 & far2) | (c3 & (sep_face > radius))))
    pick = lambda a_in, a1, a2, a3: jnp.where(inside, a_in, jnp.where(c1, a1, jnp.where(c2, a2, a3)))  # noqa
    zero = jnp.zeros_like(ni)
    pl_ = pb.v(0)
    return dict(mtype=jnp.full_like(ni, FACE_A), count=jnp.where(none, 0, 1),
                local_normal=(pick(nfx, n1x, n2x, nfx), pick(nfy, n1y, n2y, nfy)),
                local_point=(pick(face_c[0], v1[0], v2[0], face_c[0]), pick(face_c[1], v1[1], v2[1], face_c[1])),
                point_local=(pl_, pl_), point_id=(zero, zero))


def collide_circles(pa, xfa, pb, xfb):
    pax, pay = m.xf_mul(*xfa, *pa.v(0))
    pbx, pby = m.xf_mul(*xfb, *pb.v(0))
    dx, dy = pbx - pax, pby - pay
    r = pa.radius + pb.radius
    hit = ~(m.dot(dx, dy, dx, dy) > m.mul(r, r))
    zi = jnp.zeros(dx.shape, I32)
    zf = jnp.zeros_like(dx)
    return dict(mtype=jnp.full_like(zi, CIRCLES), count=hit.astype(I32), local_normal=(zf, zf),
                local_point=pa.v(0), point_local=(pb.v(0), pb.v(0)), point_id=(zi, zi))


def evaluate(geo, fa, fb, xfa, xfb, linear_slop):
    """collide.evaluate for one contact per lane."""
    pa, pb = Shape(geo, fa), Shape(geo, fb)
    pp = collide_polygons(pa, xfa, pb, xfb, linear_slop)
    pc = collide_polygon_circle(pa, xfa, pb, xfb)
    cc = collide_circles(pa, xfa, pb, xfb)
    is_pp = (pa.type == POLYGON) & (pb.type == POLYGON)
    is_pc = (pa.type == POLYGON) & (pb.type == CIRCLE)
    pick = lambda a, b, c: jnp.where(is_pp, a, jnp.where(is_pc, b, c))  # noqa: E731
    return jax.tree.map(pick, pp, pc, cc)


# ---------------------------------------------------------------- the kernel

def _kernel(cfg, geo, fin, iin, fo, io, xo, sf, vr):
    del fin, iin  # aliased with fo / io
    s = cfg.settings
    B, C = cfg.B, cfg.C
    blk = fo.shape[-1]
    lane = jax.lax.broadcasted_iota(I32, (blk,), 0)
    bf, cf_ = Tab(fo, BF, B, lane), Tab(fo, CF, C, lane, len(BF) * B)
    bi, ci = Tab(io, BI, B, lane), Tab(io, CI, C, lane, len(BI) * B)
    sc = Tab(sf, SB, B, lane)
    ORD = len(SB) * B  # island order rows (NI + 1)
    IMP = ORD + NI + 1  # island impulses ni0, ni1, ti0, ti1 per island row
    xs = len(BF) * B + len(CF) * C  # float scalars
    min_alpha, dt = fo[xs, :], fo[xs + 1, :]
    xsi = len(BI) * B + len(CI) * C
    go = io[xsi, :] != 0
    ms = io[xsi + 1, :]
    n_live = io[xsi + 2, :]
    zf = jnp.zeros((blk,), jnp.float32)
    zi = jnp.zeros((blk,), I32)
    body_of = lambda f: geo[f * NG + G_BODY].astype(I32)  # noqa: E731
    gbody = lambda f, b: bf.get(f, b)  # noqa: E731
    btype = lambda b: bi.get("type", b)  # noqa: E731

    # outputs: report rows (ni0, ni1, ti0, ti1, solved per contact), in_isl per body, walk
    def zero_x(r, _):
        xo[r, :] = zf
        return 0
    jax.lax.fori_loop(0, 5 * C + B + 1, zero_x, 0)

    # ---- 1. every body advanced to minAlpha; transforms of the advanced and of the current sweep
    def advance(b, _):
        r = lambda f: bf.row(f, b)  # noqa: E731
        ac0x, ac0y, aa0, aal0 = T._advance(r("c0x"), r("c0y"), r("a0"), r("al0"), r("cx"), r("cy"), r("a"), min_alpha)
        axf = T._sync_xf(ac0x, ac0y, aa0, r("lcx"), r("lcy"))
        oxf = T._sync_xf(r("cx"), r("cy"), r("a"), r("lcx"), r("lcy"))
        for f, v in zip(SB[:12], (ac0x, ac0y, aa0, aal0, *axf, *oxf)):
            sc.set_row(f, b, v)
        for f in ("flip", "ino", "tou", "newb", "seen"):
            sc.set_row(f, b, zf)
        return 0
    jax.lax.fori_loop(0, B, advance, 0)
    axf_of = lambda b: tuple(sc.get(f, b) for f in ("axpx", "axpy", "axqc", "axqs"))  # noqa: E731

    def manifold(slot):
        fa, fb = ci.get("fa", slot), ci.get("fb", slot)
        return evaluate(geo, fa, fb, axf_of(body_of(fa)), axf_of(body_of(fb)), s.linear_slop)

    def update(slot, man, mask):
        """toi._update_contacts on one slot per lane where mask; returns the touching flip."""
        new_count = man["count"]
        touching = new_count > 0
        old_n = ci.get("count", slot)
        oid = (ci.get("pid0", slot), ci.get("pid1", slot))
        oni = (cf_.get("ni0", slot), cf_.get("ni1", slot))
        oti = (cf_.get("ti0", slot), cf_.get("ti1", slot))
        ni, ti = [], []
        for i in range(2):
            nid = man["point_id"][i]
            m0 = (old_n > 0) & (oid[0] == nid)
            m1 = (old_n > 1) & (oid[1] == nid) & ~m0
            has = i < new_count
            ni.append(jnp.where(has, jnp.where(m0, oni[0], jnp.where(m1, oni[1], F32(0))), F32(0)))
            ti.append(jnp.where(has, jnp.where(m0, oti[0], jnp.where(m1, oti[1], F32(0))), F32(0)))
        km = mask & (new_count > 0)
        old_t = ci.get("touch", slot) != 0
        flip = mask & (touching != old_t)
        put_i = lambda f, v, mk: ci.set(f, slot, jnp.where(mk, v, ci.get(f, slot)))  # noqa: E731
        put_f = lambda f, v, mk: cf_.set(f, slot, jnp.where(mk, v, cf_.get(f, slot)))  # noqa: E731
        put_i("touch", touching.astype(I32), mask)
        put_i("count", new_count, mask)
        put_i("mtype", man["mtype"], km)
        put_f("lnx", man["local_normal"][0], km)
        put_f("lny", man["local_normal"][1], km)
        put_f("lpx", man["local_point"][0], km)
        put_f("lpy", man["local_point"][1], km)
        (p0x, p0y), (p1x, p1y) = man["point_local"]
        for f, v in zip(("p0x", "p0y", "p1x", "p1y"), (p0x, p0y, p1x, p1y)):
            put_f(f, v, km)
        put_i("pid0", man["point_id"][0], km)
        put_i("pid1", man["point_id"][1], km)
        for f, v in zip(("ni0", "ni1", "ti0", "ti1"), (*ni, *ti)):
            put_f(f, v, mask)
        return flip

    def mark_flip(slot, flip):
        for b in (body_of(ci.get("fa", slot)), body_of(ci.get("fb", slot))):
            sc.set("flip", b, jnp.where(flip, F32(1), sc.get("flip", b)))

    def pair_rule(a, b):
        dyn_a, dyn_b = btype(a) == DYNAMIC, btype(b) == DYNAMIC
        bul_a, bul_b = bi.get("bullet", a) != 0, bi.get("bullet", b) != 0
        stock = bul_a | ~dyn_a | bul_b | ~dyn_b
        if not s.unity_toi:
            return stock
        rule = (bul_a | bul_b) | jnp.where(~dyn_b, bi.get("cont", a) != 0,
                                           jnp.where(~dyn_a, bi.get("cont", b) != 0, False))
        return stock & rule

    # ---- 2, 3. minContact->Update (e = 0), then the island walk over bodyA's edges, then bodyB's (newest first);
    # one loop so the narrow phase and the contact update appear once in the kernel
    A = body_of(ci.get("fa", ms))
    Bb = body_of(ci.get("fb", ms))
    dynA, dynB = btype(A) == DYNAMIC, btype(Bb) == DYNAMIC
    old_tcount = ci.get("tcount", ms)
    sf[ORD, :] = ms.astype(jnp.float32)
    sf[ORD + NI, :] = zf  # the walk's additions so far
    nl_max = jnp.max(n_live)

    def step(e, carry):
        is0 = e == 0
        first = e <= nl_max
        k = jnp.maximum(jnp.where(first, e - 1, e - 1 - nl_max), 0)
        slot = jnp.where(is0, ms, ci.row("perm", k))
        fa, fb = ci.get("fa", slot), ci.get("fb", slot)
        sa, sb = body_of(fa), body_of(fb)
        owner = jnp.where(first, A, Bb)
        live = (k < n_live) & (ci.get("pair", slot) >= 0)
        valid = live & ((sa == owner) | (sb == owner)) & jnp.where(first, dynA, dynB)
        valid = valid & (first | ~(((sa == A) | (sb == A)) & dynA))
        valid = valid & carry[2]  # walk (go & minContact touching), known after e = 0
        other = jnp.where(sa == owner, sb, sa)
        dyn_skip = (btype(other) == DYNAMIC) & (bi.get("bullet", owner) == 0) & (bi.get("bullet", other) == 0)
        elig = valid & (slot != ms) & ~dyn_skip
        if s.unity_toi:
            elig = elig & pair_rule(owner, other)

        def visit(carry):
            cnt_c, cnt_b, t_min = carry
            man = manifold(slot)
            touch = man["count"] > 0
            t_min = jnp.where(is0, go & touch, t_min)
            add_u = elig & touch
            newb_u = add_u & (other != A) & (other != Bb) & (sc.get("seen", other) == 0)
            sc.set("seen", other, jnp.where(newb_u, F32(1), sc.get("seen", other)))
            proc = (1 + cnt_c < MAX_TOI_CONTACTS) & (2 + cnt_b < 2 * MAX_TOI_CONTACTS)
            upd_e, add_e, newb_e = elig & proc, add_u & proc, newb_u & proc
            upd = jnp.where(is0, go, upd_e)
            flip = update(slot, man, upd)
            mark_flip(slot, flip)
            en = ci.get("enabled", slot)
            ci.set("enabled", slot, jnp.where(is0, jnp.where(go, t_min.astype(I32), en), jnp.where(upd_e, 1, en)))
            for f, mk in (("ino", add_e), ("tou", upd_e), ("newb", newb_e)):
                sc.set(f, other, jnp.where(mk, F32(1), sc.get(f, other)))
            n_add = sf[ORD + NI, :].astype(I32) + add_e.astype(I32)
            row = ORD + jnp.clip(n_add, 0, NI - 1)
            sf[row, lane] = jnp.where(add_e, slot.astype(jnp.float32), sf[row, lane])
            sf[ORD + NI, :] = n_add.astype(jnp.float32)
            return cnt_c + add_u.astype(I32), cnt_b + newb_u.astype(I32), t_min
        ev = jnp.where(is0, go, elig)
        return jax.lax.cond(jnp.max(ev.astype(I32)) > 0, visit, lambda c: c, carry)

    _, _, t_min = jax.lax.fori_loop(0, 1 + 2 * nl_max, step, (zi, zi, jnp.zeros((blk,), bool)))
    walk = t_min
    n_add = sf[ORD + NI, :].astype(I32)
    ci.set("tcount", ms, jnp.where(go, old_tcount + 1, old_tcount))  # ++toiCount, toi flag cleared
    ci.set("flag", ms, jnp.where(go, 0, ci.get("flag", ms)))

    # ---- 4. body sweeps after the walk, wakes, island membership
    def sweeps(b, _):
        r = lambda f: bf.row(f, b)  # noqa: E731
        q = lambda f: sc.row(f, b)  # noqa: E731
        pair_b = ((A == b) | (Bb == b)) & go
        ino, tou = q("ino") != 0, q("tou") != 0
        keep_adv = (pair_b & t_min) | ino
        restore = (pair_b & ~t_min) | (tou & ~ino)
        for f, af in (("c0x", "ac0x"), ("c0y", "ac0y"), ("a0", "aa0"), ("al0", "aal0")):
            bf.set_row(f, b, jnp.where(keep_adv, q(af), r(f)))
        bf.set_row("cx", b, jnp.where(keep_adv, q("ac0x"), r("cx")))
        bf.set_row("cy", b, jnp.where(keep_adv, q("ac0y"), r("cy")))
        bf.set_row("a", b, jnp.where(keep_adv, q("aa0"), r("a")))
        for f, af, of in zip(("pxx", "pxy", "qc", "qs"), SB[4:8], SB[8:12]):
            bf.set_row(f, b, jnp.where(keep_adv, q(af), jnp.where(restore, q(of), r(f))))
        typ = bi.row("type", b)
        wake = (q("flip") != 0) | (pair_b & t_min) | ((q("newb") != 0) & (typ != STATIC))
        awake = bi.row("awake", b) != 0
        bi.set_row("awake", b, (awake | wake).astype(I32))
        bf.set_row("st", b, jnp.where(wake & ~awake, F32(0), r("st")))
        xo[5 * C + b, :] = ((pair_b | ino) & walk).astype(jnp.float32)
        for f, src in zip(("pcx", "pcy", "pa", "pvx", "pvy", "pw"), ("cx", "cy", "a", "vx", "vy", "w")):
            sc.set_row(f, b, bf.row(src, b))
        return 0
    jax.lax.fori_loop(0, B, sweeps, 0)
    xo[5 * C + B, :] = walk.astype(jnp.float32)
    n_isl = jnp.where(walk, 1 + n_add, 0)
    in_isl = lambda b: xo[5 * C + b, :] != 0  # noqa: E731

    # ---- 5. b2Island::SolveTOI (positions and velocities in scratch copies)
    pos = tuple(sc.bodies(f) for f in ("pcx", "pcy", "pa"))
    vel = tuple(sc.bodies(f) for f in ("pvx", "pvy", "pw"))
    n_max = jnp.max(n_isl)
    radius = lambda f: geo[f * NG + G_RAD]  # noqa: E731

    def contact_row(k):
        slot = sf[ORD + k, :].astype(I32)
        fa, fb = ci.get("fa", slot), ci.get("fb", slot)
        ia, ib = body_of(fa), body_of(fb)
        return slot, fa, fb, ia, ib

    def position_row(k):
        slot, fa, fb, ia, ib = contact_row(k)
        toi_a = lambda i: (i == A) | (i == Bb)  # noqa: E731
        g = lambda f: cf_.get(f, slot)  # noqa: E731
        return dict(ia=ia, ib=ib, mA=jnp.where(toi_a(ia), gbody("im", ia), F32(0)),
                    iA=jnp.where(toi_a(ia), gbody("ii", ia), F32(0)),
                    mB=jnp.where(toi_a(ib), gbody("im", ib), F32(0)),
                    iB=jnp.where(toi_a(ib), gbody("ii", ib), F32(0)),
                    pc_count=ci.get("count", slot), mtype=ci.get("mtype", slot), lnx=g("lnx"), lny=g("lny"),
                    lpx=g("lpx"), lpy=g("lpy"), plx=(g("p0x"), g("p1x")), ply=(g("p0y"), g("p1y")),
                    lcAx=gbody("lcx", ia), lcAy=gbody("lcy", ia), lcBx=gbody("lcx", ib), lcBy=gbody("lcy", ib),
                    radA=radius(fa), radB=radius(fb))

    okay_sep = F32(F32(-1.5) * s.linear_slop)

    def ppass(c):
        it, done = c

        def one(k, min_sep):
            row = position_row(k)
            live = (k < n_isl) & ~done
            A0, B0 = K._get2(pos, row["ia"]), K._get2(pos, row["ib"])
            cA, aA, cB, aB, sep = cs.solve_position_one(row, pos, s, s.toi_baumgarte)
            K._set2(pos, row["ia"], row["ib"], K._masked((cA, aA), A0, live), K._masked((cB, aB), B0, live), live)
            return jnp.where(live, m.bmin(min_sep, sep), min_sep)
        min_sep = jax.lax.fori_loop(0, n_max, one, zf)
        return it + 1, done | (min_sep >= okay_sep)

    jax.lax.while_loop(lambda c: (c[0] < 20) & (jnp.max((~c[1]).astype(I32)) > 0), ppass, (jnp.int32(0), ~walk))

    # leap of faith: the TOI bodies' sweep start is the solved pose
    def leap(b, _):
        pair_b = ((A == b) | (Bb == b)) & walk
        for f, src in (("c0x", "pcx"), ("c0y", "pcy"), ("a0", "pa")):
            bf.set_row(f, b, jnp.where(pair_b, sc.row(src, b), bf.row(f, b)))
        return 0
    jax.lax.fori_loop(0, B, leap, 0)

    # velocity constraints (contact_solver.prepare, no warm start) at the solved poses
    def prep(k, _):
        slot, fa, fb, ia, ib = contact_row(k)
        vc = _prepare_row(cfg, bf, ci, cf_, gbody, slot, fa, fb, ia, ib, pos, vel, radius)
        for name, off, w, kind, vec in VLAY.fields:
            vals = vc[name] if vec else (vc[name],)
            for q in range(w):
                vr[k, off + q, :] = jnp.asarray(vals[q]).astype(jnp.float32)
        for q in range(4):
            sf[IMP + 4 * k + q, :] = zf
        return 0
    jax.lax.fori_loop(0, n_max, prep, 0)

    def cvel(k, _):
        c = K.load_row(vr, k, VLAY)
        live = k < n_isl
        ni = (sf[IMP + 4 * k, :], sf[IMP + 4 * k + 1, :])
        ti = (sf[IMP + 4 * k + 2, :], sf[IMP + 4 * k + 3, :])
        A0, B0 = K._get2(vel, c["ia"]), K._get2(vel, c["ib"])
        (vA, wA, vB, wB), nik, tik = cs._solve_one(c, vel, ni, ti)
        for q, (new, old) in enumerate(((nik[0], ni[0]), (nik[1], ni[1]), (tik[0], ti[0]), (tik[1], ti[1]))):
            sf[IMP + 4 * k + q, :] = jnp.where(live, new, old)
        K._set2(vel, c["ia"], c["ib"], K._masked((vA, wA), A0, live), K._masked((vB, wB), B0, live), live)
        return 0

    def vel_iter(_, x):
        jax.lax.fori_loop(0, n_max, cvel, 0)
        return x
    jax.lax.fori_loop(0, s.velocity_iterations, vel_iter, 0)

    # integrate the island bodies over (1 - minAlpha) dt with the step's clamps; SynchronizeTransform
    h = m.mul(F32(1.0) - min_alpha, dt)
    max_t = m.mul(s.max_translation, dt) if s.clamps_per_second else s.max_translation
    max_r = m.mul(s.max_rotation, dt) if s.clamps_per_second else s.max_rotation
    def integrate(b, _):
        q = lambda f: sc.row(f, b)  # noqa: E731
        vx, vy, w = q("pvx"), q("pvy"), q("pw")
        cx, cy, a = q("pcx"), q("pcy"), q("pa")
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
        px, py, qc, qs = T._sync_xf(ncx, ncy, na, bf.row("lcx", b), bf.row("lcy", b))
        on = in_isl(b)
        for f, v in (("cx", ncx), ("cy", ncy), ("a", na), ("vx", vx), ("vy", vy), ("w", w), ("pxx", px),
                     ("pxy", py), ("qc", qc), ("qs", qs)):
            bf.set_row(f, b, jnp.where(on, v, bf.row(f, b)))
        return 0
    jax.lax.fori_loop(0, B, integrate, 0)

    # b2Island::Report: island contacts' impulses (velocity constraint pointCount points), solved flags
    def report(k, _):
        slot = sf[ORD + k, :].astype(I32)
        live = k < n_isl
        cnt = vr[k, dict((f[0], f[1]) for f in VLAY.fields)["count"], :].astype(I32)
        tgt = jnp.where(live, slot, C)
        for q in range(4):
            j = q % 2
            v = jnp.where(j < cnt, sf[IMP + 4 * k + q, :], F32(0))
            row = q * C + jnp.clip(tgt, 0, C - 1)
            xo[row, lane] = jnp.where(live, v, xo[row, lane])
        row = 4 * C + jnp.clip(tgt, 0, C - 1)
        xo[row, lane] = jnp.where(live, F32(1), xo[row, lane])
        return 0
    jax.lax.fori_loop(0, n_max, report, 0)


def _prepare_row(cfg, bf, ci, cf_, gbody, slot, fa, fb, ia, ib, pos, vel, radius):
    """contact_solver.prepare for one row per lane (warm start off, dt_ratio 1): the same operations."""
    s = cfg.settings
    mA, mB = gbody("im", ia), gbody("im", ib)
    iA, iB = gbody("ii", ia), gbody("ii", ib)
    lcAx, lcAy, lcBx, lcBy = gbody("lcx", ia), gbody("lcy", ia), gbody("lcx", ib), gbody("lcy", ib)
    rA_, rB_ = radius(fa), radius(fb)
    mtype, count = ci.get("mtype", slot), ci.get("count", slot)
    g = lambda f: cf_.get(f, slot)  # noqa: E731
    lnx, lny, lpx, lpy = g("lnx"), g("lny"), g("lpx"), g("lpy")
    pl_ = ((g("p0x"), g("p0y")), (g("p1x"), g("p1y")))
    cAx, cAy, aA = pos[0].get(ia), pos[1].get(ia), pos[2].get(ia)
    cBx, cBy, aB = pos[0].get(ib), pos[1].get(ib), pos[2].get(ib)
    xfA = cs._xf(cAx, cAy, aA, lcAx, lcAy)
    xfB = cs._xf(cBx, cBy, aB, lcBx, lcBy)
    pAx, pAy = m.xf_mul(*xfA, lpx, lpy)
    pBx, pBy = m.xf_mul(*xfB, pl_[0][0], pl_[0][1])
    dx, dy = pBx - pAx, pBy - pAy
    far = m.dot(dx, dy, dx, dy) > m.mul(m.EPS, m.EPS)
    nnx, nny, _ = m.normalize(dx, dy)
    cnx, cny = jnp.where(far, nnx, F32(1)), jnp.where(far, nny, F32(0))
    ccAx, ccAy = pAx + m.mul(rA_, cnx), pAy + m.mul(rA_, cny)
    ccBx, ccBy = pBx - m.mul(rB_, cnx), pBy - m.mul(rB_, cny)
    circ_pt = (m.mul(F32(0.5), ccAx + ccBx), m.mul(F32(0.5), ccAy + ccBy))
    fAnx, fAny = m.rot(xfA[2], xfA[3], lnx, lny)
    fAplx, fAply = m.xf_mul(*xfA, lpx, lpy)
    fBnx, fBny = m.rot(xfB[2], xfB[3], lnx, lny)
    fBplx, fBply = m.xf_mul(*xfB, lpx, lpy)
    is_c, is_a = mtype == CIRCLES, mtype == FACE_A
    pts = []
    for j in range(2):
        qx, qy = m.xf_mul(*xfB, pl_[j][0], pl_[j][1])
        d = m.dot(qx - fAplx, qy - fAply, fAnx, fAny)
        s_ = rA_ - d
        aAx, aAy = qx + m.mul(s_, fAnx), qy + m.mul(s_, fAny)
        aBx, aBy = qx - m.mul(rB_, fAnx), qy - m.mul(rB_, fAny)
        fa_pt = (m.mul(F32(0.5), aAx + aBx), m.mul(F32(0.5), aAy + aBy))
        qx, qy = m.xf_mul(*xfA, pl_[j][0], pl_[j][1])
        d = m.dot(qx - fBplx, qy - fBply, fBnx, fBny)
        s_ = rB_ - d
        bBx, bBy = qx + m.mul(s_, fBnx), qy + m.mul(s_, fBny)
        bAx, bAy = qx - m.mul(rA_, fBnx), qy - m.mul(rA_, fBny)
        fb_pt = (m.mul(F32(0.5), bAx + bBx), m.mul(F32(0.5), bAy + bBy))
        pts.append(tuple(jnp.where(is_c, cp, jnp.where(is_a, ap, bp)) for cp, ap, bp in zip(circ_pt, fa_pt, fb_pt)))
    nx = jnp.where(is_c, cnx, jnp.where(is_a, fAnx, -fBnx))
    ny = jnp.where(is_c, cny, jnp.where(is_a, fAny, -fBny))
    tx, ty = m.cross_vs(nx, ny, F32(1.0))
    vAx, vAy, wA = vel[0].get(ia), vel[1].get(ia), vel[2].get(ia)
    vBx, vBy, wB = vel[0].get(ib), vel[1].get(ib), vel[2].get(ib)
    friction, restitution = g("fr"), g("rs")
    rA, rB, nmass, tmass, bias = [], [], [], [], []
    for j in range(2):
        px_, py_ = pts[j]
        rax, ray = px_ - cAx, py_ - cAy
        rbx, rby = px_ - cBx, py_ - cBy
        rnA, rnB = m.cross(rax, ray, nx, ny), m.cross(rbx, rby, nx, ny)
        kn = ((mA + mB) + m.mul3(iA, rnA, rnA)) + m.mul3(iB, rnB, rnB)
        rtA, rtB = m.cross(rax, ray, tx, ty), m.cross(rbx, rby, tx, ty)
        kt = ((mA + mB) + m.mul3(iA, rtA, rtA)) + m.mul3(iB, rtB, rtB)
        wBx, wBy = m.cross_sv(wB, rbx, rby)
        wAx, wAy = m.cross_sv(wA, rax, ray)
        vrel = m.dot(nx, ny, ((vBx + wBx) - vAx) - wAx, ((vBy + wBy) - vAy) - wAy)
        rA.append((rax, ray))
        rB.append((rbx, rby))
        nmass.append(m.inv_if_pos(kn))
        tmass.append(m.inv_if_pos(kt))
        bias.append(jnp.where(vrel < -s.velocity_threshold, m.mul(-restitution, vrel), F32(0)))
    rn1A, rn1B = m.cross(*rA[0], nx, ny), m.cross(*rB[0], nx, ny)
    rn2A, rn2B = m.cross(*rA[1], nx, ny), m.cross(*rB[1], nx, ny)
    k11 = ((mA + mB) + m.mul3(iA, rn1A, rn1A)) + m.mul3(iB, rn1B, rn1B)
    k22 = ((mA + mB) + m.mul3(iA, rn2A, rn2A)) + m.mul3(iB, rn2B, rn2B)
    k12 = ((mA + mB) + m.mul3(iA, rn1A, rn2A)) + m.mul3(iB, rn1B, rn2B)
    safe = m.mul(k11, k11) < m.mul(F32(1000.0), m.mul(k11, k22) - m.mul(k12, k12))
    two = count == 2
    vc_count = jnp.where(two & ~safe, 1, count)
    det = m.inv_or_zero(m.mul(k11, k22) - m.mul(k12, k12))
    use_k = two & safe
    zero = jnp.zeros_like(k11)
    uk = lambda v: jnp.where(use_k, v, zero)  # noqa: E731
    return dict(ia=ia, ib=ib, mA=mA, mB=mB, iA=iA, iB=iB, nx=nx, ny=ny, rAx=(rA[0][0], rA[1][0]),
                rAy=(rA[0][1], rA[1][1]), rBx=(rB[0][0], rB[1][0]), rBy=(rB[0][1], rB[1][1]), count=vc_count,
                tangent_speed=g("ts"), tmass=tuple(tmass), friction=friction, nmass=tuple(nmass), bias=tuple(bias),
                K=(uk(k11), uk(k12), uk(k12), uk(k22)),
                NM=(uk(m.mul(det, k22)), uk(m.mul(-det, k12)), uk(m.mul(-det, k12)), uk(m.mul(det, k11))))


def _call(cfg, geo, fin, iin):
    N = fin.shape[0]
    blk = cfg.block
    Np = -(-N // blk) * blk

    fin, iin = jax.lax.optimization_barrier((fin, iin))
    B, C = cfg.B, cfg.C
    # K.SORT_LANES: worlds with an event first, grouped by live contacts (the walk's loop count)
    xsi = len(BI) * B + len(CI) * C
    order = K.lane_order(jnp.where(iin[:, xsi] != 0, iin[:, xsi + 2], 1 << 20), N, blk)
    fin, iin = (K.to_minor(x, N, Np, order) for x in (fin, iin))
    spec = lambda rows: pl.BlockSpec((rows, blk), lambda i: (0, i))  # noqa: E731
    shapes = [(fin.shape[0], Np, jnp.float32), (iin.shape[0], Np, jnp.int32), (5 * C + B + 1, Np, jnp.float32),
              (len(SB) * B + NI + 1 + 4 * NI, Np, jnp.float32)]
    kw = {}
    if not cfg.interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=cfg.num_warps, num_stages=1)
    with K.triton_mode(not cfg.interpret), K.ref_pick():
        fo, io, xo, _, _ = pl.pallas_call(
            partial(_kernel, cfg), grid=(Np // blk,),
            in_specs=[pl.BlockSpec(geo.shape, lambda i: (0,)), spec(fin.shape[0]), spec(iin.shape[0])],
            out_specs=[spec(r) for r, _, _ in shapes] + [pl.BlockSpec((NI, VLAY.width, blk), lambda i: (0, 0, i))],
            out_shape=[jax.ShapeDtypeStruct((r, n), d) for r, n, d in shapes]
            + [jax.ShapeDtypeStruct((NI, VLAY.width, Np), jnp.float32)],
            input_output_aliases={1: 0, 2: 1}, interpret=cfg.interpret, name="box2d_toi_event", **kw)(geo, fin, iin)
    return tuple(K.from_minor(x, N, order) for x in (fo, io, xo))


@jax.custom_batching.custom_vmap
def _event_call(cfg_box, geo, fin, iin):
    fo, io, xo = _call(cfg_box.cfg, geo, fin[None], iin[None])
    return fo[0], io[0], xo[0]


@_event_call.def_vmap
def _event_rule(axis_size, in_batched, cfg_box, geo, fin, iin):
    if in_batched[1]:
        raise NotImplementedError("gpu toi event: fixture geometry must be shared by all worlds")
    fin, iin = (a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((fin, iin), in_batched[2:]))
    return _call(cfg_box.cfg, geo, fin, iin), (True, True, True)


class _Static:
    def __init__(self, cfg):
        self.cfg = cfg


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.cfg), lambda cfg, _: _Static(cfg))


def geometry(model):
    """The shared per-fixture table the kernel gathers from (flat, NG values per fixture)."""
    f32 = lambda x: jnp.asarray(x).astype(jnp.float32)  # noqa: E731
    v, n = model.verts, model.normals
    g = jnp.concatenate([v[..., 0], v[..., 1], n[..., 0], n[..., 1], f32(model.count)[:, None],
                         model.radius[:, None], f32(model.shape_type)[:, None], f32(model.fixture_body)[:, None]], 1)
    return g.reshape(-1)


def event(model, state, tst, settings, dt, go, min_slot, min_alpha, axis_name=None, block=None, num_warps=None):
    """toi._event with the kernel: same arguments and results."""
    from . import pairs as gpu_pairs
    from . import solve as G
    ct, bd = state.contacts, state.bodies
    B, C = bd.type.shape[0], ct.pair.shape[0]
    fbody = model.fixture_body
    perm, _, n_live = T._list_order(ct)
    f32 = lambda x: jnp.asarray(x).astype(jnp.float32)  # noqa: E731
    i32 = lambda x: jnp.asarray(x).astype(jnp.int32)  # noqa: E731
    fin = jnp.concatenate([
        bd.c0[:, 0], bd.c0[:, 1], bd.a0, bd.c[:, 0], bd.c[:, 1], bd.a, bd.v[:, 0], bd.v[:, 1], bd.w,
        bd.xf_p[:, 0], bd.xf_p[:, 1], bd.xf_q[:, 0], bd.xf_q[:, 1], bd.local_center[:, 0], bd.local_center[:, 1],
        bd.sleep_time, bd.inv_mass, bd.inv_inertia, tst.alpha0,
        ct.local_normal[:, 0], ct.local_normal[:, 1], ct.local_point[:, 0], ct.local_point[:, 1],
        ct.point_local[:, 0, 0], ct.point_local[:, 0, 1], ct.point_local[:, 1, 0], ct.point_local[:, 1, 1],
        ct.normal_impulse[:, 0], ct.normal_impulse[:, 1], ct.tangent_impulse[:, 0], ct.tangent_impulse[:, 1],
        ct.friction, ct.restitution, ct.tangent_speed, f32(min_alpha)[None], f32(dt)[None]])
    iin = jnp.concatenate([
        i32(bd.type), i32(bd.awake), i32(bd.bullet), i32(bd.continuous),
        i32(ct.pair), i32(ct.fa), i32(ct.fb), i32(ct.touching), i32(ct.mtype), i32(ct.count), i32(ct.point_id[:, 0]),
        i32(ct.point_id[:, 1]), i32(tst.count), i32(tst.flag), i32(tst.enabled), i32(perm),
        i32(go)[None], i32(min_slot)[None], i32(n_live)[None]])
    cfg = Cfg(settings=settings, B=B, C=C, block=block or G.BLOCK, num_warps=num_warps or G.NUM_WARPS,
              interpret=G._interpret())
    fo, io, xo = _event_call(_Static(cfg), geometry(model), fin, iin)
    F = lambda q, n: fo[q * n:(q + 1) * n]  # noqa: E731
    Fc = lambda q: fo[len(BF) * B + q * C: len(BF) * B + (q + 1) * C]  # noqa: E731
    Ib = lambda q: io[q * B:(q + 1) * B]  # noqa: E731
    Ic = lambda q: io[len(BI) * B + q * C: len(BI) * B + (q + 1) * C]  # noqa: E731
    bfi = {k: F(q, B) for q, k in enumerate(BF)}
    cfi = {k: Fc(q) for q, k in enumerate(CF)}
    cii = {k: Ic(q) for q, k in enumerate(CI)}
    bd = bd._replace(c0=jnp.stack([bfi["c0x"], bfi["c0y"]], -1), a0=bfi["a0"],
                     c=jnp.stack([bfi["cx"], bfi["cy"]], -1), a=bfi["a"], v=jnp.stack([bfi["vx"], bfi["vy"]], -1),
                     w=bfi["w"], xf_p=jnp.stack([bfi["pxx"], bfi["pxy"]], -1),
                     xf_q=jnp.stack([bfi["qc"], bfi["qs"]], -1), sleep_time=bfi["st"], awake=Ib(1) != 0)
    ct = ct._replace(touching=cii["touch"] != 0, count=cii["count"], mtype=cii["mtype"],
                     local_normal=jnp.stack([cfi["lnx"], cfi["lny"]], -1),
                     local_point=jnp.stack([cfi["lpx"], cfi["lpy"]], -1),
                     point_local=jnp.stack([jnp.stack([cfi["p0x"], cfi["p0y"]], -1),
                                            jnp.stack([cfi["p1x"], cfi["p1y"]], -1)], 1),
                     point_id=jnp.stack([cii["pid0"], cii["pid1"]], -1),
                     normal_impulse=jnp.stack([cfi["ni0"], cfi["ni1"]], -1),
                     tangent_impulse=jnp.stack([cfi["ti0"], cfi["ti1"]], -1))
    alpha0, count, flag, enabled = bfi["al0"], cii["tcount"], cii["flag"] != 0, cii["enabled"] != 0
    ni_r = jnp.stack([xo[:C], xo[C:2 * C]], -1)
    ti_r = jnp.stack([xo[2 * C:3 * C], xo[3 * C:4 * C]], -1)
    solved = xo[4 * C:5 * C] != 0
    in_isl = xo[5 * C:5 * C + B] != 0
    walk = xo[5 * C + B] != 0
    state = state._replace(contacts=ct, bodies=bd)

    # toi._event's tail: synchronize fixtures, invalidate the TOIs of displaced bodies, FindNewContacts
    moved = in_isl & (bd.type == DYNAMIC)
    state = gpu_pairs.synchronize_fixtures(model, state, moved)
    ct3 = state.contacts
    touch_moved = moved[fbody[ct3.fa]] | moved[fbody[ct3.fb]]
    flag = flag & ~(touch_moved & walk)
    old_pair, old_key = ct3.pair, ct3.key
    found = gpu_pairs.find_new_contacts(model, state, walk)
    state = jax.tree.map(lambda x, y: jnp.where(walk, x, y), found, state)
    ct4 = state.contacts
    new = (ct4.pair >= 0) & ((old_pair < 0) | (old_key != ct4.key))
    tst = tst._replace(alpha0=alpha0, count=jnp.where(new, 0, count), flag=flag & ~new,
                       enabled=enabled | new, solved=tst.solved | solved,
                       normal_impulse=jnp.where(solved[:, None], ni_r, tst.normal_impulse),
                       tangent_impulse=jnp.where(solved[:, None], ti_r, tst.tangent_impulse),
                       events=tst.events + jnp.asarray(go).astype(I32))
    return state, tst
