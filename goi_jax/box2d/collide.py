"""Narrow phase: b2CollidePolygons, b2CollidePolygonAndCircle, b2CollideCircles (Box2D 2.3.1), vectorized over
contact slots. Loops over vertices become masked (…, 8) arrays; Box2D's strict `>` / `<` running max/min with
first-index ties is jnp.argmax / argmin (first occurrence), and min/max values are exact in any order.

A manifold is a dict of arrays: mtype, count, local_normal (x, y), local_point (x, y), point_local
((x0, y0), (x1, y1)), point_id (k0, k1). Where Box2D returns early it leaves pointCount = 0 and the rest of the old
manifold in place; here count = 0 and the other fields are don't-care.
"""
import jax.numpy as jnp
import numpy as np

from . import math2d as m
from .settings import K_TOL, MAX_FLOAT, MAX_POLYGON_VERTICES
from .shapes import CIRCLE, POLYGON
from .world import CIRCLES, FACE_A, FACE_B

F32 = np.float32
E_VERTEX, E_FACE = 0, 1  # b2ContactFeature::Type
_IDX = jnp.arange(MAX_POLYGON_VERTICES)


def feature_key(index_a, index_b, type_a, type_b):
    """b2ContactID::key: the four uint8 of b2ContactFeature read as a little-endian uint32 (fits int32 here)."""
    return index_a | (index_b << 8) | (type_a << 16) | (type_b << 24)


def _take(a, i):
    """a[..., i] for a per-slot index i (…,) into the last axis."""
    return jnp.take_along_axis(a, i[..., None], axis=-1)[..., 0]


def _find_max_separation(v1x, v1y, n1x, n1y, c1, xf1, v2x, v2y, c2, xf2):
    """b2FindMaxSeparation: (best edge of poly1, its separation)."""
    px, py, qc, qs = m.xf_mul_t_xf(*xf2, *xf1)  # xf = b2MulT(xf2, xf1)
    nx, ny = m.rot(qc[:, None], qs[:, None], n1x, n1y)
    wx, wy = m.xf_mul(px[:, None], py[:, None], qc[:, None], qs[:, None], v1x, v1y)
    sij = m.dot(nx[:, :, None], ny[:, :, None], v2x[:, None, :] - wx[:, :, None], v2y[:, None, :] - wy[:, :, None])
    sij = jnp.where(_IDX[None, None, :] < c2[:, None, None], sij, MAX_FLOAT)
    si = jnp.min(sij, axis=2)
    si = jnp.where(_IDX[None, :] < c1[:, None], si, -MAX_FLOAT)
    best = jnp.argmax(si, axis=1).astype(jnp.int32)
    return best, _take(si, best)


def _clip(p0, p1, id0, id1, nx, ny, offset, vertex_index_a):
    """b2ClipSegmentToLine for two input clip vertices -> (out0, out1, id_out0, id_out1, numOut)."""
    d0 = m.dot(nx, ny, p0[0], p0[1]) - offset
    d1 = m.dot(nx, ny, p1[0], p1[1]) - offset
    in0, in1 = d0 <= 0, d1 <= 0
    cross_ = m.mul(d0, d1) < 0
    interp = d0 / (d0 - d1)
    ix = p0[0] + m.mul(interp, p1[0] - p0[0])
    iy = p0[1] + m.mul(interp, p1[1] - p0[1])
    # VertexA is hitting edgeB: indexA = vertexIndexA, indexB = vIn[0].indexB, typeA vertex, typeB face.
    iid = (vertex_index_a, id0[1], jnp.full_like(id0[2], E_VERTEX), jnp.full_like(id0[3], E_FACE))
    o0 = (jnp.where(in0, p0[0], p1[0]), jnp.where(in0, p0[1], p1[1]))
    oid0 = tuple(jnp.where(in0, a, b) for a, b in zip(id0, id1))
    both = in0 & in1
    o1 = (jnp.where(both, p1[0], ix), jnp.where(both, p1[1], iy))
    oid1 = tuple(jnp.where(both, b, c) for b, c in zip(id1, iid))
    n = in0.astype(jnp.int32) + in1.astype(jnp.int32) + cross_.astype(jnp.int32)
    return o0, o1, oid0, oid1, n


def collide_polygons(sa, xfa, sb, xfb, linear_slop):
    """b2CollidePolygons(manifold, polyA, xfA, polyB, xfB); shape args are (verts_x, verts_y, normals_x, normals_y,
    count, radius) with (C, 8) arrays; transforms are (px, py, c, s)."""
    vax, vay, nax, nay, ca, ra = sa
    vbx, vby, nbx, nby, cb, rb = sb
    total_radius = ra + rb
    edge_a, sep_a = _find_max_separation(vax, vay, nax, nay, ca, xfa, vbx, vby, cb, xfb)
    edge_b, sep_b = _find_max_separation(vbx, vby, nbx, nby, cb, xfb, vax, vay, ca, xfa)
    k_tol = m.mul(K_TOL, linear_slop)
    flip = sep_b > sep_a + k_tol
    sel = lambda b_val, a_val: jnp.where(flip[..., None] if b_val.ndim == 2 else flip, b_val, a_val)
    v1x, v1y, n1x, n1y, c1 = sel(vbx, vax), sel(vby, vay), sel(nbx, nax), sel(nby, nay), sel(cb, ca)
    v2x, v2y, n2x, n2y, c2 = sel(vax, vbx), sel(vay, vby), sel(nax, nbx), sel(nay, nby), sel(ca, cb)
    xf1 = tuple(sel(b, a) for a, b in zip(xfa, xfb))
    xf2 = tuple(sel(a, b) for a, b in zip(xfa, xfb))
    edge1 = sel(edge_b, edge_a)

    # b2FindIncidentEdge: reference normal in poly2's frame, then the most anti-parallel poly2 normal.
    rnx, rny = m.rot(xf1[2], xf1[3], _take(n1x, edge1), _take(n1y, edge1))
    rnx, rny = m.rot_t(xf2[2], xf2[3], rnx, rny)
    dots = m.dot(rnx[:, None], rny[:, None], n2x, n2y)
    dots = jnp.where(_IDX[None, :] < c2[:, None], dots, MAX_FLOAT)
    i1 = jnp.argmin(dots, axis=1).astype(jnp.int32)
    i2 = jnp.where(i1 + 1 < c2, i1 + 1, 0)
    inc0 = m.xf_mul(*xf2, _take(v2x, i1), _take(v2y, i1))
    inc1 = m.xf_mul(*xf2, _take(v2x, i2), _take(v2y, i2))
    face, vert = jnp.full_like(i1, E_FACE), jnp.full_like(i1, E_VERTEX)
    id0, id1 = (edge1, i1, face, vert), (edge1, i2, face, vert)

    iv1 = edge1
    iv2 = jnp.where(edge1 + 1 < c1, edge1 + 1, 0)
    v11 = (_take(v1x, iv1), _take(v1y, iv1))
    v12 = (_take(v1x, iv2), _take(v1y, iv2))
    tx, ty, _ = m.normalize(v12[0] - v11[0], v12[1] - v11[1])  # localTangent
    lnx, lny = m.cross_vs(tx, ty, F32(1.0))  # localNormal = b2Cross(localTangent, 1.0f)
    ppx, ppy = m.mul(F32(0.5), v11[0] + v12[0]), m.mul(F32(0.5), v11[1] + v12[1])  # planePoint
    tgx, tgy = m.rot(xf1[2], xf1[3], tx, ty)  # tangent
    nx, ny = m.cross_vs(tgx, tgy, F32(1.0))  # normal
    w11 = m.xf_mul(*xf1, *v11)
    w12 = m.xf_mul(*xf1, *v12)
    front = m.dot(nx, ny, *w11)
    side1 = -m.dot(tgx, tgy, *w11) + total_radius
    side2 = m.dot(tgx, tgy, *w12) + total_radius

    c10, c11, cid10, cid11, np1 = _clip(inc0, inc1, id0, id1, -tgx, -tgy, side1, iv1)
    c20, c21, cid20, cid21, np2 = _clip(c10, c11, cid10, cid11, tgx, tgy, side2, iv2)
    ok = (np1 >= 2) & (np2 >= 2)

    pts, ids, keep = [], [], []
    for p, pid in ((c20, cid20), (c21, cid21)):
        sep = m.dot(nx, ny, *p) - front
        keep.append(sep <= total_radius)
        pts.append(m.xf_mul_t(*xf2, *p))
        ia, ib, ta, tb = pid
        ids.append(jnp.where(flip, feature_key(ib, ia, tb, ta), feature_key(ia, ib, ta, tb)))
    k0, k1 = keep
    count = jnp.where(ok, k0.astype(jnp.int32) + k1.astype(jnp.int32), 0)
    # Compaction: the first kept clip point goes to slot 0.
    p0 = tuple(jnp.where(k0, a, b) for a, b in zip(pts[0], pts[1]))
    id_0 = jnp.where(k0, ids[0], ids[1])
    count = jnp.where((sep_a > total_radius) | (sep_b > total_radius), 0, count)
    return dict(mtype=jnp.where(flip, FACE_B, FACE_A).astype(jnp.int32), count=count, local_normal=(lnx, lny),
                local_point=(ppx, ppy), point_local=(p0, pts[1]), point_id=(id_0, ids[1]))


def collide_polygon_circle(sa, xfa, sb, xfb):
    """b2CollidePolygonAndCircle(manifold, polygonA, xfA, circleB, xfB)."""
    vx, vy, nxs, nys, count, ra = sa
    bpx, bpy, _, _, _, rb = sb
    cx, cy = m.xf_mul(*xfb, bpx[:, 0], bpy[:, 0])
    clx, cly = m.xf_mul_t(*xfa, cx, cy)
    radius = ra + rb
    valid = _IDX[None, :] < count[:, None]
    s = m.dot(nxs, nys, clx[:, None] - vx, cly[:, None] - vy)
    early = jnp.any(valid & (s > radius[:, None]), axis=1)
    s = jnp.where(valid, s, -MAX_FLOAT)
    ni = jnp.argmax(s, axis=1).astype(jnp.int32)
    separation = _take(s, ni)
    i2 = jnp.where(ni + 1 < count, ni + 1, 0)
    v1 = (_take(vx, ni), _take(vy, ni))
    v2 = (_take(vx, i2), _take(vy, i2))
    nfx, nfy = _take(nxs, ni), _take(nys, ni)
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
    pick = lambda a_in, a1, a2, a3: jnp.where(inside, a_in, jnp.where(c1, a1, jnp.where(c2, a2, a3)))
    lnx = pick(nfx, n1x, n2x, nfx)
    lny = pick(nfy, n1y, n2y, nfy)
    lpx = pick(face_c[0], v1[0], v2[0], face_c[0])
    lpy = pick(face_c[1], v1[1], v2[1], face_c[1])
    zero = jnp.zeros_like(ni)
    pl = (bpx[:, 0], bpy[:, 0])
    return dict(mtype=jnp.full_like(ni, FACE_A), count=jnp.where(none, 0, 1), local_normal=(lnx, lny),
                local_point=(lpx, lpy), point_local=(pl, pl), point_id=(zero, zero))


def collide_circles(sa, xfa, sb, xfb):
    """b2CollideCircles."""
    apx, apy, _, _, _, ra = sa
    bpx, bpy, _, _, _, rb = sb
    pa = m.xf_mul(*xfa, apx[:, 0], apy[:, 0])
    pb = m.xf_mul(*xfb, bpx[:, 0], bpy[:, 0])
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    r = ra + rb
    hit = ~(m.dot(dx, dy, dx, dy) > m.mul(r, r))
    zero_i = jnp.zeros(dx.shape, jnp.int32)
    zf = jnp.zeros_like(dx)
    pl = (bpx[:, 0], bpy[:, 0])
    return dict(mtype=jnp.full_like(zero_i, CIRCLES), count=hit.astype(jnp.int32), local_normal=(zf, zf),
                local_point=(apx[:, 0], apy[:, 0]), point_local=(pl, pl), point_id=(zero_i, zero_i))


def evaluate(model, fa, fb, xfa, xfb, linear_slop):
    """b2Contact::Evaluate for each slot (fixture A, fixture B, their bodies' transforms): picks the collider by the
    shape types (polygon A + circle B, polygon pair, circle pair, as b2Contact::Create arranges them)."""
    def shape(f):
        v, n = model.verts[f], model.normals[f]
        return v[..., 0], v[..., 1], n[..., 0], n[..., 1], model.count[f], model.radius[f]
    sa, sb = shape(fa), shape(fb)
    ta, tb = model.shape_type[fa], model.shape_type[fb]
    pp = collide_polygons(sa, xfa, sb, xfb, linear_slop)
    pc = collide_polygon_circle(sa, xfa, sb, xfb)
    cc = collide_circles(sa, xfa, sb, xfb)
    is_pp = (ta == POLYGON) & (tb == POLYGON)
    is_pc = (ta == POLYGON) & (tb == CIRCLE)

    def pick(a, b, c):
        if isinstance(a, tuple):
            return tuple(pick(x, y, z) for x, y, z in zip(a, b, c))
        return jnp.where(is_pp, a, jnp.where(is_pc, b, c))
    return {k: pick(pp[k], pc[k], cc[k]) for k in pp}
