"""Shapes: host-side construction (numpy float32, b2PolygonShape / b2CircleShape v2.3.1 op for op) and AABBs in JAX.

A fixture holds one shape. Polygons store up to 8 vertices and edge normals, padded by repeating vertex 0 (pads are
masked by `count`, never read as geometry). A circle stores its centre in vertex 0 and count 1.
"""
import jax.numpy as jnp
import numpy as np

from . import math2d as m
from .settings import AABB_EXTENSION, AABB_MULTIPLIER, MAX_POLYGON_VERTICES

F32 = np.float32
CIRCLE, POLYGON = 0, 2  # b2Shape::Type (edge 1 and chain 3 are not used by the game)


def _normalize_host(x, y):
    x, y = F32(x), F32(y)
    n = F32(np.sqrt(F32(F32(x * x) + F32(y * y))))
    if n < m.EPS:
        return x, y
    inv = F32(F32(1.0) / n)
    return F32(x * inv), F32(y * inv)


def _edge_normals(v):
    """b2PolygonShape::Set's normals: Normalize(b2Cross(v[i+1] - v[i], 1.0f)) = (e.y, -e.x) normalized."""
    n = len(v)
    out = np.zeros((n, 2), F32)
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        ex, ey = F32(v[j][0] - v[i][0]), F32(v[j][1] - v[i][1])
        out[i] = _normalize_host(F32(F32(1.0) * ey), F32(F32(-1.0) * ex))
    return out


def polygon_raw(vertices):
    """Vertices taken verbatim (strictly convex, CCW), normals as Set derives them (reference capi raw=1)."""
    v = np.asarray(vertices, F32).reshape(-1, 2)
    if not 3 <= len(v) <= MAX_POLYGON_VERTICES:
        raise ValueError("polygon needs 3..8 vertices")
    for i in range(len(v)):
        e = v[(i + 1) % len(v)] - v[i]
        for k in range(2, len(v)):
            d = v[(i + k) % len(v)] - v[i]
            if F32(F32(e[0] * d[1]) - F32(e[1] * d[0])) <= 0:
                raise ValueError("polygon not strictly convex CCW")
    return v, _edge_normals(v)


def polygon_set(vertices, linear_slop):
    """b2PolygonShape::Set v2.3.1: weld points closer than (squared distance) 0.5 * linearSlop, gift-wrap hull from
    the rightmost (then lowest) point, CCW. Raises where Box2D would fall back to SetAsBox(1, 1)."""
    pts = [tuple(F32(c) for c in p) for p in np.asarray(vertices, F32).reshape(-1, 2)[:MAX_POLYGON_VERTICES]]
    weld = F32(F32(0.5) * F32(linear_slop))
    ps = []
    for v in pts:
        if all(F32(F32(F32(v[0] - q[0]) * F32(v[0] - q[0])) + F32(F32(v[1] - q[1]) * F32(v[1] - q[1]))) >= weld
               for q in ps):
            ps.append(v)
    n = len(ps)
    if n < 3:
        raise ValueError("degenerate polygon (Box2D would use SetAsBox(1, 1))")
    i0, x0 = 0, ps[0][0]
    for i in range(1, n):
        x = ps[i][0]
        if x > x0 or (x == x0 and ps[i][1] < ps[i0][1]):
            i0, x0 = i, x
    hull, ih = [], i0
    while True:
        hull.append(ih)
        ie = 0
        for j in range(1, n):
            if ie == ih:
                ie = j
                continue
            rx, ry = F32(ps[ie][0] - ps[hull[-1]][0]), F32(ps[ie][1] - ps[hull[-1]][1])
            vx, vy = F32(ps[j][0] - ps[hull[-1]][0]), F32(ps[j][1] - ps[hull[-1]][1])
            c = F32(F32(rx * vy) - F32(ry * vx))
            if c < 0:
                ie = j
            if c == 0 and F32(F32(vx * vx) + F32(vy * vy)) > F32(F32(rx * rx) + F32(ry * ry)):
                ie = j
        ih = ie
        if ie == i0:
            break
    if len(hull) < 3:
        raise ValueError("degenerate hull")
    v = np.array([ps[i] for i in hull], F32)
    return v, _edge_normals(v)


def box(hx, hy, center=(0.0, 0.0), angle=0.0):
    """b2PolygonShape::SetAsBox(hx, hy, center, angle): vertices and normals rotated by b2Rot(angle), then shifted."""
    hx, hy = F32(hx), F32(hy)
    v = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)], F32)
    n = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], F32)
    s, c = m.host_sin_cos(angle)
    px, py = F32(center[0]), F32(center[1])
    for i in range(4):
        x, y = v[i]
        v[i] = (F32(F32(F32(c * x) - F32(s * y)) + px), F32(F32(F32(s * x) + F32(c * y)) + py))
        x, y = n[i]
        n[i] = (F32(F32(c * x) - F32(s * y)), F32(F32(s * x) + F32(c * y)))
    return v, n


def pad(v, n):
    """(8, 2) vertex and normal arrays, padded with vertex/normal 0."""
    k = len(v)
    vv = np.repeat(v[:1], MAX_POLYGON_VERTICES, axis=0)
    nn = np.repeat(n[:1], MAX_POLYGON_VERTICES, axis=0)
    vv[:k], nn[:k] = v, n
    return vv, nn


# ---- AABBs (JAX)

def fixture_aabbs(model, px, py, c, s):
    """b2Shape::ComputeAABB for every fixture with its body's transform given per fixture ((F,) arrays).
    Returns (lx, ly, ux, uy). Polygons: min/max of b2Mul(xf, v_i), minus/plus the skin radius; circles: centre +- r."""
    vx, vy = model.verts[..., 0], model.verts[..., 1]  # (F, 8)
    wx, wy = m.xf_mul(px[:, None], py[:, None], c[:, None], s[:, None], vx, vy)
    valid = jnp.arange(MAX_POLYGON_VERTICES)[None, :] < model.count[:, None]
    inf = F32(np.inf)
    lx, ly = jnp.min(jnp.where(valid, wx, inf), 1), jnp.min(jnp.where(valid, wy, inf), 1)
    ux, uy = jnp.max(jnp.where(valid, wx, -inf), 1), jnp.max(jnp.where(valid, wy, -inf), 1)
    r = model.radius
    # Circle: p = xf.p + b2Mul(xf.q, m_p) (rotation first, then add, unlike b2Mul(xf, v)).
    rx, ry = m.rot(c, s, vx[:, 0], vy[:, 0])
    cx, cy = px + rx, py + ry
    circ = model.shape_type == CIRCLE
    return (jnp.where(circ, cx - r, lx - r), jnp.where(circ, cy - r, ly - r),
            jnp.where(circ, cx + r, ux + r), jnp.where(circ, cy + r, uy + r))


def fatten(lx, ly, ux, uy, dx=None, dy=None):
    """b2DynamicTree fat AABB: extend by b2_aabbExtension, then (MoveProxy) grow by 2 * displacement on its side."""
    e = AABB_EXTENSION
    lx, ly, ux, uy = lx - e, ly - e, ux + e, uy + e
    if dx is not None:
        ddx, ddy = m.mul(AABB_MULTIPLIER, dx), m.mul(AABB_MULTIPLIER, dy)
        lx = jnp.where(ddx < 0, lx + ddx, lx)
        ux = jnp.where(ddx < 0, ux, ux + ddx)
        ly = jnp.where(ddy < 0, ly + ddy, ly)
        uy = jnp.where(ddy < 0, uy, uy + ddy)
    return lx, ly, ux, uy


def overlap(a, b):
    """b2TestOverlap(AABB a, AABB b) for (lx, ly, ux, uy) tuples."""
    d1x, d1y = b[0] - a[2], b[1] - a[3]
    d2x, d2y = a[0] - b[2], a[1] - b[3]
    return ~((d1x > 0) | (d1y > 0) | (d2x > 0) | (d2y > 0))
