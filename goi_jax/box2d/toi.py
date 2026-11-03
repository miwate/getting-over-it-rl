"""Continuous collision: b2World::SolveTOI (Box2D 2.3.1) with b2TimeOfImpact, b2Distance (GJK), b2SeparationFunction,
b2Sweep::Advance and b2Island::SolveTOI, for one world (vmap it like step).

`solve_toi(model, state, settings, dt, axis_name=None)` runs after b2World::Solve when Settings.continuous_physics
is set (step.py). Settings.unity_toi applies Unity's pair rule (the oracle's b2UnityTOIPair): a pair gets TOI only if
a bullet is involved, or a Continuous dynamic body meets a static/kinematic one. Without it the stock rule applies:
a bullet, or one side static/kinematic.

How Box2D's sequential loop maps onto fixed shapes (all exact, not approximations):
- The event loop (find the earliest TOI, advance its two bodies, solve a sub-island, repeat) is a lax.while_loop.
  Sub-stepping is off (Unity's), so m_stepComplete is always true and every SolveTOI starts by resetting sweeps'
  alpha0, the contacts' TOI flags, counts and cached alphas (ContactTOI in world.py keeps the post-step values).
- Candidate search: the contact list is walked newest first. Only the "put both sweeps on the same time interval"
  Advance mutates state, and it depends only on the bodies' alpha0, never on a TOI result, so a short sequential
  pass over list positions does just those Advances and records the sweeps each contact saw; the TOI root finds then
  run for all contacts at once on the recorded sweeps. The minimum is the first in list order (strict `<`; the last
  with Settings.toi_tie_oldest, the game's closer approximation).
- Event: b2Body::Advance is idempotent for a given target (a second Advance to the same alpha changes no bit), and
  every `other` body in the sub-island walk is either kept advanced or restored to its backup, so each contact Update
  in the walk sees every body at "advanced to minAlpha". One vectorized manifold evaluation at those poses therefore
  gives every Update of the walk; the walk's order, its 32-contact / 64-body break and the island order come from
  prefix sums over the two bodies' contact-edge lists (newest first, bodyA's list then bodyB's).
- b2Island::SolveTOI reuses contact_solver: position passes (at most 20, toi_baumgarte, masses only for the two TOI
  bodies), then InitializeVelocityConstraints without warm start and velocity_iterations passes over contacts only,
  then integration with the step's clamps. Impulses are not stored in the manifold (Box2D's rule); they are kept in
  ContactTOI.normal_impulse / tangent_impulse (what b2Island::Report hands the listener, i.e. Unity's reports).

b2TimeOfImpact (20 axis iterations, 8 push-backs, 50 root steps) is one flat masked lax.while_loop over the C contact
lanes (time_of_impact), b2Distance a nested one (20 iterations): sin/cos inside a loop body is costly to compile.
With Unity's rule and a concrete Model whose bodies have no bullet / Continuous flag, SolveTOI is skipped at trace
time (Model.toi_bodies). With Settings.solver == "pallas" the search and the event run as Pallas kernels
(gpu/toi_search.py, gpu/toi_event.py; the same floats, bit-identical in interpret mode). Divergences: the contacts' e_enabledFlag is reset to true at every
step's SolveTOI (Box2D re-enables a contact only when b2Contact::Update runs on it, so a contact disabled by TOI in one
step and not updated by the next Collide stays disabled there); events are capped at `max_events` per step.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import collide as nc
from . import contact_solver as cs
from . import math2d as m
from .loops import ordered_loop
from .pairs import find_new_contacts, synchronize_fixtures
from .settings import EPSILON, MAX_POLYGON_VERTICES, MAX_SUB_STEPS, MAX_TOI_CONTACTS, PI
from .world import DYNAMIC, STATIC, empty_toi

F32 = np.float32
I32 = jnp.int32
E_UNKNOWN, E_FAILED, E_OVERLAPPED, E_TOUCHING, E_SEPARATED = 0, 1, 2, 3, 4  # b2TOIOutput::State
E_POINTS, E_FACE_A, E_FACE_B = 0, 1, 2  # b2SeparationFunction::Type
_IDX = jnp.arange(MAX_POLYGON_VERTICES)
TWO_PI = F32(F32(2.0) * PI)
EPS2 = F32(EPSILON * EPSILON)
DONE_ALPHA = F32(F32(1.0) - F32(F32(10.0) * EPSILON))  # 1.0f - 10.0f * b2_epsilon
GJK_ITERS, TOI_ITERS, ROOT_ITERS = 20, 20, 50


class _Isl(NamedTuple):  # what contact_solver.prepare reads from an island
    contact_order: jnp.ndarray
    n_contacts: jnp.ndarray


def _opaque(x, like):
    """A constant XLA cannot see (so `y / c` is not rewritten as `y * (1 / c)`)."""
    return jax.lax.optimization_barrier(jnp.full_like(like, x))


# ---------------------------------------------------------------- sweeps

def sweep_xf(sw, t):
    """b2Sweep::GetTransform(xf, beta=t). sw = (lcx, lcy, c0x, c0y, cx, cy, a0, a)."""
    lcx, lcy, c0x, c0y, cx, cy, a0, a = sw
    omb = F32(1.0) - t
    px = m.mul(omb, c0x) + m.mul(t, cx)
    py = m.mul(omb, c0y) + m.mul(t, cy)
    qc, qs = m.rot_of(m.mul(omb, a0) + m.mul(t, a))
    rx, ry = m.rot(qc, qs, lcx, lcy)
    return px - rx, py - ry, qc, qs


def _xfs(items):
    """sweep_xf for several (sweep, t) pairs of equal lane count in one evaluation (one sin/cos instance)."""
    n = items[0][1].shape[0]
    sw = tuple(jnp.concatenate([jnp.broadcast_to(it[0][k], (n,)) for it in items]) for k in range(8))
    xf = sweep_xf(sw, jnp.concatenate([it[1] for it in items]))
    return [tuple(x[i * n:(i + 1) * n] for x in xf) for i in range(len(items))]


def _normalize_sweep(sw):
    """b2Sweep::Normalize: shift a0 and a by 2 pi floor(a0 / 2 pi)."""
    lcx, lcy, c0x, c0y, cx, cy, a0, a = sw
    two_pi = _opaque(TWO_PI, a0)
    d = m.mul(two_pi, jnp.floor(a0 / two_pi))
    return lcx, lcy, c0x, c0y, cx, cy, a0 - d, a - d


def _advance(c0x, c0y, a0, alpha0, cx, cy, a, alpha):
    """b2Sweep::Advance(alpha): returns (c0x, c0y, a0, alpha0)."""
    den = F32(1.0) - alpha0
    beta = (alpha - alpha0) / jnp.where(den != 0, den, F32(1.0))
    return c0x + m.mul(beta, cx - c0x), c0y + m.mul(beta, cy - c0y), a0 + m.mul(beta, a - a0), alpha


def _sync_xf(cx, cy, a, lcx, lcy):
    """b2Body::SynchronizeTransform: q.Set(a); p = c - b2Mul(q, localCenter)."""
    qc, qs = m.rot_of(a)
    rx, ry = m.rot(qc, qs, lcx, lcy)
    return cx - rx, cy - ry, qc, qs


# ---------------------------------------------------------------- distance proxies and GJK (b2Distance.cpp)

def _vertex(proxy, i):
    vx, vy = proxy[0], proxy[1]
    i = jnp.clip(i, 0, MAX_POLYGON_VERTICES - 1)
    return (jnp.take_along_axis(vx, i[:, None], 1)[:, 0], jnp.take_along_axis(vy, i[:, None], 1)[:, 0])


def _support(proxy, dx, dy):
    """b2DistanceProxy::GetSupport: first index of the largest b2Dot(v_i, d)."""
    vx, vy, count = proxy[0], proxy[1], proxy[2]
    val = m.mul(vx, dx[:, None]) + m.mul(vy, dy[:, None])
    val = jnp.where(_IDX[None, :] < count[:, None], val, -jnp.inf)
    return jnp.argmax(val, axis=1).astype(I32)


_SFIELDS = ("wAx", "wAy", "wBx", "wBy", "wx", "wy", "a", "iA", "iB")


def _col(s, j):
    return {k: s[k][:, j] for k in _SFIELDS}


def _set(s, j, v, mask):
    """Column j of the simplex := v (dict of (N,)) where mask."""
    return {k: s[k].at[:, j].set(jnp.where(mask, v[k], s[k][:, j])) for k in _SFIELDS}


def _set_a(s, j, val, mask):
    return dict(s, a=s["a"].at[:, j].set(jnp.where(mask, val, s["a"][:, j])))


def _make_vertex(pA, xfA, iA, pB, xfB, iB):
    wAx, wAy = m.xf_mul(*xfA, *_vertex(pA, iA))
    wBx, wBy = m.xf_mul(*xfB, *_vertex(pB, iB))
    return dict(wAx=wAx, wAy=wAy, wBx=wBx, wBy=wBy, wx=wBx - wAx, wy=wBy - wAy, a=jnp.zeros_like(wAx), iA=iA, iB=iB)


def _metric(s, count):
    """b2Simplex::GetMetric."""
    w1, w2, w3 = ((s["wx"][:, j], s["wy"][:, j]) for j in range(3))
    two = m.length(w1[0] - w2[0], w1[1] - w2[1])
    three = m.cross(w2[0] - w1[0], w2[1] - w1[1], w3[0] - w1[0], w3[1] - w1[1])
    return jnp.where(count == 2, two, jnp.where(count == 3, three, F32(0.0)))


def _solve2(s, count, mask):
    w1x, w1y, w2x, w2y = s["wx"][:, 0], s["wy"][:, 0], s["wx"][:, 1], s["wy"][:, 1]
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
    (w1x, w2x, w3x), (w1y, w2y, w3y) = ((s["wx"][:, j] for j in range(3)), (s["wy"][:, j] for j in range(3)))
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
    v1, v2, v3 = _col(s, 0), _col(s, 1), _col(s, 2)
    one = F32(1.0)
    i12 = one / (d12_1 + d12_2)
    i13 = one / (d13_1 + d13_2)
    i23 = one / (d23_1 + d23_2)
    i123 = one / ((d123_1 + d123_2) + d123_3)
    # new columns (only the ones each region defines; the rest keep their values)
    a = s["a"]
    a1 = jnp.where(r_w1, one, jnp.where(r_e12, m.mul(d12_1, i12), jnp.where(r_e13, m.mul(d13_1, i13),
                                                                             m.mul(d123_1, i123))))
    s = _set_a(s, 0, a1, r_w1 | r_e12 | r_e13 | r_tri)
    s = _set_a(s, 1, jnp.where(r_e12, m.mul(d12_2, i12), jnp.where(r_w2, one, jnp.where(r_e23, m.mul(d23_1, i23),
                                                                                         m.mul(d123_2, i123)))),
               r_e12 | r_w2 | r_e23 | r_tri)
    s = _set_a(s, 2, jnp.where(r_e13, m.mul(d13_2, i13), jnp.where(r_w3, one, jnp.where(r_e23, m.mul(d23_2, i23),
                                                                                         m.mul(d123_3, i123)))),
               r_e13 | r_w3 | r_e23 | r_tri)
    del a, v1, v2, v3
    c1, c2, c3 = _col(s, 0), _col(s, 1), _col(s, 2)
    s = _set(s, 1, c3, r_e13)  # m_v2 = m_v3
    s = _set(s, 0, c2, r_w2)  # m_v1 = m_v2
    s = _set(s, 0, c3, r_w3 | r_e23)  # m_v1 = m_v3
    del c1
    count = jnp.where(r_w1 | r_w2 | r_w3, 1, jnp.where(r_e12 | r_e13 | r_e23, 2, jnp.where(r_tri, 3, count)))
    return s, count


def distance(cache, pA, xfA, pB, xfB, active):
    """b2Distance with useRadii = false. cache = (count, iA (N,3), iB (N,3), metric). Returns
    (distance, new cache); lanes not active return garbage and keep their cache."""
    ccount, ciA, ciB, cmetric = cache
    N = ccount.shape[0]
    zero = jnp.zeros((N, 3), jnp.float32)
    s = {k: zero for k in _SFIELDS[:7]}
    s.update(iA=jnp.zeros((N, 3), I32), iB=jnp.zeros((N, 3), I32))
    for j in range(3):  # ReadCache
        s = _set(s, j, _make_vertex(pA, xfA, ciA[:, j], pB, xfB, ciB[:, j]), j < ccount)
    count = ccount
    metric2 = _metric(s, count)
    flush = (count > 1) & ((metric2 < m.mul(F32(0.5), cmetric)) | (m.mul(F32(2.0), cmetric) < metric2)
                           | (metric2 < EPSILON))
    count = jnp.where(flush, 0, count)
    empty = count == 0
    z = jnp.zeros((N,), I32)
    v0 = _make_vertex(pA, xfA, z, pB, xfB, z)
    v0["a"] = jnp.ones((N,), jnp.float32)
    s = _set(s, 0, v0, empty)
    count = jnp.where(empty, 1, count)

    def body(c):
        s, count, it, run = c
        save_count, save_a, save_b = count, s["iA"], s["iB"]
        s, count = _solve2(s, count, run & (count == 2))
        s, count = _solve3(s, count, run & (count == 3))
        run = run & (count != 3)
        w1x, w1y, w2x, w2y = s["wx"][:, 0], s["wy"][:, 0], s["wx"][:, 1], s["wy"][:, 1]
        ex, ey = w2x - w1x, w2y - w1y
        sgn = m.cross(ex, ey, -w1x, -w1y)
        lx, ly = m.cross_sv(F32(1.0), ex, ey)
        rx, ry = m.cross_vs(ex, ey, F32(1.0))
        dx = jnp.where(count == 1, -w1x, jnp.where(sgn > 0, lx, rx))
        dy = jnp.where(count == 1, -w1y, jnp.where(sgn > 0, ly, ry))
        run = run & ~(m.dot(dx, dy, dx, dy) < EPS2)
        iA = _support(pA, *m.rot_t(xfA[2], xfA[3], -dx, -dy))
        iB = _support(pB, *m.rot_t(xfB[2], xfB[3], dx, dy))
        v = _make_vertex(pA, xfA, iA, pB, xfB, iB)
        for j in (1, 2):
            keep_a = s["a"][:, j]
            s = _set(s, j, v, run & (count == j))
            s = _set_a(s, j, keep_a, run & (count == j))  # the tentative vertex's `a` is not set by Box2D
        it = it + run.astype(I32)
        dup = jnp.zeros_like(run)
        for j in range(3):
            dup = dup | ((j < save_count) & (save_a[:, j] == iA) & (save_b[:, j] == iB))
        run = run & ~dup
        count = count + run.astype(I32)
        run = run & (it < GJK_ITERS)
        return s, count, it, run

    s, count, _, _ = jax.lax.while_loop(lambda c: jnp.any(c[3]), body, (s, count, jnp.zeros((N,), I32), active))
    # GetWitnessPoints, distance, WriteCache
    a1, a2, a3 = s["a"][:, 0], s["a"][:, 1], s["a"][:, 2]
    pa2 = (m.mul(a1, s["wAx"][:, 0]) + m.mul(a2, s["wAx"][:, 1]), m.mul(a1, s["wAy"][:, 0]) + m.mul(a2, s["wAy"][:, 1]))
    pb2 = (m.mul(a1, s["wBx"][:, 0]) + m.mul(a2, s["wBx"][:, 1]), m.mul(a1, s["wBy"][:, 0]) + m.mul(a2, s["wBy"][:, 1]))
    pa3 = (pa2[0] + m.mul(a3, s["wAx"][:, 2]), pa2[1] + m.mul(a3, s["wAy"][:, 2]))
    sel = lambda c1, c2, c3: jnp.where(count == 1, c1, jnp.where(count == 2, c2, c3))
    pAx, pAy = sel(s["wAx"][:, 0], pa2[0], pa3[0]), sel(s["wAy"][:, 0], pa2[1], pa3[1])
    pBx, pBy = sel(s["wBx"][:, 0], pb2[0], pa3[0]), sel(s["wBy"][:, 0], pb2[1], pa3[1])
    dist = m.length(pAx - pBx, pAy - pBy)
    keep = lambda new, old: jnp.where(active if new.ndim == 1 else active[:, None], new, old)
    cache = (keep(count, ccount), keep(s["iA"], ciA), keep(s["iB"], ciB), keep(_metric(s, count), cmetric))
    return dist, cache


# ---------------------------------------------------------------- b2SeparationFunction and b2TimeOfImpact

def _sep_init(cache, pA, xfA, pB, xfB):
    count, ciA, ciB, _ = cache
    # points
    lpA, lpB = _vertex(pA, ciA[:, 0]), _vertex(pB, ciB[:, 0])
    pax, pay = m.xf_mul(*xfA, *lpA)
    pbx, pby = m.xf_mul(*xfB, *lpB)
    ptx, pty, _ = m.normalize(pbx - pax, pby - pay)
    # face B (two points on B)
    b1, b2 = _vertex(pB, ciB[:, 0]), _vertex(pB, ciB[:, 1])
    fbx, fby, _ = m.normalize(*m.cross_vs(b2[0] - b1[0], b2[1] - b1[1], F32(1.0)))
    nbx, nby = m.rot(xfB[2], xfB[3], fbx, fby)
    lbx, lby = m.mul(F32(0.5), b1[0] + b2[0]), m.mul(F32(0.5), b1[1] + b2[1])
    qbx, qby = m.xf_mul(*xfB, lbx, lby)
    sb = m.dot(pax - qbx, pay - qby, nbx, nby)
    fbx, fby = jnp.where(sb < 0, -fbx, fbx), jnp.where(sb < 0, -fby, fby)
    # face A
    a1, a2 = _vertex(pA, ciA[:, 0]), _vertex(pA, ciA[:, 1])
    fax, fay, _ = m.normalize(*m.cross_vs(a2[0] - a1[0], a2[1] - a1[1], F32(1.0)))
    nax, nay = m.rot(xfA[2], xfA[3], fax, fay)
    lax_, lay = m.mul(F32(0.5), a1[0] + a2[0]), m.mul(F32(0.5), a1[1] + a2[1])
    qax, qay = m.xf_mul(*xfA, lax_, lay)
    sa = m.dot(pbx - qax, pby - qay, nax, nay)
    fax, fay = jnp.where(sa < 0, -fax, fax), jnp.where(sa < 0, -fay, fay)
    typ = jnp.where(count == 1, E_POINTS, jnp.where(ciA[:, 0] == ciA[:, 1], E_FACE_B, E_FACE_A))
    pick = lambda p, a, b: jnp.where(typ == E_POINTS, p, jnp.where(typ == E_FACE_A, a, b))
    zero = jnp.zeros_like(ptx)
    return dict(type=typ, axx=pick(ptx, fax, fbx), axy=pick(pty, fay, fby), lpx=pick(zero, lax_, lbx),
                lpy=pick(zero, lay, lby))


def _sep_points(f, pA, xfA, iA, pB, xfB, iB):
    """Separations of the three function types for given witness indices (FindMinSeparation / Evaluate tail)."""
    axx, axy = f["axx"], f["axy"]
    pax, pay = m.xf_mul(*xfA, *_vertex(pA, iA))
    pbx, pby = m.xf_mul(*xfB, *_vertex(pB, iB))
    s_pts = m.dot(pbx - pax, pby - pay, axx, axy)
    nax, nay = m.rot(xfA[2], xfA[3], axx, axy)
    qax, qay = m.xf_mul(*xfA, f["lpx"], f["lpy"])
    s_fa = m.dot(pbx - qax, pby - qay, nax, nay)
    nbx, nby = m.rot(xfB[2], xfB[3], axx, axy)
    qbx, qby = m.xf_mul(*xfB, f["lpx"], f["lpy"])
    s_fb = m.dot(pax - qbx, pay - qby, nbx, nby)
    t = f["type"]
    return jnp.where(t == E_POINTS, s_pts, jnp.where(t == E_FACE_A, s_fa, s_fb))


def _find_min(f, pA, xfA, pB, xfB):
    axx, axy = f["axx"], f["axy"]
    typ = f["type"]
    # points: axisA = MulT(qA, axis), axisB = MulT(qB, -axis)
    pa_i = _support(pA, *m.rot_t(xfA[2], xfA[3], axx, axy))
    pb_i = _support(pB, *m.rot_t(xfB[2], xfB[3], -axx, -axy))
    # faceA: normal = Mul(qA, axis); axisB = MulT(qB, -normal)
    nax, nay = m.rot(xfA[2], xfA[3], axx, axy)
    fa_i = _support(pB, *m.rot_t(xfB[2], xfB[3], -nax, -nay))
    # faceB: normal = Mul(qB, axis); axisA = MulT(qA, -normal)
    nbx, nby = m.rot(xfB[2], xfB[3], axx, axy)
    fb_i = _support(pA, *m.rot_t(xfA[2], xfA[3], -nbx, -nby))
    iA = jnp.where(typ == E_POINTS, pa_i, jnp.where(typ == E_FACE_A, -1, fb_i))
    iB = jnp.where(typ == E_POINTS, pb_i, jnp.where(typ == E_FACE_A, fa_i, -1))
    return _sep_points(f, pA, xfA, iA, pB, xfB, iB), iA, iB




def time_of_impact(pA, swA, pB, swB, linear_slop, active):
    """b2TimeOfImpact with tMax = 1 for N lanes. pA/pB = (verts x (N,8), verts y, count, radius); sw = sweeps.
    Returns (state, t) per lane (E_* codes).

    Box2D's three nested loops (separating axes, push-back, root finder) run as one flat lax.while_loop: each lane
    is in one phase per iteration (DIST: b2Distance at t1 and a new separating axis; FIND: deepest point at t2 and its
    separation at t1; ROOT: one secant/bisection step), so the transforms (sin/cos) appear once in the loop body."""
    swA, swB = _normalize_sweep(swA), _normalize_sweep(swB)
    N = pA[2].shape[0]
    total_radius = pA[3] + pB[3]
    target = m.bmax(linear_slop, total_radius - F32(F32(3.0) * linear_slop))
    tol = F32(F32(0.25) * linear_slop)
    one = jnp.ones((N,), jnp.float32)
    zf, zi = jnp.zeros((N,), jnp.float32), jnp.zeros((N,), I32)
    DIST, FIND, ROOT = 0, 1, 2
    c0 = dict(phase=zi, state=jnp.full((N,), E_UNKNOWN, I32), out_t=one, t1=zf, t2=one, it=zi, push=zi,
              cache=(zi, jnp.zeros((N, 3), I32), jnp.zeros((N, 3), I32), zf),
              f=dict(type=zi, axx=zf, axy=zf, lpx=zf, lpy=zf), iA=zi, iB=zi, a1=zf, a2=zf, rs1=zf, rs2=zf, cnt=zi,
              run=active)

    def body(c):
        run, ph = c["run"], c["phase"]
        dist_l, find_l, root_l = run & (ph == DIST), run & (ph == FIND), run & (ph == ROOT)
        t1, t2, f = c["t1"], c["t2"], c["f"]
        # root step time (secant on odd counts, bisection on even)
        a1, a2, rs1, rs2, cnt = c["a1"], c["a2"], c["rs1"], c["rs2"], c["cnt"]
        secant = a1 + (m.mul(target - rs1, a2 - a1) / (rs2 - rs1))
        troot = jnp.where((cnt & 1) == 1, secant, m.mul(F32(0.5), a1 + a2))
        tq1 = jnp.where(dist_l, t1, jnp.where(find_l, t2, troot))
        xA1, xB1, xA2, xB2 = _xfs([(swA, tq1), (swB, tq1), (swA, t1), (swB, t1)])
        out = dict(c)

        # ---- DIST: b2Distance at t1, then the separating function
        dist, cache = distance(c["cache"], pA, xA1, pB, xB1, dist_l)
        over = dist_l & (dist <= 0)
        touch = dist_l & ~over & (dist < target + tol)
        cont = dist_l & ~over & ~touch
        fn = _sep_init(cache, pA, xA1, pB, xB1)
        f = {k: jnp.where(cont, fn[k], f[k]) for k in f}
        state = jnp.where(over, E_OVERLAPPED, jnp.where(touch, E_TOUCHING, c["state"]))
        out_t = jnp.where(over, F32(0.0), jnp.where(touch, t1, c["out_t"]))
        run = run & ~over & ~touch
        phase = jnp.where(cont, FIND, ph)
        t2 = jnp.where(cont, one, t2)
        push = jnp.where(cont, 0, c["push"])

        # ---- FIND: deepest point at t2 (xA1/xB1 hold t2 here), separation of the witnesses at t1
        s2, fiA, fiB = _find_min(f, pA, xA1, pB, xB1)
        s1 = _sep_points(f, pA, xA2, fiA, pB, xB2, fiB)
        sep = find_l & (s2 > target + tol)
        adv = find_l & ~sep & (s2 > target - tol)
        rest = find_l & ~sep & ~adv
        fail = rest & (s1 < target - tol)
        tch = rest & ~fail & (s1 <= target + tol)
        root = rest & ~fail & ~tch
        state = jnp.where(sep, E_SEPARATED, jnp.where(fail, E_FAILED, jnp.where(tch, E_TOUCHING, state)))
        out_t = jnp.where(sep, one, jnp.where(fail | tch, t1, out_t))
        run = run & ~(sep | fail | tch)
        t1n = jnp.where(adv, t2, t1)
        iA, iB = jnp.where(root, fiA, c["iA"]), jnp.where(root, fiB, c["iB"])
        a1, a2 = jnp.where(root, t1, a1), jnp.where(root, t2, a2)
        rs1, rs2 = jnp.where(root, s1, rs1), jnp.where(root, s2, rs2)
        cnt = jnp.where(root, 0, cnt)
        phase = jnp.where(root, ROOT, phase)

        # ---- ROOT: one step of the 1D root finder (xA1/xB1 hold troot here)
        sr = _sep_points(f, pA, xA1, c["iA"], pB, xB1, c["iB"])
        cnt = jnp.where(root_l, cnt + 1, cnt)
        conv = root_l & (m.babs(sr - target) < tol)
        t2 = jnp.where(conv, troot, t2)
        go = root_l & ~conv
        up, lo = go & (sr > target), go & ~(sr > target)
        a1, rs1 = jnp.where(up, troot, a1), jnp.where(up, sr, rs1)
        a2, rs2 = jnp.where(lo, troot, a2), jnp.where(lo, sr, rs2)
        root_end = conv | (go & (cnt == ROOT_ITERS))
        push = jnp.where(root_end, push + 1, push)
        back = root_end & (push != MAX_POLYGON_VERTICES)  # next push-back iteration
        phase = jnp.where(back, FIND, phase)

        # ---- end of a separating-axis iteration (t1 advanced, or 8 push-backs): ++iter, give up at 20
        axis_end = adv | (root_end & ~back)
        it = jnp.where(axis_end, c["it"] + 1, c["it"])
        stuck = axis_end & (it == TOI_ITERS)
        state = jnp.where(stuck, E_FAILED, state)
        out_t = jnp.where(stuck, t1n, out_t)
        run = run & ~stuck
        phase = jnp.where(axis_end, DIST, phase)
        cache = tuple(jnp.where(dist_l if x.ndim == 1 else dist_l[:, None], x, y) for x, y in zip(cache, c["cache"]))
        out.update(phase=phase, state=state, out_t=out_t, t1=t1n, t2=t2, it=it, push=push, cache=cache, f=f, iA=iA,
                   iB=iB, a1=a1, a2=a2, rs1=rs1, rs2=rs2, cnt=cnt, run=run)
        return out

    c = jax.lax.while_loop(lambda c: jnp.any(c["run"]), body, c0)
    return c["state"], c["out_t"]


# ---------------------------------------------------------------- SolveTOI

def _pair_rule(bd, a, b, unity):
    """Box2D's `collideA || collideB` plus (unity) b2UnityTOIPair."""
    dyn_a, dyn_b = bd.type[a] == DYNAMIC, bd.type[b] == DYNAMIC
    stock = bd.bullet[a] | ~dyn_a | bd.bullet[b] | ~dyn_b
    if not unity:
        return stock
    bullet = bd.bullet[a] | bd.bullet[b]
    rule = bullet | jnp.where(~dyn_b, bd.continuous[a], jnp.where(~dyn_a, bd.continuous[b], False))
    return stock & rule


def _list_order(ct):
    """Slots in contact-list order (newest first = descending key), dead slots last; and each slot's position."""
    live = ct.pair >= 0
    k = jnp.where(live, ct.key, jnp.iinfo(jnp.int32).min)
    neg = jnp.where(k == jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, -k)
    perm = jnp.argsort(neg, stable=True).astype(I32)
    pos = jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0], dtype=I32))
    return perm, pos, jnp.sum(live).astype(I32)


def _proxies(model, f):
    v = model.verts[f]
    return v[..., 0], v[..., 1], model.count[f], model.radius[f]


def _search(model, state, tst, settings, go, axis_name):
    """One pass of the min-TOI search. Returns (state, tst, found, min_slot, min_alpha)."""
    ct, bd = state.contacts, state.bodies
    C = ct.pair.shape[0]
    live = ct.pair >= 0
    fb = model.fixture_body
    ia, ib = fb[ct.fa], fb[ct.fb]
    base = live & tst.enabled & (tst.count <= MAX_SUB_STEPS)
    act = bd.awake & (bd.type != STATIC)
    need = go & base & ~tst.flag & (act[ia] | act[ib]) & _pair_rule(bd, ia, ib, settings.unity_toi)
    perm, pos, n_live = _list_order(ct)

    # Sequential: the sweeps' alpha0 alignment (b2Sweep::Advance of the body that is behind), in list order.
    c0x, c0y, a0, al0 = bd.c0[:, 0], bd.c0[:, 1], bd.a0, tst.alpha0
    cx, cy, a = bd.c[:, 0], bd.c[:, 1], bd.a
    zc = jnp.zeros((C,), jnp.float32)
    rec = dict(c0xA=zc, c0yA=zc, a0A=zc, c0xB=zc, c0yB=zc, a0B=zc, alpha0=zc)

    def body(k, carry, live_k):
        (c0x, c0y, a0, al0), rec = carry
        slot = perm[k]
        n = need[slot] & live_k
        A, B = ia[slot], ib[slot]
        g = lambda arr, i: m.pick(arr, i)
        alA, alB = g(al0, A), g(al0, B)
        sA = (g(c0x, A), g(c0y, A), g(a0, A))
        sB = (g(c0x, B), g(c0y, B), g(a0, B))
        advA = n & (alA < alB)
        advB = n & ~(alA < alB) & (alB < alA)
        nA = _advance(*sA, alA, g(cx, A), g(cy, A), g(a, A), alB)
        nB = _advance(*sB, alB, g(cx, B), g(cy, B), g(a, B), alA)
        sA = tuple(jnp.where(advA, x, y) for x, y in zip(nA[:3], sA))
        sB = tuple(jnp.where(advB, x, y) for x, y in zip(nB[:3], sB))
        common = jnp.where(advA, alB, alA)
        out = []
        for arr, va, vb in zip((c0x, c0y, a0), sA, sB):
            arr = jnp.where(advA, m.put(arr, A, va), arr)
            arr = jnp.where(advB, m.put(arr, B, vb), arr)
            out.append(arr)
        al0 = jnp.where(advA, m.put(al0, A, alB), al0)
        al0 = jnp.where(advB, m.put(al0, B, alA), al0)
        vals = dict(c0xA=sA[0], c0yA=sA[1], a0A=sA[2], c0xB=sB[0], c0yB=sB[1], a0B=sB[2], alpha0=common)
        hit = (jnp.arange(C) == slot) & n
        rec = {key: jnp.where(hit, vals[key], rec[key]) for key in rec}
        return (tuple(out) + (al0,), rec)

    (c0x, c0y, a0, al0), rec = ordered_loop(n_live, C, body, ((c0x, c0y, a0, al0), rec), axis_name=axis_name)
    bd = bd._replace(c0=jnp.stack([c0x, c0y], -1), a0=a0)
    tst = tst._replace(alpha0=al0)
    state = state._replace(bodies=bd)

    lc = bd.local_center
    swA = (lc[ia, 0], lc[ia, 1], rec["c0xA"], rec["c0yA"], cx[ia], cy[ia], rec["a0A"], a[ia])
    swB = (lc[ib, 0], lc[ib, 1], rec["c0xB"], rec["c0yB"], cx[ib], cy[ib], rec["a0B"], a[ib])
    tstate, beta = time_of_impact(_proxies(model, ct.fa), swA, _proxies(model, ct.fb), swB, settings.linear_slop,
                                  need)
    al = rec["alpha0"]
    alpha = jnp.where(tstate == E_TOUCHING, m.bmin(al + m.mul(F32(1.0) - al, beta), F32(1.0)), F32(1.0))
    toi = jnp.where(need, alpha, tst.toi)
    flag = tst.flag | need
    tst = tst._replace(toi=toi, flag=flag)

    cand = go & base & flag & (toi < F32(1.0))
    min_alpha = jnp.min(jnp.where(cand, toi, F32(1.0)))
    tie = cand & (toi == min_alpha)
    if settings.toi_tie_oldest:  # the game: the last of exact ties in list order (Settings.toi_tie_oldest)
        min_slot = jnp.argmax(jnp.where(tie, pos, -1)).astype(I32)
    else:
        min_slot = jnp.argmin(jnp.where(tie, pos, C)).astype(I32)
    found = jnp.any(cand) & ~(DONE_ALPHA < min_alpha)
    return state, tst, found, min_slot, min_alpha


def _wake(bd, mask):
    """b2Body::SetAwake(true) on the masked bodies (any type, as Box2D)."""
    newly = mask & ~bd.awake
    return bd._replace(awake=bd.awake | mask, sleep_time=jnp.where(newly, F32(0.0), bd.sleep_time))


def _update_contacts(ct, man, mask):
    """b2Contact::Update on the masked slots with a freshly evaluated manifold: impulses carried over by feature key,
    touching flag. Returns (contacts, touching flipped)."""
    new_count = man["count"]
    touching = new_count > 0
    ni, ti = [], []
    for i in range(2):
        nid = man["point_id"][i]
        m0 = (ct.count > 0) & (ct.point_id[:, 0] == nid)
        m1 = (ct.count > 1) & (ct.point_id[:, 1] == nid) & ~m0
        ni.append(jnp.where(m0, ct.normal_impulse[:, 0], jnp.where(m1, ct.normal_impulse[:, 1], F32(0))))
        ti.append(jnp.where(m0, ct.tangent_impulse[:, 0], jnp.where(m1, ct.tangent_impulse[:, 1], F32(0))))
    has = jnp.arange(2)[None, :] < new_count[:, None]
    ni = jnp.where(has, jnp.stack(ni, 1), F32(0))
    ti = jnp.where(has, jnp.stack(ti, 1), F32(0))
    u, km = mask, mask & (new_count > 0)
    pl = jnp.stack([jnp.stack(man["point_local"][0], -1), jnp.stack(man["point_local"][1], -1)], 1)
    flip = mask & (touching != ct.touching)
    ct = ct._replace(
        touching=jnp.where(u, touching, ct.touching), count=jnp.where(u, new_count, ct.count),
        mtype=jnp.where(km, man["mtype"], ct.mtype),
        local_normal=jnp.where(km[:, None], jnp.stack(man["local_normal"], -1), ct.local_normal),
        local_point=jnp.where(km[:, None], jnp.stack(man["local_point"], -1), ct.local_point),
        point_local=jnp.where(km[:, None, None], pl, ct.point_local),
        point_id=jnp.where(km[:, None], jnp.stack(man["point_id"], -1), ct.point_id),
        normal_impulse=jnp.where(u[:, None], ni, ct.normal_impulse),
        tangent_impulse=jnp.where(u[:, None], ti, ct.tangent_impulse))
    return ct, flip


def _first(mask, key, n):
    """mask & (first position among mask with this key value)."""
    p = jnp.arange(mask.shape[0], dtype=I32)
    first = jnp.full((n,), mask.shape[0], I32).at[jnp.where(mask, key, n)].min(p, mode="drop")
    return mask & (first[jnp.clip(key, 0, n - 1)] == p)


def _event(model, state, tst, settings, dt, go, min_slot, min_alpha, axis_name):
    """Advance the TOI pair to min_alpha, update its contact, and (if touching) build and solve the TOI island."""
    s = settings
    ct, bd = state.contacts, state.bodies
    B, C = bd.type.shape[0], ct.pair.shape[0]
    fb = model.fixture_body
    ia, ib = fb[ct.fa], fb[ct.fb]
    A, Bb = ia[min_slot], ib[min_slot]
    bids = jnp.arange(B)
    isA, isB = (bids == A), (bids == Bb)
    lcx, lcy = bd.local_center[:, 0], bd.local_center[:, 1]

    # Every body advanced to min_alpha (b2Body::Advance); bA/bB keep it, others only where they join the island.
    adv = _advance(bd.c0[:, 0], bd.c0[:, 1], bd.a0, tst.alpha0, bd.c[:, 0], bd.c[:, 1], bd.a, min_alpha)
    ac0x, ac0y, aa0, aal0 = adv
    cat = lambda u, v: jnp.concatenate([u, v])
    both = _sync_xf(cat(ac0x, bd.c[:, 0]), cat(ac0y, bd.c[:, 1]), cat(aa0, bd.a), cat(lcx, lcx), cat(lcy, lcy))
    axf = tuple(x[:B] for x in both)
    oxf = tuple(x[B:] for x in both)  # SynchronizeTransform of a restored sweep
    xa = tuple(x[ia] for x in axf)
    xb = tuple(x[ib] for x in axf)
    man = nc.evaluate(model, ct.fa, ct.fb, xa, xb, s.linear_slop)
    touch_new = man["count"] > 0

    # minContact->Update, then the island walk (only when it touches).
    is_min = (jnp.arange(C) == min_slot) & go
    t_min = jnp.any(is_min & touch_new)
    walk = go & t_min

    perm, _, _ = _list_order(ct)
    live = ct.pair >= 0
    dynA, dynB = bd.type[A] == DYNAMIC, bd.type[Bb] == DYNAMIC
    slot = jnp.concatenate([perm, perm])
    owner = jnp.concatenate([jnp.full((C,), A, I32), jnp.full((C,), Bb, I32)])
    invA = live[perm] & ((ia[perm] == A) | (ib[perm] == A)) & dynA
    invB = live[perm] & ((ia[perm] == Bb) | (ib[perm] == Bb)) & dynB & ~invA  # a bA-bB contact: seen in bA's list
    valid = jnp.concatenate([invA, invB]) & walk
    sa, sb = ia[slot], ib[slot]
    other = jnp.where(sa == owner, sb, sa)
    dyn_skip = (bd.type[other] == DYNAMIC) & ~bd.bullet[owner] & ~bd.bullet[other]
    rule = _pair_rule(bd, owner, other, s.unity_toi) if s.unity_toi else True
    elig = valid & (slot != min_slot) & ~dyn_skip & rule
    add_u = elig & touch_new[slot]
    newb_u = add_u & (other != A) & (other != Bb)
    newb_u = _first(newb_u, other, B)
    cum_c = jnp.cumsum(add_u.astype(I32)) - add_u.astype(I32)
    cum_b = jnp.cumsum(newb_u.astype(I32)) - newb_u.astype(I32)
    proc = (1 + cum_c < MAX_TOI_CONTACTS) & (2 + cum_b < 2 * MAX_TOI_CONTACTS)
    upd_e, add_e, newb_e = elig & proc, add_u & proc, newb_u & proc
    E = 2 * C
    upd_slot = jnp.zeros((C,), bool).at[jnp.where(upd_e, slot, C)].set(True, mode="drop")
    in_isl_other = jnp.zeros((B,), bool).at[jnp.where(add_e, other, B)].set(True, mode="drop")
    touched_other = jnp.zeros((B,), bool).at[jnp.where(upd_e, other, B)].set(True, mode="drop")
    newbody = jnp.zeros((B,), bool).at[jnp.where(newb_e, other, B)].set(True, mode="drop")

    # Contact updates (min contact and the walk's).
    ct2, flip = _update_contacts(ct, man, is_min | upd_slot)
    flip_b = jnp.zeros((B,), bool).at[jnp.where(flip, ia, B)].set(True, mode="drop")
    flip_b = flip_b.at[jnp.where(flip, ib, B)].set(True, mode="drop")
    enabled = jnp.where(is_min, t_min, tst.enabled | upd_slot)
    count = jnp.where(is_min, tst.count + 1, tst.count)
    flag = tst.flag & ~is_min

    # Body sweeps after the walk.
    pair_b = (isA | isB) & go
    keep_adv = (pair_b & t_min) | in_isl_other
    restore = (pair_b & ~t_min) | (touched_other & ~in_isl_other)
    sel = lambda adv_v, rest_v, old: jnp.where(keep_adv, adv_v, jnp.where(restore, rest_v, old))
    c0 = jnp.stack([sel(ac0x, bd.c0[:, 0], bd.c0[:, 0]), sel(ac0y, bd.c0[:, 1], bd.c0[:, 1])], -1)
    a0 = sel(aa0, bd.a0, bd.a0)
    c = jnp.stack([sel(ac0x, bd.c[:, 0], bd.c[:, 0]), sel(ac0y, bd.c[:, 1], bd.c[:, 1])], -1)
    a = sel(aa0, bd.a, bd.a)
    alpha0 = sel(aal0, tst.alpha0, tst.alpha0)
    xf_p = jnp.stack([sel(axf[0], oxf[0], bd.xf_p[:, 0]), sel(axf[1], oxf[1], bd.xf_p[:, 1])], -1)
    xf_q = jnp.stack([sel(axf[2], oxf[2], bd.xf_q[:, 0]), sel(axf[3], oxf[3], bd.xf_q[:, 1])], -1)
    bd = bd._replace(c0=c0, a0=a0, c=c, a=a, xf_p=xf_p, xf_q=xf_q)
    bd = _wake(bd, flip_b | (pair_b & t_min) | (newbody & (bd.type != STATIC)))
    in_isl = (pair_b | in_isl_other) & walk

    # Island contact order: the TOI contact, then the walk's additions.
    rank = jnp.cumsum(add_e.astype(I32))  # 1-based position after the TOI contact
    order = jnp.full((C,), -1, I32).at[0].set(min_slot)
    order = order.at[jnp.where(add_e, rank, C)].set(slot, mode="drop")
    n_isl = jnp.where(walk, 1 + jnp.sum(add_e).astype(I32), 0)
    del E

    state = state._replace(contacts=ct2, bodies=bd)
    pallas = getattr(s, "solver", "xla") == "pallas"  # Settings.solver: kernels from goi_jax/box2d/gpu
    if pallas:
        from .gpu import pairs as gpu_pairs
        from .gpu.toi import solve_island as solve_island_pallas
    solve_isl = solve_island_pallas if pallas else _solve_island
    state, tst_imp = solve_isl(model, state, s, dt, min_alpha, A, Bb, order, n_isl, in_isl, walk, axis_name)
    ni_r, ti_r, solved = tst_imp
    bd = state.bodies

    # Reset island flags, synchronize fixtures and invalidate the TOIs of the displaced (dynamic) bodies.
    moved = in_isl & (bd.type == DYNAMIC)
    state = (gpu_pairs.synchronize_fixtures if pallas else synchronize_fixtures)(model, state, moved)
    ct3 = state.contacts
    touch_moved = moved[fb[ct3.fa]] | moved[fb[ct3.fb]]
    flag = flag & ~(touch_moved & walk)
    old_pair, old_key = ct3.pair, ct3.key
    found = gpu_pairs.find_new_contacts(model, state, walk) if pallas else find_new_contacts(model, state)
    state = jax.tree.map(lambda x, y: jnp.where(walk, x, y), found, state)
    ct4 = state.contacts
    new = (ct4.pair >= 0) & ((old_pair < 0) | (old_key != ct4.key))
    tst = tst._replace(alpha0=alpha0, count=jnp.where(new, 0, count), flag=flag & ~new,
                       enabled=enabled | new, solved=tst.solved | solved,
                       normal_impulse=jnp.where(solved[:, None], ni_r, tst.normal_impulse),
                       tangent_impulse=jnp.where(solved[:, None], ti_r, tst.tangent_impulse),
                       events=tst.events + go.astype(I32))
    return state, tst


def _solve_island(model, state, s, dt, min_alpha, A, Bb, order, n, in_isl, walk, axis_name):
    """b2Island::SolveTOI over the contacts `order[:n]` and the bodies `in_isl` (toi bodies A and Bb)."""
    bd = state.bodies
    B, C = bd.type.shape[0], order.shape[0]
    isl = _Isl(contact_order=jnp.where(order >= 0, order, 0), n_contacts=n)
    pos = (bd.c[:, 0], bd.c[:, 1], bd.a)
    vel = (bd.v[:, 0], bd.v[:, 1], bd.w)
    pc = _position_rows(model, state, isl)
    toi_a = lambda i: (i == A) | (i == Bb)
    pc = dict(pc, mA=jnp.where(toi_a(pc["ia"]), pc["mA"], F32(0.0)), iA=jnp.where(toi_a(pc["ia"]), pc["iA"], F32(0.0)),
              mB=jnp.where(toi_a(pc["ib"]), pc["mB"], F32(0.0)), iB=jnp.where(toi_a(pc["ib"]), pc["iB"], F32(0.0)))
    okay_sep = F32(F32(-1.5) * s.linear_slop)

    def one_pass(pos):
        def cbody(k, carry, live_k):
            pos, min_sep = carry
            c = {key: val[k] for key, val in pc.items()}
            cA, aA, cB, aB, ms = cs.solve_position_one(c, pos, s, s.toi_baumgarte)
            pos = _put_pos(pos, c["ia"], c["ib"], cA, aA, cB, aB, live_k)
            return pos, jnp.where(live_k, m.bmin(min_sep, ms), min_sep)
        pos, min_sep = ordered_loop(n, C, cbody, (pos, F32(0.0)), axis_name=axis_name)
        return pos, min_sep >= okay_sep

    def cond(c):
        it, _, done = c
        more = ~done
        if axis_name is not None:
            more = jax.lax.pmax(more.astype(I32), axis_name) > 0
        return (it < 20) & more

    def body(c):
        it, pos, done = c
        npos, okay = one_pass(pos)
        pos = tuple(jnp.where(done, x, y) for x, y in zip(pos, npos))
        return it + 1, pos, done | okay

    _, pos, _ = jax.lax.while_loop(cond, body, (jnp.int32(0), pos, ~walk))
    cx, cy, a = pos
    # Leap of faith: the TOI bodies' sweep start is the solved pose.
    pair_b = ((jnp.arange(B) == A) | (jnp.arange(B) == Bb)) & walk
    c0 = jnp.where(pair_b[:, None], jnp.stack([cx, cy], -1), bd.c0)
    a0 = jnp.where(pair_b, a, bd.a0)

    vc = cs.prepare(model, state, isl, pos, vel, False, F32(1.0), s.velocity_threshold)

    def vel_iter(_, carry):
        vel, ni, ti = carry
        return cs.solve_velocity(vc, vel, ni, ti, n, axis_name)
    vel, ni, ti = jax.lax.fori_loop(0, s.velocity_iterations, vel_iter, (vel, vc["ni"], vc["ti"]))

    # Integrate positions of every island body (h = (1 - minAlpha) dt, the step's clamps).
    h = m.mul(F32(1.0) - min_alpha, dt)
    vx, vy, w = vel
    max_t = m.mul(s.max_translation, dt) if s.clamps_per_second else jnp.asarray(s.max_translation)
    max_r = m.mul(s.max_rotation, dt) if s.clamps_per_second else jnp.asarray(s.max_rotation)
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
    lcx, lcy = bd.local_center[:, 0], bd.local_center[:, 1]
    px, py, qc, qs = _sync_xf(ncx, ncy, na, lcx, lcy)
    i2 = in_isl[:, None]
    bd = bd._replace(c0=c0, a0=a0, c=jnp.where(i2, jnp.stack([ncx, ncy], -1), bd.c), a=jnp.where(in_isl, na, bd.a),
                     v=jnp.where(i2, jnp.stack([vx, vy], -1), bd.v), w=jnp.where(in_isl, w, bd.w),
                     xf_p=jnp.where(i2, jnp.stack([px, py], -1), bd.xf_p),
                     xf_q=jnp.where(i2, jnp.stack([qc, qs], -1), bd.xf_q))

    # b2Island::Report: each island contact's impulses (velocity constraint pointCount points).
    valid = jnp.arange(C) < n
    on = valid[:, None] & (jnp.arange(2)[None, :] < vc["count"][:, None])
    tgt = jnp.where(valid, vc["slot"], C)
    ni_r = jnp.zeros((C, 2), jnp.float32).at[tgt].set(jnp.where(on, ni, F32(0.0)), mode="drop")
    ti_r = jnp.zeros((C, 2), jnp.float32).at[tgt].set(jnp.where(on, ti, F32(0.0)), mode="drop")
    solved = jnp.zeros((C,), bool).at[tgt].set(True, mode="drop")
    return state._replace(bodies=bd), (ni_r, ti_r, solved)


def _position_rows(model, state, isl):
    """The b2ContactPositionConstraint fields contact_solver.solve_position_one reads (position independent)."""
    ct, bd = state.contacts, state.bodies
    slot = isl.contact_order
    fa, fb = ct.fa[slot], ct.fb[slot]
    ia, ib = model.fixture_body[fa], model.fixture_body[fb]
    lcA, lcB = bd.local_center[ia], bd.local_center[ib]
    lp, pl = ct.local_point[slot], ct.point_local[slot]
    return dict(ia=ia, ib=ib, mA=bd.inv_mass[ia], mB=bd.inv_mass[ib], iA=bd.inv_inertia[ia], iB=bd.inv_inertia[ib],
                pc_count=ct.count[slot], mtype=ct.mtype[slot], lnx=ct.local_normal[slot, 0],
                lny=ct.local_normal[slot, 1], lpx=lp[:, 0], lpy=lp[:, 1], plx=pl[:, :, 0], ply=pl[:, :, 1],
                lcAx=lcA[:, 0], lcAy=lcA[:, 1], lcBx=lcB[:, 0], lcBy=lcB[:, 1], radA=model.radius[fa],
                radB=model.radius[fb])


def _put_pos(pos, ia, ib, cA, aA, cB, aB, live):
    cx, cy, a = pos
    hit_a = (jnp.arange(cx.shape[-1]) == ia) & live
    hit_b = (jnp.arange(cx.shape[-1]) == ib) & live
    cx = jnp.where(hit_b, cB[0], jnp.where(hit_a, cA[0], cx))
    cy = jnp.where(hit_b, cB[1], jnp.where(hit_a, cA[1], cy))
    a = jnp.where(hit_b, aB, jnp.where(hit_a, aA, a))
    return cx, cy, a


def solve_toi(model, state, settings, dt, axis_name=None, max_events=None):
    """b2World::SolveTOI (sub-stepping off). Requires the step's dt > 0 (no-op otherwise)."""
    B, C = state.bodies.type.shape[0], state.contacts.pair.shape[0]
    if C == 0 or model.pair_a.shape[0] == 0:
        return state
    tb = model.toi_bodies
    if settings.unity_toi and tb is not None and not isinstance(tb, jax.core.Tracer) and not np.any(np.asarray(tb)):
        return state  # no bullet or Continuous body: Unity's rule gives no TOI pair (state.toi is left as it was)
    keep_none = state.toi is None
    # m_stepComplete: reset alpha0 and every contact's TOI state.
    tst = empty_toi(B, C)
    if max_events is None:
        max_events = C * (MAX_SUB_STEPS + 1)
        max_events += 8
    go0 = jnp.asarray(dt, jnp.float32) > 0

    # carry: (state, tst, searching, event pending, min slot, min alpha, iteration). Iteration 0 only searches; each
    # later one solves the pending event, then searches again (one copy of each in the graph).
    def cond(c):
        return c[2] & (c[6] <= max_events)

    search, event = _search, _event
    if getattr(settings, "solver", "xla") == "pallas":  # search and event as kernels (gpu/toi_search.py, toi_event.py)
        from .gpu import toi_loop
        if toi_loop.ENABLED:  # the same loop at the batch level, only worlds with a pending event (gpu/toi_loop.py)
            state, tst = toi_loop.solve_toi(model, state, settings, dt, max_events)
            return state if keep_none else state._replace(toi=tst)
        from .gpu.toi_search import search
        from .gpu.toi_event import event

    def body(c):
        state, tst, go, pending, ms, ma, it = c
        state, tst = event(model, state, tst, settings, dt, pending, ms, ma, axis_name)
        state, tst, found, ms, ma = search(model, state, tst, settings, go, axis_name)
        return state, tst, go & found, go & found, ms, ma, it + 1

    init = (state, tst, go0, jnp.asarray(False), jnp.int32(0), F32(1.0), jnp.int32(0))
    state, tst = jax.lax.while_loop(cond, body, init)[:2]
    return state if keep_none else state._replace(toi=tst)
