"""Revolute, prismatic and weld joints of Box2D 2.3.1 (b2RevoluteJoint.cpp, b2PrismaticJoint.cpp, b2WeldJoint.cpp):
InitVelocityConstraints, SolveVelocityConstraints, SolvePositionConstraints, line by line.

Revolute has two solvers, picked by the static Settings.revolute_v24:
- True (default, the game): Box2D v2.4.1's b2RevoluteJoint, which Unity's Physics2D grafted onto its 2.3 tree
  (docs/understanding/native-box2d.md section 4; oracle: scripts/patches/box2d-2.3.1-revolute24.patch). Point
  constraint always on its own 2x2 K; motor, then lower and upper limit rows on the axial mass 1/(iA+iB), each with
  a clamped (>= 0) accumulated impulse and a speculative bias max(C, 0) * inv_dt, so they run every iteration
  whenever the limit is enabled (no limit state). The position pass re-derives the limit case from the current
  angle every iteration and its error is |C| (C clamped, slop included). impulse z is never written (stays 0) and
  limit_state is only a readout (2.3.1's classification at the step start). The impulse rows carried through the
  velocity iterations are then (J, 5): x, y, z, lower, upper; the column count is how the loops know the mode.
- False: stock 2.3.1 (3x3 block with limit states, limit impulse in z); impulse rows (J, 3).

Joints are processed in island solve order (island.py). Per-joint constants are built for all ordered joints at once
(they depend only on the step-start poses); warm starting and the iterations run one joint at a time with
loops.ordered_loop, the joint's type picking its branch with lax.switch (a select of all three under vmap).
A joint's solver row is a dict; its 3x3 matrix is stored as columns ex, ey, ez (revolute m_mass, prismatic m_K,
weld m_mass).
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import math2d as m
from .loops import ordered_loop
from .settings import PI
from .world import AT_LOWER, AT_UPPER, EQUAL_LIMITS, INACTIVE_LIMIT, PRISMATIC, REVOLUTE, WELD

F32 = np.float32
TWO_PI = F32(F32(2.0) * PI)  # 2.0f * b2_pi, folded by the compiler


def _branch(jtype):
    """lax.switch index: 0 revolute, 1 prismatic, 2 weld."""
    return jnp.where(jtype == REVOLUTE, 0, jnp.where(jtype == PRISMATIC, 1, 2)).astype(jnp.int32)


def _k_rev(mA, mB, iA, iB, rAx, rAy, rBx, rBy):
    """The revolute/weld 3x3 effective mass (InitVelocityConstraints form, iA last in each product)."""
    exx = ((mA + mB) + m.mul3(rAy, rAy, iA)) + m.mul3(rBy, rBy, iB)
    eyx = m.mul3(-rAy, rAx, iA) - m.mul3(rBy, rBx, iB)
    ezx = m.mul(-rAy, iA) - m.mul(rBy, iB)
    eyy = ((mA + mB) + m.mul3(rAx, rAx, iA)) + m.mul3(rBx, rBx, iB)
    ezy = m.mul(rAx, iA) + m.mul(rBx, iB)
    ezz = iA + iB
    return (exx, eyx, ezx), (eyx, eyy, ezy), (ezx, ezy, ezz)


def _k_prism(mA, mB, iA, iB, s1, s2, a1, a2):
    k11 = ((mA + mB) + m.mul3(iA, s1, s1)) + m.mul3(iB, s2, s2)
    k12 = m.mul(iA, s1) + m.mul(iB, s2)
    k13 = m.mul3(iA, s1, a1) + m.mul3(iB, s2, a2)
    k22 = iA + iB
    k22 = jnp.where(k22 == 0, F32(1), k22)
    k23 = m.mul(iA, a1) + m.mul(iB, a2)
    k33 = ((mA + mB) + m.mul3(iA, a1, a1)) + m.mul3(iB, a2, a2)
    return (k11, k12, k13), (k12, k22, k23), (k13, k23, k33)


def prepare(model, state, isl, pos, settings, dt, dt_ratio, warm, rot=None):
    """InitVelocityConstraints for every ordered joint, minus the velocity writes (warm_start does those).
    Returns (rows, impulse (J,3), motor_impulse (J,)) in solve order."""
    jt, bd = state.joints, state.bodies
    J = model.joint_type.shape[0]
    valid = jnp.arange(J) < isl.n_joints
    jid = jnp.where(valid, isl.joint_order, 0)
    typ = model.joint_type[jid]
    ia, ib = model.joint_a[jid], model.joint_b[jid]
    mA, mB = bd.inv_mass[ia], bd.inv_mass[ib]
    iA, iB = bd.inv_inertia[ia], bd.inv_inertia[ib]
    lcAx, lcAy = bd.local_center[ia, 0], bd.local_center[ia, 1]
    lcBx, lcBy = bd.local_center[ib, 0], bd.local_center[ib, 1]
    cx, cy, a = pos
    cAx, cAy, aA = cx[ia], cy[ia], a[ia]
    cBx, cBy, aB = cx[ib], cy[ib], a[ib]
    qc, qs = m.rot_of(a) if rot is None else rot  # once per body (elementwise: the same values), per joint
    qAc, qAs = qc[ia], qs[ia]
    qBc, qBs = qc[ib], qs[ib]
    lax_, lay = model.local_anchor_a[jid, 0] - lcAx, model.local_anchor_a[jid, 1] - lcAy
    lbx, lby = model.local_anchor_b[jid, 0] - lcBx, model.local_anchor_b[jid, 1] - lcBy
    rAx, rAy = m.rot(qAc, qAs, lax_, lay)
    rBx, rBy = m.rot(qBc, qBs, lbx, lby)
    ref = model.reference_angle[jid]
    en_motor, en_limit = jt.enable_motor[jid], jt.enable_limit[jid]
    lower, upper = jt.lower[jid], jt.upper[jid]
    state_in = jt.limit_state[jid]
    imp = jt.impulse[jid]
    ix, iy, iz = imp[:, 0], imp[:, 1], imp[:, 2]
    v24 = settings.revolute_v24
    mimp = jt.motor_impulse[jid]
    is_rev, is_pri, is_weld = typ == REVOLUTE, typ == PRISMATIC, typ == WELD

    # ---- revolute
    rev_K = _k_rev(mA, mB, iA, iB, rAx, rAy, rBx, rBy)
    fixed = (iA + iB) == 0
    rev_mm = m.inv_if_pos(iA + iB)
    rev_mimp = jnp.where(~en_motor | fixed, F32(0), mimp)
    angle = (aB - aA) - ref
    equal = m.babs(upper - lower) < m.mul(F32(2), settings.angular_slop)
    lim_on = en_limit & ~fixed
    rev_state = jnp.where(~lim_on, INACTIVE_LIMIT, jnp.where(equal, EQUAL_LIMITS, jnp.where(
        angle <= lower, AT_LOWER, jnp.where(angle >= upper, AT_UPPER, INACTIVE_LIMIT))))
    rev_zero_z = lim_on & ~equal & ((rev_state == INACTIVE_LIMIT) | (rev_state != state_in))

    # ---- prismatic
    dx = ((cBx - cAx) + rBx) - rAx
    dy = ((cBy - cAy) + rBy) - rAy
    axx, axy = m.rot(qAc, qAs, model.local_axis_a[jid, 0], model.local_axis_a[jid, 1])
    a1 = m.cross(dx + rAx, dy + rAy, axx, axy)
    a2 = m.cross(rBx, rBy, axx, axy)
    pri_mm = ((mA + mB) + m.mul3(iA, a1, a1)) + m.mul3(iB, a2, a2)
    pri_mm = jnp.where(pri_mm > 0, F32(1) / jnp.where(pri_mm > 0, pri_mm, F32(1)), pri_mm)
    pex, pey = m.rot(qAc, qAs, model.local_yaxis_a[jid, 0], model.local_yaxis_a[jid, 1])
    s1 = m.cross(dx + rAx, dy + rAy, pex, pey)
    s2 = m.cross(rBx, rBy, pex, pey)
    pri_K = _k_prism(mA, mB, iA, iB, s1, s2, a1, a2)
    trans = m.dot(axx, axy, dx, dy)
    pequal = m.babs(upper - lower) < m.mul(F32(2), settings.linear_slop)
    pri_state = jnp.where(~en_limit, INACTIVE_LIMIT, jnp.where(pequal, EQUAL_LIMITS, jnp.where(
        trans <= lower, AT_LOWER, jnp.where(trans >= upper, AT_UPPER, INACTIVE_LIMIT))))
    pri_zero_z = ~en_limit | (~pequal & ((pri_state == INACTIVE_LIMIT) | (pri_state != state_in)))
    pri_mimp = jnp.where(en_motor, mimp, F32(0))

    # ---- weld
    wK = rev_K
    soft = jt.frequency[jid] > 0
    inv22 = m.inverse22_of33(wK[0], wK[1])
    sym = m.sym_inverse33(*wK)
    invM = iA + iB
    wm = m.inv_if_pos(invM)
    C = (aB - aA) - ref
    omega = m.mul(TWO_PI, jt.frequency[jid])
    d = m.mul3(m.mul(F32(2), wm), jt.damping[jid], omega)
    k = m.mul3(wm, omega, omega)
    h = dt
    gamma = m.inv_or_zero(m.mul(h, d + m.mul(h, k)))
    bias = m.mul(m.mul3(C, h, k), gamma)
    ezz_soft = m.inv_or_zero(invM + gamma)
    use22 = soft | (wK[2][2] == 0)
    w_mass = tuple(tuple(jnp.where(use22, a_, b_) for a_, b_ in zip(ca, cb)) for ca, cb in zip(inv22, sym))
    w_mass = (w_mass[0], w_mass[1], (w_mass[2][0], w_mass[2][1], jnp.where(soft, ezz_soft, w_mass[2][2])))
    w_gamma = jnp.where(soft, gamma, F32(0))
    w_bias = jnp.where(soft, bias, F32(0))

    pick = lambda r, p, w: jnp.where(is_rev, r, jnp.where(is_pri, p, w))
    K = tuple(tuple(pick(r, p, w) for r, p, w in zip(cr, cp, cw)) for cr, cp, cw in zip(rev_K, pri_K, w_mass))
    lstate = pick(rev_state, pri_state, INACTIVE_LIMIT).astype(jnp.int32)
    iz = jnp.where(pick(False if v24 else rev_zero_z, pri_zero_z, False), F32(0), iz)
    mimp = pick(rev_mimp, pri_mimp, F32(0))
    # Warm-start scaling (or reset).
    sc = lambda x: jnp.where(warm, m.mul(x, dt_ratio), F32(0))
    if v24:
        # v2.4 revolute: limit impulses reset when the limit is off (or rotation fixed); z is left alone (0).
        lo = jnp.where(is_rev & lim_on, jt.lower_impulse[jid], F32(0))
        up = jnp.where(is_rev & lim_on, jt.upper_impulse[jid], F32(0))
        imp = jnp.stack([sc(ix), sc(iy), jnp.where(is_rev, iz, sc(iz)), sc(lo), sc(up)], 1)
    else:
        imp = jnp.stack([sc(ix), sc(iy), sc(iz)], 1)
    mimp = sc(mimp)
    inv_dt = jnp.where(dt > 0, F32(1) / jnp.where(dt > 0, dt, F32(1)), F32(0))  # b2TimeStep.inv_dt

    rows = dict(valid=valid, jid=jid, branch=_branch(typ), ia=ia, ib=ib, mA=mA, mB=mB, iA=iA, iB=iB,
                rAx=rAx, rAy=rAy, rBx=rBx, rBy=rBy, Kx=jnp.stack(K[0], 1), Ky=jnp.stack(K[1], 1),
                Kz=jnp.stack(K[2], 1), motor_mass=pick(rev_mm, pri_mm, F32(0)), limit_state=lstate,
                enable_motor=en_motor, enable_limit=en_limit, motor_speed=jt.motor_speed[jid],
                max_impulse=m.mul(dt, jt.max_motor[jid]), axx=axx, axy=axy, pex=pex, pey=pey, a1=a1, a2=a2,
                s1=s1, s2=s2, soft=soft, gamma=w_gamma, bias=w_bias, angle=angle,
                inv_dt=jnp.broadcast_to(inv_dt, valid.shape),
                # position-solve constants
                lax=lax_, lay=lay, lbx=lbx, lby=lby, ref=ref, lower=lower, upper=upper,
                lxx=model.local_axis_a[jid, 0], lxy=model.local_axis_a[jid, 1],
                lyx=model.local_yaxis_a[jid, 0], lyy=model.local_yaxis_a[jid, 1])
    return rows, imp, mimp


def _row(rows, k):
    return {key: val[k] for key, val in rows.items()}


def _cols(r):
    return tuple(r["Kx"][i] for i in range(3)), tuple(r["Ky"][i] for i in range(3)), tuple(r["Kz"][i] for i in range(3))


def _get(vel, r):
    vx, vy, w = vel
    ia, ib = r["ia"], r["ib"]
    return (m.pick(vx, ia), m.pick(vy, ia)), m.pick(w, ia), (m.pick(vx, ib), m.pick(vy, ib)), m.pick(w, ib)


def _put(vel, r, vA, wA, vB, wB, live=True):
    vx, vy, w = vel
    B = vx.shape[-1]
    ia = jnp.where(live, r["ia"], B)  # an out-of-range index makes the one-hot put a no-op
    ib = jnp.where(live, r["ib"], B)
    vx = m.put(m.put(vx, ia, vA[0]), ib, vB[0])
    vy = m.put(m.put(vy, ia, vA[1]), ib, vB[1])
    w = m.put(m.put(w, ia, wA), ib, wB)
    return vx, vy, w


def warm_start(rows, vel, imp, mimp, n):
    """The velocity part of each InitVelocityConstraints (warm starting), in joint order."""
    if rows["valid"].shape[0] == 0:
        return vel
    v24 = imp.shape[1] == 5

    def body(k, vel, live):
        r = _row(rows, k)
        vA, wA, vB, wB = _get(vel, r)
        mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
        ix, iy, iz = imp[k, 0], imp[k, 1], imp[k, 2]
        mi = mimp[k]

        def rev(_):
            Px, Py = ix, iy
            return ((vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py)),
                    wA - m.mul(iA, (m.cross(r["rAx"], r["rAy"], Px, Py) + mi) + iz),
                    (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py)),
                    wB + m.mul(iB, (m.cross(r["rBx"], r["rBy"], Px, Py) + mi) + iz))

        def rev24(_):
            Px, Py = ix, iy
            axial = (mi + imp[k, 3]) - imp[k, 4]  # motor + lower - upper
            return ((vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py)),
                    wA - m.mul(iA, m.cross(r["rAx"], r["rAy"], Px, Py) + axial),
                    (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py)),
                    wB + m.mul(iB, m.cross(r["rBx"], r["rBy"], Px, Py) + axial))

        def pri(_):
            mz = mi + iz
            Px = m.mul(ix, r["pex"]) + m.mul(mz, r["axx"])
            Py = m.mul(ix, r["pey"]) + m.mul(mz, r["axy"])
            LA = (m.mul(ix, r["s1"]) + iy) + m.mul(mz, r["a1"])
            LB = (m.mul(ix, r["s2"]) + iy) + m.mul(mz, r["a2"])
            return ((vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py)), wA - m.mul(iA, LA),
                    (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py)), wB + m.mul(iB, LB))

        def weld(_):
            Px, Py = ix, iy
            return ((vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py)),
                    wA - m.mul(iA, m.cross(r["rAx"], r["rAy"], Px, Py) + iz),
                    (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py)),
                    wB + m.mul(iB, m.cross(r["rBx"], r["rBy"], Px, Py) + iz))

        vA, wA, vB, wB = jax.lax.switch(r["branch"], (rev24 if v24 else rev, pri, weld), None)
        return _put(vel, r, vA, wA, vB, wB, live)
    return ordered_loop(n, rows["valid"].shape[0], body, vel, static=True)


def _dv(vB, wB, rbx, rby, vA, wA, rax, ray):
    bx, by = m.cross_sv(wB, rbx, rby)
    ax, ay = m.cross_sv(wA, rax, ray)
    return ((vB[0] + bx) - vA[0]) - ax, ((vB[1] + by) - vA[1]) - ay


def _rev_velocity(r, vA, wA, vB, wB, imp, mi):
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    ex, ey, ez = _cols(r)
    fixed = (iA + iB) == 0
    st = r["limit_state"]
    # motor
    on = r["enable_motor"] & (st != EQUAL_LIMITS) & ~fixed
    cdot = (wB - wA) - r["motor_speed"]
    impulse = m.mul(-r["motor_mass"], cdot)
    new_mi = m.clamp(mi + impulse, -r["max_impulse"], r["max_impulse"])
    impulse = new_mi - mi
    wA = jnp.where(on, wA - m.mul(iA, impulse), wA)
    wB = jnp.where(on, wB + m.mul(iB, impulse), wB)
    mi = jnp.where(on, new_mi, mi)
    ix, iy, iz = imp
    rax, ray, rbx, rby = r["rAx"], r["rAy"], r["rBx"], r["rBy"]
    c1x, c1y = _dv(vB, wB, rbx, rby, vA, wA, rax, ray)
    # limit active: 3x3 block
    lim = r["enable_limit"] & (st != INACTIVE_LIMIT) & ~fixed
    c2 = wB - wA
    s = m.solve33(ex, ey, ez, (c1x, c1y, c2))
    jx, jy, jz = -s[0], -s[1], -s[2]
    new_z = iz + jz
    redo = ((st == AT_LOWER) & (new_z < 0)) | ((st == AT_UPPER) & (new_z > 0))
    rx, ry = m.solve22(ex[0], ey[0], ex[1], ey[1], -c1x + m.mul(iz, ez[0]), -c1y + m.mul(iz, ez[1]))
    lx = jnp.where(redo, rx, jx)
    ly = jnp.where(redo, ry, jy)
    lz = jnp.where(redo, -iz, jz)
    l_imp = (jnp.where(redo, ix + rx, ix + jx), jnp.where(redo, iy + ry, iy + jy), jnp.where(redo, F32(0), iz + jz))
    l_vA = (vA[0] - m.mul(mA, lx), vA[1] - m.mul(mA, ly))
    l_wA = wA - m.mul(iA, m.cross(rax, ray, lx, ly) + lz)
    l_vB = (vB[0] + m.mul(mB, lx), vB[1] + m.mul(mB, ly))
    l_wB = wB + m.mul(iB, m.cross(rbx, rby, lx, ly) + lz)
    # point only: 2x2
    px, py = m.solve22(ex[0], ey[0], ex[1], ey[1], -c1x, -c1y)
    p_imp = (ix + px, iy + py, iz)
    p_vA = (vA[0] - m.mul(mA, px), vA[1] - m.mul(mA, py))
    p_wA = wA - m.mul(iA, m.cross(rax, ray, px, py))
    p_vB = (vB[0] + m.mul(mB, px), vB[1] + m.mul(mB, py))
    p_wB = wB + m.mul(iB, m.cross(rbx, rby, px, py))
    s_ = lambda a, b: jnp.where(lim, a, b)
    return ((s_(l_vA[0], p_vA[0]), s_(l_vA[1], p_vA[1])), s_(l_wA, p_wA), (s_(l_vB[0], p_vB[0]), s_(l_vB[1], p_vB[1])),
            s_(l_wB, p_wB), tuple(s_(a, b) for a, b in zip(l_imp, p_imp)), mi)


def _rev24_velocity(r, vA, wA, vB, wB, imp, mi):
    """b2RevoluteJoint::SolveVelocityConstraints, Box2D v2.4.1. imp = (x, y, z, lower, upper)."""
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    ex, ey, _ = _cols(r)
    fixed = (iA + iB) == 0
    am = r["motor_mass"]  # m_axialMass
    # motor
    on = r["enable_motor"] & ~fixed
    cdot = (wB - wA) - r["motor_speed"]
    impulse = m.mul(-am, cdot)
    new_mi = m.clamp(mi + impulse, -r["max_impulse"], r["max_impulse"])
    impulse = new_mi - mi
    wA = jnp.where(on, wA - m.mul(iA, impulse), wA)
    wB = jnp.where(on, wB + m.mul(iB, impulse), wB)
    mi = jnp.where(on, new_mi, mi)
    ix, iy, iz, lo, up = imp
    lim = r["enable_limit"] & ~fixed
    # lower limit
    C = r["angle"] - r["lower"]
    impulse = m.mul(-am, (wB - wA) + m.mul(m.bmax(C, F32(0)), r["inv_dt"]))
    new_lo = m.bmax(lo + impulse, F32(0))
    impulse = new_lo - lo
    wA = jnp.where(lim, wA - m.mul(iA, impulse), wA)
    wB = jnp.where(lim, wB + m.mul(iB, impulse), wB)
    lo = jnp.where(lim, new_lo, lo)
    # upper limit (signs flipped so C and the impulse stay positive)
    C = r["upper"] - r["angle"]
    impulse = m.mul(-am, (wA - wB) + m.mul(m.bmax(C, F32(0)), r["inv_dt"]))
    new_up = m.bmax(up + impulse, F32(0))
    impulse = new_up - up
    wA = jnp.where(lim, wA + m.mul(iA, impulse), wA)
    wB = jnp.where(lim, wB - m.mul(iB, impulse), wB)
    up = jnp.where(lim, new_up, up)
    # point to point, 2x2
    rax, ray, rbx, rby = r["rAx"], r["rAy"], r["rBx"], r["rBy"]
    c1x, c1y = _dv(vB, wB, rbx, rby, vA, wA, rax, ray)
    px, py = m.solve22(ex[0], ey[0], ex[1], ey[1], -c1x, -c1y)
    vA = (vA[0] - m.mul(mA, px), vA[1] - m.mul(mA, py))
    wA = wA - m.mul(iA, m.cross(rax, ray, px, py))
    vB = (vB[0] + m.mul(mB, px), vB[1] + m.mul(mB, py))
    wB = wB + m.mul(iB, m.cross(rbx, rby, px, py))
    return vA, wA, vB, wB, (ix + px, iy + py, iz, lo, up), mi


def _pass_limits(fn):
    """Wrap a 3-column velocity branch for the (x, y, z, lower, upper) rows: the last two pass through."""
    def run(r, vA, wA, vB, wB, imp, mi):
        vA, wA, vB, wB, i3, mi = fn(r, vA, wA, vB, wB, imp[:3], mi)
        return vA, wA, vB, wB, tuple(i3) + tuple(imp[3:]), mi
    return run


def _pri_velocity(r, vA, wA, vB, wB, imp, mi):
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    ex, ey, ez = _cols(r)
    st = r["limit_state"]
    axx, axy, pex, pey = r["axx"], r["axy"], r["pex"], r["pey"]
    a1, a2, s1, s2 = r["a1"], r["a2"], r["s1"], r["s2"]
    on = r["enable_motor"] & (st != EQUAL_LIMITS)
    cdot = (m.dot(axx, axy, vB[0] - vA[0], vB[1] - vA[1]) + m.mul(a2, wB)) - m.mul(a1, wA)
    impulse = m.mul(r["motor_mass"], r["motor_speed"] - cdot)
    new_mi = m.clamp(mi + impulse, -r["max_impulse"], r["max_impulse"])
    impulse = new_mi - mi
    Px, Py = m.mul(impulse, axx), m.mul(impulse, axy)
    LA, LB = m.mul(impulse, a1), m.mul(impulse, a2)
    vA = (jnp.where(on, vA[0] - m.mul(mA, Px), vA[0]), jnp.where(on, vA[1] - m.mul(mA, Py), vA[1]))
    wA = jnp.where(on, wA - m.mul(iA, LA), wA)
    vB = (jnp.where(on, vB[0] + m.mul(mB, Px), vB[0]), jnp.where(on, vB[1] + m.mul(mB, Py), vB[1]))
    wB = jnp.where(on, wB + m.mul(iB, LB), wB)
    mi = jnp.where(on, new_mi, mi)

    c1x = (m.dot(pex, pey, vB[0] - vA[0], vB[1] - vA[1]) + m.mul(s2, wB)) - m.mul(s1, wA)
    c1y = wB - wA
    ix, iy, iz = imp
    lim = r["enable_limit"] & (st != INACTIVE_LIMIT)
    # limit: 3x3 then clamp the axial impulse and re-solve the other two
    c2 = (m.dot(axx, axy, vB[0] - vA[0], vB[1] - vA[1]) + m.mul(a2, wB)) - m.mul(a1, wA)
    df = m.solve33(ex, ey, ez, (-c1x, -c1y, -c2))
    nx_, ny_, nz_ = ix + df[0], iy + df[1], iz + df[2]
    nz_ = jnp.where(st == AT_LOWER, m.bmax(nz_, F32(0)), jnp.where(st == AT_UPPER, m.bmin(nz_, F32(0)), nz_))
    bx = -c1x - m.mul(nz_ - iz, ez[0])
    by = -c1y - m.mul(nz_ - iz, ez[1])
    fx, fy = m.solve22(ex[0], ey[0], ex[1], ey[1], bx, by)
    fx, fy = fx + ix, fy + iy
    dfx, dfy, dfz = fx - ix, fy - iy, nz_ - iz
    l_imp = (fx, fy, nz_)
    lPx = m.mul(dfx, pex) + m.mul(dfz, axx)
    lPy = m.mul(dfx, pey) + m.mul(dfz, axy)
    lLA = (m.mul(dfx, s1) + dfy) + m.mul(dfz, a1)
    lLB = (m.mul(dfx, s2) + dfy) + m.mul(dfz, a2)
    # no limit: 2x2
    gx, gy = m.solve22(ex[0], ey[0], ex[1], ey[1], -c1x, -c1y)
    p_imp = (ix + gx, iy + gy, iz)
    pPx, pPy = m.mul(gx, pex), m.mul(gx, pey)
    pLA = m.mul(gx, s1) + gy
    pLB = m.mul(gx, s2) + gy
    s_ = lambda a, b: jnp.where(lim, a, b)
    Px, Py, LA, LB = s_(lPx, pPx), s_(lPy, pPy), s_(lLA, pLA), s_(lLB, pLB)
    vA = (vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py))
    wA = wA - m.mul(iA, LA)
    vB = (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py))
    wB = wB + m.mul(iB, LB)
    return vA, wA, vB, wB, tuple(s_(a, b) for a, b in zip(l_imp, p_imp)), mi


def _weld_velocity(r, vA, wA, vB, wB, imp, mi):
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    ex, ey, ez = _cols(r)
    rax, ray, rbx, rby = r["rAx"], r["rAy"], r["rBx"], r["rBy"]
    ix, iy, iz = imp
    # soft: angular row with gamma/bias, then the 2x2 point rows
    c2 = wB - wA
    i2 = m.mul(-ez[2], (c2 + r["bias"]) + m.mul(r["gamma"], iz))
    s_iz = iz + i2
    s_wA = wA - m.mul(iA, i2)
    s_wB = wB + m.mul(iB, i2)
    c1x, c1y = _dv(vB, s_wB, rbx, rby, vA, s_wA, rax, ray)
    q = m.mat33_mul22(ex, ey, c1x, c1y)
    i1x, i1y = -q[0], -q[1]
    s_imp = (ix + i1x, iy + i1y, s_iz)
    s_vA = (vA[0] - m.mul(mA, i1x), vA[1] - m.mul(mA, i1y))
    s_wA = s_wA - m.mul(iA, m.cross(rax, ray, i1x, i1y))
    s_vB = (vB[0] + m.mul(mB, i1x), vB[1] + m.mul(mB, i1y))
    s_wB = s_wB + m.mul(iB, m.cross(rbx, rby, i1x, i1y))
    # rigid: 3x3
    d1x, d1y = _dv(vB, wB, rbx, rby, vA, wA, rax, ray)
    q3 = m.mat33_mul(ex, ey, ez, (d1x, d1y, wB - wA))
    jx, jy, jz = -q3[0], -q3[1], -q3[2]
    h_imp = (ix + jx, iy + jy, iz + jz)
    h_vA = (vA[0] - m.mul(mA, jx), vA[1] - m.mul(mA, jy))
    h_wA = wA - m.mul(iA, m.cross(rax, ray, jx, jy) + jz)
    h_vB = (vB[0] + m.mul(mB, jx), vB[1] + m.mul(mB, jy))
    h_wB = wB + m.mul(iB, m.cross(rbx, rby, jx, jy) + jz)
    s_ = lambda a, b: jnp.where(r["soft"], a, b)
    return ((s_(s_vA[0], h_vA[0]), s_(s_vA[1], h_vA[1])), s_(s_wA, h_wA), (s_(s_vB[0], h_vB[0]), s_(s_vB[1], h_vB[1])),
            s_(s_wB, h_wB), tuple(s_(a, b) for a, b in zip(s_imp, h_imp)), mi)


def solve_velocity(rows, vel, imp, mimp, n):
    """One pass of SolveVelocityConstraints over the first n ordered joints."""
    if rows["valid"].shape[0] == 0:
        return vel, imp, mimp
    W = imp.shape[1]
    if W == 5:
        branches = (_rev24_velocity, _pass_limits(_pri_velocity), _pass_limits(_weld_velocity))
    else:
        branches = (_rev_velocity, _pri_velocity, _weld_velocity)

    def body(k, carry, live):
        vel, imp, mimp = carry
        r = _row(rows, k)
        vA, wA, vB, wB = _get(vel, r)
        args = (r, vA, wA, vB, wB, tuple(imp[k, i] for i in range(W)), mimp[k])
        vA, wA, vB, wB, ik, mk = jax.lax.switch(r["branch"], tuple((lambda a, f=f: f(*a)) for f in branches), args)
        vel = _put(vel, r, vA, wA, vB, wB, live)
        return (vel, imp.at[k].set(jnp.where(live, jnp.stack(ik), imp[k])),
                mimp.at[k].set(jnp.where(live, mk, mimp[k])))
    return ordered_loop(n, rows["valid"].shape[0], body, (vel, imp, mimp), static=True)


def _rev_position(r, cA, aA, cB, aB, s):
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    fixed = (iA + iB) == 0
    st = r["limit_state"]
    lim = r["enable_limit"] & (st != INACTIVE_LIMIT) & ~fixed
    angle = (aB - aA) - r["ref"]
    mac = s.max_angular_correction
    ce = m.clamp(angle - r["lower"], -mac, mac)
    cl = angle - r["lower"]
    cu = angle - r["upper"]
    C = jnp.where(st == EQUAL_LIMITS, ce, jnp.where(st == AT_LOWER, m.clamp(cl + s.angular_slop, -mac, F32(0)),
                                                    m.clamp(cu - s.angular_slop, F32(0), mac)))
    ang_err = jnp.where(st == EQUAL_LIMITS, m.babs(ce), jnp.where(st == AT_LOWER, -cl, cu))
    limit_impulse = m.mul(-r["motor_mass"], C)
    aA = jnp.where(lim, aA - m.mul(iA, limit_impulse), aA)
    aB = jnp.where(lim, aB + m.mul(iB, limit_impulse), aB)
    ang_err = jnp.where(lim, ang_err, F32(0))
    return _rev_point_position(r, cA, aA, cB, aB, s, ang_err)


def _rev24_position(r, cA, aA, cB, aB, s):
    """b2RevoluteJoint::SolvePositionConstraints, Box2D v2.4.1: the limit case comes from the current angle."""
    iA, iB = r["iA"], r["iB"]
    fixed = (iA + iB) == 0
    lim = r["enable_limit"] & ~fixed
    angle = (aB - aA) - r["ref"]
    lower, upper = r["lower"], r["upper"]
    mac = s.max_angular_correction
    equal = m.babs(upper - lower) < m.mul(F32(2), s.angular_slop)
    C = jnp.where(equal, m.clamp(angle - lower, -mac, mac),
                  jnp.where(angle <= lower, m.clamp((angle - lower) + s.angular_slop, -mac, F32(0)),
                            jnp.where(angle >= upper, m.clamp((angle - upper) - s.angular_slop, F32(0), mac), F32(0))))
    limit_impulse = m.mul(-r["motor_mass"], C)
    aA = jnp.where(lim, aA - m.mul(iA, limit_impulse), aA)
    aB = jnp.where(lim, aB + m.mul(iB, limit_impulse), aB)
    ang_err = jnp.where(lim, m.babs(C), F32(0))
    return _rev_point_position(r, cA, aA, cB, aB, s, ang_err)


def _rev_point_position(r, cA, aA, cB, aB, s, ang_err):
    """The point-to-point part of the revolute position solve (the same code in 2.3.1 and 2.4.1)."""
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    qAc, qAs = m.rot_of(aA)
    qBc, qBs = m.rot_of(aB)
    rax, ray = m.rot(qAc, qAs, r["lax"], r["lay"])
    rbx, rby = m.rot(qBc, qBs, r["lbx"], r["lby"])
    Cx = ((cB[0] + rbx) - cA[0]) - rax
    Cy = ((cB[1] + rby) - cA[1]) - ray
    pos_err = m.length(Cx, Cy)
    kxx = ((mA + mB) + m.mul3(iA, ray, ray)) + m.mul3(iB, rby, rby)
    kxy = m.mul3(-iA, rax, ray) - m.mul3(iB, rbx, rby)
    kyy = ((mA + mB) + m.mul3(iA, rax, rax)) + m.mul3(iB, rbx, rbx)
    jx, jy = m.solve22(kxx, kxy, kxy, kyy, Cx, Cy)
    jx, jy = -jx, -jy
    cA = (cA[0] - m.mul(mA, jx), cA[1] - m.mul(mA, jy))
    aA = aA - m.mul(iA, m.cross(rax, ray, jx, jy))
    cB = (cB[0] + m.mul(mB, jx), cB[1] + m.mul(mB, jy))
    aB = aB + m.mul(iB, m.cross(rbx, rby, jx, jy))
    return cA, aA, cB, aB, (pos_err <= s.linear_slop) & (ang_err <= s.angular_slop)


def _pri_position(r, cA, aA, cB, aB, s):
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    qAc, qAs = m.rot_of(aA)
    qBc, qBs = m.rot_of(aB)
    rax, ray = m.rot(qAc, qAs, r["lax"], r["lay"])
    rbx, rby = m.rot(qBc, qBs, r["lbx"], r["lby"])
    dx = ((cB[0] + rbx) - cA[0]) - rax
    dy = ((cB[1] + rby) - cA[1]) - ray
    axx, axy = m.rot(qAc, qAs, r["lxx"], r["lxy"])
    a1 = m.cross(dx + rax, dy + ray, axx, axy)
    a2 = m.cross(rbx, rby, axx, axy)
    pex, pey = m.rot(qAc, qAs, r["lyx"], r["lyy"])
    s1 = m.cross(dx + rax, dy + ray, pex, pey)
    s2 = m.cross(rbx, rby, pex, pey)
    c1x = m.dot(pex, pey, dx, dy)
    c1y = (aB - aA) - r["ref"]
    lin_err, ang_err = m.babs(c1x), m.babs(c1y)
    lo, hi = r["lower"], r["upper"]
    t = m.dot(axx, axy, dx, dy)
    mlc = s.max_linear_correction
    eq = m.babs(hi - lo) < m.mul(F32(2), s.linear_slop)
    at_lo, at_hi = t <= lo, t >= hi
    en = r["enable_limit"]
    active = en & (eq | at_lo | at_hi)
    c2 = jnp.where(eq, m.clamp(t, -mlc, mlc), jnp.where(at_lo, m.clamp((t - lo) + s.linear_slop, -mlc, F32(0)),
                                                        m.clamp((t - hi) - s.linear_slop, F32(0), mlc)))
    e2 = jnp.where(eq, m.babs(t), jnp.where(at_lo, lo - t, t - hi))
    lin_err = jnp.where(active, m.bmax(lin_err, e2), lin_err)
    ex, ey, ez = _k_prism(mA, mB, iA, iB, s1, s2, a1, a2)
    i3 = m.solve33(ex, ey, ez, (-c1x, -c1y, -c2))
    k11, k12, k22 = ex[0], ex[1], ey[1]
    g = m.solve22(k11, k12, k12, k22, -c1x, -c1y)
    ix = jnp.where(active, i3[0], g[0])
    iy = jnp.where(active, i3[1], g[1])
    iz = jnp.where(active, i3[2], F32(0))
    Px = m.mul(ix, pex) + m.mul(iz, axx)
    Py = m.mul(ix, pey) + m.mul(iz, axy)
    LA = (m.mul(ix, s1) + iy) + m.mul(iz, a1)
    LB = (m.mul(ix, s2) + iy) + m.mul(iz, a2)
    cA = (cA[0] - m.mul(mA, Px), cA[1] - m.mul(mA, Py))
    aA = aA - m.mul(iA, LA)
    cB = (cB[0] + m.mul(mB, Px), cB[1] + m.mul(mB, Py))
    aB = aB + m.mul(iB, LB)
    return cA, aA, cB, aB, (lin_err <= s.linear_slop) & (ang_err <= s.angular_slop)


def _weld_position(r, cA, aA, cB, aB, s):
    mA, mB, iA, iB = r["mA"], r["mB"], r["iA"], r["iB"]
    qAc, qAs = m.rot_of(aA)
    qBc, qBs = m.rot_of(aB)
    rax, ray = m.rot(qAc, qAs, r["lax"], r["lay"])
    rbx, rby = m.rot(qBc, qBs, r["lbx"], r["lby"])
    ex, ey, ez = _k_rev(mA, mB, iA, iB, rax, ray, rbx, rby)
    c1x = ((cB[0] + rbx) - cA[0]) - rax
    c1y = ((cB[1] + rby) - cA[1]) - ray
    pos_err = m.length(c1x, c1y)
    c2 = (aB - aA) - r["ref"]
    p22 = m.solve22(ex[0], ey[0], ex[1], ey[1], c1x, c1y)
    p33 = m.solve33(ex, ey, ez, (c1x, c1y, c2))
    soft = r["soft"]
    use33 = ~soft & (ez[2] > 0)
    ix = jnp.where(use33, -p33[0], -p22[0])
    iy = jnp.where(use33, -p33[1], -p22[1])
    iz = jnp.where(use33, -p33[2], F32(0))
    ang_err = jnp.where(soft, F32(0), m.babs(c2))
    # soft: aA -= iA * cross(rA, P) (no angular term); rigid adds impulse.z (0 in the 2x2 fallback, exact either way)
    tA = jnp.where(soft, m.cross(rax, ray, ix, iy), m.cross(rax, ray, ix, iy) + iz)
    tB = jnp.where(soft, m.cross(rbx, rby, ix, iy), m.cross(rbx, rby, ix, iy) + iz)
    cA = (cA[0] - m.mul(mA, ix), cA[1] - m.mul(mA, iy))
    aA = aA - m.mul(iA, tA)
    cB = (cB[0] + m.mul(mB, ix), cB[1] + m.mul(mB, iy))
    aB = aB + m.mul(iB, tB)
    return cA, aA, cB, aB, (pos_err <= s.linear_slop) & (ang_err <= s.angular_slop)


def solve_position_one(r, pos, settings):
    cx, cy, a = pos
    ia, ib = r["ia"], r["ib"]
    cA, aA = (m.pick(cx, ia), m.pick(cy, ia)), m.pick(a, ia)
    cB, aB = (m.pick(cx, ib), m.pick(cy, ib)), m.pick(a, ib)
    args = (r, cA, aA, cB, aB)
    rev = _rev24_position if settings.revolute_v24 else _rev_position
    return jax.lax.switch(r["branch"], (lambda x: rev(*x, settings), lambda x: _pri_position(*x, settings),
                                        lambda x: _weld_position(*x, settings)), args)


def store(state, rows, imp, mimp, n):
    """Write the ordered impulses and limit states back to the joint table (joint index order)."""
    jt = state.joints
    J = jt.impulse.shape[0]
    if J == 0:
        return jt
    tgt = jnp.where(rows["valid"], rows["jid"], J)
    if imp.shape[1] == 5:  # v2.4 revolute rows: x, y, z, lower, upper
        jt = jt._replace(lower_impulse=jt.lower_impulse.at[tgt].set(imp[:, 3], mode="drop"),
                         upper_impulse=jt.upper_impulse.at[tgt].set(imp[:, 4], mode="drop"))
        imp = imp[:, :3]
    return jt._replace(impulse=jt.impulse.at[tgt].set(imp, mode="drop"),
                       motor_impulse=jt.motor_impulse.at[tgt].set(mimp, mode="drop"),
                       limit_state=jt.limit_state.at[tgt].set(rows["limit_state"], mode="drop"))
