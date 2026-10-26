"""b2ContactSolver (Box2D 2.3.1): velocity constraints with the 2-point block solver, warm start, impulse storage and
the non-linear Gauss-Seidel position solve, over the contacts of every island in solve order (island.py).

Per-constraint data is built for all ordered contacts at once (it depends only on the step-start poses), then the
sequential parts (warm start, 300 velocity passes, position passes) walk the order one contact per iteration (loops.ordered_loop), reading and writing the two bodies' entries of the shared velocity/position arrays: exactly the
Gauss-Seidel sweep of b2ContactSolver, island by island interleaved (islands share no non-static body).
Body state inside the solver is flat (B,) arrays: vx, vy, w and cx, cy, a.
"""
import jax.numpy as jnp
import numpy as np

from . import math2d as m
from .loops import ordered_loop
from .settings import MAX_CONDITION_NUMBER
from .world import CIRCLES, FACE_A

F32 = np.float32


def _xf(cx, cy, a, lcx, lcy):
    """xf.q.Set(a); xf.p = c - b2Mul(xf.q, localCenter)."""
    qc, qs = m.rot_of(a)
    return _xf_q(cx, cy, qc, qs, lcx, lcy)


def _xf_q(cx, cy, qc, qs, lcx, lcy):
    """_xf with xf.q given."""
    rx, ry = m.rot(qc, qs, lcx, lcy)
    return cx - rx, cy - ry, qc, qs


def prepare(model, state, isl, pos, vel, warm, dt_ratio, velocity_threshold, rot=None):
    """b2ContactSolver constructor + InitializeVelocityConstraints for the ordered contacts. pos = (cx, cy, a),
    vel = (vx, vy, w) are the island arrays after velocity integration. rot: m.rot_of(a) if the caller has it."""
    ct, bd = state.contacts, state.bodies
    C = ct.pair.shape[0]
    k_idx = jnp.arange(C)
    valid = k_idx < isl.n_contacts
    slot = jnp.where(valid, isl.contact_order, 0)
    fa, fb = ct.fa[slot], ct.fb[slot]
    ia, ib = model.fixture_body[fa], model.fixture_body[fb]
    mA, mB = bd.inv_mass[ia], bd.inv_mass[ib]
    iA, iB = bd.inv_inertia[ia], bd.inv_inertia[ib]
    lcA, lcB = bd.local_center[ia], bd.local_center[ib]
    rA_, rB_ = model.radius[fa], model.radius[fb]
    mtype, count = ct.mtype[slot], ct.count[slot]
    ln, lp, pl = ct.local_normal[slot], ct.local_point[slot], ct.point_local[slot]
    ni = jnp.where(warm, m.mul(dt_ratio, ct.normal_impulse[slot]), F32(0))
    ti = jnp.where(warm, m.mul(dt_ratio, ct.tangent_impulse[slot]), F32(0))
    cx, cy, a = pos
    vx, vy, w = vel
    cAx, cAy = cx[ia], cy[ia]
    cBx, cBy = cx[ib], cy[ib]
    # xf.q.Set(a) once per body, gathered per contact (the same values as per contact: rot_of is elementwise; the
    # float-float sin/cos over (C, 2) instead of (B,) was a third of the step's XLA time at 16k worlds on the 3090)
    qc, qs = m.rot_of(a) if rot is None else rot
    xfA = _xf_q(cAx, cAy, qc[ia], qs[ia], lcA[:, 0], lcA[:, 1])
    xfB = _xf_q(cBx, cBy, qc[ib], qs[ib], lcB[:, 0], lcB[:, 1])

    # b2WorldManifold::Initialize
    lnx, lny = ln[:, 0], ln[:, 1]
    # circles
    pAx, pAy = m.xf_mul(*xfA, lp[:, 0], lp[:, 1])
    pBx, pBy = m.xf_mul(*xfB, pl[:, 0, 0], pl[:, 0, 1])
    dx, dy = pBx - pAx, pBy - pAy
    far = m.dot(dx, dy, dx, dy) > m.mul(m.EPS, m.EPS)
    nnx, nny, _ = m.normalize(dx, dy)
    cnx, cny = jnp.where(far, nnx, F32(1)), jnp.where(far, nny, F32(0))
    ccAx, ccAy = pAx + m.mul(rA_, cnx), pAy + m.mul(rA_, cny)
    ccBx, ccBy = pBx - m.mul(rB_, cnx), pBy - m.mul(rB_, cny)
    circ_pt = (m.mul(F32(0.5), ccAx + ccBx), m.mul(F32(0.5), ccAy + ccBy))
    # faceA
    fAnx, fAny = m.rot(xfA[2], xfA[3], lnx, lny)
    fAplx, fAply = m.xf_mul(*xfA, lp[:, 0], lp[:, 1])
    # faceB
    fBnx, fBny = m.rot(xfB[2], xfB[3], lnx, lny)
    fBplx, fBply = m.xf_mul(*xfB, lp[:, 0], lp[:, 1])
    is_c, is_a = mtype == CIRCLES, mtype == FACE_A
    pts = []
    for j in range(2):
        # faceA: clip point on B
        qx, qy = m.xf_mul(*xfB, pl[:, j, 0], pl[:, j, 1])
        d = m.dot(qx - fAplx, qy - fAply, fAnx, fAny)
        s_ = rA_ - d
        aAx, aAy = qx + m.mul(s_, fAnx), qy + m.mul(s_, fAny)
        aBx, aBy = qx - m.mul(rB_, fAnx), qy - m.mul(rB_, fAny)
        fa_pt = (m.mul(F32(0.5), aAx + aBx), m.mul(F32(0.5), aAy + aBy))
        # faceB: clip point on A
        qx, qy = m.xf_mul(*xfA, pl[:, j, 0], pl[:, j, 1])
        d = m.dot(qx - fBplx, qy - fBply, fBnx, fBny)
        s_ = rB_ - d
        bBx, bBy = qx + m.mul(s_, fBnx), qy + m.mul(s_, fBny)
        bAx, bAy = qx - m.mul(rA_, fBnx), qy - m.mul(rA_, fBny)
        fb_pt = (m.mul(F32(0.5), bAx + bBx), m.mul(F32(0.5), bAy + bBy))
        pts.append(tuple(jnp.where(is_c, cp, jnp.where(is_a, ap, bp)) for cp, ap, bp in zip(circ_pt, fa_pt, fb_pt)))
    nx = jnp.where(is_c, cnx, jnp.where(is_a, fAnx, -fBnx))
    ny = jnp.where(is_c, cny, jnp.where(is_a, fAny, -fBny))

    tx, ty = m.cross_vs(nx, ny, F32(1.0))
    vAx, vAy, wA = vx[ia], vy[ia], w[ia]
    vBx, vBy, wB = vx[ib], vy[ib], w[ib]
    friction, restitution = ct.friction[slot], ct.restitution[slot]
    out = dict(valid=valid, slot=slot, ia=ia, ib=ib, mA=mA, mB=mB, iA=iA, iB=iB, nx=nx, ny=ny,
               friction=friction, tangent_speed=ct.tangent_speed[slot])
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
        bias.append(jnp.where(vrel < -velocity_threshold, m.mul(-restitution, vrel), F32(0)))
    # 2-point block solver set-up
    rn1A, rn1B = m.cross(*rA[0], nx, ny), m.cross(*rB[0], nx, ny)
    rn2A, rn2B = m.cross(*rA[1], nx, ny), m.cross(*rB[1], nx, ny)
    k11 = ((mA + mB) + m.mul3(iA, rn1A, rn1A)) + m.mul3(iB, rn1B, rn1B)
    k22 = ((mA + mB) + m.mul3(iA, rn2A, rn2A)) + m.mul3(iB, rn2B, rn2B)
    k12 = ((mA + mB) + m.mul3(iA, rn1A, rn2A)) + m.mul3(iB, rn1B, rn2B)
    safe = m.mul(k11, k11) < m.mul(MAX_CONDITION_NUMBER, m.mul(k11, k22) - m.mul(k12, k12))
    two = count == 2
    vc_count = jnp.where(two & ~safe, 1, count)
    det = m.inv_or_zero(m.mul(k11, k22) - m.mul(k12, k12))
    use_k = two & safe
    zero = jnp.zeros_like(k11)
    out.update(
        count=vc_count, pc_count=count,
        rAx=jnp.stack([rA[0][0], rA[1][0]], 1), rAy=jnp.stack([rA[0][1], rA[1][1]], 1),
        rBx=jnp.stack([rB[0][0], rB[1][0]], 1), rBy=jnp.stack([rB[0][1], rB[1][1]], 1),
        nmass=jnp.stack(nmass, 1), tmass=jnp.stack(tmass, 1), bias=jnp.stack(bias, 1),
        ni=ni, ti=ti,
        K=jnp.stack([jnp.where(use_k, k11, zero), jnp.where(use_k, k12, zero), jnp.where(use_k, k12, zero),
                     jnp.where(use_k, k22, zero)], 1),  # ex.x, ex.y, ey.x, ey.y
        NM=jnp.stack([jnp.where(use_k, m.mul(det, k22), zero), jnp.where(use_k, m.mul(-det, k12), zero),
                      jnp.where(use_k, m.mul(-det, k12), zero), jnp.where(use_k, m.mul(det, k11), zero)], 1),
        # position constraint data
        mtype=mtype, lnx=lnx, lny=lny, lpx=lp[:, 0], lpy=lp[:, 1], plx=pl[:, :, 0], ply=pl[:, :, 1],
        lcAx=lcA[:, 0], lcAy=lcA[:, 1], lcBx=lcB[:, 0], lcBy=lcB[:, 1], radA=rA_, radB=rB_)
    return out


def _row(vc, k):
    return {key: val[k] for key, val in vc.items()}


def _apply(vel, c, dvAx, dvAy, dwA, dvBx, dvBy, dwB, live=True):
    """Write back the two bodies (A first, then B, as Box2D's stores); nothing where not live."""
    vx, vy, w = vel
    B = vx.shape[-1]
    ia, ib = jnp.where(live, c["ia"], B), jnp.where(live, c["ib"], B)
    vx = m.put(m.put(vx, ia, dvAx), ib, dvBx)
    vy = m.put(m.put(vy, ia, dvAy), ib, dvBy)
    w = m.put(m.put(w, ia, dwA), ib, dwB)
    return vx, vy, w


def warm_start(vc, vel, n, axis_name=None):
    def body(k, vel, live):
        c = _row(vc, k)
        vx, vy, w = vel
        vAx, vAy, wA = m.pick(vx, c["ia"]), m.pick(vy, c["ia"]), m.pick(w, c["ia"])
        vBx, vBy, wB = m.pick(vx, c["ib"]), m.pick(vy, c["ib"]), m.pick(w, c["ib"])
        nx, ny = c["nx"], c["ny"]
        tx, ty = m.cross_vs(nx, ny, F32(1.0))
        for j in range(2):
            on = j < c["count"]
            Px = m.mul(c["ni"][j], nx) + m.mul(c["ti"][j], tx)
            Py = m.mul(c["ni"][j], ny) + m.mul(c["ti"][j], ty)
            wA = jnp.where(on, wA - m.mul(c["iA"], m.cross(c["rAx"][j], c["rAy"][j], Px, Py)), wA)
            vAx = jnp.where(on, vAx - m.mul(c["mA"], Px), vAx)
            vAy = jnp.where(on, vAy - m.mul(c["mA"], Py), vAy)
            wB = jnp.where(on, wB + m.mul(c["iB"], m.cross(c["rBx"][j], c["rBy"][j], Px, Py)), wB)
            vBx = jnp.where(on, vBx + m.mul(c["mB"], Px), vBx)
            vBy = jnp.where(on, vBy + m.mul(c["mB"], Py), vBy)
        return _apply(vel, c, vAx, vAy, wA, vBx, vBy, wB, live)
    return ordered_loop(n, vc["valid"].shape[0], body, vel, axis_name=axis_name)


def _dv(vB, wB, rbx, rby, vA, wA, rax, ray):
    """vB + b2Cross(wB, rB) - vA - b2Cross(wA, rA)."""
    bx, by = m.cross_sv(wB, rbx, rby)
    ax, ay = m.cross_sv(wA, rax, ray)
    return ((vB[0] + bx) - vA[0]) - ax, ((vB[1] + by) - vA[1]) - ay


def _solve_one(c, vel_in, ni, ti):
    """b2ContactSolver::SolveVelocityConstraints for one contact. Returns new body velocities and impulses."""
    vx, vy, w = vel_in
    vA, wA = (m.pick(vx, c["ia"]), m.pick(vy, c["ia"])), m.pick(w, c["ia"])
    vB, wB = (m.pick(vx, c["ib"]), m.pick(vy, c["ib"])), m.pick(w, c["ib"])
    mA, mB, iA, iB = c["mA"], c["mB"], c["iA"], c["iB"]
    nx, ny = c["nx"], c["ny"]
    tx, ty = m.cross_vs(nx, ny, F32(1.0))
    rax, ray, rbx, rby = c["rAx"], c["rAy"], c["rBx"], c["rBy"]
    count = c["count"]
    ni = list(ni[j] for j in range(2))
    ti = list(ti[j] for j in range(2))

    # Tangent constraints first.
    for j in range(2):
        on = j < count
        dvx, dvy = _dv(vB, wB, rbx[j], rby[j], vA, wA, rax[j], ray[j])
        vt = m.dot(dvx, dvy, tx, ty) - c["tangent_speed"]
        lam = m.mul(c["tmass"][j], -vt)
        maxf = m.mul(c["friction"], ni[j])
        new = m.clamp(ti[j] + lam, -maxf, maxf)
        lam = new - ti[j]
        Px, Py = m.mul(lam, tx), m.mul(lam, ty)
        vA = tuple(jnp.where(on, va - m.mul(mA, p), va) for va, p in zip(vA, (Px, Py)))
        wA = jnp.where(on, wA - m.mul(iA, m.cross(rax[j], ray[j], Px, Py)), wA)
        vB = tuple(jnp.where(on, vb + m.mul(mB, p), vb) for vb, p in zip(vB, (Px, Py)))
        wB = jnp.where(on, wB + m.mul(iB, m.cross(rbx[j], rby[j], Px, Py)), wB)
        ti[j] = jnp.where(on, new, ti[j])

    # --- single point (pointCount == 1)
    dvx, dvy = _dv(vB, wB, rbx[0], rby[0], vA, wA, rax[0], ray[0])
    vn = m.dot(dvx, dvy, nx, ny)
    lam = m.mul(-c["nmass"][0], vn - c["bias"][0])
    new = m.bmax(ni[0] + lam, F32(0))
    lam = new - ni[0]
    Px, Py = m.mul(lam, nx), m.mul(lam, ny)
    s_vA = (vA[0] - m.mul(mA, Px), vA[1] - m.mul(mA, Py))
    s_wA = wA - m.mul(iA, m.cross(rax[0], ray[0], Px, Py))
    s_vB = (vB[0] + m.mul(mB, Px), vB[1] + m.mul(mB, Py))
    s_wB = wB + m.mul(iB, m.cross(rbx[0], rby[0], Px, Py))
    s_ni0 = new

    # --- block solver (pointCount == 2)
    K, NM = c["K"], c["NM"]
    ax_, ay_ = ni[0], ni[1]
    dv1 = _dv(vB, wB, rbx[0], rby[0], vA, wA, rax[0], ray[0])
    dv2 = _dv(vB, wB, rbx[1], rby[1], vA, wA, rax[1], ray[1])
    vn1, vn2 = m.dot(*dv1, nx, ny), m.dot(*dv2, nx, ny)
    bx = vn1 - c["bias"][0]
    by = vn2 - c["bias"][1]
    bx = bx - (m.mul(K[0], ax_) + m.mul(K[2], ay_))
    by = by - (m.mul(K[1], ax_) + m.mul(K[3], ay_))
    # case 1: both points active
    x1 = (-(m.mul(NM[0], bx) + m.mul(NM[2], by)), -(m.mul(NM[1], bx) + m.mul(NM[3], by)))
    ok1 = (x1[0] >= 0) & (x1[1] >= 0)
    # case 2: point 1 only
    x2 = (m.mul(-c["nmass"][0], bx), F32(0))
    ok2 = (x2[0] >= 0) & ((m.mul(K[1], x2[0]) + by) >= 0)
    # case 3: point 2 only
    x3 = (F32(0), m.mul(-c["nmass"][1], by))
    ok3 = (x3[1] >= 0) & ((m.mul(K[2], x3[1]) + bx) >= 0)
    # case 4: neither
    ok4 = (bx >= 0) & (by >= 0)
    c1, c2 = ok1, ~ok1 & ok2
    c3 = ~ok1 & ~ok2 & ok3
    c4 = ~ok1 & ~ok2 & ~ok3 & ok4
    applied = c1 | c2 | c3 | c4
    xx = jnp.where(c1, x1[0], jnp.where(c2, x2[0], F32(0)))
    xy = jnp.where(c1, x1[1], jnp.where(c3, x3[1], F32(0)))
    dx_, dy_ = xx - ax_, xy - ay_
    P1x, P1y = m.mul(dx_, nx), m.mul(dx_, ny)
    P2x, P2y = m.mul(dy_, nx), m.mul(dy_, ny)
    b_vA = (vA[0] - m.mul(mA, P1x + P2x), vA[1] - m.mul(mA, P1y + P2y))
    b_wA = wA - m.mul(iA, m.cross(rax[0], ray[0], P1x, P1y) + m.cross(rax[1], ray[1], P2x, P2y))
    b_vB = (vB[0] + m.mul(mB, P1x + P2x), vB[1] + m.mul(mB, P1y + P2y))
    b_wB = wB + m.mul(iB, m.cross(rbx[0], rby[0], P1x, P1y) + m.cross(rbx[1], rby[1], P2x, P2y))
    blk = (count == 2) & applied
    single = count == 1
    sel = lambda s_val, b_val, keep: jnp.where(single, s_val, jnp.where(blk, b_val, keep))
    vA = (sel(s_vA[0], b_vA[0], vA[0]), sel(s_vA[1], b_vA[1], vA[1]))
    wA = sel(s_wA, b_wA, wA)
    vB = (sel(s_vB[0], b_vB[0], vB[0]), sel(s_vB[1], b_vB[1], vB[1]))
    wB = sel(s_wB, b_wB, wB)
    ni0 = sel(s_ni0, xx, ni[0])
    ni1 = jnp.where(blk, xy, ni[1])
    return (vA, wA, vB, wB), jnp.stack([ni0, ni1]), jnp.stack(ti)


def solve_velocity(vc, vel, ni, ti, n, axis_name=None):
    """One b2ContactSolver::SolveVelocityConstraints pass over the first n ordered contacts."""
    def body(k, carry, live):
        vel, ni, ti = carry
        c = _row(vc, k)
        (vA, wA, vB, wB), nik, tik = _solve_one(c, vel, ni[k], ti[k])
        vel = _apply(vel, c, vA[0], vA[1], wA, vB[0], vB[1], wB, live)
        return vel, ni.at[k].set(jnp.where(live, nik, ni[k])), ti.at[k].set(jnp.where(live, tik, ti[k]))
    return ordered_loop(n, vc["valid"].shape[0], body, (vel, ni, ti), axis_name=axis_name)


def store_impulses(contacts, vc, ni, ti):
    """b2ContactSolver::StoreImpulses: points j < the velocity constraint's pointCount (a 2-point manifold reduced
    to one keeps its second point's carried-over impulse)."""
    C = contacts.pair.shape[0]
    on = vc["valid"][:, None] & (jnp.arange(2)[None, :] < vc["count"][:, None])
    tgt = jnp.where(vc["valid"], vc["slot"], C)
    cur_n = contacts.normal_impulse[vc["slot"]]
    cur_t = contacts.tangent_impulse[vc["slot"]]
    new_n = jnp.where(on, ni, cur_n)
    new_t = jnp.where(on, ti, cur_t)
    return contacts._replace(normal_impulse=contacts.normal_impulse.at[tgt].set(new_n, mode="drop"),
                             tangent_impulse=contacts.tangent_impulse.at[tgt].set(new_t, mode="drop"))


def solve_position_one(c, pos, settings, baumgarte):
    """b2ContactSolver::SolvePositionConstraints for one contact: returns new positions of A and B and the
    contact's minimum separation (starting from 0, as minSeparation does)."""
    cx, cy, a = pos
    cA, aA = (m.pick(cx, c["ia"]), m.pick(cy, c["ia"])), m.pick(a, c["ia"])
    cB, aB = (m.pick(cx, c["ib"]), m.pick(cy, c["ib"])), m.pick(a, c["ib"])
    mA, mB, iA, iB = c["mA"], c["mB"], c["iA"], c["iB"]
    min_sep = F32(0)
    for j in range(2):
        on = j < c["pc_count"]
        xfA = _xf(cA[0], cA[1], aA, c["lcAx"], c["lcAy"])
        xfB = _xf(cB[0], cB[1], aB, c["lcBx"], c["lcBy"])
        # b2PositionSolverManifold::Initialize
        pAx, pAy = m.xf_mul(*xfA, c["lpx"], c["lpy"])
        pBx, pBy = m.xf_mul(*xfB, c["plx"][0], c["ply"][0])
        cnx, cny, _ = m.normalize(pBx - pAx, pBy - pAy)
        c_pt = (m.mul(F32(0.5), pAx + pBx), m.mul(F32(0.5), pAy + pBy))
        c_sep = (m.dot(pBx - pAx, pBy - pAy, cnx, cny) - c["radA"]) - c["radB"]
        anx, any_ = m.rot(xfA[2], xfA[3], c["lnx"], c["lny"])
        aplx, aply = m.xf_mul(*xfA, c["lpx"], c["lpy"])
        aqx, aqy = m.xf_mul(*xfB, c["plx"][j], c["ply"][j])
        a_sep = (m.dot(aqx - aplx, aqy - aply, anx, any_) - c["radA"]) - c["radB"]
        bnx, bny = m.rot(xfB[2], xfB[3], c["lnx"], c["lny"])
        bplx, bply = m.xf_mul(*xfB, c["lpx"], c["lpy"])
        bqx, bqy = m.xf_mul(*xfA, c["plx"][j], c["ply"][j])
        b_sep = (m.dot(bqx - bplx, bqy - bply, bnx, bny) - c["radA"]) - c["radB"]
        is_c, is_a = c["mtype"] == CIRCLES, c["mtype"] == FACE_A
        pick = lambda x, y, z: jnp.where(is_c, x, jnp.where(is_a, y, z))
        nx, ny = pick(cnx, anx, -bnx), pick(cny, any_, -bny)
        px_, py_ = pick(c_pt[0], aqx, bqx), pick(c_pt[1], aqy, bqy)
        sep = pick(c_sep, a_sep, b_sep)
        rax, ray = px_ - cA[0], py_ - cA[1]
        rbx, rby = px_ - cB[0], py_ - cB[1]
        min_sep = jnp.where(on, m.bmin(min_sep, sep), min_sep)
        C_ = m.clamp(m.mul(baumgarte, sep + settings.linear_slop), -settings.max_linear_correction, F32(0))
        rnA, rnB = m.cross(rax, ray, nx, ny), m.cross(rbx, rby, nx, ny)
        K = ((mA + mB) + m.mul3(iA, rnA, rnA)) + m.mul3(iB, rnB, rnB)
        imp = jnp.where(K > 0, -C_ / jnp.where(K > 0, K, F32(1)), F32(0))
        Px, Py = m.mul(imp, nx), m.mul(imp, ny)
        ncA = (cA[0] - m.mul(mA, Px), cA[1] - m.mul(mA, Py))
        naA = aA - m.mul(iA, m.cross(rax, ray, Px, Py))
        ncB = (cB[0] + m.mul(mB, Px), cB[1] + m.mul(mB, Py))
        naB = aB + m.mul(iB, m.cross(rbx, rby, Px, Py))
        cA = (jnp.where(on, ncA[0], cA[0]), jnp.where(on, ncA[1], cA[1]))
        aA = jnp.where(on, naA, aA)
        cB = (jnp.where(on, ncB[0], cB[0]), jnp.where(on, ncB[1], cB[1]))
        aB = jnp.where(on, naB, aB)
    return cA, aA, cB, aB, min_sep
