"""b2Island::SolveTOI (toi._solve_island) with its two sequential loops in the island kernel (kernel.py).

The TOI sub-island has contacts only and is one island per world: up to 20 position passes (toi_baumgarte, done at
min separation >= -1.5 linearSlop, masses only for the two TOI bodies), then InitializeVelocityConstraints without
warm start and `velocity_iterations` passes, then integration over (1 - minAlpha) dt with the step's clamps. Here
the position passes are one kernel call (no velocity iterations, no integration), the constraint set-up stays in
XLA (contact_solver.prepare at the solved poses), and the velocity passes + integration are a second kernel call
(no position passes). Everything around them is toi._solve_island's code. Bit-identical to it in interpret mode.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from .. import contact_solver as cs
from .. import math2d as m
from ..toi import _Isl, _position_rows, _sync_xf
from . import kernel as K
from . import solve as G

F32 = np.float32
I32 = jnp.int32


def _world(B, vel, pos, in_isl, n, h, max_t, max_r):
    f32 = lambda x: jnp.asarray(x).astype(jnp.float32)
    return jnp.concatenate([*vel, *pos, f32(in_isl),
                            jnp.stack([f32(n), F32(0), f32(n > 0), f32(h), f32(max_t), f32(max_r)])])


def _dummy_joints(W):
    rows = dict(ia=jnp.zeros((1,), jnp.int32), ib=jnp.zeros((1,), jnp.int32), branch=jnp.zeros((1,), jnp.int32),
                isl=jnp.zeros((1,), jnp.int32))
    return rows, jnp.zeros((1, W + 1), jnp.float32)


def solve_island(model, state, s, dt, min_alpha, A, Bb, order, n, in_isl, walk, axis_name):
    """toi._solve_island with the kernel; same arguments and results."""
    bd = state.bodies
    B, C = bd.type.shape[0], order.shape[0]
    isl = _Isl(contact_order=jnp.where(order >= 0, order, 0), n_contacts=n)
    pos = (bd.c[:, 0], bd.c[:, 1], bd.a)
    vel = (bd.v[:, 0], bd.v[:, 1], bd.w)
    pc = _position_rows(model, state, isl)
    toi_a = lambda i: (i == A) | (i == Bb)
    pc = dict(pc, mA=jnp.where(toi_a(pc["ia"]), pc["mA"], F32(0.0)), iA=jnp.where(toi_a(pc["ia"]), pc["iA"], F32(0.0)),
              mB=jnp.where(toi_a(pc["ib"]), pc["mB"], F32(0.0)), iB=jnp.where(toi_a(pc["ib"]), pc["iB"], F32(0.0)),
              isl=jnp.zeros((C,), jnp.int32))
    W = 5 if s.revolute_v24 else 3
    jr, ji = _dummy_joints(W)
    jlay = K.layout_of(jr)
    interp = G._interpret()
    no_t = jnp.asarray(F32(0))

    # ---- position passes: island 0 = the TOI island, live while `walk` (n_isl = 1 then, else 0)
    pc = jax.lax.optimization_barrier(pc)
    clay = K.layout_of(pc)
    ps = replace(s, velocity_iterations=0, position_iterations=20, warm_starting=False, baumgarte=s.toi_baumgarte)
    cfg = K.Config(settings=ps, clay=clay, jlay=jlay, B=B, C=C, J=1, W=W, block=G.BLOCK, num_warps=G.NUM_WARPS,
                   interpret=interp, parts=("pos", "nojpos"), okay_factor=F32(-1.5))
    wf = _world(B, vel, pos, jnp.zeros((B,), bool), jnp.where(walk, n, 0), no_t, no_t, no_t)
    wo, _, _ = K.island_solve(cfg, K.pack(pc, clay), K.pack(jr, jlay), wf, jnp.zeros((C, 4), jnp.float32), ji)
    cx, cy, a = wo[3 * B:4 * B], wo[4 * B:5 * B], wo[5 * B:6 * B]
    pos = (cx, cy, a)

    # Leap of faith: the TOI bodies' sweep start is the solved pose.
    pair_b = ((jnp.arange(B) == A) | (jnp.arange(B) == Bb)) & walk
    c0 = jnp.where(pair_b[:, None], jnp.stack([cx, cy], -1), bd.c0)
    a0 = jnp.where(pair_b, a, bd.a0)

    vc = cs.prepare(model, state, isl, pos, vel, False, F32(1.0), s.velocity_threshold)

    # ---- velocity passes (contacts only, no warm start) + integration of the island bodies
    h = m.mul(F32(1.0) - min_alpha, dt)
    max_t = m.mul(s.max_translation, dt) if s.clamps_per_second else jnp.asarray(s.max_translation)
    max_r = m.mul(s.max_rotation, dt) if s.clamps_per_second else jnp.asarray(s.max_rotation)
    crows = dict(vc, isl=jnp.zeros((C,), jnp.int32))
    crows = jax.lax.optimization_barrier(crows)
    clay = K.layout_of(crows, G._SKIP)
    vs = replace(s, position_iterations=0, warm_starting=False)
    cfg = K.Config(settings=vs, clay=clay, jlay=jlay, B=B, C=C, J=1, W=W, block=G.BLOCK, num_warps=G.NUM_WARPS,
                   interpret=interp, parts=("cvel",))
    wf = _world(B, vel, pos, in_isl, n, h, max_t, max_r)
    wo, co, _ = K.island_solve(cfg, K.pack(crows, clay), K.pack(jr, jlay), wf,
                               jnp.concatenate([vc["ni"], vc["ti"]], 1), ji)
    vx, vy, w = wo[:B], wo[B:2 * B], wo[2 * B:3 * B]
    ncx, ncy, na = wo[3 * B:4 * B], wo[4 * B:5 * B], wo[5 * B:6 * B]
    ni, ti = co[:, :2], co[:, 2:]

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
