"""b2World::Solve with the island solver in one Pallas kernel (kernel.py); everything else as step.solve.

The set-up (islands, velocity integration, contact and joint constraint data) and the wrap-up (impulse storage,
transform sync, sleeping, fixture sync, FindNewContacts) are step.solve's code; only the sequential middle (warm
start, velocity iterations, position integration, position iterations) runs in the kernel. Keep this in step with
step.solve: tests/box2d/test_gpu.py checks the two bit for bit on every test scene (interpret mode on CPU).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .. import contact_solver as cs
from .. import joints as jn
from .. import math2d as m
from ..island import find_islands
from .pairs import find_new_contacts as fnc_pallas
from .pairs import synchronize_fixtures as sync_pallas
from ..world import DYNAMIC, STATIC
from . import kernel as K

F32 = np.float32
_SKIP = ("valid", "slot", "jid", "ni", "ti")
BLOCK = 128  # worlds per kernel program (lanes; one per thread with 4 warps); read at trace time
NUM_WARPS = 4


def solve(model, state, settings, dt, inv_dt, dt_ratio, axis_name=None):
    """step.solve with the Pallas island solver. axis_name is unused (the kernel walks each world's own counts)."""
    s = settings
    from . import islands as GI
    isl = (GI.find_islands if GI.ENABLED else find_islands)(model, state)
    bd = state.bodies._replace(awake=isl.awake, sleep_time=isl.sleep_time)
    B = bd.type.shape[0]
    in_isl = isl.body_island >= 0
    h = dt

    # Integrate velocities and apply damping (dynamic bodies); c0 = c for every island body.
    dyn = in_isl & (bd.type == DYNAMIC)
    gs = bd.gravity_scale
    g = state.gravity
    ax = m.mul(gs, g[0]) + m.mul(bd.inv_mass, bd.force[:, 0])
    ay = m.mul(gs, g[1]) + m.mul(bd.inv_mass, bd.force[:, 1])
    vx = bd.v[:, 0] + m.mul(h, ax)
    vy = bd.v[:, 1] + m.mul(h, ay)
    w = bd.w + m.mul(m.mul(h, bd.inv_inertia), bd.torque)
    ld = F32(1) / (F32(1) + m.mul(h, bd.linear_damping))
    ad = F32(1) / (F32(1) + m.mul(h, bd.angular_damping))
    vx, vy, w = m.mul(vx, ld), m.mul(vy, ld), m.mul(w, ad)
    vx = jnp.where(dyn, vx, bd.v[:, 0])
    vy = jnp.where(dyn, vy, bd.v[:, 1])
    w = jnp.where(dyn, w, bd.w)
    cx, cy, a = bd.c[:, 0], bd.c[:, 1], bd.a
    c0 = jnp.where(in_isl[:, None], bd.c, bd.c0)
    a0 = jnp.where(in_isl, bd.a, bd.a0)
    state = state._replace(bodies=bd._replace(c0=c0, a0=a0))

    warm = jnp.asarray(s.warm_starting)
    pos, vel = (cx, cy, a), (vx, vy, w)
    rot = m.rot_of(a)  # xf.q of every body at the island poses, shared by the set-ups (and the fixture sync)
    vc = cs.prepare(model, state, isl, pos, vel, warm, dt_ratio, s.velocity_threshold, rot=rot)
    jrows, jimp, jmimp = jn.prepare(model, state, isl, pos, s, dt, dt_ratio, warm, rot=rot)

    # ---- the kernel
    C, J, W = vc["valid"].shape[0], jrows["valid"].shape[0], jimp.shape[1]
    crows = dict(vc, isl=isl.contact_island[vc["slot"]])
    jr = dict(jrows, isl=isl.joint_island[jrows["jid"]])
    if C == 0:  # the kernel wants at least one row; n_contacts is 0 so it is never live
        crows = {k: jnp.zeros((1,) + v.shape[1:], v.dtype) for k, v in crows.items()}
    if J == 0:
        jr = {k: jnp.zeros((1,) + v.shape[1:], v.dtype) for k, v in jr.items()}
    clay, jlay = K.layout_of(crows, _SKIP), K.layout_of(jr, _SKIP)
    # Materialize the constraint data before packing: without the barrier XLA fuses the set-up arithmetic (sin/cos
    # included) into the packing concatenates and recomputes it per packed field (minutes of compile, ~1 s/step).
    crows, jr = jax.lax.optimization_barrier((crows, jr))
    max_t = m.mul(s.max_translation, h) if s.clamps_per_second else jnp.asarray(s.max_translation)
    max_r = m.mul(s.max_rotation, h) if s.clamps_per_second else jnp.asarray(s.max_rotation)
    f32 = lambda x: jnp.asarray(x).astype(jnp.float32)
    wf = jnp.concatenate([vx, vy, w, cx, cy, a, f32(in_isl),
                          jnp.stack([f32(isl.n_contacts), f32(isl.n_joints), f32(isl.n_islands),
                                     f32(h), f32(max_t), f32(max_r)])])
    ci = jnp.concatenate([vc["ni"], vc["ti"]], 1) if C else jnp.zeros((1, 4), jnp.float32)
    ji = jnp.concatenate([jimp, jmimp[:, None]], 1) if J else jnp.zeros((1, W + 1), jnp.float32)
    cfg = K.Config(settings=s, clay=clay, jlay=jlay, B=B, C=max(C, 1), J=max(J, 1), W=W,
                   block=BLOCK, num_warps=NUM_WARPS, interpret=_interpret())
    wo, co, jo = K.island_solve(cfg, K.pack(crows, clay), K.pack(jr, jlay), wf, ci, ji)
    vx, vy, w, cx, cy, a = (wo[f * B:(f + 1) * B] for f in range(6))
    solved = wo[6 * B:7 * B] != 0
    ni, ti = co[:C, :2], co[:C, 2:]
    jimp, jmimp = jo[:J, :W], jo[:J, W]

    contacts = cs.store_impulses(state.contacts, vc, ni, ti)
    joints = jn.store(state, jrows, jimp, jmimp, isl.n_joints)

    # Write back + SynchronizeTransform (island bodies).
    bd = state.bodies
    c = jnp.where(in_isl[:, None], jnp.stack([cx, cy], -1), bd.c)
    a = jnp.where(in_isl, a, bd.a)
    v = jnp.where(in_isl[:, None], jnp.stack([vx, vy], -1), bd.v)
    w = jnp.where(in_isl, w, bd.w)
    qc, qs = m.rot_of(a)
    rx, ry = m.rot(qc, qs, bd.local_center[:, 0], bd.local_center[:, 1])
    xf_p = jnp.where(in_isl[:, None], jnp.stack([c[:, 0] - rx, c[:, 1] - ry], -1), bd.xf_p)
    xf_q = jnp.where(in_isl[:, None], jnp.stack([qc, qs], -1), bd.xf_q)
    bd = bd._replace(c=c, a=a, v=v, w=w, xf_p=xf_p, xf_q=xf_q)

    # Sleep, per island.
    if s.allow_sleep:
        lin_tol = m.mul(s.linear_sleep_tolerance, s.linear_sleep_tolerance)
        ang_tol = m.mul(s.angular_sleep_tolerance, s.angular_sleep_tolerance)
        nonstatic = in_isl & (bd.type != STATIC)
        moving = ~bd.allow_sleep | (m.mul(bd.w, bd.w) > ang_tol) | (m.dot(bd.v[:, 0], bd.v[:, 1], bd.v[:, 0], bd.v[:, 1]) > lin_tol)
        st = jnp.where(moving, F32(0), bd.sleep_time + h)
        st = jnp.where(nonstatic, st, bd.sleep_time)
        min_st = jnp.full(B, np.finfo(np.float32).max, jnp.float32)  # b2_maxFloat
        bi = jnp.where(nonstatic, isl.body_island, B)
        min_st = min_st.at[bi].min(jnp.where(moving, F32(0), st), mode="drop")
        isl_sleep = (min_st >= s.time_to_sleep) & solved
        sleep = nonstatic & isl_sleep[jnp.clip(isl.body_island, 0, B - 1)]
        bd = bd._replace(sleep_time=jnp.where(sleep, F32(0), st), awake=bd.awake & ~sleep,
                         v=jnp.where(sleep[:, None], F32(0), bd.v), w=jnp.where(sleep, F32(0), bd.w),
                         force=jnp.where(sleep[:, None], F32(0), bd.force), torque=jnp.where(sleep, F32(0), bd.torque))
    else:
        isl_sleep = jnp.zeros(B, bool)

    from ..step import _static_last_island
    last = _static_last_island(model, state, isl, B)
    hit = (bd.type == STATIC) & (last >= 0)
    sleeps = isl_sleep[jnp.clip(last, 0, B - 1)]
    bd = bd._replace(awake=jnp.where(hit, ~sleeps, bd.awake), sleep_time=jnp.where(hit, F32(0), bd.sleep_time))

    state = state._replace(bodies=bd, contacts=contacts, joints=joints)
    # the bodies it moves have a0 = the island pose's a (set above), so their sweep-start rotation is `rot`
    state = sync_pallas(model, state, in_isl & (bd.type != STATIC), block=BLOCK,
                        rot0=rot if sync_reuses_rotation() else None)
    return fnc_pallas(model, state, block=BLOCK)


def sync_reuses_rotation():
    from . import pairs as P
    return P.SYNC_ROT


def _interpret():
    return jax.default_backend() == "cpu"
