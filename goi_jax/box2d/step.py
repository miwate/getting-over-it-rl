"""b2World::Step (Box2D 2.3.1) for one world: FindNewContacts if fixtures were added, Collide, Solve (islands,
b2Island::Solve per island), synchronize fixtures + FindNewContacts, SolveTOI (toi.py, when Settings.continuous_physics), clear forces.

`step(model, state, settings, dt, axis_name=None)` is pure (settings is a hashable constant). Batch it with vmap;
naming the vmap axis lets the sweeps share one trip count across the batch (loops.py), which is much faster:

    run = jax.jit(jax.vmap(lambda s: step(model, s, settings, dt, axis_name="w"), axis_name="w"))

TOI: b2World::SolveTOI is in toi.py and runs when Settings.continuous_physics is set (Settings.game() does, with
Unity's pair rule); with it off the step equals Box2D with continuous physics off.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import contact_solver as cs
from . import joints as jn
from . import math2d as m
from .island import find_islands
from .loops import ordered_loop
from .pairs import collide, find_new_contacts, synchronize_fixtures
from .toi import solve_toi
from .world import DYNAMIC, STATIC

F32 = np.float32


def step(model, state, settings, dt, axis_name=None):
    dt = jnp.asarray(dt, jnp.float32)
    # b2World::Step: new fixtures -> FindNewContacts first.
    state = start_new_contacts(model, state, settings, axis_name)
    state = state._replace(new_fixture=jnp.asarray(False))
    inv_dt = jnp.where(dt > 0, F32(1) / jnp.where(dt > 0, dt, F32(1)), F32(0))
    dt_ratio = m.mul(state.inv_dt0, dt)
    state = collide(model, state, settings)
    if getattr(settings, "solver", "xla") == "pallas":  # island solver as one Pallas kernel (gpu/)
        from .gpu.solve import solve as solve_pallas
        state = solve_pallas(model, state, settings, dt, inv_dt, dt_ratio, axis_name)
    else:
        state = solve(model, state, settings, dt, inv_dt, dt_ratio, axis_name)
    if settings.continuous_physics:  # b2World::SolveTOI (toi.py)
        state = solve_toi(model, state, settings, dt, axis_name)
    state = state._replace(inv_dt0=jnp.where(dt > 0, inv_dt, state.inv_dt0))
    if settings.auto_clear_forces:
        b = state.bodies
        state = state._replace(bodies=b._replace(force=jnp.zeros_like(b.force), torque=jnp.zeros_like(b.torque)))
    return state


def start_new_contacts(model, state, settings, axis_name=None):
    """b2World::Step's FindNewContacts where State.new_fixture (the state unchanged elsewhere). With the Pallas
    solver the candidate scan is a kernel run only where new_fixture; with the "idle" GPU speed option and a named
    batch axis, the whole call (scan, placement and the merge) is skipped when no world of the batch has a new
    fixture (normally every step of a training rollout): a lax.cond on the batch-wide flag, same results."""
    pallas = getattr(settings, "solver", "xla") == "pallas"

    def run(st):
        if pallas:  # candidate scan as a kernel, only where new_fixture
            from .gpu.pairs import find_new_contacts as fnc_pallas
            found = fnc_pallas(model, st, st.new_fixture)
        else:
            found = find_new_contacts(model, st)
        return jax.tree.map(lambda a, b: jnp.where(st.new_fixture, a, b), found, st)

    if pallas and axis_name is not None:
        from .gpu import pairs as gpu_pairs
        if gpu_pairs.SKIP_IDLE_START:
            any_new = jax.lax.pmax(jnp.asarray(state.new_fixture).astype(jnp.int32), axis_name) > 0
            return jax.lax.cond(any_new, run, lambda st: st, state)
    return run(state)


def solve(model, state, settings, dt, inv_dt, dt_ratio, axis_name=None):
    """b2World::Solve: every awake island at once (island.py explains why that is exact)."""
    s = settings
    isl = find_islands(model, state)
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
    vc = cs.prepare(model, state, isl, pos, vel, warm, dt_ratio, s.velocity_threshold)
    if s.warm_starting:
        vel = cs.warm_start(vc, vel, isl.n_contacts, axis_name)
    jrows, jimp, jmimp = jn.prepare(model, state, isl, pos, s, dt, dt_ratio, warm)
    vel = jn.warm_start(jrows, vel, jimp, jmimp, isl.n_joints)

    def vel_iter(_, carry):
        vel, jimp, jmimp, ni, ti = carry
        vel, jimp, jmimp = jn.solve_velocity(jrows, vel, jimp, jmimp, isl.n_joints)
        vel, ni, ti = cs.solve_velocity(vc, vel, ni, ti, isl.n_contacts, axis_name)
        return vel, jimp, jmimp, ni, ti
    vel, jimp, jmimp, ni, ti = jax.lax.fori_loop(0, s.velocity_iterations, vel_iter,
                                                 (vel, jimp, jmimp, vc["ni"], vc["ti"]))
    contacts = cs.store_impulses(state.contacts, vc, ni, ti)
    joints = jn.store(state, jrows, jimp, jmimp, isl.n_joints)

    # Integrate positions with the per-step translation / rotation clamps.
    vx, vy, w = vel
    max_t = m.mul(s.max_translation, h) if s.clamps_per_second else jnp.asarray(s.max_translation)
    max_r = m.mul(s.max_rotation, h) if s.clamps_per_second else jnp.asarray(s.max_rotation)
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
    cx, cy, a = jnp.where(in_isl, ncx, cx), jnp.where(in_isl, ncy, cy), jnp.where(in_isl, na, a)
    vx, vy, w = (jnp.where(in_isl, x, y) for x, y in zip((vx, vy, w), vel))

    # Position iterations; an island stops at its first pass with contacts and joints within tolerance.
    pos, solved = _solve_positions(model, isl, vc, jrows, (cx, cy, a), s, axis_name)
    cx, cy, a = pos

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

    # Static bodies: Box2D adds them to each island that touches them (SetAwake(true) when popped) and puts them to
    # sleep with a sleeping island, so their awake flag ends as the last such island left it. No physics effect.
    last = _static_last_island(model, state, isl, B)
    hit = (bd.type == STATIC) & (last >= 0)
    sleeps = isl_sleep[jnp.clip(last, 0, B - 1)]
    bd = bd._replace(awake=jnp.where(hit, ~sleeps, bd.awake), sleep_time=jnp.where(hit, F32(0), bd.sleep_time))

    state = state._replace(bodies=bd, contacts=contacts, joints=joints)
    state = synchronize_fixtures(model, state, in_isl & (bd.type != STATIC))
    return find_new_contacts(model, state)


def _static_last_island(model, state, isl, B):
    """For each body, the highest island id among solved contacts and joints that touch it (-1 if none)."""
    ct = state.contacts
    last = jnp.full(B, -1, jnp.int32)
    fb = model.fixture_body
    for body in (fb[ct.fa], fb[ct.fb]):
        last = last.at[jnp.where(isl.contact_island >= 0, body, B)].max(isl.contact_island, mode="drop")
    for body in (model.joint_a, model.joint_b):
        last = last.at[jnp.where(isl.joint_island >= 0, body, B)].max(isl.joint_island, mode="drop")
    return last


def _solve_positions(model, isl, vc, jrows, pos, s, axis_name=None):
    """The position loop of b2Island::Solve with per-island early exit: islands still unsolved run contacts then
    joints; after each pass an island whose min separation >= -3 linearSlop and whose joints all report OK is done.
    Returns the positions and each island's positionSolved."""
    B = pos[0].shape[0]
    C = vc["valid"].shape[0]
    J = jrows["valid"].shape[0]
    c_isl = isl.contact_island[vc["slot"]]
    j_isl = isl.joint_island[jrows["jid"]]
    n_isl = isl.n_islands

    def one_pass(pos, done):
        def cbody(k, carry, live_k):
            pos, min_sep = carry
            c = {key: val[k] for key, val in vc.items()}
            i = c_isl[k]
            live = live_k & ~done[i]
            cA, aA, cB, aB, ms = cs.solve_position_one(c, pos, s, s.baumgarte)
            pos = _put_pos(pos, c["ia"], c["ib"], cA, aA, cB, aB, live)
            min_sep = min_sep.at[i].set(jnp.where(live, m.bmin(min_sep[i], ms), min_sep[i]))
            return pos, min_sep
        min_sep = jnp.zeros(B, jnp.float32)
        if C:
            pos, min_sep = ordered_loop(isl.n_contacts, C, cbody, (pos, min_sep), axis_name=axis_name)

        def jbody(k, carry, live_k):
            pos, ok = carry
            r = {key: val[k] for key, val in jrows.items()}
            i = j_isl[k]
            live = live_k & ~done[i]
            cA, aA, cB, aB, jok = jn.solve_position_one(r, pos, s)
            pos = _put_pos(pos, r["ia"], r["ib"], cA, aA, cB, aB, live)
            ok = ok.at[i].set(jnp.where(live, ok[i] & jok, ok[i]))
            return pos, ok
        jok = jnp.ones(B, bool)
        if J:
            pos, jok = ordered_loop(isl.n_joints, J, jbody, (pos, jok), static=True)
        okay = (min_sep >= m.mul(F32(-3), s.linear_slop)) & jok
        return pos, okay

    def cond(carry):
        it, pos, done = carry
        more = ~jnp.all(done)
        if axis_name is not None:  # one shared trip count for the batch; finished worlds are masked by `done`
            more = jax.lax.pmax(more.astype(jnp.int32), axis_name) > 0
        return (it < s.position_iterations) & more

    def body(carry):
        it, pos, done = carry
        pos, okay = one_pass(pos, done)
        return it + 1, pos, done | okay

    done0 = jnp.arange(B) >= n_isl
    _, pos, done = jax.lax.while_loop(cond, body, (jnp.int32(0), pos, done0))
    return pos, done


def _put_pos(pos, ia, ib, cA, aA, cB, aB, live):
    cx, cy, a = pos
    hit_a = (jnp.arange(cx.shape[-1]) == ia) & live
    hit_b = (jnp.arange(cx.shape[-1]) == ib) & live
    cx = jnp.where(hit_b, cB[0], jnp.where(hit_a, cA[0], cx))
    cy = jnp.where(hit_b, cB[1], jnp.where(hit_a, cA[1], cy))
    a = jnp.where(hit_b, aB, jnp.where(hit_a, aA, a))
    return cx, cy, a
