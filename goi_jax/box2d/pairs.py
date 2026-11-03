"""Contact pairs without a dynamic tree: b2ContactManager::Collide / FindNewContacts and b2Fixture::Synchronize over
a fixed candidate list (Model.pair_a/pair_b) and C contact slots.

How this reproduces b2BroadPhase (2.3.1):
- Each fixture keeps its fat AABB (Fixtures.fat), updated exactly as b2DynamicTree::MoveProxy does: unchanged while
  it contains the swept shape AABB, else the swept AABB + b2_aabbExtension + 2 * displacement on the moving side.
- Box2D creates a pair in UpdatePairs only for proxies that moved, querying their fat AABB. A pair whose fat AABBs
  overlap and has no contact must have started overlapping when one of them moved (fat AABBs change only then), so
  "all candidates whose fat AABBs overlap and have no contact" is the same set. Collide destroys a contact when the
  fat AABBs stop overlapping, but only if one body is awake and not static, as Box2D does; "awake" includes wakes
  by contacts visited earlier in Collide's newest-first walk (_active_in_list_order).
- New contacts get keys in ascending candidate order (= UpdatePairs' sorted (proxyA, proxyB) order); Box2D prepends
  each to the world list and both bodies' edge lists, so newest-first = descending key.
Divergences: proxy ids are assumed to be fixture creation order (true while no proxy is destroyed and re-created,
e.g. by deactivating a body); sensor pairs are not tracked (no physics effect); a full table drops new contacts and
counts them in State.overflow.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import collide as nc
from . import math2d as m
from . import shapes
from .world import DYNAMIC, STATIC, fixture_fat, fixture_friction, fixture_restitution, is_compact

F32 = np.float32


def body_xf(bodies, idx):
    return bodies.xf_p[idx, 0], bodies.xf_p[idx, 1], bodies.xf_q[idx, 0], bodies.xf_q[idx, 1]


def _fat(model, fx, idx):
    return fixture_fat(model, fx, idx)


def collide(model, state, settings):
    """b2ContactManager::Collide: destroy slots whose fat AABBs separated, update the manifold of the rest
    (b2Contact::Update: evaluate, carry impulses over by feature key, touching flag, wake on a touch change)."""
    ct, bd = state.contacts, state.bodies
    fb_ = model.fixture_body
    live = ct.pair >= 0
    ba, bb = fb_[ct.fa], fb_[ct.fb]
    over = shapes.overlap(_fat(model, state.fixtures, ct.fa), _fat(model, state.fixtures, ct.fb))
    man = nc.evaluate(model, ct.fa, ct.fb, body_xf(bd, ba), body_xf(bd, bb), settings.linear_slop)
    new_count = man["count"]
    touching = new_count > 0
    # Which contacts Collide visits depends on wakes earlier in its walk (list order, newest first): an Update
    # whose touching flag flips wakes both bodies, and so does destroying a contact that still has manifold
    # points (b2Contact::Destroy), so a later contact of a woken sleeper is updated in the same pass.
    wakes = jnp.where(over, touching != ct.touching, ct.count > 0)
    active, awake = _active_in_list_order(live, ct.key, ba, bb, bd.awake, bd.type != STATIC, wakes)
    destroy = active & ~over
    update = active & over
    # Warm-start carry-over: the first old point with the same id key.
    old_ids, old_n = ct.point_id, ct.count
    ni, ti = [], []
    for i in range(2):
        nid = man["point_id"][i]
        match0 = (old_n > 0) & (old_ids[:, 0] == nid)
        match1 = (old_n > 1) & (old_ids[:, 1] == nid) & ~match0
        ni.append(jnp.where(match0, ct.normal_impulse[:, 0], jnp.where(match1, ct.normal_impulse[:, 1], F32(0))))
        ti.append(jnp.where(match0, ct.tangent_impulse[:, 0], jnp.where(match1, ct.tangent_impulse[:, 1], F32(0))))
    has = jnp.arange(2)[None, :] < new_count[:, None]
    ni = jnp.where(has, jnp.stack(ni, 1), F32(0))
    ti = jnp.where(has, jnp.stack(ti, 1), F32(0))

    u = update
    u2 = u[:, None]
    u3 = u[:, None, None]
    pl = jnp.stack([jnp.stack(man["point_local"][0], -1), jnp.stack(man["point_local"][1], -1)], 1)
    keep_m = new_count > 0  # an early-out Evaluate only zeroes pointCount
    ct = ct._replace(
        touching=jnp.where(u, touching, ct.touching),
        count=jnp.where(u, new_count, ct.count),
        mtype=jnp.where(u & keep_m, man["mtype"], ct.mtype),
        local_normal=jnp.where(u2 & keep_m[:, None], jnp.stack(man["local_normal"], -1), ct.local_normal),
        local_point=jnp.where(u2 & keep_m[:, None], jnp.stack(man["local_point"], -1), ct.local_point),
        point_local=jnp.where(u3 & keep_m[:, None, None], pl, ct.point_local),
        point_id=jnp.where(u2 & keep_m[:, None], jnp.stack(man["point_id"], -1), ct.point_id),
        normal_impulse=jnp.where(u2, ni, ct.normal_impulse),
        tangent_impulse=jnp.where(u2, ti, ct.tangent_impulse))
    ct = _free(ct, destroy)
    newly = awake & ~bd.awake  # SetAwake(true) zeroes sleepTime only if the body was asleep
    bodies = bd._replace(awake=awake, sleep_time=jnp.where(newly, F32(0), bd.sleep_time))
    return state._replace(contacts=ct, bodies=bodies)


def _active_in_list_order(live, key, ba, bb, awake, movable, wakes):
    """Collide's per-contact `activeA || activeB` with the wakes of contacts visited earlier in the walk.

    W[b] = list position at which body b becomes awake (-1 if awake at the start, INF if never). A contact at
    position k is visited if a movable body of it has W < k; a visited contact with `wakes` sets both its bodies'
    W to at most k. Iterating from the start-of-walk flags only adds wakes that also happen sequentially, at their
    true positions, and stops at the sequential result (the earliest missing wake would be found by the next
    pass), so the loop is exact; it runs once more than the longest wake chain (normally 1-2 passes).
    Returns (active per slot, awake per body after the walk)."""
    C, B = key.shape[0], awake.shape[0]
    INF = jnp.int32(C + 1)
    # list position = number of live contacts with a larger (newer) key; keys are unique, dead slots never visit
    pos = jnp.sum(live[None, :] & (key[None, :] > key[:, None]), axis=1, dtype=jnp.int32)
    w0 = jnp.where(awake, jnp.int32(-1), INF)

    bodies = jnp.arange(B, dtype=jnp.int32)
    hit = (ba[:, None] == bodies) | (bb[:, None] == bodies)  # (C, B) one-hot: no batched scatter under vmap

    def visit(W):
        return live & (((W[ba] < pos) & movable[ba]) | ((W[bb] < pos) & movable[bb]))

    def relax(W):
        k = jnp.where(visit(W) & wakes, pos, INF)
        return jnp.minimum(w0, jnp.min(jnp.where(hit, k[:, None], INF), axis=0))

    # Carry (W, previous W); W1 = relax(w0) equals w0 on almost every step, which ends the loop at once.
    W, _ = jax.lax.while_loop(lambda c: jnp.any(c[0] != c[1]), lambda c: (relax(c[0]), c[0]), (relax(w0), w0))
    return visit(W), W < INF



def _free(ct, mask):
    return ct._replace(pair=jnp.where(mask, -1, ct.pair), touching=ct.touching & ~mask,
                       count=jnp.where(mask, 0, ct.count))


def _wake(bd, mask, ba, bb):
    """b2Body::SetAwake(true) on both bodies of the masked slots (zeroes sleepTime only if it was asleep)."""
    B = bd.type.shape[0]
    hit = jnp.zeros(B, bool).at[ba].max(mask).at[bb].max(mask)
    newly = hit & ~bd.awake
    return bd._replace(awake=bd.awake | hit, sleep_time=jnp.where(newly, F32(0), bd.sleep_time))


def find_new_contacts(model, state):
    """b2ContactManager::FindNewContacts: create a contact for every candidate whose fat AABBs overlap and has none,
    in candidate order, into the free slots (lowest slot first)."""
    P = model.pair_a.shape[0]
    if P == 0:
        return state
    ct, bd, fx = state.contacts, state.bodies, state.fixtures
    C = ct.pair.shape[0]
    pa, pb = model.pair_a, model.pair_b
    ba, bb = model.fixture_body[pa], model.fixture_body[pb]
    over = shapes.overlap(_fat(model, fx, pa), _fat(model, fx, pb))
    # b2Body::ShouldCollide (one body dynamic; joints are excluded statically) and active bodies (have proxies).
    can = ((bd.type[ba] == DYNAMIC) | (bd.type[bb] == DYNAMIC)) & bd.active[ba] & bd.active[bb]
    exists = jnp.zeros(P, bool).at[jnp.where(ct.pair >= 0, ct.pair, P)].set(True, mode="drop")
    new = over & can & ~exists
    rank = jnp.cumsum(new) - 1  # creation order among the new pairs
    n_new = jnp.sum(new)
    free = ct.pair < 0
    free_rank = jnp.cumsum(free) - 1
    n_free = jnp.sum(free)
    # slot_of_rank[r] = r-th free slot
    slot_of_rank = jnp.full(C, C, jnp.int32).at[jnp.where(free, free_rank, C)].set(jnp.arange(C, dtype=jnp.int32),
                                                                                mode="drop")
    placed = new & (rank < n_free)
    target = jnp.where(placed, slot_of_rank[jnp.clip(rank, 0, C - 1)], C)
    cand = jnp.arange(P, dtype=jnp.int32)
    fa = model.pair_a
    fb = model.pair_b
    fr_a, fr_b = fixture_friction(model, fx, fa), fixture_friction(model, fx, fb)
    rs_a, rs_b = fixture_restitution(model, fx, fa), fixture_restitution(model, fx, fb)
    fr = jnp.sqrt(m.mul(fr_a, fr_b))  # b2MixFriction
    rs = jnp.where(rs_a > rs_b, rs_a, rs_b)  # b2MixRestitution
    put = lambda arr, val: arr.at[target].set(val, mode="drop")
    zero2 = jnp.zeros((P, 2), F32)
    ct = ct._replace(
        pair=put(ct.pair, cand), fa=put(ct.fa, fa), fb=put(ct.fb, fb),
        key=put(ct.key, state.next_key + rank.astype(jnp.int32)), touching=put(ct.touching, jnp.zeros(P, bool)),
        count=put(ct.count, jnp.zeros(P, jnp.int32)), normal_impulse=put(ct.normal_impulse, zero2),
        tangent_impulse=put(ct.tangent_impulse, zero2), point_id=put(ct.point_id, jnp.zeros((P, 2), jnp.int32)),
        friction=put(ct.friction, fr), restitution=put(ct.restitution, rs),
        tangent_speed=put(ct.tangent_speed, jnp.zeros(P, F32)))
    bodies = _wake(bd, placed, ba, bb)  # AddPair wakes both bodies (no sensors here)
    n_placed = jnp.minimum(n_new, n_free)
    return state._replace(contacts=ct, bodies=bodies, next_key=state.next_key + n_new.astype(jnp.int32),
                          overflow=state.overflow + (n_new - n_placed).astype(jnp.int32))


def synchronize_fixtures(model, state, moved):
    """b2Body::SynchronizeFixtures for bodies in `moved` (island bodies that are not static): xf1 from the sweep
    start (c0, a0), xf2 = m_xf; b2Fixture::Synchronize combines both AABBs and b2DynamicTree::MoveProxy re-fattens
    the proxy only if its fat AABB no longer contains the result. Compact layout (world.compact_model): only the
    rows of State.fixtures (non-static fixtures; a static body never moves) are computed."""
    bd, fx = state.bodies, state.fixtures
    fbody = model.fixture_body
    if is_compact(model, fx):
        rows = model.slot_fixture
        fbody = fbody[rows]
        model = model._replace(verts=model.verts[rows], count=model.count[rows], radius=model.radius[rows],
                               shape_type=model.shape_type[rows])
    c1, s1 = m.rot_of(bd.a0)
    lcx, lcy = bd.local_center[:, 0], bd.local_center[:, 1]
    rx, ry = m.rot(c1, s1, lcx, lcy)
    p1x, p1y = bd.c0[:, 0] - rx, bd.c0[:, 1] - ry
    g = lambda a: a[fbody]
    a1 = shapes.fixture_aabbs(model, g(p1x), g(p1y), g(c1), g(s1))
    a2 = shapes.fixture_aabbs(model, *(g(x) for x in body_xf(bd, jnp.arange(bd.type.shape[0]))))
    lx, ly = m.bmin(a1[0], a2[0]), m.bmin(a1[1], a2[1])
    ux, uy = m.bmax(a1[2], a2[2]), m.bmax(a1[3], a2[3])
    dx, dy = g(bd.xf_p[:, 0]) - g(p1x), g(bd.xf_p[:, 1]) - g(p1y)
    old = fx.fat
    contains = (old[:, 0] <= lx) & (old[:, 1] <= ly) & (ux <= old[:, 2]) & (uy <= old[:, 3])
    nfat = jnp.stack(shapes.fatten(lx, ly, ux, uy, dx, dy), -1)
    upd = g(moved) & ~contains
    return state._replace(fixtures=fx._replace(fat=jnp.where(upd[:, None], nfat, old)))
