"""Island discovery: the depth-first search of b2World::Solve (v2.3.1), done once per step for the whole world.

Box2D solves island after island. The islands share no non-static body and static bodies are never changed by a
constraint (zero inverse mass), so solving every island's constraints in one pass gives the same floats, provided
(1) each island's constraints keep Box2D's order relative to each other, and (2) what Box2D decides per island stays
per island: the position-iteration early exit and falling asleep. This module returns that order and the island
labels; contact_solver/joints/step use them.

The search, as in b2World::Solve: seeds are tried in body-list order (newest body first = descending index); a
seed must be unvisited, awake, active and not static. Popping body b (LIFO stack) wakes it, then walks b's contact
edges newest first (descending key): each enabled, touching, non-sensor contact not yet in an island joins the
island in that order and pushes its other body if unvisited; then b's joint edges newest first (descending joint
index) the same way (joints to inactive bodies are skipped). Static bodies are never pushed here: Box2D pushes
them but does not expand them, so that changes no order.

Vectorized form: each loop iteration either pops one body or starts one island; a body's edges are handled at
once with prefix sums over the edges in list order. 2*B iterations always suffice (one seed and one pop per body);
the loop ends as soon as the stack is empty and no seed is left.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .world import STATIC


class Islands(NamedTuple):
    contact_order: jnp.ndarray  # i32[C] contact slots in solve order (first n_contacts valid, rest -1)
    n_contacts: jnp.ndarray  # i32
    joint_order: jnp.ndarray  # i32[J] joints in solve order (first n_joints valid)
    n_joints: jnp.ndarray
    body_island: jnp.ndarray  # i32[B] island of each body, -1 if in none (asleep, inactive, static)
    contact_island: jnp.ndarray  # i32[C] by slot, -1 if not solved
    joint_island: jnp.ndarray  # i32[J] by joint, -1 if not solved
    n_islands: jnp.ndarray
    awake: jnp.ndarray  # bool[B] after the search's SetAwake(true)
    sleep_time: jnp.ndarray  # f32[B]


def find_islands(model, state):
    bd, ct = state.bodies, state.contacts
    B, C, J = bd.type.shape[0], ct.pair.shape[0], model.joint_type.shape[0]
    fbody = model.fixture_body
    ca, cb = fbody[ct.fa], fbody[ct.fb]
    c_ok = (ct.pair >= 0) & ct.touching  # enabled always; sensors never get a slot
    # Contact-edge list order: descending key; invalid slots last.
    sort_key = jnp.where(c_ok, ct.key, jnp.iinfo(jnp.int32).min)
    perm = jnp.argsort(_neg(sort_key), stable=True).astype(jnp.int32)
    s_ok, s_a, s_b = c_ok[perm], ca[perm], cb[perm]
    # Joint-edge order: descending joint index.
    jperm = jnp.arange(J - 1, -1, -1, dtype=jnp.int32)
    ja, jb = model.joint_a[jperm], model.joint_b[jperm]
    static = bd.type == STATIC
    seedable = bd.active & ~static

    carry = dict(
        flag=jnp.zeros(B, bool), cflag=jnp.zeros(C, bool), jflag=jnp.zeros(J, bool),
        stack=jnp.zeros(B, jnp.int32), sp=jnp.int32(0),
        corder=jnp.full(C, -1, jnp.int32), nc=jnp.int32(0), jorder=jnp.full(J, -1, jnp.int32), nj=jnp.int32(0),
        body_island=jnp.full(B, -1, jnp.int32), cisl=jnp.full(C, -1, jnp.int32), jisl=jnp.full(J, -1, jnp.int32),
        nisl=jnp.int32(0), awake=bd.awake, sleep_time=bd.sleep_time)

    def first_pos(mask, other, n):
        """For each element, is it the first in order (lowest position) among `mask` with its `other` value."""
        pos = jnp.arange(mask.shape[0], dtype=jnp.int32)
        big = jnp.int32(mask.shape[0])
        first = jnp.full(n, big, jnp.int32).at[jnp.where(mask, other, n)].min(pos, mode="drop")
        return mask & (first[jnp.clip(other, 0, n - 1)] == pos)

    def push(k, stack, sp, flag, mask, other):
        rank = jnp.cumsum(mask) - 1
        stack = stack.at[jnp.where(mask, sp + rank, B)].set(other, mode="drop")
        flag = flag.at[jnp.where(mask, other, B)].set(True, mode="drop")
        return stack, sp + jnp.sum(mask).astype(jnp.int32), flag

    def body_fn(k, s):
        empty = s["sp"] == 0
        # --- start an island: highest-index unvisited seedable awake body
        cand = seedable & s["awake"] & ~s["flag"]
        has_seed = jnp.any(cand)
        seed = (B - 1 - jnp.argmax(cand[::-1])).astype(jnp.int32)
        start = empty & has_seed
        # --- pop
        pop = ~empty
        b = s["stack"][jnp.maximum(s["sp"] - 1, 0)]
        isl = s["nisl"] - 1

        # contacts of b, in edge order
        involves = (s_a == b) | (s_b == b)
        elig = pop & s_ok & involves & ~s["cflag"][perm]
        rank = jnp.cumsum(elig) - 1
        corder = s["corder"].at[jnp.where(elig, s["nc"] + rank, C)].set(perm, mode="drop")
        cisl = s["cisl"].at[jnp.where(elig, perm, C)].set(isl, mode="drop")
        cflag = s["cflag"].at[jnp.where(elig, perm, C)].set(True, mode="drop")
        nc = s["nc"] + jnp.sum(elig).astype(jnp.int32)
        other = jnp.where(s_a == b, s_b, s_a)
        pushable = elig & ~s["flag"][other] & ~static[other]
        pushable = first_pos(pushable, other, B)
        sp0 = s["sp"] - pop.astype(jnp.int32)
        stack, sp, flag = push(k, s["stack"], sp0, s["flag"], pushable, other)

        # joints of b, in edge order (after the contact pushes)
        jinv = (ja == b) | (jb == b)
        jother = jnp.where(ja == b, jb, ja)
        jel = pop & jinv & ~s["jflag"][jperm] & bd.active[jother]
        jrank = jnp.cumsum(jel) - 1
        jorder = s["jorder"].at[jnp.where(jel, s["nj"] + jrank, J)].set(jperm, mode="drop")
        jisl = s["jisl"].at[jnp.where(jel, jperm, J)].set(isl, mode="drop")
        jflag = s["jflag"].at[jnp.where(jel, jperm, J)].set(True, mode="drop")
        nj = s["nj"] + jnp.sum(jel).astype(jnp.int32)
        jpush = first_pos(jel & ~flag[jother] & ~static[jother], jother, B)
        stack, sp, flag = push(k, stack, sp, flag, jpush, jother)

        # island membership + SetAwake(true) of the popped body
        body_island = s["body_island"].at[jnp.where(pop, b, B)].set(isl, mode="drop")
        was = s["awake"][b]
        awake = s["awake"].at[jnp.where(pop, b, B)].set(True, mode="drop")
        sleep_time = s["sleep_time"].at[jnp.where(pop & ~was, b, B)].set(0.0, mode="drop")

        # seed branch (only when the stack is empty)
        stack = jnp.where(start, stack.at[0].set(seed), stack)
        sp = jnp.where(start, 1, sp)
        flag = jnp.where(start, flag.at[seed].set(True), flag)
        nisl = s["nisl"] + start.astype(jnp.int32)
        return dict(flag=flag, cflag=cflag, jflag=jflag, stack=stack, sp=sp, corder=corder, nc=nc,
                    jorder=jorder, nj=nj, body_island=body_island, cisl=cisl, jisl=jisl, nisl=nisl,
                    awake=awake, sleep_time=sleep_time)

    # 2*B iterations always suffice; the loop stops early once the stack is empty and no seed is left (the remaining
    # iterations would change nothing). Under vmap it runs until every world is done.
    def more(c):
        k, s = c
        left = (s["sp"] > 0) | jnp.any(seedable & s["awake"] & ~s["flag"])
        return (k < 2 * B) & left

    _, s = jax.lax.while_loop(more, lambda c: (c[0] + 1, body_fn(c[0], c[1])), (jnp.int32(0), carry))
    return Islands(contact_order=s["corder"], n_contacts=s["nc"], joint_order=s["jorder"], n_joints=s["nj"],
                   body_island=s["body_island"], contact_island=s["cisl"], joint_island=s["jisl"],
                   n_islands=s["nisl"], awake=s["awake"], sleep_time=s["sleep_time"])


def _neg(k):
    """-k without int32 overflow at INT_MIN (descending sort key)."""
    return jnp.where(k == jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, -k)
