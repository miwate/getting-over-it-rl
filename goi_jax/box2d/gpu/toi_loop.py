"""b2World::SolveTOI's event loop over a batch of worlds with the active worlds compacted (Settings.solver "pallas").

toi.solve_toi runs per world under vmap: its lax.while_loop becomes one batched loop that keeps running the search
and the event (the Pallas kernels of toi_search.py / toi_event.py plus their XLA packing, the fixture sync and
FindNewContacts) on every world until the world with the most TOI events of the step is done, and merges every
world's state with a select each iteration. Most worlds have no event, or one, so most of that work is on idle
worlds.

Here the loop runs at the batch level (a custom_vmap rule): one search over all worlds, then each iteration gathers
up to M worlds that still have a pending event (lowest index first), runs event + search on just those M lanes and
scatters them back. Each world's events are its own business (nothing in the search or the event reads another
world), so doing a world's k-th event in a later iteration than another world's changes no bit of either: every
world goes through exactly the sequence of event / search calls toi.solve_toi gives it (including the per-world
cap of max_events iterations). The event call of solve_toi's first iteration has no pending event and changes
nothing; it is skipped. Unbatched calls run toi.solve_toi's own loop.

M = the batch / DIVISOR (at least MIN_LANES, at most the batch); read at trace time. SMALL_DIVISOR adds a second,
smaller lane count for the iterations with few active worlds.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .. import toi as T
from ..world import empty_toi
from . import speed_option

ENABLED = True
DIVISOR = 4
MIN_LANES = 256
# SMALL_DIVISOR > 0: an iteration with at most batch / SMALL_DIVISOR active worlds (rounded up to a kernel block)
# runs event + search on that many lanes instead of M (a lax.cond between the two sizes; same bits, the lanes only
# change which worlds share a launch). 0 (default): always M. GOI_GPU_SPEED=small (or all) sets 16.
SMALL_DIVISOR = 16 if speed_option("small") else 0
F32 = np.float32


class _Static:
    def __init__(self, v):
        self.v = v


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.v), lambda v, _: _Static(v))


def _per_world(model, state, tst, dt, settings, max_events):
    """toi.solve_toi's loop (pallas kernels), one world."""
    from .toi_event import event
    from .toi_search import search
    go0 = jnp.asarray(dt, jnp.float32) > 0

    def cond(c):
        return c[2] & (c[6] <= max_events)

    def body(c):
        state, tst, go, pending, ms, ma, it = c
        state, tst = event(model, state, tst, settings, dt, pending, ms, ma, None)
        state, tst, found, ms, ma = search(model, state, tst, settings, go, None)
        return state, tst, go & found, go & found, ms, ma, it + 1

    init = (state, tst, go0, jnp.asarray(False), jnp.int32(0), F32(1.0), jnp.int32(0))
    state, tst = jax.lax.while_loop(cond, body, init)[:2]
    return state, tst


@jax.custom_batching.custom_vmap
def _loop(box, model, state, tst, dt):
    settings, max_events = box.v
    return _per_world(model, state, tst, dt, settings, max_events)


@_loop.def_vmap
def _loop_rule(axis_size, in_batched, box, model, state, tst, dt):
    settings, max_events = box.v
    N = axis_size
    bcast = lambda t, b: jax.tree.map(lambda x, bb: x if bb else jnp.broadcast_to(x, (N,) + jnp.shape(x)), t, b)  # noqa
    state, tst, dt = bcast(state, in_batched[2]), bcast(tst, in_batched[3]), bcast(dt, in_batched[4])
    if any(jax.tree.leaves(in_batched[1])):  # a per-world Model: no compaction
        out = jax.vmap(lambda md, st, ts, d: _per_world(md, st, ts, d, settings, max_events))(
            bcast(model, in_batched[1]), state, tst, dt)
        return out, jax.tree.map(lambda _: True, out)
    from .toi_event import event
    from .toi_search import search
    vsearch = jax.vmap(lambda st, ts, g: search(model, st, ts, settings, g, "w"), axis_name="w")
    vevent = jax.vmap(lambda st, ts, d, g, s_, a_: event(model, st, ts, settings, d, g, s_, a_, "w"),
                      axis_name="w")

    go0 = dt > 0
    # solve_toi's iteration 0 (its event has nothing pending): the search on every world
    state, tst, found, ms, ma = vsearch(state, tst, go0)
    act = go0 & found & (1 <= max_events)   # a world goes on while searching and its iteration count <= max_events
    it = jnp.ones((N,), jnp.int32)
    from . import solve as G
    M = int(min(N, max(MIN_LANES, -(-N // DIVISOR))))
    M = int(min(N, -(-M // G.BLOCK) * G.BLOCK)) if M < N else N

    def cond(c):
        return jnp.any(c[2])

    def body_m(c, M):
        state, tst, act, ms, ma, it = c
        idx = jnp.nonzero(act, size=M, fill_value=N)[0]
        ok = idx < N
        j = jnp.minimum(idx, N - 1)
        take = lambda t: jax.tree.map(lambda x: x[j], t)  # noqa: E731
        st, ts = take(state), take(tst)
        st, ts = vevent(st, ts, dt[j], ok, ms[j], ma[j])
        st, ts, fnd, ms_s, ma_s = vsearch(st, ts, ok)
        dst = jnp.where(ok, idx, N)
        put = lambda full, part: jax.tree.map(lambda x, y: x.at[dst].set(y, mode="drop"), full, part)  # noqa: E731
        state, tst = put(state, st), put(tst, ts)
        it_n = it[j] + 1
        it = it.at[dst].set(it_n, mode="drop")
        act = act.at[dst].set(fnd & (it_n <= max_events), mode="drop")
        ms = ms.at[dst].set(ms_s, mode="drop")
        ma = ma.at[dst].set(ma_s, mode="drop")
        return state, tst, act, ms, ma, it

    Ms = int(min(M, -(-max(1, -(-N // SMALL_DIVISOR)) // G.BLOCK) * G.BLOCK)) if SMALL_DIVISOR else M
    if Ms < M:  # few worlds left with an event (the loop's later iterations): the same iteration on Ms lanes
        def body(c):
            return jax.lax.cond(jnp.sum(c[2].astype(jnp.int32)) <= Ms, lambda c_: body_m(c_, Ms),
                                lambda c_: body_m(c_, M), c)
    else:
        def body(c):
            return body_m(c, M)

    state, tst = jax.lax.while_loop(cond, body, (state, tst, act, ms, ma, it))[:2]
    out = (state, tst)
    return out, jax.tree.map(lambda _: True, out)


def solve_toi(model, state, settings, dt, max_events):
    """toi.solve_toi's loop for Settings.solver "pallas" (state.toi is replaced by the fresh bookkeeping)."""
    B, C = state.bodies.type.shape[0], state.contacts.pair.shape[0]
    st, tst = _loop(_Static((settings, int(max_events))), model, state._replace(toi=None), empty_toi(B, C),
                    jnp.asarray(dt, jnp.float32))
    return st, tst
