"""GOI_GPU_SPEED option "graph": keep the training update's loops inside the CUDA graph (no host round trips).

With XLA command buffers on (scripts/cuda_run.sh), every while loop whose body is all kernels becomes a CUDA-graph
conditional node and iterates on the device. XLA:GPU however turns a dynamic-update-slice whose update is a plain
buffer (a scan's stacked output, a one-row scatter such as `x.at[j].set(v)` under vmap) into an async memcpy on a
side stream ("dynamic-slice-copy-fusion-async-wrapper"), and an async start/done pair in a loop body keeps that loop
on the host: each iteration copies its predicate back and relaunches its body graph. In the training update that
held the rollout's step loop, the K physics sub-steps per env step and the GAE scan on the host (runs/speed2/hlo6;
scripts/graph_check.py lists such loops from a thunk dump). No XLA option removes the wrapper in jax 0.12.

The helpers below write the same bits without a copy-only dynamic-update-slice:
- `opaque(z, tree)`: each leaf through an XOR of its bits with `z`, an int32 that is 0 at run time but not a
  compile-time constant; the write then fuses with a (no-op) computation and is no longer a copy;
- `set_rows(x, rows, v)`: x.at[rows].set(v) for static row indices as a select over the whole (small) array;
- `set_at(x, k, v)`: x.at[k].set(v) for a traced index k known to be in range (a loop counter), as a select. At
  32,768 envs XLA split sim.py's callback-dispatch loop (`out.at[k].set`) so that its write became an async copy and
  the rollout and sub-step loops went back to the host (runs/speed3/hlo/full_32768_4, review of speed round 3).
With the option off they are the plain operations (defaults unchanged). scripts/graph_probe.py checks the pattern on
a toy update (compile only).
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import speed_option

ENABLED = speed_option("graph")
_UINT = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}


def runtime_zero(i):
    """0 at run time for an int32 i >= 0 (its sign bit), not foldable by the compiler."""
    return jax.lax.shift_right_arithmetic(jnp.asarray(i, jnp.int32), jnp.int32(31))


def _opaque_leaf(z, x):
    x = jnp.asarray(x)
    if x.dtype == jnp.bool_:
        return x ^ (z != 0)
    u = _UINT[x.dtype.itemsize]
    return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(x, u) ^ z.astype(u), x.dtype)


def opaque(z, tree):
    """The tree unchanged, bit for bit (when ENABLED: through an XOR with the run-time zero z)."""
    if not ENABLED:
        return tree
    return jax.tree.map(lambda x: None if x is None else _opaque_leaf(z, x), tree)


def set_rows(x, rows, v):
    """x.at[rows].set(v) for static row indices (int or sequence of ints) of a small array; when ENABLED as a
    select over the whole array (identical values, no dynamic-update-slice)."""
    if not ENABLED:
        return x.at[jnp.asarray(rows) if not np.isscalar(rows) else rows].set(v)
    n = x.shape[0]
    idx = np.atleast_1d(np.asarray(rows))
    if np.isscalar(rows) or np.ndim(rows) == 0:
        mask = np.arange(n) == int(rows)
        val = jnp.broadcast_to(jnp.asarray(v, x.dtype), x.shape[1:])[None]
    else:
        # same rule as .at[].set with sequence indices: v broadcasts to (len(idx),) + x.shape[1:]; one value per row
        assert len(set(idx.tolist())) == len(idx), "set_rows: duplicate rows"
        vv = jnp.broadcast_to(jnp.asarray(v, x.dtype), (len(idx),) + x.shape[1:])
        pos = np.full(n, 0, np.int64)
        pos[idx] = np.arange(len(idx))
        mask = np.zeros(n, bool)
        mask[idx] = True
        val = vv[jnp.asarray(pos)]
    mask = jnp.asarray(mask).reshape((n,) + (1,) * (x.ndim - 1))
    return jnp.where(mask, val, x)


def set_at(x, k, v):
    """x.at[k].set(v) along axis 0 for a traced int index 0 <= k < x.shape[0] (e.g. a fori_loop counter); when
    ENABLED as a select over the whole (small) array (identical values for in-range k, no dynamic-update-slice)."""
    if not ENABLED:
        return x.at[k].set(v)
    mask = (jnp.arange(x.shape[0]) == k).reshape((x.shape[0],) + (1,) * (x.ndim - 1))
    return jnp.where(mask, jnp.broadcast_to(jnp.asarray(v, x.dtype), x.shape[1:])[None], x)
