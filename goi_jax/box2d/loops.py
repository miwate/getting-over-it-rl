"""The loop shape of the sequential Gauss-Seidel sweeps (one constraint per iteration, in solve order).

`ordered_loop(n, size, body, carry, static, axis_name)` runs `carry = body(k, carry, live)` for k = 0, 1, ...; the
body must leave the carry unchanged where `live` is false (it masks only the entries it touches). Same results in
every form; they differ in cost:
- one world, dynamic: lax.fori_loop(0, n), live always true.
- static (joints): `size` iterations, live = k < n. The joint table is nearly always full, and under vmap a per-world
  trip count turns the loop into a while loop that selects the whole carry every iteration (player rig, 256 worlds,
  CPU: 296 -> 131 ms per step from this alone).
- batched, dynamic (contacts, under `jax.vmap(..., axis_name=axis_name)`): max(n) over the batch via lax.pmax, so the
  trip count is shared and only touched entries are masked. The contact table is sized for the worst case and mostly
  empty, so a static loop would solve every empty slot (131 -> 774 ms).
"""
import jax


def ordered_loop(n, size, body, carry, static=False, axis_name=None):
    if size == 0:
        return carry
    if static:
        return jax.lax.fori_loop(0, size, lambda k, c: body(k, c, k < n), carry)
    if axis_name is not None:
        return jax.lax.fori_loop(0, jax.lax.pmax(n, axis_name), lambda k, c: body(k, c, k < n), carry)
    return jax.lax.fori_loop(0, n, lambda k, c: body(k, c, True), carry)
