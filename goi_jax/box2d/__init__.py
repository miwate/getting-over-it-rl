"""float32 JAX port of Box2D 2.3.1's b2World::Step, the engine under Unity 2020.3's Physics2D, for Getting Over It.

    from goi_jax import box2d as jb
    d = jb.WorldDef(jb.Settings.game())            # same builder calls as goi_jax.reference.box2d.World
    g = d.add_body(jb.STATIC); d.add_box(g, (10, 0.5), friction=1.0)
    b = d.add_body(jb.DYNAMIC, (0, 1.2), 0.3, mass=2, inertia=1 / 3); d.add_box(b, (0.5, 0.5))
    model, state = d.build(contact_capacity=16)
    state = jb.step(model, state, d.settings, jb.GAME_DT)
    run = jax.jit(jax.vmap(lambda s: jb.step(model, s, settings, dt, axis_name="w"), axis_name="w"))

Parity: against the oracle (goi_jax/reference, Box2D 2.3.1 built with -ffp-contract=off and correctly rounded
sin/cos, trig_mode=1) every test scene in tests/box2d is bit-identical at every step (120-1200 steps checked):
bodies (transform and sweep centre), contact impulses, manifold geometry and feature ids, contact-list order,
joint impulses and limit states, awake flags and sleep timers. Prismatic axes are normalized twice, as the
oracle's C API does (world.py).

Layout (world.py)
- Model: what never changes in a scene (fixture geometry, joint frames, the candidate pair list); shared by all
  worlds. State: everything per world (bodies, fixture friction and fat AABBs, joint parameters and impulses, a
  fixed table of C contact slots, gravity, inv_dt0). Fixed capacities B, F, J, C, P; masks instead of lists.
- Bodies, fixtures and joints are indexed in creation order. Contacts live in slots; each has a creation key.

Floating point (math2d.py)
- One rounding per C operation, in C's order. XLA CPU fuses `a * b + c` into an FMA, so every product goes
  through math2d.mul (a select stops the fusion); tests/box2d lints for raw `x + a * b`.
- sin/cos are correctly rounded float32 (float-float evaluation); the oracle's trig_mode=1 matches. glibc's sinf
  misrounds ~1.4% of angles and the game's MSVC sinf is unknown: with the oracle on glibc sinf, scenes diverge
  after 6-40 steps but stay within ~2e-5 over 10 s (pile, revolute, player rig).
- Never divide by a constant (XLA rewrites x / c as x * (1/c)). XLA CPU flushes denormals to zero; Box2D does not.

Ordering rules (what makes Gauss-Seidel bit-identical)
- Contacts (pairs.py): created by FindNewContacts in candidate order = b2BroadPhase::UpdatePairs order, ascending
  (lower, higher) proxy id, with proxy id = fixture creation order; keys ascend in that order, and Box2D's world list
  and every body's edge list are newest first = descending key. Fixture A/B as b2Contact::Create orders them
  (polygon before circle). Fat AABBs follow b2DynamicTree::MoveProxy exactly, so pairs appear and disappear on the
  same step as in Box2D (contacts are created at the end of a step and get their first manifold in the next).
- Collide (pairs.py) visits the contact list newest first and a visit can wake bodies (touching flips, or a
  destroyed contact still had points), which makes later contacts of those bodies active in the same pass; the
  port resolves that order dependence exactly with a short fixed-point loop over list positions.
- Islands (island.py): b2World::Solve's DFS, seeded from the highest body index (body list newest first), contact
  edges newest first, then joint edges newest first (descending joint index). All islands are solved in one pass
  over that order; per-island decisions (position-iteration early exit, sleeping) stay per island.
- Within a velocity iteration: all joints, then all contacts, in island order (b2Island::Solve).

Known divergences / limits
- TOI (b2World::SolveTOI, toi.py) runs when Settings.continuous_physics is set (Settings.game(), with Unity's pair
  rule); tests/box2d/test_toi.py checks it bit for bit against the oracle with continuous physics on.
- Sensor pairs are not tracked (no effect on the physics; trigger callbacks need them later).
- Proxy ids are assumed to be fixture creation order. Box2D re-allocates ids when a proxy is destroyed and
  re-created (fixture removal, body SetActive(false/true)), which would change creation order of later pairs.
- Filtering is resolved at build time (layer matrix, collideConnected, same body); a contact is never re-filtered.
- A full contact table drops new contacts (State.overflow counts them; size C for the worst case).
- Not ported: edge/chain shapes, other joint types, tangent speed (always 0), sub-stepping, ShiftOrigin,
  joint reaction readouts (impulse * inv_dt), b2ContactListener callbacks, everything Unity does around Step
  (MovePosition, transform sync, material refresh). The game's own Box2D build (Unity's fork, MSVC) may differ from
  2.3.1 in ways only game recordings can show.

Modules: settings (constants), math2d (float32 b2Math, trig, one-hot body access), shapes (construction, AABBs),
collide (narrow phase), pairs (broad phase emulation, manifold update), island (DFS order), contact_solver, joints,
loops (sweep loop shapes for one world vs a batch), step (b2World::Step), world (pytrees, builder).
"""
import numpy as np

from .settings import Settings
from .step import step
from .world import DYNAMIC, KINEMATIC, PRISMATIC, REVOLUTE, STATIC, WELD, Model, State, WorldDef

GAME_DT = np.float32(0.008333334)  # Time.fixedDeltaTime (float32 nearest 1/120)
