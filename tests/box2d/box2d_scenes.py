"""Test scenes, each built through harness.Twin (so the JAX world and the Box2D oracle get the same calls).
Units: metres, kilograms, radians. All use the game's settings (300 velocity / 30 position iterations)."""
import json

import numpy as np

from goi_jax.box2d.world import DYNAMIC, KINEMATIC, STATIC
from goi_jax.reference import box2d as b2
from goi_jax.reference import scene as ref_scene

from box2d_harness import Twin

F32 = np.float32


def _ngon(n, r, phase=0.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.stack([r * np.cos(a), r * np.sin(a)], 1).astype(F32)


def free_fall():
    """Free bodies: semi-implicit Euler with gravity and damping; a spinning one and a kinematic mover."""
    t = Twin()
    a = t.add_body(DYNAMIC, (1.25, 7.5), 0.0, (3, 2), 0.0, mass=4, inertia=1, linear_damping=0.2)
    t.add_box(a, (0.3, 0.2))
    b = t.add_body(DYNAMIC, (-2, 3), 0.7, (-1, 4), 5.0, mass=1.5, center=(0.1, -0.05), inertia=0.3,
                   linear_damping=0.05, angular_damping=0.5)
    t.add_polygon(b, _ngon(5, 0.4))
    k = t.add_body(KINEMATIC, (5, 0), 0.1, (0.5, -0.25), 1.0)
    t.add_box(k, (0.5, 0.1))
    return t


def box_on_ground():
    t = Twin()
    g = t.add_body(STATIC)
    t.add_box(g, (10, 0.5), friction=1.0)
    b = t.add_body(DYNAMIC, (0, 1.015), mass=2, inertia=F32(2) / F32(6))
    t.add_box(b, (0.5, 0.5), friction=0.5)
    return t


def box_stack():
    t = Twin()
    g = t.add_body(STATIC)
    t.add_box(g, (10, 0.5), friction=0.6)
    for i in range(4):
        b = t.add_body(DYNAMIC, (0.03 * i, 1.0 + 0.995 * i + 0.01), 0.002 * i, mass=1, inertia=F32(1) / F32(6))
        t.add_box(b, (0.5, 0.49), friction=0.6)
    return t


def polygons_at_angles():
    """Polygon-polygon manifolds: boxes, a hexagon and a triangle landing at assorted angles on a flat and a
    sloped static polygon, plus two boxes hitting each other mid-air."""
    t = Twin(contact_capacity=24)
    g = t.add_body(STATIC)
    t.add_box(g, (12, 0.5), friction=0.8)
    t.add_polygon(g, [(4, 0.5), (9, 0.5), (9, 2.3)], friction=0.8)  # ramp, about 20 degrees
    for i, ang in enumerate((0.0, 0.3, np.pi / 4, 1.4, -0.7)):
        b = t.add_body(DYNAMIC, (-8 + 2.2 * i, 1.6 + 0.1 * i), ang, (0, -1), 0.5 * (i - 2), mass=1 + 0.3 * i,
                       inertia=0.2 + 0.05 * i)
        t.add_box(b, (0.45, 0.3 + 0.05 * i), friction=0.5)
    h = t.add_body(DYNAMIC, (6.5, 2.2), 0.1, mass=2, inertia=0.4)
    t.add_polygon(h, _ngon(6, 0.45, 0.2), friction=0.4)
    tri = t.add_body(DYNAMIC, (3.0, 1.2), 2.0, mass=1, inertia=0.1)
    t.add_polygon(tri, [(0.4, 0.0), (-0.3, 0.35), (-0.25, -0.3)], friction=0.9)
    p = t.add_body(DYNAMIC, (0.0, 4.0), 0.2, (2, 0), 0, mass=1, inertia=0.15)
    t.add_box(p, (0.3, 0.3))
    q = t.add_body(DYNAMIC, (1.2, 4.1), -0.3, (-2, 0), 1, mass=1.5, inertia=0.2)
    t.add_box(q, (0.35, 0.25))
    return t


def polygon_circle():
    """Polygon-circle (face and both vertex regions) and circle-circle manifolds."""
    t = Twin(contact_capacity=24)
    g = t.add_body(STATIC)
    t.add_box(g, (12, 0.5), friction=0.7)
    t.add_polygon(g, [(-9, 0.5), (-4, 0.5), (-9, 2.0)], friction=0.7)  # slope facing right
    c1 = t.add_body(DYNAMIC, (0, 0.82), mass=1, inertia=0.05)
    t.add_circle(c1, 0.3, friction=0.5)
    c2 = t.add_body(DYNAMIC, (-6.5, 2.0), 0.0, (0, 0), -2.0, mass=1, inertia=0.04)  # rolls down the slope
    t.add_circle(c2, 0.25, (0.02, 0.0), friction=0.9)
    box = t.add_body(STATIC, (4, 1.5), 0.3)
    t.add_box(box, (0.5, 0.2))
    c3 = t.add_body(DYNAMIC, (4.6, 2.4), 0.0, (-0.5, -1), 0, mass=0.5, inertia=0.01)  # lands on the box's corner
    t.add_circle(c3, 0.2)
    c4 = t.add_body(DYNAMIC, (1.5, 3.0), 0.0, (1.5, 0), 0, mass=0.8, inertia=0.02)
    t.add_circle(c4, 0.25)
    c5 = t.add_body(DYNAMIC, (2.6, 3.05), 0.0, (-1.5, 0), 0, mass=1.0, inertia=0.03)
    t.add_circle(c5, 0.3)
    s1 = t.add_body(DYNAMIC, (-2, 0.81), mass=1, inertia=0.05)
    t.add_circle(s1, 0.3)
    s2 = t.add_body(DYNAMIC, (-1.98, 1.4), mass=0.5, inertia=0.02)  # circle stacked on a circle
    t.add_circle(s2, 0.28)
    hb = t.add_body(DYNAMIC, (7, 1.4), 0.6, (0, -0.5), 0, mass=1, inertia=0.1)
    t.add_box(hb, (0.4, 0.3))
    hc = t.add_body(DYNAMIC, (7.1, 2.3), 0.0, (0, -1), 0, mass=0.3, inertia=0.005)  # circle on a dynamic box
    t.add_circle(hc, 0.15)
    return t


def revolute():
    """Pendulums on a static anchor: motor + limits (motor drives into the upper limit), a free one swinging
    through both limits, and a dynamic-dynamic hinge chain."""
    t = Twin()
    g = t.add_body(STATIC)
    t.add_box(g, (0.2, 0.2), (0, 5))
    a = t.add_body(DYNAMIC, (1.0, 5.0), 0.0, mass=1, inertia=0.1)
    t.add_box(a, (0.5, 0.1))
    t.add_revolute(g, a, (0, 5), (-0.5, 0), 0.0, limits=(-0.6, 0.4), motor_speed=2.0, max_motor_torque=30.0,
                   enable_motor=True)
    b = t.add_body(DYNAMIC, (-1.0, 5.0), 0.0, mass=2, inertia=0.3)
    t.add_box(b, (0.6, 0.1))
    t.add_revolute(g, b, (0, 5), (0.6, 0), 0.0, limits=(-0.5, 0.5))
    c = t.add_body(DYNAMIC, (-2.2, 5.0), 0.0, (0, 1), 0, mass=1, inertia=0.1)
    t.add_box(c, (0.6, 0.08))
    t.add_revolute(b, c, (-0.6, 0), (0.6, 0), 0.0, motor_speed=-1.0, max_motor_torque=5.0, enable_motor=True)
    return t


def prismatic():
    """A slider on an angled axis from a static body (motor driving into the upper limit), a dynamic-dynamic
    prismatic with limits, and one with equal limits."""
    t = Twin()
    g = t.add_body(STATIC)
    t.add_box(g, (0.2, 0.2))
    s = t.add_body(DYNAMIC, (0.5, 0.5), 0.3, mass=2, inertia=0.2)
    t.add_box(s, (0.3, 0.1))
    ang = 0.7
    t.add_prismatic(g, s, (0, 0), (0, 0), (np.cos(ang), np.sin(ang)), 0.3, limits=(-0.2, 1.0), motor_speed=1.5,
                    max_motor_force=200.0, enable_motor=True)
    u = t.add_body(DYNAMIC, (3, 2), 0.1, (0.5, 0), 0.2, mass=1, inertia=0.1)
    t.add_box(u, (0.3, 0.3))
    v = t.add_body(DYNAMIC, (3.6, 2.05), 0.1, mass=0.5, inertia=0.05)
    t.add_box(v, (0.2, 0.2))
    t.add_prismatic(u, v, (0.3, 0), (-0.2, 0), (1, 0), 0.0, limits=(-0.05, 0.25), motor_speed=-0.5,
                    max_motor_force=10.0, enable_motor=True)
    e = t.add_body(DYNAMIC, (-2, 1), 0.0, mass=1, inertia=0.1)
    t.add_box(e, (0.2, 0.2))
    t.add_prismatic(g, e, (-2, 1), (0, 0), (0, 1), 0.0, limits=(0.0, 0.0))
    return t


def weld():
    """Rigid and soft (frequency/damping) welds: cantilevers from a static body and a welded dynamic pair."""
    t = Twin()
    g = t.add_body(STATIC)
    t.add_box(g, (0.2, 0.5), (0, 3))
    a = t.add_body(DYNAMIC, (0.7, 3.0), 0.0, mass=1, inertia=0.1)
    t.add_box(a, (0.5, 0.1))
    t.add_weld(g, a, (0.2, 3.0), (-0.5, 0.0), 0.0)
    b = t.add_body(DYNAMIC, (-0.7, 3.0), 0.0, mass=1, inertia=0.1)
    t.add_box(b, (0.5, 0.1))
    t.add_weld(g, b, (-0.2, 3.0), (0.5, 0.0), 0.0, frequency=5.0, damping=0.7)
    c = t.add_body(DYNAMIC, (3, 3), 0.2, (1, 0), 1.0, mass=1, inertia=0.1)
    t.add_box(c, (0.3, 0.1))
    d = t.add_body(DYNAMIC, (3.6, 3.12), 0.2, mass=2, inertia=0.2)
    t.add_box(d, (0.3, 0.1))
    t.add_weld(c, d, (0.3, 0), (-0.3, 0), 0.0, frequency=70.0, damping=1.0)
    e = t.add_body(DYNAMIC, (5, 3), 0.0, (0, 0), -2.0, mass=1, inertia=0.1)
    t.add_box(e, (0.3, 0.1))
    f = t.add_body(DYNAMIC, (5.6, 3.0), 0.0, mass=1, inertia=0.1)
    t.add_box(f, (0.3, 0.1))
    t.add_weld(e, f, (0.3, 0), (-0.3, 0), 0.0)
    return t


def pile():
    """Twelve mixed bodies (boxes, polygons, circles) dropped into a static V-shaped bin: many simultaneous
    contacts, islands merging, contact creation and destruction, sleeping at the end."""
    t = Twin(contact_capacity=64)
    g = t.add_body(STATIC)
    t.add_box(g, (6, 0.3), friction=0.6)
    t.add_polygon(g, [(-6, 0.3), (-5.5, 0.3), (-6, 3.0)], friction=0.6)
    t.add_polygon(g, [(5.5, 0.3), (6, 0.3), (6, 3.0)], friction=0.6)
    rng = np.random.default_rng(7)
    for i in range(12):
        x, y = F32(rng.uniform(-3, 3)), F32(1.2 + 0.9 * i)
        ang, w = F32(rng.uniform(-1.5, 1.5)), F32(rng.uniform(-2, 2))
        b = t.add_body(DYNAMIC, (x, y), ang, (F32(rng.uniform(-1, 1)), 0), w, mass=F32(0.5 + 0.1 * i),
                       inertia=F32(0.05 + 0.01 * i))
        kind = i % 3
        if kind == 0:
            t.add_box(b, (F32(0.25 + 0.02 * i), F32(0.2)), friction=0.5)
        elif kind == 1:
            t.add_polygon(b, _ngon(3 + i % 5, 0.3), friction=0.4)
        else:
            t.add_circle(b, F32(0.2 + 0.01 * i), friction=0.6)
    return t


def misc():
    """Less common paths: a kinematic paddle pushing a box (kinematic bodies join islands), a box that starts asleep
    and is woken by a falling one, a bouncy circle (restitution), gravity scale, a never-sleeping body, and an
    inactive body overlapping the ground (no contacts, not simulated)."""
    t = Twin(contact_capacity=24)
    g = t.add_body(STATIC)
    t.add_box(g, (12, 0.5), friction=0.6)
    k = t.add_body(KINEMATIC, (-4.0, 0.9), 0.0, (1.5, 0.0), 0.3)
    t.add_box(k, (0.2, 0.3))
    pushed = t.add_body(DYNAMIC, (-3.2, 0.81), mass=1, inertia=0.05)
    t.add_box(pushed, (0.3, 0.3), friction=0.4)
    sleeper = t.add_body(DYNAMIC, (2.0, 0.995), mass=1, inertia=0.1, awake=False)
    t.add_box(sleeper, (0.5, 0.49))
    faller = t.add_body(DYNAMIC, (2.1, 2.6), 0.2, mass=0.5, inertia=0.03)
    t.add_box(faller, (0.25, 0.25))
    ball = t.add_body(DYNAMIC, (5.0, 2.5), 0.0, (0.3, 0.0), 0.0, mass=0.4, inertia=0.004, gravity_scale=0.5)
    t.add_circle(ball, 0.2, restitution=0.6)
    awake_box = t.add_body(DYNAMIC, (-7.0, 1.2), 0.1, mass=1, inertia=0.1, allow_sleep=False)
    t.add_box(awake_box, (0.3, 0.3))
    ghost = t.add_body(DYNAMIC, (8.0, 0.6), 0.0, mass=1, inertia=0.1, active=False)
    t.add_box(ghost, (0.3, 0.3))
    return t


PLAYER_BODIES = ("Player", "Player/Hub", "Player/Hub/Slider", "Player/Hub/Slider/Handle",
                 "Player/Hub/Slider/Handle/PoleMiddle", "Player/Hub/Slider/Handle/PoleMiddle/Tip")


def player_rig():
    """The climber's 6-body chain (pot, hub, slider, handle, pole, tip) with the dump's masses, inertias, centres,
    damping and joints (dumps/engine-dump-start.jsonl), resting on a static polygon ground. Shapes are stand-ins: a box
    over the bounding box of the pot bottom's and of the tip's render vertices. Continuous (TOI) is off.
    Layers as in the game (Player 8, Terrain 10, the dump's layer matrix)."""
    if not ref_scene.DEFAULT_DUMP.exists():
        return None
    rows = [json.loads(l) for l in ref_scene.DEFAULT_DUMP.read_text().splitlines() if l.strip()]
    srow = next(r for r in rows if r["kind"] == "settings")
    body = {r["path"]: r for r in rows if r["kind"] == "body"}
    joints = [r for r in rows if r["kind"] == "joint" and r["enabled"] and r["path"] in PLAYER_BODIES]
    t = Twin(layer_masks=list(srow["layerCollisionMasks"]))
    d = b2.deg2rad
    cols = {r["path"]: r for r in rows if r["kind"] == "collider" and r["active"] and r.get("verts")}

    def local_box(cpath, bpath):
        """Half extents and centre of the collider's render vertices' bounding box in its body's frame."""
        r = body[bpath]
        v = ref_scene._world_to_local(np.asarray(cols[cpath]["verts"], np.float64).reshape(-1, 2), r["position"],
                                      d(r["rotation"])).astype(np.float64)
        lo, hi = v.min(0), v.max(0)
        return tuple((hi - lo) / 2), tuple((hi + lo) / 2)

    lowest = min(np.asarray(cols[c]["verts"]).reshape(-1, 2)[:, 1].min()
                 for c in ("Player/PotCollider", "Player/Hub/Slider/Handle/PoleMiddle/Tip"))
    gy = float(lowest) - 0.004
    ground = t.add_body(STATIC)
    t.add_polygon(ground, [(-50, gy - 1), (-38, gy - 1), (-38, gy), (-50, gy)], friction=10.0, layer=10)
    ids = {}
    for path in PLAYER_BODIES:
        r = body[path]
        ids[path] = t.add_body(DYNAMIC, r["position"], d(r["rotation"]), r["velocity"], d(r["angularVelocity"]),
                               mass=r["mass"], center=r["centerOfMass"], inertia=r["inertia"],
                               linear_damping=r["drag"], angular_damping=r["angularDrag"],
                               gravity_scale=r["gravityScale"], fixed_rotation=r["constraints"] == "FreezeRotation",
                               allow_sleep=r["sleepMode"] != "NeverSleep")
    he, c = local_box("Player/PotCollider", "Player")
    t.add_box(ids["Player"], he, c, friction=0.3, layer=8)  # PotBottomFriction
    he, c = local_box("Player/Hub/Slider/Handle/PoleMiddle/Tip", "Player/Hub/Slider/Handle/PoleMiddle/Tip")
    t.add_box(ids["Player/Hub/Slider/Handle/PoleMiddle/Tip"], he, c, friction=0.7, layer=8)  # SlidingFriction
    for r in joints:
        a, bb = ids[r["path"]], ids[r["connected"]]
        common = dict(anchor_a=r["anchor"], anchor_b=r["connectedAnchor"], reference_angle=d(r["referenceAngle"]),
                      collide=r["enableCollision"])
        if r["type"] == "HingeJoint2D":
            t.add_revolute(a, bb, **common, limits=(d(r["limits"][0]), d(r["limits"][1])) if r["useLimits"] else None,
                           motor_speed=d(r["motorSpeed"]), max_motor_torque=r["maxMotorTorque"],
                           enable_motor=r["useMotor"])
        elif r["type"] == "SliderJoint2D":
            ang = d(r["angle"])
            t.add_prismatic(a, bb, **common, axis=(np.cos(ang), np.sin(ang)),
                            limits=tuple(r["limits"]) if r["useLimits"] else None, motor_speed=r["motorSpeed"],
                            max_motor_force=r["maxMotorForce"], enable_motor=r["useMotor"])
        else:
            t.add_weld(a, bb, **common, frequency=r["frequency"], damping=r["dampingRatio"])
    return t


SCENES = dict(free_fall=free_fall, box_on_ground=box_on_ground, box_stack=box_stack,
              polygons_at_angles=polygons_at_angles, polygon_circle=polygon_circle, revolute=revolute,
              prismatic=prismatic, weld=weld, pile=pile, misc=misc, player_rig=player_rig)


# ---- adversarial scenes (review): paths the scenes above do not reach

def late_wake():
    """b2ContactManager::Collide wakes bodies while it walks the contact list: a box (B) asleep on a short ledge,
    overhanging its right edge, is touched by a floating box (A, gravity scale 0, constant slide). The A-B contact
    is newer, so Collide reaches it first, wakes B, and then also re-evaluates the older B-ledge contact in the same
    pass. B drifted in the step it fell asleep and its manifold point at the ledge edge is a clip point (it moves
    with B), so the stale and the fresh manifold differ. The A-B pair exists before B sleeps (fat AABBs overlap from
    the start), so the only wake is the touch."""
    t = Twin()
    g = t.add_body(STATIC)
    t.add_box(g, (0.5, 0.5), friction=0.0)
    # B drifts at 5 mm/s on the frictionless ledge: below the sleep tolerance, but its pose changes every step
    b = t.add_body(DYNAMIC, (0.4, 0.815), 0.0, (0.005, 0.0), 0.0, mass=1, inertia=F32(1) / F32(6))
    t.add_box(b, (0.3, 0.3), friction=0.0)
    a = t.add_body(DYNAMIC, (1.02, 0.9), 0.0, (-0.12, 0.0), 0.0, mass=0.5, inertia=0.02, gravity_scale=0.0)
    t.add_box(a, (0.2, 0.15), friction=0.3)
    return t


def thin_plates():
    """Thin rotated polygons: 50:1 plates and a needle triangle at small and steep angles, a plate landing almost
    parallel on an equal plate (separations on both polygons within k_tol, so the reference face can flip between
    A and B from step to step), and a thin plate on a slightly rotated static sliver."""
    t = Twin(contact_capacity=32)
    g = t.add_body(STATIC)
    t.add_box(g, (12, 0.5), friction=0.6)
    t.add_box(g, (1.5, 0.02), (6.0, 0.9), 0.004, friction=0.5)  # static sliver, slightly tilted
    base = t.add_body(DYNAMIC, (0.0, 0.54), 0.0, mass=1, inertia=0.3)
    t.add_box(base, (1.0, 0.02), friction=0.5)
    top = t.add_body(DYNAMIC, (0.01, 0.62), 0.0008, (0, -0.2), 0.0, mass=1, inertia=0.3)
    t.add_box(top, (1.0, 0.02), friction=0.5)  # equal plate, near-parallel faces
    for i, ang in enumerate((0.02, -0.15, 1.2, np.pi / 2 - 0.01)):
        p = t.add_body(DYNAMIC, (-7 + 2.5 * i, 1.4 + 0.2 * i), ang, (0.1 * i, -0.5), 0.3 * (i - 1), mass=0.4,
                       inertia=0.04)
        t.add_box(p, (0.6, 0.012), friction=0.4)
    n = t.add_body(DYNAMIC, (6.0, 1.3), 0.05, mass=0.2, inertia=0.01)
    t.add_polygon(n, [(-0.7, -0.03), (0.7, 0.0), (-0.7, 0.03)], friction=0.4)  # needle
    s = t.add_body(DYNAMIC, (6.4, 1.6), -0.01, mass=0.3, inertia=0.02)
    t.add_box(s, (0.8, 0.015), friction=0.5)
    return t


def limits_mid_run():
    """Joint limits that engage and release mid-run while contacts act on the same bodies: a hinged arm (limit
    +-0.3) on a cart that drops onto the ground, a motor that reverses into the lower limit, and a prismatic whose
    body falls into its lower limit and bounces off a box."""
    t = Twin(contact_capacity=24)
    g = t.add_body(STATIC)
    t.add_box(g, (12, 0.5), friction=0.8)
    cart = t.add_body(DYNAMIC, (0.0, 1.2), 0.05, (0.5, 0), 0.0, mass=3, inertia=0.5)
    t.add_box(cart, (0.6, 0.15), friction=0.7)
    arm = t.add_body(DYNAMIC, (0.6, 1.6), 0.0, (0, 0), 3.0, mass=0.5, inertia=0.05)
    t.add_box(arm, (0.05, 0.4), friction=0.5)
    t.add_revolute(cart, arm, (0.5, 0.0), (0.0, -0.4), 0.0, limits=(-0.3, 0.3))
    arm2 = t.add_body(DYNAMIC, (-0.6, 1.6), 0.0, mass=0.5, inertia=0.05)
    t.add_box(arm2, (0.05, 0.4))
    t.add_revolute(cart, arm2, (-0.5, 0.0), (0.0, -0.4), 0.0, limits=(-0.5, 0.2), motor_speed=-4.0,
                   max_motor_torque=2.0, enable_motor=True)
    post = t.add_body(STATIC, (4.0, 2.5))
    t.add_box(post, (0.1, 0.1))
    s = t.add_body(DYNAMIC, (4.0, 2.2), 0.0, (0, 0.5), 0.0, mass=1, inertia=0.1)
    t.add_box(s, (0.15, 0.15), friction=0.5)
    t.add_prismatic(post, s, (4.0, 2.5), (0, 0), (0.1, -1.0), 0.0, limits=(0.1, 1.2))
    box = t.add_body(DYNAMIC, (4.2, 0.81), 0.0, mass=1, inertia=0.1)
    t.add_box(box, (0.3, 0.3), friction=0.5)
    return t


SCENES.update(late_wake=late_wake, thin_plates=thin_plates, limits_mid_run=limits_mid_run)
SCENE_STEPS = dict(late_wake=240)  # steps a scene needs before its event (default 120)
