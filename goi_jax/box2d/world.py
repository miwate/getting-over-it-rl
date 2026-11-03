"""World layout: a static Model (geometry, joint frames, candidate pairs) and a per-world State, both pytrees of
fixed-shape arrays, plus WorldDef, a host-side builder whose methods mirror goi_jax/reference/box2d.World.

Capacities are fixed when the world is built: B bodies, F fixtures, J joints, P candidate pairs, C contact slots.
`jax.vmap(step, in_axes=(None, 0))` batches States over one shared Model.

Indices follow creation order, which Box2D uses for its orderings (see step.py and the package docstring):
- body i is the i-th created; Box2D's body list is newest first, so islands are seeded from the highest index;
- fixture i is the i-th created and gets broad-phase proxy id order i (leaf ids grow with creation order while no
  proxy is destroyed);
- joint i is the i-th created; each body's joint list is newest first.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import shapes
from .math2d import host_sin_cos
from .settings import MAX_POLYGON_VERTICES, Settings

F32, I32 = np.float32, np.int32
STATIC, KINEMATIC, DYNAMIC = 0, 1, 2  # b2BodyType
REVOLUTE, PRISMATIC, WELD = 1, 2, 8  # b2JointType
INACTIVE_LIMIT, AT_LOWER, AT_UPPER, EQUAL_LIMITS = 0, 1, 2, 3  # b2LimitState
CIRCLES, FACE_A, FACE_B = 0, 1, 2  # b2Manifold::Type


class Model(NamedTuple):
    """Scene constants shared by every world."""
    fixture_body: jnp.ndarray  # i32[F]
    shape_type: jnp.ndarray  # i32[F] shapes.CIRCLE / POLYGON
    radius: jnp.ndarray  # f32[F] b2Shape::m_radius (polygon skin)
    count: jnp.ndarray  # i32[F] vertex count (circle 1)
    verts: jnp.ndarray  # f32[F, 8, 2] local vertices (circle: centre in [0])
    normals: jnp.ndarray  # f32[F, 8, 2]
    joint_type: jnp.ndarray  # i32[J]
    joint_a: jnp.ndarray  # i32[J] body A
    joint_b: jnp.ndarray  # i32[J] body B
    local_anchor_a: jnp.ndarray  # f32[J, 2]
    local_anchor_b: jnp.ndarray  # f32[J, 2]
    local_axis_a: jnp.ndarray  # f32[J, 2] prismatic m_localXAxisA (normalized)
    local_yaxis_a: jnp.ndarray  # f32[J, 2] prismatic m_localYAxisA = b2Cross(1, x axis)
    reference_angle: jnp.ndarray  # f32[J]
    pair_a: jnp.ndarray  # i32[P] fixture A of each candidate contact, after b2Contact::Create's type swap
    pair_b: jnp.ndarray  # i32[P] fixture B; candidates are sorted in b2BroadPhase::UpdatePairs order
    # bool[B] bullet or Continuous when built. toi.py skips SolveTOI at trace time (Unity rule) when the model is
    # a concrete constant and no body has either flag; flags set on a State later need a model built with them.
    toi_bodies: jnp.ndarray = None
    # Compact fixture layout (compact_model; None = full layout): a static body's fixtures never move, so their fat
    # AABB, friction and restitution live here, shared by every world, and State.fixtures holds only the rows of
    # fixtures on non-static bodies (fixture_rows). Read per-fixture values through fixture_fat / fixture_friction /
    # fixture_restitution, which accept either layout.
    fixture_slot: jnp.ndarray = None  # i32[F] row of the fixture in State.fixtures, -1 for a static body's
    slot_fixture: jnp.ndarray = None  # i32[Fd] fixture of each State.fixtures row (ascending)
    static_fat: jnp.ndarray = None  # f32[F, 4] fat AABB of each static-body fixture (rows of the others: unused)
    static_friction: jnp.ndarray = None  # f32[F]
    static_restitution: jnp.ndarray = None  # f32[F]
    # Culled candidate scan for the GPU FindNewContacts kernel (gpu/pairs.py; compact layout only, scan_tables):
    # the candidates regrouped into SCAN_SEGMENTS contiguous ranges of the candidate order, inside each range by the
    # non-static fixture they pair with (spatially sorted), in chunks of SCAN_CHUNK with the union of the chunk's
    # static fat AABBs. A world tests a chunk's candidates only where that fixture's fat AABB overlaps the union.
    scan_cand: jnp.ndarray = None  # i32[Q] candidate index in scan order (-1: padding), Q = segments x chunks x CHUNK
    scan_row: jnp.ndarray = None  # i32[Q / CHUNK] State.fixtures row of the chunk's non-static fixture (-1: test all,
    #                                   both sides non-static; -2: padding)
    scan_box: jnp.ndarray = None  # f32[Q / CHUNK, 4] union of the chunk's static fat AABBs


class Bodies(NamedTuple):
    type: jnp.ndarray  # i32[B]
    active: jnp.ndarray  # bool[B] (Rigidbody2D.simulated)
    awake: jnp.ndarray  # bool[B]
    allow_sleep: jnp.ndarray  # bool[B] e_autoSleepFlag
    bullet: jnp.ndarray  # bool[B] b2Body e_bulletFlag (TOI with every body, toi.py)
    continuous: jnp.ndarray  # bool[B] Unity Continuous (TOI against static/kinematic, Settings.unity_toi)
    sleep_time: jnp.ndarray  # f32[B]
    xf_p: jnp.ndarray  # f32[B, 2] body origin (m_xf.p)
    xf_q: jnp.ndarray  # f32[B, 2] (cos, sin) of the angle (m_xf.q)
    local_center: jnp.ndarray  # f32[B, 2] m_sweep.localCenter
    c0: jnp.ndarray  # f32[B, 2] centre of mass at the start of the step
    c: jnp.ndarray  # f32[B, 2] centre of mass
    a0: jnp.ndarray  # f32[B]
    a: jnp.ndarray  # f32[B] angle (radians, not wrapped)
    v: jnp.ndarray  # f32[B, 2] linear velocity of the centre of mass
    w: jnp.ndarray  # f32[B]
    force: jnp.ndarray  # f32[B, 2]
    torque: jnp.ndarray  # f32[B]
    mass: jnp.ndarray
    inv_mass: jnp.ndarray
    inertia: jnp.ndarray  # about the centre of mass
    inv_inertia: jnp.ndarray
    linear_damping: jnp.ndarray
    angular_damping: jnp.ndarray
    gravity_scale: jnp.ndarray


class Fixtures(NamedTuple):
    friction: jnp.ndarray  # f32[F] (a contact mixes it once, at creation, as b2Contact does)
    restitution: jnp.ndarray  # f32[F]
    fat: jnp.ndarray  # f32[F, 4] broad-phase fat AABB (lower x, lower y, upper x, upper y)


class Joints(NamedTuple):
    enable_motor: jnp.ndarray  # bool[J]
    motor_speed: jnp.ndarray  # f32[J] rad/s or m/s
    max_motor: jnp.ndarray  # f32[J] maxMotorTorque / maxMotorForce
    enable_limit: jnp.ndarray  # bool[J]
    lower: jnp.ndarray  # f32[J] angle or translation
    upper: jnp.ndarray  # f32[J]
    frequency: jnp.ndarray  # f32[J] weld frequencyHz (0 rigid)
    damping: jnp.ndarray  # f32[J] weld dampingRatio
    impulse: jnp.ndarray  # f32[J, 3] accumulated m_impulse (revolute x, y, limit; prismatic perp, angle, limit;
    #                       weld x, y, angle)
    motor_impulse: jnp.ndarray  # f32[J]
    limit_state: jnp.ndarray  # i32[J] b2LimitState (revolute with Settings.revolute_v24: readout only)
    lower_impulse: jnp.ndarray  # f32[J] revolute v2.4 accumulated lower-limit impulse (>= 0; impulse z stays 0)
    upper_impulse: jnp.ndarray  # f32[J] revolute v2.4 accumulated upper-limit impulse (>= 0)


class Contacts(NamedTuple):
    """C slots, each a live b2Contact or empty (pair < 0). `key` orders them: Box2D's contact list and every body's
    contact-edge list are newest first, i.e. descending key."""
    pair: jnp.ndarray  # i32[C] candidate index, -1 empty
    fa: jnp.ndarray  # i32[C] fixture A
    fb: jnp.ndarray  # i32[C] fixture B
    key: jnp.ndarray  # i32[C] creation order (larger = newer)
    touching: jnp.ndarray  # bool[C] e_touchingFlag
    mtype: jnp.ndarray  # i32[C] b2Manifold::Type
    count: jnp.ndarray  # i32[C] manifold pointCount (0..2)
    local_normal: jnp.ndarray  # f32[C, 2]
    local_point: jnp.ndarray  # f32[C, 2]
    point_local: jnp.ndarray  # f32[C, 2, 2] b2ManifoldPoint::localPoint
    point_id: jnp.ndarray  # i32[C, 2] b2ContactID::key (indexA | indexB << 8 | typeA << 16 | typeB << 24)
    normal_impulse: jnp.ndarray  # f32[C, 2]
    tangent_impulse: jnp.ndarray  # f32[C, 2]
    friction: jnp.ndarray  # f32[C] mixed at creation
    restitution: jnp.ndarray  # f32[C]
    tangent_speed: jnp.ndarray  # f32[C]


class ContactTOI(NamedTuple):
    """b2World::SolveTOI bookkeeping of the last step (toi.py). SolveTOI resets it at its start (Box2D's
    m_stepComplete is always true without sub-stepping), so nothing in it feeds the next step; it is the post-step
    state Box2D leaves behind, kept for parity checks and for Unity's TOI-time contact reports."""
    alpha0: jnp.ndarray  # f32[B] b2Sweep::alpha0
    count: jnp.ndarray  # i32[C] b2Contact::m_toiCount
    toi: jnp.ndarray  # f32[C] m_toi (cached alpha; meaningful where flag)
    flag: jnp.ndarray  # bool[C] e_toiFlag
    enabled: jnp.ndarray  # bool[C] e_enabledFlag (SolveTOI disables a TOI contact that turns out not touching)
    solved: jnp.ndarray  # bool[C] solved in a TOI sub-step this step
    normal_impulse: jnp.ndarray  # f32[C, 2] impulses of the last TOI sub-step solve (b2Island::Report), not stored
    tangent_impulse: jnp.ndarray  # f32[C, 2]   in the manifold (Box2D keeps the manifold's own values)
    events: jnp.ndarray  # i32 TOI events solved in the step


def empty_toi(B, C):
    return ContactTOI(alpha0=jnp.zeros((B,), jnp.float32), count=jnp.zeros((C,), jnp.int32),
                      toi=jnp.ones((C,), jnp.float32), flag=jnp.zeros((C,), bool), enabled=jnp.ones((C,), bool),
                      solved=jnp.zeros((C,), bool), normal_impulse=jnp.zeros((C, 2), jnp.float32),
                      tangent_impulse=jnp.zeros((C, 2), jnp.float32), events=jnp.asarray(np.int32(0)))


class State(NamedTuple):
    bodies: Bodies
    fixtures: Fixtures
    joints: Joints
    contacts: Contacts
    gravity: jnp.ndarray  # f32[2]
    inv_dt0: jnp.ndarray  # f32 previous step's 1/dt (dtRatio = inv_dt0 * dt); 0 in a fresh world
    next_key: jnp.ndarray  # i32 key of the next created contact
    new_fixture: jnp.ndarray  # bool: run FindNewContacts at the start of the next step (b2World e_newFixture)
    overflow: jnp.ndarray  # i32 contacts dropped because all C slots were taken (must stay 0)
    toi: ContactTOI = None  # continuous-collision bookkeeping (toi.py); None in a State built without it


def _filter_ok(fa, fb, masks):
    """Reference Filter::ShouldCollide (capi.cpp): the layer matrix when given, else b2ContactFilter's groups."""
    if masks is not None:
        return bool((int(masks[fa["layer"]]) & 0xFFFFFFFF) >> fb["layer"] & 1)
    if fa["group"] == fb["group"] and fa["group"] != 0:
        return fa["group"] > 0
    return (fa["mask"] & fb["category"]) != 0 and (fa["category"] & fb["mask"]) != 0


class WorldDef:
    """Host-side builder. Methods and defaults mirror goi_jax/reference/box2d.World so a test can build the same
    scene in both; the float32 arithmetic (mass data, box rotation, axis normalization) follows the same C code."""

    def __init__(self, settings=None, layer_masks=None):
        self.settings = settings or Settings()
        self.layer_masks = layer_masks
        self.bodies, self.fixtures, self.joints = [], [], []

    # ---- bodies

    def add_body(self, type=DYNAMIC, position=(0, 0), angle=0.0, velocity=(0, 0), angular_velocity=0.0, *,
                 mass=None, center=(0, 0), inertia=0.0, exact_inertia=True, linear_damping=0.0,
                 angular_damping=0.0, gravity_scale=1.0, fixed_rotation=False, bullet=False, continuous=False,
                 awake=True, allow_sleep=True, active=True):
        a = F32(angle)
        s, c = host_sin_cos(a)
        b = dict(type=type, active=active, awake=awake, allow_sleep=allow_sleep, bullet=bullet,
                 continuous=continuous, fixed_rotation=fixed_rotation, p=np.array(position, F32), q=(c, s),
                 local_center=np.zeros(2, F32), c=np.array(position, F32), a=a, v=np.zeros(2, F32), w=F32(0),
                 mass=F32(1 if type == DYNAMIC else 0), inv_mass=F32(1 if type == DYNAMIC else 0),
                 inertia=F32(0), inv_inertia=F32(0), linear_damping=F32(linear_damping),
                 angular_damping=F32(angular_damping), gravity_scale=F32(gravity_scale), sleep_time=F32(0))
        self.bodies.append(b)
        bid = len(self.bodies) - 1
        if mass is not None:
            self.set_mass(bid, mass, center, inertia, exact_inertia)
        if type != STATIC and (velocity[0] or velocity[1] or angular_velocity):
            b["v"], b["w"] = np.array(velocity, F32), F32(angular_velocity)
            b["awake"] = True
            if not awake:  # reference add_body re-sleeps it: SetAwake(false) zeroes the velocity
                b["awake"], b["v"], b["w"] = False, np.zeros(2, F32), F32(0)
        return bid

    def set_mass(self, body, mass, center=(0, 0), inertia=0.0, exact=True):
        """Reference goi_b2_body_set_mass: b2MassData.I = I_com + m |c|^2, then b2Body::SetMassData, then (exact)
        the given I_com verbatim."""
        b = self.bodies[body]
        if b["type"] != DYNAMIC:
            return
        mass, cx, cy, icom = F32(mass), F32(center[0]), F32(center[1]), F32(inertia)
        cc = F32(F32(cx * cx) + F32(cy * cy))
        md_i = F32(icom + F32(mass * cc))
        m = mass if mass > 0 else F32(1)
        b["mass"], b["inv_mass"] = m, F32(F32(1) / m)
        b["inertia"], b["inv_inertia"] = F32(0), F32(0)
        if md_i > 0 and not b["fixed_rotation"]:
            b["inertia"] = F32(md_i - F32(m * cc))
            b["inv_inertia"] = F32(F32(1) / b["inertia"])
        if exact and b["inertia"] > 0:
            b["inertia"], b["inv_inertia"] = icom, F32(F32(1) / icom)
        old = b["c"].copy()
        c, s = b["q"]
        px, py = b["p"]
        b["local_center"] = np.array([cx, cy], F32)
        b["c"] = np.array([F32(F32(F32(c * cx) - F32(s * cy)) + px), F32(F32(F32(s * cx) + F32(c * cy)) + py)], F32)
        dx, dy = F32(b["c"][0] - old[0]), F32(b["c"][1] - old[1])
        w = b["w"]
        b["v"] = np.array([F32(b["v"][0] + F32(-w * dy)), F32(b["v"][1] + F32(w * dx))], F32)

    # ---- fixtures

    def _add_fixture(self, body, shape_type, verts, normals, radius, friction, restitution, sensor, layer,
                     category, mask, group):
        vv, nn = shapes.pad(verts, normals)
        self.fixtures.append(dict(body=body, shape_type=shape_type, verts=vv, normals=nn, count=len(verts),
                                  radius=F32(radius), friction=F32(friction), restitution=F32(restitution),
                                  sensor=bool(sensor), layer=int(layer) & 31, category=category, mask=mask,
                                  group=group))
        return len(self.fixtures) - 1

    def add_polygon(self, body, vertices, radius=None, raw=False, *, friction=0.2, restitution=0.0, density=0.0,
                    sensor=False, layer=0, category=1, mask=0xFFFF, group=0):
        v, n = shapes.polygon_raw(vertices) if raw else shapes.polygon_set(vertices, self.settings.linear_slop)
        r = self.settings.polygon_radius if radius is None else radius
        return self._add_fixture(body, shapes.POLYGON, v, n, r, friction, restitution, sensor, layer, category,
                                 mask, group)

    def add_box(self, body, half_extents, center=(0, 0), angle=0.0, radius=None, *, friction=0.2, restitution=0.0,
                density=0.0, sensor=False, layer=0, category=1, mask=0xFFFF, group=0):
        v, n = shapes.box(half_extents[0], half_extents[1], center, angle)
        r = self.settings.polygon_radius if radius is None else radius
        return self._add_fixture(body, shapes.POLYGON, v, n, r, friction, restitution, sensor, layer, category,
                                 mask, group)

    def add_circle(self, body, radius, center=(0, 0), *, friction=0.2, restitution=0.0, density=0.0, sensor=False,
                   layer=0, category=1, mask=0xFFFF, group=0):
        v = np.array([center], F32)
        return self._add_fixture(body, shapes.CIRCLE, v, np.zeros((1, 2), F32), radius, friction, restitution,
                                 sensor, layer, category, mask, group)

    # ---- joints (anchors local to each body's origin, angles in radians)

    def _add_joint(self, **j):
        self.joints.append(j)
        return len(self.joints) - 1

    def add_revolute(self, body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0), reference_angle=0.0, *,
                     limits=None, motor_speed=0.0, max_motor_torque=0.0, enable_motor=False, collide=False):
        lo, hi = limits if limits is not None else (0.0, 0.0)
        return self._add_joint(type=REVOLUTE, a=body_a, b=body_b, la=anchor_a, lb=anchor_b, ref=reference_angle,
                               axis=(1, 0), enable_limit=limits is not None, lower=lo, upper=hi,
                               motor_speed=motor_speed, max_motor=max_motor_torque, enable_motor=enable_motor,
                               frequency=0.0, damping=0.0, collide=collide)

    def add_prismatic(self, body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0), axis=(1, 0), reference_angle=0.0, *,
                      limits=None, motor_speed=0.0, max_motor_force=0.0, enable_motor=False, collide=False):
        lo, hi = limits if limits is not None else (0.0, 0.0)
        return self._add_joint(type=PRISMATIC, a=body_a, b=body_b, la=anchor_a, lb=anchor_b, ref=reference_angle,
                               axis=axis, enable_limit=limits is not None, lower=lo, upper=hi,
                               motor_speed=motor_speed, max_motor=max_motor_force, enable_motor=enable_motor,
                               frequency=0.0, damping=0.0, collide=collide)

    def add_weld(self, body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0), reference_angle=0.0, *, frequency=0.0,
                 damping=0.0, collide=False):
        return self._add_joint(type=WELD, a=body_a, b=body_b, la=anchor_a, lb=anchor_b, ref=reference_angle,
                               axis=(1, 0), enable_limit=False, lower=0.0, upper=0.0, motor_speed=0.0,
                               max_motor=0.0, enable_motor=False, frequency=frequency, damping=damping,
                               collide=collide)

    # ---- build

    def candidate_pairs(self):
        """Fixture pairs that can ever form a contact, in b2BroadPhase::UpdatePairs order (ascending (lower, higher)
        proxy id), each as (fixture A, fixture B) after b2Contact::Create puts the polygon first in a
        polygon-circle pair. Excluded for good: same body, both bodies static, a joint between the bodies with
        collideConnected false, the contact filter, and sensors (sensor contacts only feed callbacks; they never
        reach the solver or wake bodies)."""
        no_collide = {frozenset((j["a"], j["b"])) for j in self.joints if not j["collide"]}
        out = []
        for i, fi in enumerate(self.fixtures):
            for k in range(i + 1, len(self.fixtures)):
                fk = self.fixtures[k]
                if fi["body"] == fk["body"] or fi["sensor"] or fk["sensor"]:
                    continue
                if self.bodies[fi["body"]]["type"] == STATIC and self.bodies[fk["body"]]["type"] == STATIC:
                    continue
                if frozenset((fi["body"], fk["body"])) in no_collide or not _filter_ok(fi, fk, self.layer_masks):
                    continue
                if fi["shape_type"] == shapes.CIRCLE and fk["shape_type"] == shapes.POLYGON:
                    out.append((k, i))
                else:
                    out.append((i, k))
        return out

    def build(self, contact_capacity=16):
        """(Model, State) as numpy-backed jnp arrays. Fat AABBs are those b2BroadPhase::CreateProxy gives (shape AABB
        at the body transform, extended); contacts start empty with new_fixture set, as after CreateFixture."""
        B, F, J, C = len(self.bodies), len(self.fixtures), len(self.joints), contact_capacity
        bs, fs, js = self.bodies, self.fixtures, self.joints
        pairs = self.candidate_pairs()
        arr = lambda key, dt=F32: np.array([x[key] for x in bs], dt)
        q = np.array([b["q"] for b in bs], F32).reshape(B, 2)
        bodies = Bodies(
            type=arr("type", I32), active=arr("active", bool), awake=arr("awake", bool),
            allow_sleep=arr("allow_sleep", bool), bullet=arr("bullet", bool), continuous=arr("continuous", bool),
            sleep_time=arr("sleep_time"), xf_p=np.array([b["p"] for b in bs], F32).reshape(B, 2), xf_q=q,
            local_center=np.array([b["local_center"] for b in bs], F32).reshape(B, 2),
            c0=np.array([b["c"] for b in bs], F32).reshape(B, 2), c=np.array([b["c"] for b in bs], F32).reshape(B, 2),
            a0=arr("a"), a=arr("a"), v=np.array([b["v"] for b in bs], F32).reshape(B, 2), w=arr("w"),
            force=np.zeros((B, 2), F32), torque=np.zeros(B, F32), mass=arr("mass"), inv_mass=arr("inv_mass"),
            inertia=arr("inertia"), inv_inertia=arr("inv_inertia"), linear_damping=arr("linear_damping"),
            angular_damping=arr("angular_damping"), gravity_scale=arr("gravity_scale"))
        farr = lambda key, dt=F32, shape=(): np.array([f[key] for f in fs], dt).reshape((F,) + shape)
        jarr = lambda key, dt=F32, shape=(): np.array([j[key] for j in js], dt).reshape((J,) + shape)
        axes = []
        for j in js:  # prismatic: m_localXAxisA = Normalize(axis); m_localYAxisA = b2Cross(1.0f, x axis)
            # Normalized twice, as the oracle does (goi_b2_joint_add_prismatic normalizes localAxisA, then the
            # b2PrismaticJoint constructor again); the second pass changes the last bit of ~40% of axes.
            x, y = shapes._normalize_host(*shapes._normalize_host(F32(j["axis"][0]), F32(j["axis"][1])))
            axes.append(((x, y), (F32(-F32(1) * y), F32(F32(1) * x))))
        model = Model(
            fixture_body=farr("body", I32), shape_type=farr("shape_type", I32), radius=farr("radius"),
            count=farr("count", I32), verts=farr("verts", F32, (MAX_POLYGON_VERTICES, 2)),
            normals=farr("normals", F32, (MAX_POLYGON_VERTICES, 2)),
            joint_type=jarr("type", I32), joint_a=jarr("a", I32), joint_b=jarr("b", I32),
            local_anchor_a=jarr("la", F32, (2,)), local_anchor_b=jarr("lb", F32, (2,)),
            local_axis_a=np.array([a[0] for a in axes], F32).reshape(J, 2),
            local_yaxis_a=np.array([a[1] for a in axes], F32).reshape(J, 2),
            reference_angle=jarr("ref"),
            pair_a=np.array([p[0] for p in pairs], I32).reshape(-1), pair_b=np.array([p[1] for p in pairs], I32).reshape(-1),
            toi_bodies=bodies.bullet | bodies.continuous)
        model = Model(*(None if x is None else jnp.asarray(x) for x in model))
        fb = model.fixture_body
        lx, ly, ux, uy = shapes.fixture_aabbs(model, jnp.asarray(bodies.xf_p[:, 0])[fb], jnp.asarray(bodies.xf_p[:, 1])[fb],
                                              jnp.asarray(q[:, 0])[fb], jnp.asarray(q[:, 1])[fb])
        fat = jnp.stack(shapes.fatten(lx, ly, ux, uy), axis=-1)
        fixtures = Fixtures(friction=farr("friction"), restitution=farr("restitution"), fat=fat)
        joints = Joints(
            enable_motor=jarr("enable_motor", bool), motor_speed=jarr("motor_speed"), max_motor=jarr("max_motor"),
            enable_limit=jarr("enable_limit", bool), lower=jarr("lower"), upper=jarr("upper"),
            frequency=jarr("frequency"), damping=jarr("damping"), impulse=np.zeros((J, 3), F32),
            motor_impulse=np.zeros(J, F32), limit_state=np.zeros(J, I32), lower_impulse=np.zeros(J, F32),
            upper_impulse=np.zeros(J, F32))
        contacts = empty_contacts(C)
        state = State(bodies=Bodies(*(jnp.asarray(x) for x in bodies)), fixtures=Fixtures(*(jnp.asarray(x) for x in fixtures)),
                      joints=Joints(*(jnp.asarray(x) for x in joints)), contacts=contacts,
                      gravity=jnp.asarray(np.array(self.settings.gravity, F32)), inv_dt0=jnp.asarray(F32(0)),
                      next_key=jnp.asarray(I32(0)), new_fixture=jnp.asarray(len(fs) > 0),
                      overflow=jnp.asarray(I32(0)), toi=empty_toi(B, C))
        return model, state


def empty_contacts(C):
    z = lambda *s: jnp.zeros((C,) + s, jnp.float32)
    zi = lambda *s: jnp.zeros((C,) + s, jnp.int32)
    return Contacts(pair=jnp.full((C,), -1, jnp.int32), fa=zi(), fb=zi(), key=zi(),
                    touching=jnp.zeros((C,), bool), mtype=zi(), count=zi(), local_normal=z(2), local_point=z(2),
                    point_local=z(2, 2), point_id=zi(2), normal_impulse=z(2), tangent_impulse=z(2), friction=z(),
                    restitution=z(), tangent_speed=z())


# ---------------------------------------------------------------- compact fixture layout

def is_compact(model, fixtures):
    """True when `fixtures` (a State.fixtures, any leading batch axes) holds only the non-static rows of a compact
    Model. A full-layout State still works with a compact Model (every reader checks the shape)."""
    return model.fixture_slot is not None and fixtures.fat.shape[-2] != model.fixture_body.shape[-1]


def fixture_rows(model, fixtures):
    """The fixture of each row of `fixtures`: model.slot_fixture (compact) or arange(F) (full)."""
    if is_compact(model, fixtures):
        return model.slot_fixture
    return jnp.arange(model.fixture_body.shape[-1], dtype=jnp.int32)


def _per_fixture(model, fixtures, idx, mine, shared):
    if not is_compact(model, fixtures):
        return mine[idx]
    slot = model.fixture_slot[idx]
    return jnp.where(slot >= 0, mine[jnp.maximum(slot, 0)], shared[idx])


def fixture_fat(model, fixtures, idx):
    """Fat AABB (lower x, lower y, upper x, upper y) of fixtures `idx`, each shaped like idx, in either layout."""
    if not is_compact(model, fixtures):
        return tuple(fixtures.fat[idx, k] for k in range(4))
    slot = model.fixture_slot[idx]
    dyn, s = slot >= 0, jnp.maximum(slot, 0)
    return tuple(jnp.where(dyn, fixtures.fat[s, k], model.static_fat[idx, k]) for k in range(4))


def fixture_friction(model, fixtures, idx):
    return _per_fixture(model, fixtures, idx, fixtures.friction, model.static_friction)


def fixture_restitution(model, fixtures, idx):
    return _per_fixture(model, fixtures, idx, fixtures.restitution, model.static_restitution)


def compact_model(model, state):
    """Host: the Model with the compact fixture tables, from a full-layout State (any leading batch axes, numpy or
    concrete jax arrays) whose static-body fixture rows are the same in every world (checked; they never change in a
    step: synchronize_fixtures only moves fixtures of island bodies, which are never static). Unchanged when every
    fixture is on a non-static body or the model is already compact.

    Bit-exactness: the values are the same floats, only stored once; candidate pairs, contact slots and their order
    are untouched (they index fixtures, not rows)."""
    if model.fixture_slot is not None:
        return model
    btype = np.asarray(state.bodies.type)
    btype = btype.reshape(-1, btype.shape[-1])
    if not (btype == btype[0]).all():
        raise ValueError("compact_model: body types differ between worlds")
    fb = np.asarray(model.fixture_body)
    static = btype[0][fb] == STATIC
    if not static.any():
        return model
    fx = state.fixtures
    F = fb.shape[0]
    fat = np.asarray(fx.fat).reshape(-1, F, 4)
    fr = np.asarray(fx.friction).reshape(-1, F)
    rs = np.asarray(fx.restitution).reshape(-1, F)
    for name, a in (("fat", fat), ("friction", fr), ("restitution", rs)):
        if not (a[:, static] == a[:1, static]).all():
            raise ValueError(f"compact_model: static fixtures' {name} differs between worlds")
    rows = np.nonzero(~static)[0].astype(I32)
    slot = np.full(F, -1, I32)
    slot[rows] = np.arange(len(rows), dtype=I32)
    keep = lambda a: np.where(static.reshape((F, *((1,) * (a.ndim - 1)))), a, 0).astype(a.dtype)  # noqa: E731
    sfat = keep(fat[0])
    sc, sr, sbx = scan_tables(np.asarray(model.pair_a), np.asarray(model.pair_b), slot, sfat)
    return model._replace(fixture_slot=jnp.asarray(slot), slot_fixture=jnp.asarray(rows),
                          static_fat=jnp.asarray(sfat), static_friction=jnp.asarray(keep(fr[0])),
                          static_restitution=jnp.asarray(keep(rs[0])), scan_cand=jnp.asarray(sc),
                          scan_row=jnp.asarray(sr), scan_box=jnp.asarray(sbx))


SCAN_SEGMENTS, SCAN_CHUNK = 16, 16


def _morton(x, y):
    def q(v):
        span = float(v.max() - v.min()) if v.size else 0.0
        return ((v - v.min()) / (span or 1.0) * 65535).astype(np.uint64)

    def spread(v):
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
        return (v | (v << np.uint64(1))) & np.uint64(0x55555555)
    return spread(q(x)) | (spread(q(y)) << np.uint64(1))


def scan_tables(pair_a, pair_b, slot, static_fat, segments=SCAN_SEGMENTS, chunk=SCAN_CHUNK):
    """Host: Model.scan_cand / scan_row / scan_box (see Model) for the compact layout. Segment s holds the candidates
    with index in [s P / S, (s + 1) P / S); every segment gets the same number of chunks (padding: -1)."""
    P = pair_a.shape[0]
    ra, rb = slot[pair_a], slot[pair_b]
    both = (ra >= 0) & (rb >= 0)
    mrow = np.where(both, -1, np.where(ra >= 0, ra, rb)).astype(I32)
    sb = np.where((ra < 0)[:, None], static_fat[pair_a], static_fat[pair_b]).astype(F32)
    key = _morton((sb[:, 0] + sb[:, 2]) * 0.5, (sb[:, 1] + sb[:, 3]) * 0.5) if P else np.zeros(0, np.uint64)
    seg = np.arange(P) * segments // max(P, 1)
    per_seg = []
    for s in range(segments):
        chunks = []
        for r in np.unique(mrow[seg == s]):
            idx = np.nonzero((seg == s) & (mrow == r))[0]
            if r >= 0:
                idx = idx[np.argsort(key[idx], kind="stable")]
            for c in range(0, len(idx), chunk):
                part = idx[c:c + chunk]
                box = (np.array([-np.inf, -np.inf, np.inf, np.inf], F32) if r < 0 else
                       np.concatenate([sb[part, :2].min(0), sb[part, 2:].max(0)]).astype(F32))
                chunks.append((np.pad(part, (0, chunk - len(part)), constant_values=-1), int(r), box))
        per_seg.append(chunks)
    n = max(1, max(len(c) for c in per_seg))
    cand = np.full((segments, n, chunk), -1, I32)
    row = np.full((segments, n), -2, I32)  # padding chunks: -2, skipped
    box = np.tile(np.array([np.inf, np.inf, -np.inf, -np.inf], F32), (segments, n, 1))
    for s, chunks in enumerate(per_seg):
        for j, (part, r, b) in enumerate(chunks):
            cand[s, j], row[s, j], box[s, j] = part, r, b
    return cand.reshape(-1), row.reshape(-1), box.reshape(-1, 4)


def compact_state(model, state, check=True):
    """A full-layout State (any leading batch axes) in the compact layout of `model` (compact_model): only the
    non-static fixtures' rows. Idempotent; a full Model leaves the state as it is. check: verify (host, concrete
    arrays only) that the dropped static rows equal the Model's."""
    fx = state.fixtures
    if model.fixture_slot is None or is_compact(model, fx):
        return state
    rows = np.asarray(model.slot_fixture)
    if check:
        try:
            got = [np.asarray(x) for x in (fx.fat, fx.friction, fx.restitution)]
        except jax.errors.TracerArrayConversionError:  # under a trace: nothing to check
            got = None
        if got is not None:
            F = model.fixture_body.shape[-1]
            static = np.asarray(model.fixture_slot) < 0
            for name, a, shared in zip(("fat", "friction", "restitution"), got,
                                       (model.static_fat, model.static_friction, model.static_restitution)):
                a = a.reshape((-1, F) + a.shape[a.ndim - (2 if name == "fat" else 1) + 1:])
                if not (a[:, static] == np.asarray(shared)[static][None]).all():
                    raise ValueError(f"compact_state: static fixtures' {name} differs from the model's")
    take = lambda a, ax: (np.take(a, rows, axis=ax) if isinstance(a, np.ndarray)  # noqa: E731
                          else jnp.take(jnp.asarray(a), jnp.asarray(rows), axis=ax))
    fx = Fixtures(friction=take(fx.friction, -1), restitution=take(fx.restitution, -1), fat=take(fx.fat, -2))
    return state._replace(fixtures=fx)


def expand_state(model, state):
    """The full-layout State of a compact one (static rows from the Model); any leading batch axes."""
    fx = state.fixtures
    if not is_compact(model, fx):
        return state
    rows = jnp.asarray(model.slot_fixture)
    lead = fx.fat.shape[:-2]
    put = lambda shared, mine, ax: jnp.broadcast_to(jnp.asarray(shared), lead + shared.shape).at[  # noqa: E731
        (Ellipsis, rows, *((slice(None),) * (-1 - ax)))].set(mine)
    return state._replace(fixtures=Fixtures(friction=put(model.static_friction, fx.friction, -1),
                                            restitution=put(model.static_restitution, fx.restitution, -1),
                                            fat=put(model.static_fat, fx.fat, -2)))

