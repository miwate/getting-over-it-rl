"""ctypes wrapper over patched Box2D 2.3.1 (build/box2d/libgoi_box2d.so): the CPU oracle for the JAX engine.

It separates "JAX vs Box2D" error from "Box2D vs Unity" error: the JAX step is checked against this first, and this
against the game. Build with scripts/build_box2d.sh (fetches and patches the sources); `load()` does it on demand.

Units are Box2D's: radians, body-origin poses, mass data about the centre of mass. Unity-side conversions (degrees,
layer matrix, Rigidbody2D fields) live in goi_jax/reference/scene.py. All values cross the boundary as float32.

    from goi_jax.reference import box2d as b2
    w = b2.World(b2.Settings.game())
    body = w.add_body(b2.DYNAMIC, (0, 1), mass=1, inertia=0.1)
    w.add_box(body, (0.5, 0.5))
    w.step(b2.GAME_DT)
"""
import ctypes as C
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = REPO_ROOT / "build" / "box2d" / "libgoi_box2d.so"

STATIC, KINEMATIC, DYNAMIC = 0, 1, 2
CIRCLE, POLYGON = 0, 2  # b2Shape::Type
REVOLUTE, PRISMATIC, WELD = 1, 2, 8  # b2JointType
GAME_DT = np.float32(0.008333334)  # Time.fixedDeltaTime as dumped (float32 nearest 1/120)

F32 = np.float32
_f = C.POINTER(C.c_float)
_i = C.POINTER(C.c_int32)


def deg2rad(deg):
    """Unity's degree conversion (value * Mathf.Deg2Rad in float32). Which rounding Unity's native side uses for the
    Physics2D settings is not known; stock Box2D's macros compute deg / 180 * pi instead (see Settings.stock)."""
    return F32(F32(deg) * F32(0.0174532924))


class Contact(C.Structure):
    """Mirror of GoiContact in capi.cpp (one live contact, world contact-list order)."""
    _fields_ = [(n, C.c_int32) for n in ("fixtureA", "fixtureB", "childA", "childB", "bodyA", "bodyB", "flags",
                                          "manifoldType", "pointCount", "toiCount")] + \
               [(n, C.c_float) for n in ("friction", "restitution", "tangentSpeed", "toi")] + \
               [("localNormal", C.c_float * 2), ("localPoint", C.c_float * 2),
                ("pointLocal", (C.c_float * 2) * 2), ("normalImpulse", C.c_float * 2), ("tangentImpulse", C.c_float * 2),
                ("key", C.c_uint32 * 2), ("worldNormal", C.c_float * 2), ("worldPoints", (C.c_float * 2) * 2),
                ("separations", C.c_float * 2)]

    def to_dict(self):
        n = self.pointCount
        return {
            "fixtures": (self.fixtureA, self.fixtureB), "children": (self.childA, self.childB),
            "bodies": (self.bodyA, self.bodyB), "touching": bool(self.flags & 2), "enabled": bool(self.flags & 4),
            "flags": self.flags, "toi_count": self.toiCount, "toi": self.toi,
            "friction": self.friction, "restitution": self.restitution, "tangent_speed": self.tangentSpeed,
            "manifold_type": self.manifoldType,  # 0 circles, 1 faceA, 2 faceB
            "local_normal": np.array(self.localNormal, F32), "local_point": np.array(self.localPoint, F32),
            "points": [{"local_point": np.array(self.pointLocal[i], F32), "normal_impulse": F32(self.normalImpulse[i]),
                        "tangent_impulse": F32(self.tangentImpulse[i]), "key": int(self.key[i]),
                        "world_point": np.array(self.worldPoints[i], F32), "separation": F32(self.separations[i])}
                       for i in range(n)],
            "world_normal": np.array(self.worldNormal, F32),
        }


class Event(C.Structure):
    """Mirror of Event in capi.cpp: kind 0 BeginContact, 1 EndContact, 2 PostSolve (impulses of one solve)."""
    _fields_ = [(n, C.c_int32) for n in ("kind", "fixtureA", "fixtureB", "childA", "childB", "count")] + \
               [("normalImpulses", C.c_float * 2), ("tangentImpulses", C.c_float * 2)]


_lib = None


def load(build=True):
    """The shared library, built first (scripts/build_box2d.sh) if it is missing and build is true."""
    global _lib
    if _lib is not None:
        return _lib
    if not LIB_PATH.exists():
        if not build:
            raise FileNotFoundError(f"{LIB_PATH} missing; run scripts/build_box2d.sh")
        subprocess.run([str(REPO_ROOT / "scripts" / "build_box2d.sh")], check=True, stdout=subprocess.DEVNULL)
    lib = C.CDLL(str(LIB_PATH))
    sig = {
        "goi_b2_set_param": (C.c_int, [C.c_char_p, C.c_float]),
        "goi_b2_get_param": (C.c_int, [C.c_char_p, _f]),
        "goi_b2_world_create": (C.c_void_p, [C.c_float, C.c_float]),
        "goi_b2_world_destroy": (None, [C.c_void_p]),
        "goi_b2_world_set_flags": (None, [C.c_void_p, _i]),
        "goi_b2_world_set_gravity": (None, [C.c_void_p, C.c_float, C.c_float]),
        "goi_b2_world_set_layer_masks": (None, [C.c_void_p, C.POINTER(C.c_uint32)]),
        "goi_b2_world_get_inv_dt0": (C.c_float, [C.c_void_p]),
        "goi_b2_world_set_inv_dt0": (None, [C.c_void_p, C.c_float]),
        "goi_b2_world_step": (None, [C.c_void_p, C.c_float, C.c_int32, C.c_int32]),
        "goi_b2_world_clear_forces": (None, [C.c_void_p]),
        "goi_b2_world_counts": (None, [C.c_void_p, _i]),
        "goi_b2_body_add": (C.c_int, [C.c_void_p, _f, _i]),
        "goi_b2_body_set_mass": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float, C.c_float, C.c_int32]),
        "goi_b2_body_get": (C.c_int, [C.c_void_p, C.c_int, _f]),
        "goi_b2_body_set_transform": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float]),
        "goi_b2_body_set_velocity": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float]),
        "goi_b2_body_set_sleep": (C.c_int, [C.c_void_p, C.c_int, C.c_int32, C.c_float]),
        "goi_b2_body_set_flag": (C.c_int, [C.c_void_p, C.c_int, C.c_int32, C.c_int32]),
        "goi_b2_body_set_damping": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float]),
        "goi_b2_body_apply_force": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float, C.c_int32]),
        "goi_b2_body_move_to": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float, C.c_float]),
        "goi_b2_fixture_add_polygon": (C.c_int, [C.c_void_p, C.c_int, _f, C.c_int32, C.c_float, C.c_int32, _f, _i]),
        "goi_b2_fixture_add_box": (C.c_int, [C.c_void_p, C.c_int] + [C.c_float] * 6 + [_f, _i]),
        "goi_b2_fixture_add_circle": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float, C.c_float, _f, _i]),
        "goi_b2_fixture_set_friction": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_int32]),
        "goi_b2_fixture_set_restitution": (C.c_int, [C.c_void_p, C.c_int, C.c_float]),
        "goi_b2_fixture_set_filter": (C.c_int, [C.c_void_p, C.c_int, _i]),
        "goi_b2_fixture_get": (C.c_int, [C.c_void_p, C.c_int, _f, _f]),
        "goi_b2_fixture_shape": (C.c_int, [C.c_void_p, C.c_int, _f]),
        "goi_b2_joint_add_revolute": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int32, _f, _i]),
        "goi_b2_joint_add_prismatic": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int32, _f, _i]),
        "goi_b2_joint_add_weld": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int32, _f]),
        "goi_b2_joint_set": (C.c_int, [C.c_void_p, C.c_int, C.c_int32, C.c_float, C.c_float]),
        "goi_b2_joint_get": (C.c_int, [C.c_void_p, C.c_int, C.c_float, _f]),
        "goi_b2_joint_set_impulses": (C.c_int, [C.c_void_p, C.c_int] + [C.c_float] * 4),
        "goi_b2_joint_set_limit_impulses": (C.c_int, [C.c_void_p, C.c_int, C.c_float, C.c_float]),
        "goi_b2_joint_frame": (C.c_int, [C.c_void_p, C.c_int, _f]),
        "goi_b2_joint_solver": (C.c_int, [C.c_void_p, C.c_int, _f]),
        "goi_b2_body_set_raw": (C.c_int, [C.c_void_p, C.c_int, _f]),
        "goi_b2_contact_struct_size": (C.c_int32, []),
        "goi_b2_contacts": (C.c_int32, [C.c_void_p, C.POINTER(Contact), C.c_int32]),
        "goi_b2_event_struct_size": (C.c_int32, []),
        "goi_b2_events": (C.c_int32, [C.c_void_p, C.POINTER(Event), C.c_int32]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    assert lib.goi_b2_contact_struct_size() == C.sizeof(Contact), "Contact layout differs from capi.cpp"
    assert lib.goi_b2_event_struct_size() == C.sizeof(Event), "Event layout differs from capi.cpp"
    for name in _PARAMS.values():
        out = C.c_float()
        lib.goi_b2_get_param(name.encode(), C.byref(out))
        _STOCK[name] = F32(out.value)
    _lib = lib
    return lib


def _fa(*v):
    return (C.c_float * len(v))(*(float(F32(x)) for x in v))


def _ia(*v):
    return (C.c_int32 * len(v))(*(int(x) for x in v))


_STOCK = {}


def _stock(name):
    """Stock 2.3.1 value of a b2Settings global, as the library had it before any world changed it."""
    load()
    return _STOCK[name]


# Box2D name of each Settings field that is a b2Settings global (process-wide in the library).
_PARAMS = {
    "linear_slop": "linearSlop", "angular_slop": "angularSlop", "polygon_radius": "polygonRadius",
    "velocity_threshold": "velocityThreshold", "max_linear_correction": "maxLinearCorrection",
    "max_angular_correction": "maxAngularCorrection", "max_translation": "maxTranslation",
    "max_rotation": "maxRotation", "baumgarte": "baumgarte", "toi_baumgarte": "toiBaumgarte",
    "time_to_sleep": "timeToSleep", "linear_sleep_tolerance": "linearSleepTolerance",
    "angular_sleep_tolerance": "angularSleepTolerance",
}


@dataclass
class Settings:
    """World configuration. None for a b2Settings field means the library's stock 2.3.1 value, read from the .so.

    max_translation / max_rotation are Box2D's per-step clamps (metres, radians per step). Unity's
    maxTranslationSpeed / maxRotationSpeed may be per second instead (open question T1); clamps_per_second=True
    multiplies them by dt at every step.
    """
    gravity: tuple = (0.0, -10.0)
    velocity_iterations: int = 8
    position_iterations: int = 3
    linear_slop: float = None
    angular_slop: float = None
    polygon_radius: float = None
    velocity_threshold: float = None
    max_linear_correction: float = None
    max_angular_correction: float = None
    max_translation: float = None
    max_rotation: float = None
    baumgarte: float = None
    toi_baumgarte: float = None
    time_to_sleep: float = None
    linear_sleep_tolerance: float = None
    angular_sleep_tolerance: float = None
    clamps_per_second: bool = False
    unity_toi: bool = False  # TOI only where a continuous body meets a static/kinematic one (or a bullet)
    friction_mix: int = 0  # 0 sqrt(f1*f2) (stock), 1 min, 2 max, 3 mean, 4 product
    trig_mode: int = 0  # b2Rot sin/cos: 0 the C library's sinf/cosf (stock), 1 correctly rounded (goi_jax.box2d)
    # b2RevoluteJoint solver: True Box2D v2.4.1's (Unity's graft, the game; scripts/patches/*-revolute24.patch),
    # False stock v2.3.1's 3x3 block with limit states. Default True even for stock(): the oracle models the game.
    revolute_v24: bool = True
    allow_sleep: bool = True
    warm_starting: bool = True
    continuous_physics: bool = True
    sub_stepping: bool = False
    auto_clear_forces: bool = True
    layer_masks: list = field(default=None)  # 32 Physics2D.GetLayerCollisionMask values; None = stock filtering

    @classmethod
    def stock(cls, **kw):
        return cls(**kw)

    @classmethod
    def game(cls, **kw):
        """Getting Over It's Physics2D settings (dumps/engine-dump-start.jsonl settings row, overview §1.5)."""
        s = cls(gravity=(0.0, -30.0), velocity_iterations=300, position_iterations=30,
                velocity_threshold=1.0, max_linear_correction=0.2, max_angular_correction=deg2rad(8),
                max_translation=100.0, max_rotation=deg2rad(360), baumgarte=0.4, toi_baumgarte=0.75,
                time_to_sleep=0.5, linear_sleep_tolerance=0.01, angular_sleep_tolerance=deg2rad(2),
                linear_slop=0.005, polygon_radius=0.01, unity_toi=True)
        return replace(s, **kw)

    def resolved(self, name):
        v = getattr(self, name)
        return _stock(_PARAMS[name]) if v is None else F32(v)


class World:
    """One Box2D world. Ids returned by add_* index bodies, fixtures and joints in creation order."""

    def __init__(self, settings=None):
        self.lib = load()
        self.settings = settings or Settings()
        g = self.settings.gravity
        self.ptr = self.lib.goi_b2_world_create(F32(g[0]), F32(g[1]))
        self._apply_world_flags()

    def __del__(self):
        if getattr(self, "ptr", None):
            self.lib.goi_b2_world_destroy(self.ptr)
            self.ptr = None

    # ---- configuration

    def _apply_world_flags(self):
        s = self.settings
        self.lib.goi_b2_world_set_flags(self.ptr, _ia(s.allow_sleep, s.warm_starting, s.continuous_physics,
                                                      s.sub_stepping, s.auto_clear_forces))
        self.lib.goi_b2_world_set_gravity(self.ptr, F32(s.gravity[0]), F32(s.gravity[1]))
        if s.layer_masks is None:
            self.lib.goi_b2_world_set_layer_masks(self.ptr, None)
        else:
            masks = (C.c_uint32 * 32)(*(int(m) & 0xFFFFFFFF for m in s.layer_masks))
            self.lib.goi_b2_world_set_layer_masks(self.ptr, masks)

    def _apply_globals(self, dt):
        # b2Settings globals are shared by every world in the process, so each step re-applies this world's.
        s = self.settings
        for name, b2name in _PARAMS.items():
            v = s.resolved(name)
            if s.clamps_per_second and name in ("max_translation", "max_rotation"):
                v = F32(v * F32(dt))
            self.lib.goi_b2_set_param(b2name.encode(), v)
        self.lib.goi_b2_set_param(b"unityTOI", float(s.unity_toi))
        self.lib.goi_b2_set_param(b"frictionMix", float(s.friction_mix))
        self.lib.goi_b2_set_param(b"trigMode", float(s.trig_mode))
        self.lib.goi_b2_set_param(b"revoluteSolver", float(bool(s.revolute_v24)))

    def configure(self, **kw):
        """Change settings in place (world flags, gravity and layer masks take effect now, globals at next step)."""
        self.settings = replace(self.settings, **kw)
        self._apply_world_flags()

    @property
    def inv_dt0(self):
        """Previous step's 1/dt (Box2D's dtRatio = inv_dt0 * dt scales warm starts); 0 in a fresh world."""
        return F32(self.lib.goi_b2_world_get_inv_dt0(self.ptr))

    @inv_dt0.setter
    def inv_dt0(self, v):
        self.lib.goi_b2_world_set_inv_dt0(self.ptr, F32(v))

    def counts(self):
        out = (C.c_int32 * 5)()
        self.lib.goi_b2_world_counts(self.ptr, out)
        return dict(zip(("bodies", "fixtures", "joints", "contacts", "events"), out))

    # ---- stepping

    def step(self, dt=GAME_DT, velocity_iterations=None, position_iterations=None):
        s = self.settings
        self._apply_globals(dt)
        self.lib.goi_b2_world_step(self.ptr, F32(dt), velocity_iterations or s.velocity_iterations,
                                   position_iterations or s.position_iterations)

    def clear_forces(self):
        self.lib.goi_b2_world_clear_forces(self.ptr)

    # ---- bodies

    def add_body(self, type=DYNAMIC, position=(0, 0), angle=0.0, velocity=(0, 0), angular_velocity=0.0, *,
                 mass=None, center=(0, 0), inertia=0.0, exact_inertia=True, linear_damping=0.0,
                 angular_damping=0.0, gravity_scale=1.0, fixed_rotation=False, bullet=False, continuous=False,
                 awake=True, allow_sleep=True, active=True):
        """A body with explicit mass data (inertia about the centre of mass, as Rigidbody2D.inertia). Fixtures are
        added with density 0, so the mass set here is never re-derived from shapes."""
        bid = self.lib.goi_b2_body_add(self.ptr, _fa(position[0], position[1], angle, linear_damping,
                                                     angular_damping, gravity_scale),
                                       _ia(type, fixed_rotation, bullet, continuous, awake, allow_sleep, active))
        if mass is not None:
            self.set_mass(bid, mass, center, inertia, exact_inertia)
        # After the mass: SetMassData shifts the velocity when the centre moves under a spinning body.
        if type != STATIC and (velocity[0] or velocity[1] or angular_velocity):
            self.set_velocity(bid, velocity, angular_velocity)
            if not awake:
                self.set_sleep(bid, awake=False)
        return bid

    def set_mass(self, body, mass, center=(0, 0), inertia=0.0, exact=True):
        self._check(self.lib.goi_b2_body_set_mass(self.ptr, body, F32(mass), F32(center[0]), F32(center[1]),
                                                  F32(inertia), int(exact)))

    _BODY_FIELDS = ("x", "y", "angle", "cx", "cy", "vx", "vy", "w", "awake", "sleep_time", "type", "active",
                    "mass", "inv_mass", "inertia", "inv_inertia", "local_cx", "local_cy", "c0x", "c0y", "a0",
                    "alpha0", "flags", "bullet_continuous")

    def body_array(self, body):
        """float32[24] in _BODY_FIELDS order: origin pose, centre, velocities, sleep state, mass data, sweep."""
        out = np.zeros(24, F32)
        self._check(self.lib.goi_b2_body_get(self.ptr, body, out.ctypes.data_as(_f)))
        return out

    def body(self, body):
        a = self.body_array(body)
        d = dict(zip(self._BODY_FIELDS, a))
        d["position"], d["center"], d["velocity"] = a[0:2].copy(), a[3:5].copy(), a[5:7].copy()
        for k in ("awake", "active"):
            d[k] = bool(d[k])
        d["type"] = int(d["type"])
        return d

    def bodies_state(self):
        """float32[n, 8]: x, y, angle, vx, vy, w, awake, sleep_time for every body (the per-step state)."""
        n = self.counts()["bodies"]
        return np.stack([self.body_array(i)[[0, 1, 2, 5, 6, 7, 8, 9]] for i in range(n)]) if n else np.zeros((0, 8), F32)

    def set_transform(self, body, position, angle):
        self._check(self.lib.goi_b2_body_set_transform(self.ptr, body, F32(position[0]), F32(position[1]), F32(angle)))

    def set_body_raw(self, body, xf, c0, c, a0, a, alpha0=0.0):
        """Write the transform (p.x, p.y, q.s, q.c) and sweep verbatim, e.g. a recorded native body. Fixtures and
        proxies are not re-synced: use on bodies without fixtures, or call set_transform first."""
        f = np.array([*xf, *c0, *c, a0, a, alpha0], F32)
        self._check(self.lib.goi_b2_body_set_raw(self.ptr, body, f.ctypes.data_as(_f)))

    def set_velocity(self, body, velocity, angular_velocity=0.0):
        self._check(self.lib.goi_b2_body_set_velocity(self.ptr, body, F32(velocity[0]), F32(velocity[1]),
                                                      F32(angular_velocity)))

    def set_sleep(self, body, awake=None, sleep_time=None):
        self._check(self.lib.goi_b2_body_set_sleep(self.ptr, body, -1 if awake is None else int(awake),
                                                   F32(-1 if sleep_time is None else sleep_time)))

    def set_body_flag(self, body, what, value):
        """what: bullet, continuous, active, fixed_rotation, allow_sleep, type. fixed_rotation and type reset the
        mass to Box2D's shape-derived value: call set_mass again afterwards."""
        code = ("bullet", "continuous", "active", "fixed_rotation", "allow_sleep", "type").index(what)
        self._check(self.lib.goi_b2_body_set_flag(self.ptr, body, code, int(value)))

    def set_damping(self, body, linear, angular, gravity_scale=1.0):
        self._check(self.lib.goi_b2_body_set_damping(self.ptr, body, F32(linear), F32(angular), F32(gravity_scale)))

    def apply_force(self, body, force, torque=0.0, wake=True):
        self._check(self.lib.goi_b2_body_apply_force(self.ptr, body, F32(force[0]), F32(force[1]), F32(torque),
                                                     int(wake)))

    def move_to(self, body, position, angle=None, dt=GAME_DT):
        """Kinematic MovePosition/MoveRotation emulation: velocity = (target - pose) / dt for the next step."""
        if angle is None:
            angle = self.body_array(body)[2]
        self._check(self.lib.goi_b2_body_move_to(self.ptr, body, F32(position[0]), F32(position[1]), F32(angle),
                                                 F32(dt)))

    # ---- fixtures

    def _material(self, friction, restitution, density):
        return _fa(friction, restitution, density)

    def _filter(self, sensor, layer, category, mask, group):
        return _ia(sensor, layer, category, mask, group)

    def add_polygon(self, body, vertices, radius=None, raw=False, *, friction=0.2, restitution=0.0, density=0.0,
                    sensor=False, layer=0, category=1, mask=0xFFFF, group=0):
        """Convex piece in body-local coordinates (3..8 vertices). raw=True keeps the given CCW order instead of
        b2PolygonShape::Set's hull order. radius defaults to the settings' polygon_radius (b2_polygonRadius)."""
        v = np.ascontiguousarray(vertices, F32).reshape(-1, 2)
        r = self.settings.resolved("polygon_radius") if radius is None else F32(radius)
        fid = self.lib.goi_b2_fixture_add_polygon(self.ptr, body, v.ctypes.data_as(_f), len(v), r, int(raw),
                                                  self._material(friction, restitution, density),
                                                  self._filter(sensor, layer, category, mask, group))
        return self._check(fid)

    def add_box(self, body, half_extents, center=(0, 0), angle=0.0, radius=None, *, friction=0.2, restitution=0.0,
                density=0.0, sensor=False, layer=0, category=1, mask=0xFFFF, group=0):
        r = self.settings.resolved("polygon_radius") if radius is None else F32(radius)
        fid = self.lib.goi_b2_fixture_add_box(self.ptr, body, F32(half_extents[0]), F32(half_extents[1]),
                                              F32(center[0]), F32(center[1]), F32(angle), r,
                                              self._material(friction, restitution, density),
                                              self._filter(sensor, layer, category, mask, group))
        return self._check(fid)

    def add_circle(self, body, radius, center=(0, 0), *, friction=0.2, restitution=0.0, density=0.0, sensor=False,
                   layer=0, category=1, mask=0xFFFF, group=0):
        fid = self.lib.goi_b2_fixture_add_circle(self.ptr, body, F32(center[0]), F32(center[1]), F32(radius),
                                                 self._material(friction, restitution, density),
                                                 self._filter(sensor, layer, category, mask, group))
        return self._check(fid)

    def set_friction(self, fixture, friction, refresh_contacts=False):
        """Box2D mixes friction when a contact is created; refresh_contacts re-mixes existing ones too (T5)."""
        self._check(self.lib.goi_b2_fixture_set_friction(self.ptr, fixture, F32(friction), int(refresh_contacts)))

    def set_restitution(self, fixture, restitution):
        self._check(self.lib.goi_b2_fixture_set_restitution(self.ptr, fixture, F32(restitution)))

    def set_filter(self, fixture, *, sensor=False, layer=0, category=1, mask=0xFFFF, group=0):
        self._check(self.lib.goi_b2_fixture_set_filter(self.ptr, fixture,
                                                       self._filter(sensor, layer, category, mask, group)))

    def fixture(self, fixture):
        o, v = np.zeros(8, F32), np.zeros(16, F32)
        self._check(self.lib.goi_b2_fixture_get(self.ptr, fixture, o.ctypes.data_as(_f), v.ctypes.data_as(_f)))
        return {"body": int(o[0]), "type": int(o[1]), "child_count": int(o[2]), "vertices": v[:2 * int(o[3])].reshape(-1, 2),
                "radius": o[4], "friction": o[5], "restitution": o[6], "layer": int(o[7])}

    def fixture_shape(self, fixture):
        """Exact shape data: vertex count, skin radius, centroid, vertices and edge normals (float32, as stored)."""
        o = np.zeros(36, F32)
        self._check(self.lib.goi_b2_fixture_shape(self.ptr, fixture, o.ctypes.data_as(_f)))
        n = int(o[0])
        return {"count": n, "radius": o[1], "centroid": o[2:4].copy(), "vertices": o[4:20].reshape(8, 2)[:n].copy(),
                "normals": o[20:36].reshape(8, 2)[:n].copy()}

    # ---- joints (anchors local to each body's origin, angles in radians)

    def add_revolute(self, body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0), reference_angle=0.0, *,
                     limits=None, motor_speed=0.0, max_motor_torque=0.0, enable_motor=False, collide=False):
        lo, hi = limits if limits is not None else (0.0, 0.0)
        jid = self.lib.goi_b2_joint_add_revolute(self.ptr, body_a, body_b, int(collide),
                                                 _fa(*anchor_a, *anchor_b, reference_angle, lo, hi, motor_speed,
                                                     max_motor_torque),
                                                 _ia(limits is not None, enable_motor))
        return self._check(jid)

    def add_prismatic(self, body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0), axis=(1, 0), reference_angle=0.0, *,
                      limits=None, motor_speed=0.0, max_motor_force=0.0, enable_motor=False, collide=False):
        lo, hi = limits if limits is not None else (0.0, 0.0)
        jid = self.lib.goi_b2_joint_add_prismatic(self.ptr, body_a, body_b, int(collide),
                                                  _fa(*anchor_a, *anchor_b, *axis, reference_angle, lo, hi,
                                                      motor_speed, max_motor_force),
                                                  _ia(limits is not None, enable_motor))
        return self._check(jid)

    def add_weld(self, body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0), reference_angle=0.0, *, frequency=0.0,
                 damping=0.0, collide=False):
        jid = self.lib.goi_b2_joint_add_weld(self.ptr, body_a, body_b, int(collide),
                                             _fa(*anchor_a, *anchor_b, reference_angle, frequency, damping))
        return self._check(jid)

    def _joint_set(self, joint, what, value, value2=0.0):
        self._check(self.lib.goi_b2_joint_set(self.ptr, joint, what, F32(value), F32(value2)))

    def set_motor_speed(self, joint, speed):
        self._joint_set(joint, 0, speed)

    def set_max_motor(self, joint, value):
        """maxMotorTorque (revolute) or maxMotorForce (prismatic)."""
        self._joint_set(joint, 1, value)

    def enable_motor(self, joint, flag=True):
        self._joint_set(joint, 2, float(flag))

    def enable_limit(self, joint, flag=True):
        self._joint_set(joint, 3, float(flag))

    def set_limits(self, joint, lower, upper):
        self._joint_set(joint, 4, lower, upper)

    def set_weld_spring(self, joint, frequency, damping):
        self._joint_set(joint, 5, frequency)
        self._joint_set(joint, 6, damping)

    _JOINT_FIELDS = ("type", "body_a", "body_b", "fx", "fy", "torque", "ix", "iy", "iz", "motor_impulse",
                     "limit_state", "value", "speed", "motor_speed", "max_motor", "gamma", "bias",
                     "lower_impulse", "upper_impulse")

    def joint(self, joint, inv_dt=None):
        """Reaction force/torque at inv_dt (default 1/GAME_DT, as Joint2D.reactionForce uses the last step's dt),
        accumulated impulses (revolute/prismatic: point x, y and limit z, plus motor; weld: x, y, angle),
        limit state (0 inactive, 1 lower, 2 upper, 3 equal), angle or translation, speed. Revolute with the v2.4
        solver: lower_impulse / upper_impulse hold the limit impulses (iz stays 0, so torque reads 0 as in Unity) and
        limit_state is a readout only (2.3.1's classification of the step-start angle; the v2.4 solver has none)."""
        inv_dt = F32(1) / GAME_DT if inv_dt is None else F32(inv_dt)
        o = np.zeros(19, F32)
        self._check(self.lib.goi_b2_joint_get(self.ptr, joint, inv_dt, o.ctypes.data_as(_f)))
        d = dict(zip(self._JOINT_FIELDS, o))
        for k in ("type", "body_a", "body_b", "limit_state"):
            d[k] = int(d[k])
        return d

    def set_joint_impulses(self, joint, impulse=(0, 0, 0), motor_impulse=0.0, lower_impulse=None, upper_impulse=None):
        """Overwrite the accumulated (warm-start) impulses. lower/upper (revolute only) are left alone when None."""
        self._check(self.lib.goi_b2_joint_set_impulses(self.ptr, joint, *(F32(x) for x in impulse), F32(motor_impulse)))
        if lower_impulse is not None or upper_impulse is not None:
            j = self.joint(joint)
            lo = j["lower_impulse"] if lower_impulse is None else lower_impulse
            hi = j["upper_impulse"] if upper_impulse is None else upper_impulse
            self._check(self.lib.goi_b2_joint_set_limit_impulses(self.ptr, joint, F32(lo), F32(hi)))

    def joint_frame(self, joint):
        """The stored joint frame: local_anchor_a/b, reference_angle, local_axis / local_perp (prismatic
        localXAxisA / localYAxisA after Box2D's normalization; zeros otherwise), lower, upper, collide."""
        o = np.zeros(12, F32)
        self._check(self.lib.goi_b2_joint_frame(self.ptr, joint, o.ctypes.data_as(_f)))
        return {"local_anchor_a": o[0:2].copy(), "local_anchor_b": o[2:4].copy(), "reference_angle": o[4],
                "local_axis": o[5:7].copy(), "local_perp": o[7:9].copy(), "lower": o[9], "upper": o[10],
                "collide": bool(o[11])}

    def joint_solver(self, joint):
        """Solver temporaries of the last step's InitVelocityConstraints: rA, rB, K (2x2 point mass as ex.x, ex.y,
        ey.x, ey.y; v2.4 revolute m_K, else the 2x2 block of m_mass), angle (v2.4 revolute m_angle, the step-start
        aB - aA - ref; else 0), axial_mass (v2.4 m_axialMass / v2.3.1 m_motorMass), lower/upper impulses, island
        indices. These are the fields Unity's native revolute joint holds (docs/understanding/native-box2d.md)."""
        o = np.zeros(14, F32)
        self._check(self.lib.goi_b2_joint_solver(self.ptr, joint, o.ctypes.data_as(_f)))
        return {"rA": o[0:2].copy(), "rB": o[2:4].copy(), "K": o[4:8].copy(), "angle": o[8], "axial_mass": o[9],
                "lower_impulse": o[10], "upper_impulse": o[11], "index_a": int(o[12]), "index_b": int(o[13])}

    # ---- contacts and events

    def contacts_raw(self):
        n = self.counts()["contacts"]
        buf = (Contact * max(n, 1))()
        got = self.lib.goi_b2_contacts(self.ptr, buf, len(buf))
        return list(buf[:min(n, got)])

    def contacts(self, touching_only=False):
        """Every live contact in world contact-list order (newest first), with manifold and world manifold."""
        cs = [c.to_dict() for c in self.contacts_raw()]
        return [c for c in cs if c["touching"]] if touching_only else cs

    def events(self):
        """Last step's events: (kind, fixtureA, fixtureB, childA, childB, normalImpulses, tangentImpulses);
        kind 'begin', 'end' or 'post_solve' (one per solve of a touching contact, TOI sub-steps included)."""
        n = self.counts()["events"]
        buf = (Event * max(n, 1))()
        self.lib.goi_b2_events(self.ptr, buf, len(buf))
        kinds = ("begin", "end", "post_solve")
        return [(kinds[e.kind], e.fixtureA, e.fixtureB, e.childA, e.childB,
                 np.array(e.normalImpulses[:e.count], F32), np.array(e.tangentImpulses[:e.count], F32))
                for e in buf[:n]]

    @staticmethod
    def _check(r):
        if r < 0:
            raise ValueError(f"Box2D C API rejected the call (code {r})")
        return r
