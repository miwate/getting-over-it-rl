"""Load a plugin engine dump (dumps/engine-dump-start.jsonl, written by plugin/EngineDump.cs) into the Box2D oracle.

Unity -> Box2D mapping used here (each choice that is a guess is named; overview.md section numbers in brackets):
- Rigidbody2D.position is the body origin, rotation is degrees, centerOfMass is local, inertia is about the centre
  of mass (T7: passed verbatim). drag / angularDrag -> linear / angular damping. FreezeRotation -> fixedRotation.
  Continuous -> the patched per-body continuous flag (Unity's TOI rule, settings.unity_toi). NeverSleep -> no sleep.
- Colliders without a Rigidbody2D sit on one static ground body at the origin, as Unity does.
- Joints: body A is the joint component's body, B the connected one (ground if none); anchors are local to each;
  angles, limits and motor speeds convert from degrees. Unity's sign convention for hinge angles is not verified (a
  hinge with limits mirrored would show up as a limit fighting the motor on the first steps).
- The 32x32 layer matrix replaces Box2D's uint16 filter (custom b2ContactFilter in capi.cpp).
- Friction: collider material, else body material, else DefaultFriction 1.0 (priority INFERRED, §1.5).

Collider geometry. The dump has world-space render triangles of each collider's shapes, not the convex b2 pieces
(R2). Until the plugin dumps the real pieces, a collider becomes:
- exact pieces from `pieces=` if given (JSON lines, one per collider:
  {"path": ..., "index": k, "frame": "world"|"body", "pieces": [{"verts": [[x, y], ...], "radius": r}, ...]}
  where index is the k-th dump row with that path, since paths repeat, e.g. Player/PotCollider/Sides);
- else one polygon of all its vertices when it has one shape of <= 8 vertices and they are convex (right
  outline, vertex order and therefore contact feature ids still unconfirmed);
- else one triangle fixture per render triangle. This is APPROXIMATE: wrong internal edges and contact order.
Circles take centre and radius from the mesh ring (the dump has the unscaled radius and no Transform).
Scene.approximate lists every collider that is not exact.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import box2d as b2

F32 = np.float32
DEFAULT_DUMP = b2.REPO_ROOT / "dumps" / "engine-dump-start.jsonl"
DEFAULT_FRICTION = 1.0  # DefaultFriction asset [GGM]
_TYPES = {"Static": b2.STATIC, "Kinematic": b2.KINEMATIC, "Dynamic": b2.DYNAMIC}


@dataclass
class Scene:
    world: b2.World
    settings_row: dict
    body_rows: list
    joint_rows: list  # all dump joints, including disabled ones (not built)
    collider_rows: list
    bodies: dict = field(default_factory=dict)  # body path -> body id (ground is "<ground>")
    joints: list = field(default_factory=list)  # (joint row index, joint id) for every built joint
    fixtures: list = field(default_factory=list)  # (collider row index, fixture id, how) for every fixture
    approximate: list = field(default_factory=list)  # (collider path, reason)
    skipped: list = field(default_factory=list)  # (collider path, reason)

    def body_id(self, path):
        return self.bodies[path]

    def joint_id(self, path, type_name):
        for k, jid in self.joints:
            r = self.joint_rows[k]
            if r["path"] == path and r["type"] == type_name:
                return jid
        raise KeyError((path, type_name))


def parse_material(s):
    """'Name friction=F bounciness=B' (EngineDump.Mat) -> (name, friction, bounciness), or None."""
    if not s:
        return None
    name, rest = s.split(" friction=")
    fr, bo = rest.split(" bounciness=")
    return name, float(fr), float(bo)


def settings_from_row(row, **kw):
    d = b2.deg2rad
    return b2.Settings.game(
        gravity=tuple(row["gravity"]), velocity_iterations=row["velocityIterations"],
        position_iterations=row["positionIterations"], velocity_threshold=row["velocityThreshold"],
        max_linear_correction=row["maxLinearCorrection"], max_angular_correction=d(row["maxAngularCorrection"]),
        max_translation=row["maxTranslationSpeed"], max_rotation=d(row["maxRotationSpeed"]),
        baumgarte=row["baumgarteScale"], toi_baumgarte=row["baumgarteTOIScale"], time_to_sleep=row["timeToSleep"],
        linear_sleep_tolerance=row["linearSleepTolerance"], angular_sleep_tolerance=d(row["angularSleepTolerance"]),
        polygon_radius=row["defaultContactOffset"],  # T2: INFERRED to be b2_polygonRadius
        layer_masks=list(row["layerCollisionMasks"]), **kw)


def _world_to_local(pts, pos, angle):
    # Inverse of b2Transform (b2MulT) in float32, as Box2D would compute it.
    c, s = F32(np.cos(F32(angle))), F32(np.sin(F32(angle)))
    d = pts.astype(F32) - np.asarray(pos, F32)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1).astype(F32)


def _hull_order(v):
    """v (float32) as a strictly convex CCW polygon starting at the rightmost point (lowest y on ties), the order
    b2PolygonShape::Set produces; None if some point is not a strict hull vertex. Mesh vertex order is not the
    outline order, so the outline is rebuilt here, then passed raw (Set's 5 cm weld would eat small pieces)."""
    v = np.asarray(v, F32)
    p = sorted(set(map(tuple, v.astype(np.float64))))
    if len(p) < 3 or len(p) < len(v):
        return None

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]

    hull = half(p) + half(p[::-1])  # CCW from the leftmost point
    if len(hull) != len(p):
        return None
    i0 = max(range(len(hull)), key=lambda i: (hull[i][0], -hull[i][1]))
    return np.asarray(hull[i0:] + hull[:i0], F32)


def _load_pieces(path):
    out = {}
    if path is None:
        return out
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["path"], r.get("index", 0))] = r
    return out


def load_scene(dump=DEFAULT_DUMP, pieces=None, settings=None, inactive_bodies=True, min_triangle_area=1e-7):
    """Build the dump's world. settings overrides the dump's settings row; pieces is an optional exact-pieces file.
    Bodies are created in dump order (sorted by path), which is not necessarily Unity's creation order (R10)."""
    rows = [json.loads(l) for l in Path(dump).read_text().splitlines() if l.strip()]
    srow = next(r for r in rows if r["kind"] == "settings")
    sc = Scene(world=None, settings_row=srow, body_rows=[r for r in rows if r["kind"] == "body"],
               joint_rows=[r for r in rows if r["kind"] == "joint"],
               collider_rows=[r for r in rows if r["kind"] == "collider"])
    w = sc.world = b2.World(settings or settings_from_row(srow))
    piece_map = _load_pieces(pieces)

    sc.bodies["<ground>"] = w.add_body(b2.STATIC)
    poses = {"<ground>": ((0.0, 0.0), F32(0))}
    body_mat = {}
    for r in sc.body_rows:
        cons = r["constraints"]
        if cons not in ("None", "FreezeRotation"):
            raise NotImplementedError(f"{r['path']}: constraints {cons}")  # position freezes need a joint
        active = r["active"] and r["simulated"]
        if not active and not inactive_bodies:
            continue
        angle = b2.deg2rad(r["rotation"])
        bid = w.add_body(
            _TYPES[r["bodyType"]], r["position"], angle, r["velocity"], b2.deg2rad(r["angularVelocity"]),
            mass=r["mass"], center=r["centerOfMass"], inertia=r["inertia"],
            linear_damping=r["drag"], angular_damping=r["angularDrag"], gravity_scale=r["gravityScale"],
            fixed_rotation=cons == "FreezeRotation", continuous=r["collisionDetection"] == "Continuous",
            awake=r["awake"], allow_sleep=r["sleepMode"] != "NeverSleep", active=active)
        sc.bodies[r["path"]] = bid
        poses[r["path"]] = (r["position"], angle)
        body_mat[r["path"]] = parse_material(r["material"])

    seen = {}
    for k, r in enumerate(sc.collider_rows):
        path = r["path"]
        index = seen[path] = seen.get(path, -1) + 1
        if not r["active"]:
            sc.skipped.append((path, "inactive"))
            continue
        owner = r["body"] or "<ground>"
        if owner not in sc.bodies:
            sc.skipped.append((path, "body not built"))
            continue
        bid = sc.bodies[owner]
        mat = parse_material(r["material"]) or body_mat.get(owner)
        opts = dict(friction=mat[1] if mat else DEFAULT_FRICTION, restitution=mat[2] if mat else 0.0,
                    sensor=r["trigger"], layer=r["layer"])
        pos, angle = poses[owner]

        if (path, index) in piece_map:
            pr = piece_map[(path, index)]
            for p in pr["pieces"]:
                v = np.asarray(p["verts"], F32)
                if pr.get("frame", "world") == "world":
                    v = _world_to_local(v, pos, angle)
                sc.fixtures.append((k, w.add_polygon(bid, v, p.get("radius"), raw=True, **opts), "piece"))
            continue

        verts = np.asarray(r.get("verts", []), np.float64).reshape(-1, 2)
        tris = np.asarray(r.get("tris", []), int).reshape(-1, 3)
        if len(verts) == 0:
            sc.skipped.append((path, "no shapes"))
            continue
        local = _world_to_local(verts, pos, angle)

        if r["type"] == "CircleCollider2D":
            c = local.astype(np.float64).mean(axis=0)
            rad = float(np.linalg.norm(local - c, axis=1).max())
            sc.fixtures.append((k, w.add_circle(bid, rad, c, **opts), "circle"))
            sc.approximate.append((path, "circle centre/radius from render ring"))
        elif r["shapeCount"] == 1 and len(local) <= 8 and _hull_order(local) is not None:
            sc.fixtures.append((k, w.add_polygon(bid, _hull_order(local), raw=True, **opts), "outline"))
            if r["type"] != "BoxCollider2D":
                sc.approximate.append((path, "single-piece outline (vertex order unconfirmed)"))
        else:
            sc.approximate.append((path, f"{len(tris)} render triangles for {r['shapeCount']} pieces"))
            for t in tris:
                v = _hull_order(local[t])
                e1, e2 = (v[1] - v[0], v[2] - v[0]) if v is not None else ((0, 0), (0, 0))
                if v is None or abs(float(e1[0]) * float(e2[1]) - float(e1[1]) * float(e2[0])) * 0.5 < min_triangle_area:
                    sc.skipped.append((path, "degenerate triangle"))
                    continue
                try:
                    sc.fixtures.append((k, w.add_polygon(bid, v, raw=True, **opts), "triangle"))
                except ValueError:  # convex in float64, not in Box2D's float32 test
                    sc.skipped.append((path, "degenerate triangle"))

    ground = sc.bodies["<ground>"]
    d = b2.deg2rad
    for k, r in enumerate(sc.joint_rows):
        if not r["enabled"] or r["path"] not in sc.bodies:
            continue
        a = sc.bodies[r["path"]]
        bb = sc.bodies.get(r["connected"], ground) if r["connected"] else ground
        common = dict(anchor_a=r["anchor"], anchor_b=r["connectedAnchor"], reference_angle=d(r["referenceAngle"]),
                      collide=r["enableCollision"])
        if r["type"] == "HingeJoint2D":
            jid = w.add_revolute(a, bb, **common, limits=(d(r["limits"][0]), d(r["limits"][1])) if r["useLimits"] else None,
                                 motor_speed=d(r["motorSpeed"]), max_motor_torque=r["maxMotorTorque"],
                                 enable_motor=r["useMotor"])
        elif r["type"] == "SliderJoint2D":
            ang = d(r["angle"])
            jid = w.add_prismatic(a, bb, **common, axis=(np.cos(ang), np.sin(ang)),
                                  limits=tuple(r["limits"]) if r["useLimits"] else None,
                                  motor_speed=r["motorSpeed"], max_motor_force=r["maxMotorForce"],
                                  enable_motor=r["useMotor"])
        elif r["type"] == "FixedJoint2D":
            jid = w.add_weld(a, bb, **common, frequency=r["frequency"], damping=r["dampingRatio"])
        else:
            raise NotImplementedError(r["type"])
        sc.joints.append((k, jid))
    return sc
