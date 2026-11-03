"""Twin worlds: the same scene built in goi_jax.box2d (JAX) and goi_jax.reference (Box2D 2.3.1 oracle), stepped side
by side and compared per step.

The oracle runs with trig_mode=1 (correctly rounded sin/cos, which the JAX port reproduces) and continuous physics off
(TOI is not ported). Everything else is Settings.game() unless a scene says otherwise.

Errors are reported per quantity as max |jax - box2d| and max ulp distance (number of float32 values between them).
"""
import time

import jax
import numpy as np

from goi_jax import box2d as jb
from goi_jax.box2d.world import WorldDef
from goi_jax.reference import box2d as b2

F32 = np.float32
H = b2.GAME_DT
QUANTITIES = ("x", "y", "angle", "vx", "vy", "w", "normal_impulse", "tangent_impulse", "joint_impulse",
              "motor_impulse", "limit_impulse")


def ulp_distance(a, b):
    """Float32 values between a and b (0 = identical bits; +0 and -0 count as equal)."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    oa = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ob = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(oa - ob)


class Twin:
    """Forwards each builder call to both worlds and checks that they hand out the same ids."""

    def __init__(self, settings=None, layer_masks=None, contact_capacity=16, **ref_kw):
        self.settings = settings or jb.Settings.game()
        kw = self.settings.reference_kwargs()
        kw.update(continuous_physics=False, layer_masks=layer_masks)
        kw.update(ref_kw)
        self.ref = b2.World(b2.Settings(**kw))
        self.defn = WorldDef(self.settings, layer_masks=layer_masks)
        self.capacity = contact_capacity

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def both(*a, **k):
            r = getattr(self.ref, name)(*a, **k)
            j = getattr(self.defn, name)(*a, **k)
            assert r == j, (name, r, j)
            return r
        return both

    def build(self):
        self.model, self.state = self.defn.build(self.capacity)
        s, model = self.settings, self.model
        self._step = jax.jit(lambda st: jb.step(model, st, s, H))
        return self


# ---- snapshots

def ref_snapshot(w):
    n = w.counts()["bodies"]
    bodies = np.stack([w.body_array(i) for i in range(n)])
    joints = [w.joint(j) for j in range(w.counts()["joints"])]
    contacts = w.contacts()
    return dict(
        x=bodies[:, 0], y=bodies[:, 1], angle=bodies[:, 2], cx=bodies[:, 3], cy=bodies[:, 4], vx=bodies[:, 5],
        vy=bodies[:, 6], w=bodies[:, 7], awake=bodies[:, 8].astype(bool), sleep_time=bodies[:, 9],
        joint_impulse=np.array([[j["ix"], j["iy"], j["iz"]] for j in joints], F32).reshape(-1, 3),
        motor_impulse=np.array([j["motor_impulse"] for j in joints], F32),
        limit_impulse=np.array([[j["lower_impulse"], j["upper_impulse"]] for j in joints], F32).reshape(-1, 2),
        limit_state=np.array([j["limit_state"] for j in joints], int),
        contacts=[dict(f=c["fixtures"], touching=c["touching"], count=len(c["points"]), mtype=c["manifold_type"],
                       ni=np.array([p["normal_impulse"] for p in c["points"]], F32),
                       ti=np.array([p["tangent_impulse"] for p in c["points"]], F32),
                       ids=[p["key"] for p in c["points"]], ln=c["local_normal"], lp=c["local_point"],
                       pl=np.array([p["local_point"] for p in c["points"]], F32).reshape(-1, 2)) for c in contacts])


def jax_snapshot(st):
    st = jax.tree.map(np.asarray, st)
    b, ct, jt = st.bodies, st.contacts, st.joints
    live = np.nonzero(ct.pair >= 0)[0]
    live = live[np.argsort(-ct.key[live], kind="stable")]  # Box2D list order: newest first
    contacts = []
    for k in live:
        n = int(ct.count[k])
        contacts.append(dict(f=(int(ct.fa[k]), int(ct.fb[k])), touching=bool(ct.touching[k]), count=n,
                             mtype=int(ct.mtype[k]), ni=ct.normal_impulse[k, :n].copy(),
                             ti=ct.tangent_impulse[k, :n].copy(), ids=[int(i) & 0xFFFFFFFF for i in ct.point_id[k, :n]],
                             ln=ct.local_normal[k], lp=ct.local_point[k], pl=ct.point_local[k, :n]))
    return dict(x=b.xf_p[:, 0], y=b.xf_p[:, 1], angle=b.a, cx=b.c[:, 0], cy=b.c[:, 1], vx=b.v[:, 0], vy=b.v[:, 1],
                w=b.w, awake=b.awake, sleep_time=b.sleep_time, joint_impulse=jt.impulse, motor_impulse=jt.motor_impulse,
                limit_state=jt.limit_state, contacts=contacts, overflow=int(st.overflow),
                limit_impulse=np.stack([jt.lower_impulse, jt.upper_impulse], 1))


def compare(r, j):
    """Per-quantity (max abs, max ulp) and the list of structural mismatches for one step."""
    err, issues = {}, []
    for q in ("x", "y", "angle", "vx", "vy", "w", "joint_impulse", "motor_impulse", "limit_impulse"):
        a, b = np.asarray(r[q], F32), np.asarray(j[q], F32)
        err[q] = (float(np.max(np.abs(a - b), initial=0)), int(np.max(ulp_distance(a, b), initial=0)))
    for q in ("cx", "cy", "sleep_time"):  # sweep centres and sleep timers: bit-identical or reported
        a, b = np.asarray(r[q], F32), np.asarray(j[q], F32)
        if np.max(ulp_distance(a, b), initial=0):
            issues.append(f"{q} {a} vs {b}")
    if not np.array_equal(r["awake"], j["awake"]):
        issues.append(f"awake {r['awake'].astype(int)} vs {j['awake'].astype(int)}")
    if not np.array_equal(r["limit_state"], j["limit_state"]):
        issues.append(f"limit_state {r['limit_state']} vs {j['limit_state']}")
    if j.get("overflow"):
        issues.append("contact table overflow")
    rf, jf = [c["f"] for c in r["contacts"]], [c["f"] for c in j["contacts"]]
    if rf != jf:
        issues.append(f"contact list {rf} vs {jf}")
    ni, ti = [(0.0, 0)], [(0.0, 0)]
    jmap = {c["f"]: c for c in j["contacts"]}
    for rc in r["contacts"]:
        jc = jmap.get(rc["f"])
        if jc is None:
            continue
        if (rc["touching"], rc["count"]) != (jc["touching"], jc["count"]):
            issues.append(f"contact {rc['f']} touching/count {rc['touching'], rc['count']} vs {jc['touching'], jc['count']}")
            continue
        if rc["count"] and (rc["ids"] != jc["ids"] or rc["mtype"] != jc["mtype"]):
            issues.append(f"contact {rc['f']} ids/type {rc['ids'], rc['mtype']} vs {jc['ids'], jc['mtype']}")
            continue
        if rc["count"] and not all(np.array_equal(np.asarray(rc[k], F32), np.asarray(jc[k], F32))
                                   for k in ("ln", "lp", "pl")):
            issues.append(f"contact {rc['f']} manifold geometry {rc['ln'], rc['lp'], rc['pl']} vs "
                          f"{jc['ln'], jc['lp'], jc['pl']}")
        for key, acc in (("ni", ni), ("ti", ti)):
            a, b = rc[key], jc[key]
            acc.append((float(np.max(np.abs(a - b), initial=0)), int(np.max(ulp_distance(a, b), initial=0))))
    err["normal_impulse"] = (max(e[0] for e in ni), max(e[1] for e in ni))
    err["tangent_impulse"] = (max(e[0] for e in ti), max(e[1] for e in ti))
    return err, issues


def run(twin, steps=120):
    """Step both worlds; per-step errors. Returns dict(errors=[per step dict], issues=[(step, msg)], timing)."""
    if not hasattr(twin, "model"):
        twin.build()
    st = twin.state
    t0 = time.perf_counter()
    st = twin._step(st)
    jax.block_until_ready(st)
    compile_s = time.perf_counter() - t0
    errors, issues, contacts_seen = [], [], 0
    for k in range(steps):
        if k:
            st = twin._step(st)
        twin.ref.step(H)
        r, j = ref_snapshot(twin.ref), jax_snapshot(st)
        contacts_seen = max(contacts_seen, sum(c["touching"] for c in r["contacts"]))
        e, iss = compare(r, j)
        errors.append(e)
        issues += [(k + 1, m) for m in iss]
    t0 = time.perf_counter()
    for _ in range(20):
        st = twin._step(st)
    jax.block_until_ready(st)
    step_s = (time.perf_counter() - t0) / 20
    return dict(errors=errors, issues=issues, compile_s=compile_s, step_s=step_s, max_touching=contacts_seen,
                state=st)


def summarize(res, at=(1, 120)):
    """{quantity: {"1": (abs, ulp), "120": (abs, ulp), "max": (abs, ulp)}}."""
    out = {}
    for q in QUANTITIES:
        row = {}
        for s in at:
            if s <= len(res["errors"]):
                row[str(s)] = res["errors"][s - 1][q]
        row["max"] = (max(e[q][0] for e in res["errors"]), max(e[q][1] for e in res["errors"]))
        out[q] = row
    return out


def initial_mismatches(twin):
    """Bodies and shapes of the two builds, bit for bit, before any step."""
    if not hasattr(twin, "model"):
        twin.build()
    bad = []
    st = jax.tree.map(np.asarray, twin.state)
    b = st.bodies
    for i in range(twin.ref.counts()["bodies"]):
        r = twin.ref.body(i)
        pairs = dict(x=(r["x"], b.xf_p[i, 0]), y=(r["y"], b.xf_p[i, 1]), angle=(r["angle"], b.a[i]),
                     cx=(r["cx"], b.c[i, 0]), cy=(r["cy"], b.c[i, 1]), vx=(r["vx"], b.v[i, 0]), vy=(r["vy"], b.v[i, 1]),
                     w=(r["w"], b.w[i]), mass=(r["mass"], b.mass[i]), inv_mass=(r["inv_mass"], b.inv_mass[i]),
                     inertia=(r["inertia"], b.inertia[i]), inv_inertia=(r["inv_inertia"], b.inv_inertia[i]),
                     lcx=(r["local_cx"], b.local_center[i, 0]), lcy=(r["local_cy"], b.local_center[i, 1]))
        for k, (u, v) in pairs.items():
            if F32(u).tobytes() != F32(v).tobytes() and not (u == 0 and v == 0):
                bad.append(f"body {i} {k}: {u!r} vs {v!r}")
        if bool(r["awake"]) != bool(b.awake[i]):
            bad.append(f"body {i} awake")
    m_ = jax.tree.map(np.asarray, twin.model)
    for f in range(twin.ref.counts()["fixtures"]):
        s = twin.ref.fixture_shape(f)
        n = s["count"]
        if n != m_.count[f] or F32(s["radius"]) != m_.radius[f]:
            bad.append(f"fixture {f} count/radius")
            continue
        if not np.array_equal(s["vertices"], m_.verts[f, :n]):
            bad.append(f"fixture {f} vertices {s['vertices']} vs {m_.verts[f, :n]}")
        if m_.shape_type[f] == 2 and not np.array_equal(s["normals"], m_.normals[f, :n]):
            bad.append(f"fixture {f} normals {s['normals']} vs {m_.normals[f, :n]}")
    return bad
