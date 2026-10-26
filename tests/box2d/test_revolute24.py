"""The v2.4.1 revolute solver (Unity's graft onto Box2D 2.3, docs/understanding/native-box2d.md section 4): the oracle
patch (scripts/patches/box2d-2.3.1-revolute24.patch), the JAX port (goi_jax/box2d/joints.py, Settings.revolute_v24)
and the game's native joint fields in the parity recordings.

- Twin scenes with hinges are bit-identical JAX vs oracle at the game's 300/30 iterations with either solver
  (lower/upper limit impulses included, via box2d_harness's limit_impulse quantity).
- The C API's joint frame and solver readouts match the JAX model and the v2.4 formulas.
- Every solved hinge in dumps/parity (4200 recorded steps) stores exactly what v2.4.1's InitVelocityConstraints
  computes from the recorded step-start bodies: m_angle, m_axialMass, m_K, rA, rB; impulse.z is never written; and
  the pot hinge carries a lower-limit impulse while still above its limit (v2.4's speculative row, impossible in
  2.3.1), with the converged velocity landing exactly on the limit.
CPU, about 2-3 minutes.
"""
import math
from pathlib import Path

import numpy as np
import pytest

import box2d_scenes
from box2d_harness import QUANTITIES, Twin, initial_mismatches, run, summarize
from goi_jax import box2d as jb
from goi_jax.reference import box2d as b2

F32 = np.float32
H = b2.GAME_DT
HINGE_SCENES = ("revolute", "limits_mid_run", "player_rig")
PARITY_ROOT = Path(__file__).resolve().parents[2] / "dumps" / "parity"


def _scene(name, settings):
    """A scene from box2d_scenes built with the given settings (the scene functions construct Twin themselves)."""
    orig = box2d_scenes.Twin
    box2d_scenes.Twin = lambda *a, **k: Twin(*a, settings=settings, **k)
    try:
        return box2d_scenes.SCENES[name]()
    finally:
        box2d_scenes.Twin = orig


def _assert_bit_exact(twin, steps=120):
    assert initial_mismatches(twin) == []
    res = run(twin, steps)
    assert res["issues"] == []
    summ = summarize(res)
    for q in QUANTITIES:
        assert summ[q]["max"] == (0.0, 0), (q, summ[q])
    return res


def test_solver_switch_reaches_the_library():
    lib = b2.load()
    v = (b2.C.c_float * 1)()
    for flag in (False, True):
        w = b2.World(b2.Settings.game(revolute_v24=flag))
        w.step(H)
        assert lib.goi_b2_get_param(b"revoluteSolver", v) == 0 and v[0] == float(flag)
    assert jb.Settings.game().revolute_v24 and jb.Settings().revolute_v24  # the game's solver by default
    assert jb.Settings.game(revolute_v24=False).reference_kwargs()["revolute_v24"] is False


@pytest.mark.parametrize("v24", [True, False], ids=["v24", "v231"])
@pytest.mark.parametrize("name", HINGE_SCENES)
def test_hinge_scene_parity(name, v24):
    twin = _scene(name, jb.Settings.game(revolute_v24=v24))
    if twin is None:
        pytest.skip("no dumps/engine-dump-start.jsonl")
    res = _assert_bit_exact(twin)
    if name == "revolute":
        st = res["state"].joints
        if v24:
            # both limit rows got used, and the 2.3.1 limit slot (z) stayed 0
            assert np.asarray(st.lower_impulse).max() > 0 or np.asarray(st.upper_impulse).max() > 0
            assert np.all(np.asarray(st.impulse)[:, 2] == 0)
        else:
            assert np.all(np.asarray(st.lower_impulse) == 0) and np.all(np.asarray(st.upper_impulse) == 0)


def test_solvers_differ():
    """The switch changes the physics (the revolute scene drives a motor into its limit)."""
    out = []
    for v24 in (True, False):
        twin = _scene("revolute", jb.Settings.game(revolute_v24=v24))
        for _ in range(60):
            twin.ref.step(H)
        out.append(twin.ref.bodies_state())
    assert not np.array_equal(out[0], out[1])


def test_limit_impulses_every_step():
    """lower/upper impulses at every step of the two limit scenes: non-negative, z and the reaction torque stay 0
    (GetReactionTorque reads z, as Unity's does), and both rows get used somewhere (the motor-driven pendulum sits on
    its upper limit; limits_mid_run's motor reverses into its lower limit)."""
    lo_max = up_max = F32(0)
    for name in ("revolute", "limits_mid_run"):
        twin = _scene(name, jb.Settings.game())
        w = twin.ref
        hinges = [j for j in range(w.counts()["joints"]) if w.joint(j)["type"] == b2.REVOLUTE]
        for _ in range(120):
            w.step(H)
            for j in hinges:
                info = w.joint(j)
                assert info["iz"] == 0 and info["torque"] == 0
                assert info["lower_impulse"] >= 0 and info["upper_impulse"] >= 0
                lo_max, up_max = max(lo_max, info["lower_impulse"]), max(up_max, info["upper_impulse"])
    assert lo_max > 0 and up_max > 0


def test_joint_frame_and_solver_readouts():
    twin = _scene("player_rig", jb.Settings.game())
    if twin is None:
        pytest.skip("no dumps/engine-dump-start.jsonl")
    twin.build()
    m = twin.model
    w = twin.ref
    for j in range(w.counts()["joints"]):
        f = w.joint_frame(j)
        assert np.array_equal(f["local_anchor_a"], np.asarray(m.local_anchor_a[j]))
        assert np.array_equal(f["local_anchor_b"], np.asarray(m.local_anchor_b[j]))
        assert f["reference_angle"] == np.asarray(m.reference_angle[j])
        if int(m.joint_type[j]) == b2.PRISMATIC:
            assert np.array_equal(f["local_axis"], np.asarray(m.local_axis_a[j]))
            assert np.array_equal(f["local_perp"], np.asarray(m.local_yaxis_a[j]))
    before = [w.body(i) for i in range(w.counts()["bodies"])]
    w.step(H)
    for j in range(w.counts()["joints"]):
        info = w.joint(j)
        if info["type"] != b2.REVOLUTE:
            continue
        s = w.joint_solver(j)
        A, B = before[info["body_a"]], before[info["body_b"]]
        assert s["angle"] == F32(F32(B["angle"] - A["angle"]) - w.joint_frame(j)["reference_angle"])
        iA, iB = F32(A["inv_inertia"]), F32(B["inv_inertia"])
        assert s["axial_mass"] == (F32(1) / F32(iA + iB) if iA + iB > 0 else 0)
        assert (s["lower_impulse"], s["upper_impulse"]) == (info["lower_impulse"], info["upper_impulse"])
        assert s["K"][1] == s["K"][2]


# ---- the game's native joints (dumps/parity recordings)

_J = 0x98  # derived b2Joint fields start (goi_jax/parity/states.py JOINT_FIELDS)
_SOLVER_FIELDS = {"rA": (_J + 0x44, "<f4", 2), "rB": (_J + 0x4C, "<f4", 2), "lcA": (_J + 0x54, "<f4", 2),
                  "lcB": (_J + 0x5C, "<f4", 2), "inv": (_J + 0x64, "<f4", 4), "K": (_J + 0x74, "<f4", 4)}


def _rot(angle, v):
    """b2Mul(b2Rot(angle), v) with correctly rounded sin/cos (trig_mode 1)."""
    s, c = F32(math.sin(float(angle))), F32(math.cos(float(angle)))
    return np.array([F32(F32(c * v[0]) - F32(s * v[1])), F32(F32(s * v[0]) + F32(c * v[1]))], F32)


def _k24(inv, rA, rB):
    mA, mB, iA, iB = inv
    exx = F32(F32(F32(mA + mB) + F32(F32(rA[1] * rA[1]) * iA)) + F32(F32(rB[1] * rB[1]) * iB))
    eyx = F32(F32(F32(-rA[1] * rA[0]) * iA) - F32(F32(rB[1] * rB[0]) * iB))
    eyy = F32(F32(F32(mA + mB) + F32(F32(rA[0] * rA[0]) * iA)) + F32(F32(rB[0] * rB[0]) * iB))
    return np.array([exx, eyx, eyx, eyy], F32)


def _native_hinge_rows():
    """(joint name, step-start bodies A/B, stored joint fields after Simulate) for every solved hinge step."""
    from goi_jax.parity.states import decode, iter_chains
    for _, c in iter_chains(PARITY_ROOT):
        names = c.names["body_names"]
        ex = decode(c.joints_raw[1], _SOLVER_FIELDS)
        b0 = [c.body(0, i) for i in range(len(names))]
        b1 = [c.body(1, i) for i in range(len(names))]
        for ji, jname in enumerate(c.names["joints"]):
            if not jname.endswith("HingeJoint2D"):
                continue
            A = names.index(jname.split("#")[0])  # Unity: bodyA is the joint's own Rigidbody2D
            hit = c.joint_index[1] == ji
            slot = np.where(hit.any(1), hit.argmax(1), -1)
            jf = c.joint(1, ji)
            x = {k: v[np.arange(len(slot)), np.maximum(slot, 0)] for k, v in ex.items()}
            for s in np.nonzero((slot >= 0) & (jf["indexA"] >= 0))[0]:
                inv = x["inv"][s]
                # bodyB: the body whose mass data the joint copied and whose anchor meets A's
                cands = [i for i in range(len(names)) if i != A and b0[i]["invMass"][s] == inv[1]
                         and b0[i]["invI"][s] == inv[3] and np.array_equal(b0[i]["localCenter"][s], x["lcB"][s])]

                def gap(i):
                    def world(b, la):
                        p = b["xf"][s]
                        return np.array([p[0] + p[3] * la[0] - p[2] * la[1], p[1] + p[2] * la[0] + p[3] * la[1]])
                    return np.hypot(*(world(b0[A], jf["localAnchorA"][s]) - world(b0[i], jf["localAnchorB"][s])))
                cands = [i for i in sorted(cands, key=gap)[:1] if gap(i) < 0.05]
                if not cands:
                    continue
                B = cands[0]
                yield jname, {k: v[s] for k, v in b0[A].items()}, {k: v[s] for k, v in b0[B].items()}, \
                    {k: v[s] for k, v in b1[A].items()}, {k: v[s] for k, v in b1[B].items()}, \
                    {**{k: v[s] for k, v in jf.items()}, **{k: v[s] for k, v in x.items()}}


@pytest.mark.skipif(not (PARITY_ROOT / "index.json").exists(), reason="no dumps/parity recordings")
def test_native_hinges_store_what_v24_computes():
    n, per_joint, spec = 0, {}, []
    for jname, A0, B0, A1, B1, j in _native_hinge_rows():
        n += 1
        per_joint[jname] = per_joint.get(jname, 0) + 1
        # InitVelocityConstraints from the step-start sweep (NAT0): bit for bit
        assert j["angle"] == F32(F32(B0["a"] - A0["a"]) - j["referenceAngle"]), jname
        iA, iB = j["inv"][2], j["inv"][3]
        assert j["axialMass"] == (F32(1) / F32(iA + iB) if iA + iB > 0 else F32(0)), jname
        assert np.array_equal(j["rA"], _rot(A0["a"], j["localAnchorA"] - j["lcA"])), jname
        assert np.array_equal(j["rB"], _rot(B0["a"], j["localAnchorB"] - j["lcB"])), jname
        assert np.array_equal(j["K"], _k24(j["inv"], j["rA"], j["rB"])), jname
        # v2.4 never writes impulse.z; limit impulses are clamped accumulations; the motor stays within dt * max
        assert j["impulse"][2] == 0 and j["lowerImpulse"] >= 0 and j["upperImpulse"] >= 0, jname
        assert abs(j["motorImpulse"]) <= F32(H * j["maxMotorTorque"]) * (1 + 1e-6), jname
        if not j["enableLimit"]:
            assert j["lowerImpulse"] == 0 and j["upperImpulse"] == 0, jname
        C = F32(j["angle"] - j["lowerAngle"])
        if j["lowerImpulse"] > 0 and C > 0:
            # speculative lower limit: pushed while still above it, so that the step ends exactly on it
            spec.append(float(F32(B1["w"] - A1["w"]) + C / H))
    assert n > 20000 and len(per_joint) >= 5, per_joint
    spec = np.abs(np.array(spec))
    assert len(spec) >= 50, len(spec)
    assert np.mean(spec < 1e-5) > 0.7, np.mean(spec < 1e-5)
