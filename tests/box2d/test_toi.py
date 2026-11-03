"""Continuous collision (goi_jax/box2d/toi.py) against the Box2D 2.3.1 oracle with continuous physics on.

Every scene is stepped in both worlds and compared bit for bit at every step: the harness quantities (bodies, sweep
centres, contact list, manifolds, impulses, joints, awake flags) plus the TOI state Box2D leaves behind: each body's
sweep c0 / a0 / alpha0 and each contact's toiCount, e_toiFlag, e_enabledFlag and cached toi (where flagged).
CPU; one compile per scene.
"""
import jax
import numpy as np
import pytest

from goi_jax import box2d as jb
from goi_jax.box2d.world import DYNAMIC, KINEMATIC, STATIC

from box2d_harness import H, QUANTITIES, Twin, compare, jax_snapshot, ref_snapshot

F32 = np.float32


def _twin(unity=True, gravity=(0.0, -30.0), capacity=16, **kw):
    s = jb.Settings.game(unity_toi=unity, gravity=gravity, **kw)
    return Twin(s, contact_capacity=capacity, continuous_physics=True, unity_toi=unity)


def ball_wall():
    """A continuous ball at 900 m/s (7.5 m per step) towards a 0.1 m static wall: stopped, not tunnelled."""
    t = _twin(gravity=(0.0, 0.0))
    w = t.add_body(STATIC, (5.0, 0.0))
    t.add_box(w, (0.05, 2.0))
    b = t.add_body(DYNAMIC, (0.0, 0.1), 0.0, (900.0, 3.0), 0.0, mass=1.0, inertia=0.005, continuous=True)
    t.add_circle(b, 0.1)
    d = t.add_body(DYNAMIC, (0.0, -1.0), 0.0, (900.0, 0.0), 0.0, mass=1.0, inertia=0.005)  # discrete: tunnels
    t.add_circle(d, 0.1)
    return t


def poly_angles():
    """Fast continuous boxes and a triangle hitting static polygons at several angles and spins."""
    t = _twin(capacity=24)
    g = t.add_body(STATIC)
    t.add_box(g, (30.0, 0.5), friction=0.8)
    t.add_polygon(g, [(10, 0.5), (16, 0.5), (16, 3.0)], friction=0.6)
    for i, (ang, vel, w) in enumerate([(0.0, (0, -400), 0.0), (0.4, (60, -350), 3.0), (1.1, (-80, -500), -8.0),
                                        (2.5, (200, -300), 20.0)]):
        b = t.add_body(DYNAMIC, (-12.0 + 6 * i, 4.0 + 0.3 * i), ang, vel, w, mass=1.0 + 0.2 * i,
                       inertia=0.1 + 0.02 * i, continuous=True)
        t.add_box(b, (0.3, 0.12 + 0.03 * i), friction=0.5)
    tri = t.add_body(DYNAMIC, (13.0, 6.0), 0.3, (20, -420), 5.0, mass=0.8, inertia=0.05, continuous=True)
    t.add_polygon(tri, [(0.3, 0.0), (-0.2, 0.25), (-0.2, -0.2)], friction=0.9)
    return t


def hammer_chain():
    """A pivot (kinematic) - arm - head chain on revolute joints, the head Continuous, driven by a fast motor so the
    head swings into the terrain (the hammer Tip case: TOI moves the head without its joints)."""
    t = _twin(capacity=16)
    g = t.add_body(STATIC)
    t.add_box(g, (10.0, 0.5), friction=0.7, layer=0)
    t.add_polygon(g, [(-1.0, 0.5), (1.5, 0.5), (0.2, 1.4)], friction=0.7)
    hub = t.add_body(KINEMATIC, (0.0, 2.6))
    t.add_circle(hub, 0.05, group=-1)
    arm = t.add_body(DYNAMIC, (0.9, 2.6), 0.0, mass=2.0, center=(0.0, 0.0), inertia=0.3)
    t.add_box(arm, (0.9, 0.04), group=-1)
    head = t.add_body(DYNAMIC, (1.9, 2.6), 0.0, mass=0.5, inertia=0.004, continuous=True)
    t.add_box(head, (0.1, 0.06), friction=0.7, group=-1)
    t.add_revolute(hub, arm, (0.0, 0.0), (-0.9, 0.0), motor_speed=-40.0, max_motor_torque=5000.0, enable_motor=True)
    t.add_revolute(arm, head, (0.9, 0.0), (-0.1, 0.0), limits=(-0.3, 0.3))
    return t


def multi_event():
    """Several TOI events in one step: two continuous balls each hitting thin walls, one zig-zagging between two
    close walls with restitution (several sub-steps of the same pair and of different pairs)."""
    t = _twin(gravity=(0.0, 0.0), capacity=16)
    w = t.add_body(STATIC)
    t.add_box(w, (0.05, 3.0), (3.0, 0.0), restitution=1.0)
    t.add_box(w, (0.05, 3.0), (3.6, 0.0), restitution=1.0)
    t.add_box(w, (0.05, 3.0), (-3.0, 5.0))
    a = t.add_body(DYNAMIC, (3.3, 0.0), 0.0, (300.0, 40.0), 1.0, mass=1.0, inertia=0.01, continuous=True)
    t.add_circle(a, 0.08, restitution=1.0, friction=0.0)  # no friction: tangential decay into denormals (XLA flushes them)
    b = t.add_body(DYNAMIC, (0.0, 5.0), 0.3, (-700.0, -20.0), 4.0, mass=1.0, inertia=0.02, continuous=True)
    t.add_box(b, (0.1, 0.1))
    return t


def stock_bullets():
    """Stock pair rule (unity_toi off): a fast bullet against a dynamic box and the ground, and an ordinary dynamic
    body against the ground (stock Box2D gives every dynamic-static pair TOI)."""
    t = _twin(unity=False, capacity=16)
    g = t.add_body(STATIC)
    t.add_box(g, (10.0, 0.5))
    box = t.add_body(DYNAMIC, (3.0, 1.1), mass=4.0, inertia=0.6)
    t.add_box(box, (0.5, 0.5))
    bul = t.add_body(DYNAMIC, (-3.0, 1.2), 0.0, (800.0, 0.0), 0.0, mass=0.2, inertia=0.001, bullet=True)
    t.add_circle(bul, 0.05)
    fall = t.add_body(DYNAMIC, (-6.0, 6.0), 0.2, (0.0, -600.0), 0.0, mass=1.0, inertia=0.1)
    t.add_box(fall, (0.2, 0.2))
    return t


def island_walk():
    """The TOI island walk: a continuous box sliding on the ground (already touching) into a two-piece wall, a second
    continuous box, a plain prop in the path and a kinematic paddle (several contacts per TOI island, up to 4)."""
    t = _twin(capacity=32)
    g = t.add_body(STATIC)
    t.add_box(g, (20.0, 0.5), friction=0.6)
    t.add_box(g, (0.05, 2.0), (6.0, 2.5))
    t.add_box(g, (0.05, 2.0), (6.3, 2.5))
    k = t.add_body(KINEMATIC, (-6.0, 1.5), 0.0, (0.0, 0.0), 0.0)
    t.add_box(k, (0.05, 1.0))
    b = t.add_body(DYNAMIC, (0.0, 0.6), 0.0, (500.0, 0.0), 0.0, mass=1.0, inertia=0.05, continuous=True)
    t.add_box(b, (0.2, 0.1), friction=0.5)
    c = t.add_body(DYNAMIC, (-1.0, 0.75), 0.0, (-450.0, 0.0), 0.0, mass=1.0, inertia=0.05, continuous=True)
    t.add_box(c, (0.2, 0.25), friction=0.5)
    s = t.add_body(DYNAMIC, (3.0, 0.62), 0.0, (0.0, 0.0), 0.0, mass=1.0, inertia=0.05)
    t.add_box(s, (0.1, 0.1))
    return t


SCENES = dict(ball_wall=ball_wall, poly_angles=poly_angles, hammer_chain=hammer_chain, multi_event=multi_event,
              stock_bullets=stock_bullets, island_walk=island_walk)


def _toi_issues(ref, st):
    issues = []
    st = jax.tree.map(np.asarray, st)
    b, tt, ct = st.bodies, st.toi, st.contacts
    n = ref.counts()["bodies"]
    for i in range(n):
        r = ref.body(i)
        for name, a, j in (("c0x", r["c0x"], b.c0[i, 0]), ("c0y", r["c0y"], b.c0[i, 1]), ("a0", r["a0"], b.a0[i]),
                           ("alpha0", r["alpha0"], tt.alpha0[i])):
            if F32(a).tobytes() != F32(j).tobytes() and not (a == 0 and j == 0):
                issues.append(f"body {i} {name} {a!r} vs {j!r}")
    slots = {(int(ct.fa[k]), int(ct.fb[k])): k for k in np.nonzero(ct.pair >= 0)[0]}
    for c in ref.contacts():
        k = slots.get(tuple(c["fixtures"]))
        if k is None:
            continue
        flag = bool(c["flags"] & 0x20)
        got = (int(tt.count[k]), bool(tt.flag[k]), bool(tt.enabled[k]))
        if got != (c["toi_count"], flag, c["enabled"]):
            issues.append(f"contact {c['fixtures']} toi count/flag/enabled {got} vs {c['toi_count'], flag, c['enabled']}")
        elif flag and F32(c["toi"]).tobytes() != F32(tt.toi[k]).tobytes():
            issues.append(f"contact {c['fixtures']} toi {c['toi']!r} vs {tt.toi[k]!r}")
    return issues


def run_toi(twin, steps):
    twin.build()
    st = twin.state
    errors, issues, events, most, counts, solved = [], [], 0, 0, 0, 0
    for k in range(steps):
        st = twin._step(st)
        twin.ref.step(H)
        e, iss = compare(ref_snapshot(twin.ref), jax_snapshot(st))
        iss += _toi_issues(twin.ref, st)
        events += int(st.toi.events)
        most = max(most, int(st.toi.events))
        counts = max(counts, int(np.max(np.asarray(st.toi.count), initial=0)))
        solved = max(solved, int(np.sum(np.asarray(st.toi.solved))))
        errors.append(e)
        issues += [(k + 1, x) for x in iss]
    return dict(errors=errors, issues=issues, events=events, most_events=most, max_toi_count=counts,
                most_solved=solved, state=st)


@pytest.mark.parametrize("name", list(SCENES))
def test_toi_parity(name):
    res = run_toi(SCENES[name](), 90)
    assert res["issues"][:5] == []
    for q in QUANTITIES:
        assert max(e[q][1] for e in res["errors"]) == 0, q
    assert res["events"] > 0  # the scene exercises TOI
    if name == "multi_event":
        assert res["most_events"] > 2 and res["max_toi_count"] > 1  # several events, same pair sub-stepped
    if name == "island_walk":
        assert res["most_solved"] > 2  # TOI islands with walk-added contacts


def test_ball_stopped_by_wall():
    twin = ball_wall()
    res = run_toi(twin, 3)
    x = np.asarray(res["state"].bodies.c[:, 0])
    assert x[1] < 5.0 and x[2] > 5.0  # continuous ball stopped, discrete ball tunnelled


def test_default_settings_skip_toi():
    """Settings() keeps continuous physics off: the step never touches the TOI state."""
    s = jb.Settings()
    assert not s.continuous_physics and jb.Settings.game().continuous_physics and jb.Settings.game().unity_toi
