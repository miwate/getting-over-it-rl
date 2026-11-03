"""The Box2D 2.3.1 oracle (goi_jax/reference): integration rule, contacts, joints, the goi patches, the dump scene.
CPU only, a few seconds. Builds build/box2d/libgoi_box2d.so on first use (scripts/build_box2d.sh)."""
import numpy as np
import pytest

from goi_jax.reference import box2d as b2
from goi_jax.reference import scene

F32 = np.float32
H = b2.GAME_DT


def resting_box(settings=None, y=1.015):
    w = b2.World(settings or b2.Settings.game())
    ground = w.add_body(b2.STATIC)
    w.add_box(ground, (10, 0.5), friction=1.0)
    box = w.add_body(b2.DYNAMIC, (0, y), mass=2, inertia=F32(2) / F32(6))
    fix = w.add_box(box, (0.5, 0.5), friction=0.5)
    return w, box, fix


@pytest.mark.parametrize("drag", [0.0, 0.2])
def test_free_fall_is_semi_implicit_euler_bit_exact(drag):
    w = b2.World(b2.Settings.game())
    g = F32(-30)
    body = w.add_body(b2.DYNAMIC, (1.25, 7.5), velocity=(3, 2), mass=4, inertia=1, linear_damping=drag)
    x, y, vx, vy = F32(1.25), F32(7.5), F32(3), F32(2)
    damp = F32(1) / (F32(1) + H * F32(drag))  # 2.3.1: v *= 1 / (1 + h * c)
    for _ in range(300):
        w.step(H)
        vx = (vx + H * (F32(1) * F32(0) + F32(0))) * damp
        vy = (vy + H * (F32(1) * g + F32(0))) * damp
        x, y = x + H * vx, y + H * vy
        s = w.body_array(body)
        assert (s[0], s[1], s[5], s[6]) == (x, y, vx, vy)


def test_box_rests_and_sleeps():
    w, box, _ = resting_box()
    start = w.body_array(box)
    for _ in range(240):
        w.step(H)
    end = w.body(box)
    assert abs(end["x"] - start[0]) < 1e-5 and abs(end["angle"]) < 1e-5
    assert abs(end["y"] - start[1]) < 1e-3  # only settles into the linear-slop overlap
    assert not end["awake"]  # timeToSleep 0.5 s at rest


def test_contact_impulses_carry_the_weight():
    w, box, fix = resting_box()
    for _ in range(30):
        w.step(H)
    cs = w.contacts(touching_only=True)
    assert len(cs) == 1
    c = cs[0]
    assert c["manifold_type"] in (1, 2) and len(c["points"]) == 2
    assert len({p["key"] for p in c["points"]}) == 2  # distinct feature ids
    total = sum(p["normal_impulse"] for p in c["points"])
    assert total == pytest.approx(2 * 30 * H, rel=1e-3)  # m g dt per step at rest
    assert c["friction"] == pytest.approx(np.sqrt(F32(0.5) * F32(1.0)), rel=1e-6)
    assert -0.02 < min(p["separation"] for p in c["points"]) < 0  # rests inside the 2 x 0.01 skin
    posts = [e for e in w.events() if e[0] == "post_solve"]
    assert posts and float(posts[-1][5].sum()) == pytest.approx(float(total), rel=1e-6)


def test_set_friction_refresh():
    w, box, fix = resting_box()
    for _ in range(5):
        w.step(H)
    w.set_friction(fix, 0.0)
    assert w.contacts()[0]["friction"] == pytest.approx(np.sqrt(0.5), rel=1e-6)  # Box2D keeps the mixed value
    w.set_friction(fix, 0.25, refresh_contacts=True)
    assert w.contacts()[0]["friction"] == pytest.approx(0.5, rel=1e-6)  # sqrt(0.25 * 1)


def test_layer_matrix_filters_pairs():
    masks = [-1] * 32
    masks[3] &= ~(1 << 5)
    masks[5] &= ~(1 << 3)
    w = b2.World(b2.Settings.game(layer_masks=masks))
    ground = w.add_body(b2.STATIC)
    w.add_box(ground, (10, 0.5), layer=3)
    box = w.add_body(b2.DYNAMIC, (0, 1.015), mass=1, inertia=0.1)
    w.add_box(box, (0.5, 0.5), layer=5)
    for _ in range(30):
        w.step(H)
    assert w.counts()["contacts"] == 0 and w.body(box)["y"] < 0.9  # falls through


def test_pendulum_conserves_energy_and_joint():
    w = b2.World(b2.Settings.game())
    pivot = w.add_body(b2.STATIC)
    L, m, inertia = 2.0, 1.0, 0.01
    bob = w.add_body(b2.DYNAMIC, (L, 0), mass=m, inertia=inertia)
    j = w.add_revolute(pivot, bob, (0, 0), (-L, 0))
    lowest = None
    for i in range(240):
        w.step(H)
        s = w.body(bob)
        r = np.hypot(s["x"], s["y"])
        assert abs(r - L) < 1e-3
        if lowest is None and s["x"] <= 0:
            lowest = s
    assert lowest is not None
    v = np.hypot(lowest["vx"], lowest["vy"])
    # Semi-implicit Euler plus the joint's position projection bleeds a little energy each step (~2% here).
    assert v == pytest.approx(np.sqrt(2 * 30 * L / (1 + inertia / (m * L * L))), rel=0.03)
    info = w.joint(j)
    assert info["type"] == b2.REVOLUTE and np.isfinite([info["fx"], info["fy"]]).all()


def test_revolute_limit_and_motor():
    w = b2.World(b2.Settings.game(gravity=(0, 0)))
    pivot = w.add_body(b2.STATIC)
    arm = w.add_body(b2.DYNAMIC, (1, 0), mass=1, inertia=0.1)
    j = w.add_revolute(pivot, arm, (0, 0), (-1, 0), limits=(-0.5, 0.5), motor_speed=2.0, max_motor_torque=1000,
                       enable_motor=True)
    for _ in range(12):
        w.step(H)
    assert w.joint(j)["speed"] == pytest.approx(2.0, rel=1e-4)
    for _ in range(120):
        w.step(H)
    info = w.joint(j)
    assert info["limit_state"] == 2 and info["value"] == pytest.approx(0.5, abs=b2.deg2rad(2))
    w.set_motor_speed(j, -2.0)
    w.step(H)
    assert w.joint(j)["motor_speed"] == F32(-2.0)


def _bullet_through_wall(**kw):
    s = b2.Settings.game(gravity=(0, 0))
    w = b2.World(s)
    wall = w.add_body(b2.STATIC, (0, 0))
    w.add_box(wall, (0.05, 2))
    ball = w.add_body(b2.DYNAMIC, (-1, 0), velocity=(240, 0), mass=1, inertia=0.01, **kw)
    w.add_circle(ball, 0.05)
    for _ in range(3):
        w.step(H)
    return w.body(ball)["x"]


def test_unity_toi_rule():
    w = b2.World(b2.Settings.stock())
    assert w.settings.resolved("baumgarte") == F32(0.2)  # stock value read back from the library
    # Stock 2.3.1: TOI for every dynamic-vs-static pair. Unity: only for Continuous bodies.
    assert _bullet_through_wall() > 0  # game settings use the Unity rule: discrete body tunnels
    assert _bullet_through_wall(continuous=True) < 0
    w = b2.World(b2.Settings.game(gravity=(0, 0), unity_toi=False))
    wall = w.add_body(b2.STATIC)
    w.add_box(wall, (0.05, 2))
    ball = w.add_body(b2.DYNAMIC, (-1, 0), velocity=(240, 0), mass=1, inertia=0.01)
    w.add_circle(ball, 0.05)
    for _ in range(3):
        w.step(H)
    assert w.body(ball)["x"] < 0


def test_dump_scene_loads_and_steps():
    if not scene.DEFAULT_DUMP.exists():
        pytest.skip("no dumps/engine-dump-start.jsonl")
    sc = scene.load_scene()
    w = sc.world
    assert len(sc.body_rows) == 16 and len(sc.joint_rows) == 9
    assert w.counts()["bodies"] == 17  # plus the static ground body
    assert w.counts()["joints"] == 8  # the disabled Handle hinge is not built
    # Mapping check: Box2D's joint coordinates at load equal what Unity reported.
    slider = sc.joint_id("Player/Hub/Slider", "SliderJoint2D")
    assert w.joint(slider)["value"] == pytest.approx(-0.356999725, abs=1e-6)
    assert w.joint(sc.joint_id("Player/Hub", "HingeJoint2D"))["value"] == pytest.approx(0, abs=1e-6)
    tip = sc.body_id("Player/Hub/Slider/Handle/PoleMiddle/Tip")
    assert w.body(tip)["bullet_continuous"] == 2 and w.body(tip)["inertia"] == F32(0.0572354943)
    for _ in range(120):
        w.step(H)
        assert np.isfinite(w.bodies_state()).all()
    assert w.counts()["contacts"] > 0
