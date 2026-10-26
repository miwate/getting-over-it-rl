"""Engine constants: Box2D 2.3.1's b2Settings plus the Physics2D values Getting Over It sets.

Every float is the float32 Box2D itself would hold, computed the way its source computes it, so the JAX step and the
oracle (goi_jax/reference, built from third_party/box2d-2.3.1) start from bit-identical constants:
- stock values are b2Settings.h v2.3.1 expressions evaluated in float32 (e.g. b2_angularSlop = 2.0f / 180.0f * b2_pi);
- game values are dumps/engine-dump-start.jsonl row 1 (docs/understanding/overview.md §1.5), converted as
  goi_jax/reference/box2d.py Settings.game() does (degrees * Mathf.Deg2Rad in float32).
Settings is a frozen dataclass of Python scalars and numpy float32: hashable, so it is a static jit argument.
"""
from dataclasses import dataclass, replace

import numpy as np

F32 = np.float32

# b2Settings.h v2.3.1 compile-time constants (not exposed by Unity).
PI = F32(3.14159265359)  # b2_pi
EPSILON = F32(np.finfo(np.float32).eps)  # b2_epsilon = FLT_EPSILON
MAX_FLOAT = F32(np.finfo(np.float32).max)  # b2_maxFloat = FLT_MAX
MAX_MANIFOLD_POINTS = 2  # b2_maxManifoldPoints
MAX_POLYGON_VERTICES = 8  # b2_maxPolygonVertices
AABB_EXTENSION = F32(0.1)  # b2_aabbExtension (fat AABB margin)
AABB_MULTIPLIER = F32(2.0)  # b2_aabbMultiplier (predictive fat AABB growth along the displacement)
MAX_SUB_STEPS = 8  # b2_maxSubSteps (TOI)
MAX_TOI_CONTACTS = 32  # b2_maxTOIContacts
BLOCK_SOLVE = True  # g_blockSolve (b2ContactSolver.cpp)
MAX_CONDITION_NUMBER = F32(1000.0)  # k_maxConditionNumber, 2-point block solver
K_TOL = F32(0.1)  # b2CollidePolygons k_tol = 0.1f * b2_linearSlop


def deg2rad(deg):
    """Unity's degree conversion, value * Mathf.Deg2Rad in float32 (same as goi_jax/reference/box2d.py)."""
    return F32(F32(deg) * F32(0.0174532924))


def _stock_angle(deg):
    """b2Settings.h form `(deg / 180.0f * b2_pi)`, left to right in float32."""
    return F32(F32(F32(deg) / F32(180.0)) * PI)


@dataclass(frozen=True)
class Settings:
    """World constants (b2Settings globals plus b2World flags). Defaults are stock Box2D 2.3.1; Settings.game() is
    Getting Over It. max_translation / max_rotation are Box2D's per-step clamps (Unity's per-second reading, T1, is
    clamps_per_second=True, which multiplies them by dt as the oracle does)."""
    gravity: tuple = (F32(0.0), F32(-10.0))
    velocity_iterations: int = 8
    position_iterations: int = 3
    linear_slop: F32 = F32(0.005)
    angular_slop: F32 = _stock_angle(2.0)
    polygon_radius: F32 = F32(F32(2.0) * F32(0.005))
    velocity_threshold: F32 = F32(1.0)
    max_linear_correction: F32 = F32(0.2)
    max_angular_correction: F32 = _stock_angle(8.0)
    max_translation: F32 = F32(2.0)
    max_rotation: F32 = F32(F32(0.5) * PI)
    baumgarte: F32 = F32(0.2)
    toi_baumgarte: F32 = F32(0.75)
    time_to_sleep: F32 = F32(0.5)
    linear_sleep_tolerance: F32 = F32(0.01)
    angular_sleep_tolerance: F32 = _stock_angle(2.0)
    clamps_per_second: bool = False
    allow_sleep: bool = True
    warm_starting: bool = True
    continuous_physics: bool = False  # b2World::SolveTOI (toi.py); Box2D's default is on, off keeps the discrete step
    unity_toi: bool = False  # Unity's TOI pair rule (reference patch b2UnityTOIPair); False = stock Box2D's
    # SolveTOI's min-TOI search when several contacts share the exact minimum alpha: False = stock Box2D (the first in
    # contact-list order, i.e. the newest, strict `<`), True = the last (the oldest, as `<=`). Game recordings: of 26
    # exact ties (two pieces sharing the hitting vertex, or duplicate rocks) the game took the oldest 20 times, the
    # newest 5, and 1 is ambiguous (two of three tied contacts got events) (goi_jax/parity/diagnose_toi.py --ties). The
    # tied contacts' order in the game's own recorded contact list equals the engine's in all 26, so list order does
    # not explain the exceptions; Unity's real rule is unknown. True is the better approximation (single-step parity:
    # 11 TOI steps fixed, 1 broken). The CPU oracle (goi_jax/reference) keeps the stock rule.
    toi_tie_oldest: bool = False
    auto_clear_forces: bool = True
    # b2RevoluteJoint solver (joints.py): True Box2D v2.4.1's, which Unity's Physics2D grafted onto 2.3 (the game;
    # docs/understanding/native-box2d.md section 4), False stock 2.3.1's. Default True in stock() too.
    revolute_v24: bool = True
    # Island solver: "xla" (step.solve) or "pallas" (goi_jax/box2d/gpu: one kernel, one world per lane; same floats).
    solver: str = "xla"

    @classmethod
    def stock(cls, **kw):
        return replace(cls(), **{k: _cast(v) for k, v in kw.items()})

    @classmethod
    def game(cls, **kw):
        """Getting Over It's Physics2D settings (overview §1.5), as goi_jax/reference/box2d.py Settings.game()."""
        s = cls(gravity=(F32(0.0), F32(-30.0)), velocity_iterations=300, position_iterations=30,
                velocity_threshold=F32(1.0), max_linear_correction=F32(0.2), max_angular_correction=deg2rad(8),
                max_translation=F32(100.0), max_rotation=deg2rad(360), baumgarte=F32(0.4), toi_baumgarte=F32(0.75),
                time_to_sleep=F32(0.5), linear_sleep_tolerance=F32(0.01), angular_sleep_tolerance=deg2rad(2),
                linear_slop=F32(0.005), polygon_radius=F32(0.01), continuous_physics=True, unity_toi=True,
                toi_tie_oldest=True)
        return replace(s, **{k: _cast(v) for k, v in kw.items()})

    def reference_kwargs(self):
        """The same configuration for goi_jax.reference.box2d.Settings (plus trig_mode=1, the rounding ported here)."""
        names = ("linear_slop", "angular_slop", "polygon_radius", "velocity_threshold", "max_linear_correction",
                 "max_angular_correction", "max_translation", "max_rotation", "baumgarte", "toi_baumgarte",
                 "time_to_sleep", "linear_sleep_tolerance", "angular_sleep_tolerance")
        kw = {n: float(getattr(self, n)) for n in names}
        kw.update(gravity=tuple(float(g) for g in self.gravity), velocity_iterations=self.velocity_iterations,
                  position_iterations=self.position_iterations, clamps_per_second=self.clamps_per_second,
                  allow_sleep=self.allow_sleep, warm_starting=self.warm_starting, trig_mode=1,
                  revolute_v24=self.revolute_v24, continuous_physics=self.continuous_physics, unity_toi=self.unity_toi)
        return kw


def _cast(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, int, np.integer)) and not isinstance(v, (float, np.floating)):
        return v
    if isinstance(v, tuple):
        return tuple(F32(x) for x in v)
    return F32(v)
