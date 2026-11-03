"""Box2D's b2Math in float32 JAX, one rounding per C operation, in the C evaluation order.

Why every product goes through `mul`: XLA on CPU contracts `a * b + c` into a fused multiply-add (one rounding),
which changed 23% of random float32 `a*b + c` results [measured, jax 0.12.0 CPU]; Box2D is built with
-ffp-contract=off (scripts/build_box2d.sh), as MSVC x64 SSE is assumed to be. `lax.optimization_barrier` and
`reduce_precision` do not stop the contraction; a select on the product does (cuphead-rl sim/common/trig.py,
`rounded`). So: never write `a * b` in engine code that feeds an addition; write `mul(a, b)`.

Vectors are (x, y) tuples of arrays, so each C expression maps to a visible sequence of float32 ops. C's `a + b + c`
is `(a + b) + c`; `-a * b` is `(-a) * b` (equal to -(a*b) bit for bit).

sin/cos are correctly rounded float32 (the oracle's trig_mode=1, scripts/patches/box2d-2.3.1-trig.patch), from
cuphead-rl's float-float `sin_cos` (copied, not imported: that package is a different project).
"""
import jax.numpy as jnp
import numpy as np
from jax import lax

f32, i32, u32 = jnp.float32, jnp.int32, jnp.uint32
F32 = np.float32
_NAN = np.float32(np.nan)
EPS = F32(np.finfo(np.float32).eps)


def rounded(v):
    """v rounded to float32 before any later use (a select keeps XLA from fusing it into an add)."""
    return jnp.where(jnp.isnan(v), _NAN, v)


def mul(a, b):
    return rounded(a * b)


def mul3(a, b, c):
    """C's a * b * c = (a * b) * c."""
    return mul(mul(a, b), c)


def dot(ax, ay, bx, by):
    """b2Dot: a.x * b.x + a.y * b.y."""
    return mul(ax, bx) + mul(ay, by)


def cross(ax, ay, bx, by):
    """b2Cross(vec, vec): a.x * b.y - a.y * b.x."""
    return mul(ax, by) - mul(ay, bx)


def cross_vs(ax, ay, s):
    """b2Cross(a, s) = (s * a.y, -s * a.x)."""
    return mul(s, ay), mul(-s, ax)


def cross_sv(s, ax, ay):
    """b2Cross(s, a) = (-s * a.y, s * a.x)."""
    return mul(-s, ay), mul(s, ax)


def smul(s, x, y):
    """s * v."""
    return mul(s, x), mul(s, y)


def rot(c, s, x, y):
    """b2Mul(q, v) = (q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y)."""
    return mul(c, x) - mul(s, y), mul(s, x) + mul(c, y)


def rot_t(c, s, x, y):
    """b2MulT(q, v) = (q.c * v.x + q.s * v.y, -q.s * v.x + q.c * v.y)."""
    return mul(c, x) + mul(s, y), mul(-s, x) + mul(c, y)


def xf_mul(px, py, c, s, x, y):
    """b2Mul(T, v) = ((q.c * v.x - q.s * v.y) + p.x, (q.s * v.x + q.c * v.y) + p.y)."""
    return (mul(c, x) - mul(s, y)) + px, (mul(s, x) + mul(c, y)) + py


def xf_mul_t(px, py, c, s, x, y):
    """b2MulT(T, v): d = v - p, then (q.c * d.x + q.s * d.y, -q.s * d.x + q.c * d.y)."""
    dx, dy = x - px, y - py
    return mul(c, dx) + mul(s, dy), mul(-s, dx) + mul(c, dy)


def rot_mul_t(qc, qs, rc, rs):
    """b2MulT(q, r): s = q.c * r.s - q.s * r.c, c = q.c * r.c + q.s * r.s. Returns (c, s)."""
    return mul(qc, rc) + mul(qs, rs), mul(qc, rs) - mul(qs, rc)


def xf_mul_t_xf(apx, apy, ac, as_, bpx, bpy, bc, bs):
    """b2MulT(A, B) for transforms: q = A.q^T B.q, p = A.q^T (B.p - A.p). Returns (px, py, c, s)."""
    c, s = rot_mul_t(ac, as_, bc, bs)
    px, py = rot_t(ac, as_, bpx - apx, bpy - apy)
    return px, py, c, s


def length(x, y):
    """b2Vec2::Length: sqrtf(x * x + y * y)."""
    return jnp.sqrt(mul(x, x) + mul(y, y))


def normalize(x, y):
    """b2Vec2::Normalize: unchanged (length 0 returned) below b2_epsilon, else times 1 / length."""
    n = length(x, y)
    small = n < EPS
    inv = F32(1.0) / jnp.where(small, F32(1.0), n)
    return jnp.where(small, x, mul(x, inv)), jnp.where(small, y, mul(y, inv)), jnp.where(small, F32(0.0), n)


def clamp(a, lo, hi):
    """b2Clamp(a, low, high) = b2Max(low, b2Min(a, high)) with b2Min(a, b) = a < b ? a : b."""
    m = jnp.where(a < hi, a, hi)
    return jnp.where(lo > m, lo, m)


def bmin(a, b):
    """b2Min: a < b ? a : b (keeps b on ties and NaN, unlike jnp.minimum)."""
    return jnp.where(a < b, a, b)


def bmax(a, b):
    """b2Max: a > b ? a : b."""
    return jnp.where(a > b, a, b)


def babs(a):
    """b2Abs: a > 0 ? a : -a."""
    return jnp.where(a > 0, a, -a)


def inv_or_zero(x):
    """`x != 0 ? 1 / x : 0` style guards (det, gamma, mass)."""
    nz = x != 0
    return jnp.where(nz, F32(1.0) / jnp.where(nz, x, F32(1.0)), F32(0.0))


def inv_if_pos(x):
    """`k > 0 ? 1 / k : 0`."""
    p = x > 0
    return jnp.where(p, F32(1.0) / jnp.where(p, x, F32(1.0)), F32(0.0))


# ---- body-array access inside the sequential solver loops

def pick(arr, i):
    """arr[i] for one index per world, as a chain of selects over the B static columns: exact, and elementwise,
    so XLA fuses it with the arithmetic around it (a gather or a reduction with a batched index would each be a
    separate kernel per use under vmap)."""
    out = arr[..., 0]
    for b in range(1, arr.shape[-1]):
        out = jnp.where(i == b, arr[..., b], out)
    return out


def put(arr, i, val):
    """arr with arr[i] = val (one-hot select)."""
    return jnp.where(jnp.arange(arr.shape[-1]) == i, val, arr)


# ---- b2Mat22 / b2Mat33 (column-major as in Box2D: ex, ey, ez columns)

def solve22(a11, a12, a21, a22, bx, by):
    """b2Mat22::Solve and b2Mat33::Solve22 (same code): det = a11*a22 - a12*a21, inverted unless 0."""
    det = inv_or_zero(mul(a11, a22) - mul(a12, a21))
    return mul(det, mul(a22, bx) - mul(a12, by)), mul(det, mul(a11, by) - mul(a21, bx))


def _cross3(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (mul(ay, bz) - mul(az, by), mul(az, bx) - mul(ax, bz), mul(ax, by) - mul(ay, bx))


def _dot3(a, b):
    return (mul(a[0], b[0]) + mul(a[1], b[1])) + mul(a[2], b[2])


def solve33(ex, ey, ez, b):
    """b2Mat33::Solve33 (Cramer): columns ex, ey, ez as 3-tuples, b a 3-tuple."""
    det = inv_or_zero(_dot3(ex, _cross3(ey, ez)))
    return (mul(det, _dot3(b, _cross3(ey, ez))), mul(det, _dot3(ex, _cross3(b, ez))),
            mul(det, _dot3(ex, _cross3(ey, b))))


def inverse22_of33(ex, ey):
    """b2Mat33::GetInverse22 -> columns (ex, ey, ez) of the 3x3 result (z entries 0)."""
    a, b, c, d = ex[0], ey[0], ex[1], ey[1]
    det = inv_or_zero(mul(a, d) - mul(b, c))
    z = jnp.zeros_like(a)
    return (mul(det, d), mul(-det, c), z), (mul(-det, b), mul(det, a), z), (z, z, z)


def sym_inverse33(ex, ey, ez):
    """b2Mat33::GetSymInverse33 -> columns (ex, ey, ez)."""
    det = inv_or_zero(_dot3(ex, _cross3(ey, ez)))
    a11, a12, a13 = ex[0], ey[0], ez[0]
    a22, a23 = ey[1], ez[1]
    a33 = ez[2]
    exx = mul(det, mul(a22, a33) - mul(a23, a23))
    exy = mul(det, mul(a13, a23) - mul(a12, a33))
    exz = mul(det, mul(a12, a23) - mul(a13, a22))
    eyy = mul(det, mul(a11, a33) - mul(a13, a13))
    eyz = mul(det, mul(a13, a12) - mul(a11, a23))
    ezz = mul(det, mul(a11, a22) - mul(a12, a12))
    return (exx, exy, exz), (exy, eyy, eyz), (exz, eyz, ezz)


def mat33_mul(ex, ey, ez, v):
    """b2Mul(Mat33, Vec3) = v.x * ex + v.y * ey + v.z * ez (vector sums left to right)."""
    return tuple((mul(v[0], ex[k]) + mul(v[1], ey[k])) + mul(v[2], ez[k]) for k in range(3))


def mat33_mul22(ex, ey, vx, vy):
    """b2Mul22(Mat33, Vec2) = (ex.x * v.x + ey.x * v.y, ex.y * v.x + ey.y * v.y)."""
    return mul(ex[0], vx) + mul(ey[0], vy), mul(ex[1], vx) + mul(ey[1], vy)


# ---- correctly rounded sin/cos (float-float evaluation, one final rounding)

def _barrier(x):
    return lax.optimization_barrier(x)


def _split(a):
    bits = lax.bitcast_convert_type(jnp.asarray(a, f32), u32)
    hi = lax.bitcast_convert_type(bits & u32(0xFFFFF000), f32)
    return hi, a - hi


def _two_prod(a, b):
    p = rounded(a * b)
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _two_sum(a, b):
    s = _barrier(a + b)
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _fast_two_sum(a, b):
    s = _barrier(a + b)
    return s, b - (s - a)


def _add(ah, al, bh, bl):
    s, e1 = _two_sum(ah, bh)
    t, e2 = _two_sum(al, bl)
    s, e1 = _fast_two_sum(s, e1 + t)
    return _fast_two_sum(s, e1 + e2)


def _mul2(ah, al, bh, bl):
    p, e = _two_prod(ah, bh)
    return _fast_two_sum(p, e + (ah * bl + al * bh))


def _host_pair(value):
    hi = np.float32(value)
    return hi, np.float32(np.float64(value) - np.float64(hi))


def _parts(decimal):
    from fractions import Fraction
    exact, parts = Fraction(decimal), []
    for _ in range(3):
        p = np.float32(float(exact))
        parts.append(p)
        exact -= Fraction(float(p))
    return parts


_PIO2 = _parts("1.57079632679489661923132169163975144209858469968755291048747229615390820314310449931401741267105853399107404326")
_TWO_OVER_PI = np.float32(2.0 / np.pi)
_SIN_COEFFS = [(-1.0) ** k / float(np.prod(np.arange(1, 2 * k + 2, dtype=np.float64))) for k in range(10)]
_COS_COEFFS = [(-1.0) ** k / float(np.prod(np.arange(1, 2 * k + 1, dtype=np.float64))) for k in range(10)]


def _poly(xh, xl, coefficients):
    ch, cl = _host_pair(coefficients[-1])
    sh, sl = jnp.full_like(xh, ch), jnp.full_like(xh, cl)
    for c in reversed(coefficients[:-1]):
        ph, pl = _mul2(sh, sl, xh, xl)
        kh, kl = _host_pair(c)
        sh, sl = _add(ph, pl, jnp.full_like(xh, kh), jnp.full_like(xh, kl))
    return sh, sl


def sin_cos(x):
    """(sin x, cos x) rounded once to float32, i.e. (float)sin((double)x): Cody-Waite reduction by pi/2 in three
    float32 parts, float-float Taylor series (cuphead-rl sim/common/trig.py: exact on every sample to |x| = 1e4).
    sin(+-0) keeps the sign, as C does."""
    x = jnp.asarray(x, f32)
    k = jnp.round(x * _TWO_OVER_PI)
    p1h, p1l = _two_prod(k, jnp.full_like(k, _PIO2[0]))
    rh, rl = _add(x, jnp.zeros_like(x), -p1h, -p1l)
    p2h, p2l = _two_prod(k, jnp.full_like(k, _PIO2[1]))
    rh, rl = _add(rh, rl, -p2h, -p2l)
    rh, rl = _add(rh, rl, -(k * _PIO2[2]), jnp.zeros_like(x))
    r2h, r2l = _mul2(rh, rl, rh, rl)
    sh, sl = _poly(r2h, r2l, _SIN_COEFFS)
    sh, sl = _mul2(rh, rl, sh, sl)
    ch, cl = _poly(r2h, r2l, _COS_COEFFS)
    s, c = (sh + sl).astype(f32), (ch + cl).astype(f32)
    q = jnp.mod(k.astype(i32), 4)
    sin_x = jnp.where(q == 0, s, jnp.where(q == 1, c, jnp.where(q == 2, -s, -c)))
    cos_x = jnp.where(q == 0, c, jnp.where(q == 1, -s, jnp.where(q == 2, -c, s)))
    return jnp.where(x == 0, x, sin_x), cos_x


def rot_of(angle):
    """b2Rot(angle) -> (c, s)."""
    s, c = sin_cos(angle)
    return c, s


# ---- host-side float32 helpers (scene building), same operation order

def host_sin_cos(a):
    """(float)sin((double)a), (float)cos((double)a) with the C library's double sin/cos (Python's math)."""
    import math
    a = float(np.float32(a))
    return F32(math.sin(a)), F32(math.cos(a))
