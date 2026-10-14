"""Progress along a route that was actually climbed, instead of along the level's spline (ProgressMeter).

The spline's nearest point says where the climber is along the level, but in places the way up first goes backward
along it: the winning path dropped 11 under the anvil's boulder, swung on the bucket 11 and 25 below its record, went
down an ice slope and onto a platform below the mast. A reward for new spline records pays nothing for those moves and
pays for dead ends ahead of them (the pit below the anvil reads +2). A route is a climbed path's pot positions (every
third of a second, runs/climbs/win-e4-route.npz); progress along it is the arc length up to the route point the climber
is at, followed point to point: it moves on only to points within --near of the pot (no leaping across a gap) and
looks no further than `reach` points either way.
"""
import numpy as np


class Route:
    def __init__(self, path, near=2.0, reach=30):
        z = np.load(path)
        self.x, self.y, self.p, self.s = z["x"], z["y"], z["p"], z["s"]
        self.near2, self.reach, self.i = near * near, reach, 0

    def start(self, x, y, progress):
        """The route point a state is at: the nearest among those where the route's spline progress is within 6."""
        cand = np.flatnonzero(np.abs(self.p - progress) < 6)
        if not len(cand):
            cand = np.arange(len(self.x))
        d = (self.x[cand] - x) ** 2 + (self.y[cand] - y) ** 2
        self.i = int(cand[np.argmin(d)])
        return float(self.s[self.i])

    def step(self, x, y):
        """Progress along the route after a step: the nearest point within reach, moving forward only when close."""
        lo, hi = max(0, self.i - self.reach), min(len(self.x), self.i + self.reach + 1)
        d = (self.x[lo:hi] - x) ** 2 + (self.y[lo:hi] - y) ** 2
        j = lo + int(np.argmin(d))
        if j > self.i and d[j - lo] > self.near2:  # ahead of the route but not on it (mid-air): stay
            j = self.i
        self.i = j
        return float(self.s[j])

    def scale(self, lo, hi):
        """Spline units per route unit over [lo, hi): route rewards then sum to the stretch's spline length."""
        m = (self.p >= lo) & (self.p < hi)
        return (hi - lo) / max(self.s[m].max() - self.s[m].min(), 1e-6) if m.any() else 1.0


if __name__ == "__main__":  # python -m goi_env.route: a straight route followed forward, a gap not leapt
    import os
    import tempfile
    f = os.path.join(tempfile.mkdtemp(), "r.npz")
    xs = np.arange(0, 20.0)
    np.savez(f, x=xs, y=np.zeros_like(xs), p=xs, s=xs)
    r = Route(f, near=2.0, reach=5)
    assert r.start(0.2, 0.1, 0.0) == 0
    assert r.step(1.1, 0.0) == 1 and r.step(2.9, 0.5) == 3  # along it
    assert r.step(6.0, 5.0) == 3  # 5 above the route: not on it, stays
    assert r.step(5.0, 0.0) == 5 and r.step(3.0, 0.0) == 3  # on it, forward; back down, back
    print("route: ok")
