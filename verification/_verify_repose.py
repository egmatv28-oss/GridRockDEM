import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
r"""Angle-of-repose test for loose single blocks.

Pours one-cell blocks from a point, lets the pile settle and fits the free
surface slope on both flanks. Compares it with the Coulomb value atan(mu) and
reports blocks that rest in mid-air.

Note: y grows DOWNWARDS in this solver (gravity is +y), so the free surface of
a column is min(y - bsz/2) and its height above the floor is ROWS - that.

Run:
    C:\py312\python.exe _verify_repose.py
"""
import math
import os
os.environ.setdefault("GRIDROCKDEM_ARCH", "cpu")
import numpy as np
import taichi as ti  # noqa: F401

from physstate import (COLS, ROWS, bcount, bx, by, bvx, bvy, bsz, nbond,
                       LCELL, DT, add_block_func, refillOff, P)
import solver
from gui import clear_scene


@ti.kernel
def add_one(x: ti.f32, y: ti.f32):
    i = add_block_func(x, y, 1.0, 0)
    if i >= 0:
        bvx[i] = (ti.random() - 0.5) * 0.10


def pour(nblocks=130, every=28, settle=700):
    """Slow point deposition just above the current peak (classic repose test)."""
    solver.apply_gravity(1.0)
    solver.apply_rock(10.0, 4.0, 0.62)
    clear_scene()
    refillOff[None] = 1
    P[None].walls = 1
    P[None].depth = 0.0
    rng = np.random.default_rng(11)
    cx = COLS * 0.5
    for k in range(nblocks):
        n = int(bcount[None])
        if n > 0:
            xa = bx.to_numpy()[:n] / LCELL
            ya = by.to_numpy()[:n] / LCELL
            sa = bsz.to_numpy()[:n] / LCELL
            m = np.abs(xa - cx) < 6.0
            edge = float(np.min(ya[m] - 0.5 * sa[m])) if m.any() else ROWS - 1.0
            y_place = max(1.0, edge - 1.6)
        else:
            y_place = ROWS - 2.0
        add_one(cx + float(rng.uniform(-0.8, 0.8)), y_place)
        for _ in range(every):
            solver.step_physics(0, DT)
    for _ in range(settle):
        solver.step_physics(0, DT)
    n = int(bcount[None])
    v = np.sqrt(bvx.to_numpy()[:n] ** 2 + bvy.to_numpy()[:n] ** 2)
    print("settled: max |v|=%.4f m/s" % float(v.max()))
    return float(P[None].mu)


def height_profile():
    n = int(bcount[None])
    xa = bx.to_numpy()[:n] / LCELL
    ya = by.to_numpy()[:n] / LCELL
    sa = bsz.to_numpy()[:n] / LCELL
    edge = np.full(COLS, 1e9)
    for i in range(n):
        c = int(xa[i])
        if 0 <= c < COLS:
            edge[c] = min(edge[c], ya[i] - 0.5 * sa[i])
    h = np.where(edge < 1e8, ROWS - edge, 0.0)
    return xa, ya, sa, h


def flank(h, side):
    """Envelope (angle-of-repose) slope: steepest line from the heap apex to
    the base of that side. Robust for a rough heap of discrete cubes."""
    peak = int(np.argmax(h))
    hp = h[peak]
    best = None
    if side == 'left':
        cols = range(0, peak)
    else:
        cols = range(peak + 1, COLS)
    for x in cols:
        if 0.6 <= h[x] <= 1.8:
            dist = abs((x + 0.5) - (peak + 0.5))
            if dist < 2.0:
                continue
            ang = math.degrees(math.atan((hp - h[x]) / dist))
            if best is None or ang > best[0]:
                best = (ang, x, hp, h[x])
    return best


def floating(xa, ya, sa, tol=1.1):
    n = len(xa)
    vya = bvy.to_numpy()[:n]
    bad = []
    for i in range(n):
        if abs(vya[i]) > 0.05:
            continue
        if ya[i] > ROWS - 1.5:          # on the floor
            continue
        support = False
        for j in range(n):
            if j == i:
                continue
            if abs(xa[j] - xa[i]) < tol and 0.3 < (ya[j] - ya[i]) < 1.6:
                support = True
                break
        if not support:
            bad.append((i, xa[i], ya[i], int(nbond[i])))
    return bad


def main():
    mu = pour()
    xa, ya, sa, h = height_profile()
    n = len(xa)
    print("blocks=%d  mu=%.2f  expected atan(mu)=%.1f deg" %
          (n, mu, math.degrees(math.atan(mu))))
    peak = int(np.argmax(h))
    print("pile: x=[%.1f..%.1f]  peak h=%.1f at col %d  cols=%d" %
          (xa.min(), xa.max(), h[peak], peak, int((h > 0.2).sum())))
    for side in ('left', 'right'):
        r = flank(h, side)
        if r is None:
            print("%s flank: insufficient data" % side)
        else:
            ang, xb, hpk, hb = r
            print("%s flank: envelope=%.1f deg (apex h=%.1f -> base col %d h=%.1f)" %
                  (side, ang, hpk, xb, hb))
    cols = [x for x in range(COLS) if h[x] > 0.2]
    print("heights:", " ".join("%d:%.0f" % (x, h[x]) for x in cols))
    # ASCII profile (higher = taller)
    for y in range(int(h.max()), 0, -1):
        row = "".join("#" if h[x] >= y else "." for x in range(COLS))
        print("h=%2d %s" % (y, row))
    bad = floating(xa, ya, sa)
    print("resting without support below: %d" % len(bad))
    for i, x, y, nb in bad[:10]:
        print("   i=%d x=%.2f y=%.2f nbond=%d" % (i, x, y, nb))


if __name__ == "__main__":
    main()
