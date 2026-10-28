import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_DIAG
"""Диагностика-4: куда уходит боковое сжатие внизу.

По строкам: сколько блоков получает sideF, сумма |sideF|, смещение левых
крайних блоков от стены, сжатие горизонтальных связей по геометрии (d-brest).

Запуск:  C:\py312\python.exe _diag_cracks4.py
"""
import render  # noqa: F401
from render import generate
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       bsz, bSxx, bSyy, sideF, bfixed, brest, nbond,
                       COLS, ROWS, LCELL, BOND_DEAD, BOND_JOINT)
from solver import step_physics, apply_rock, apply_gravity, DT
import numpy as np


def main():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 600.0
    P[None].k0 = 1.0
    generate(0, 0)
    for f in range(300):
        step_physics(0, DT)

    n = bcount[None]
    xb = bx.to_numpy()[:n]
    yb = by.to_numpy()[:n]
    bs = bsz.to_numpy()[:n]
    xs = bSxx.to_numpy()[:n]
    sfl = sideF.to_numpy()[:n]
    fix = bfixed.to_numpy()[:n]

    m = bondCount[None]
    ia = bondIntact.to_numpy()[:m]
    aa = bondA.to_numpy()[:m]
    bb = bondB.to_numpy()[:m]
    rest = brest.to_numpy()[:m]
    d = np.sqrt((xb[bb[:m]] - xb[aa[:m]])**2 + (yb[bb[:m]] - yb[aa[:m]])**2)
    horiz = np.abs(xb[bb[:m]] - xb[aa[:m]]) >= np.abs(yb[bb[:m]] - yb[aa[:m]])
    comp = (rest[:m] - d)   # >0 = сжатие, м
    live = (ia[:m] != BOND_DEAD)

    print("row | nsideF | sum|sideF| | bSxx_mean | hcomp_um | left_edge_bx | nfix")
    for r in range(ROWS):
        mb = (yb >= r * LCELL) & (yb < (r + 1) * LCELL)
        if not mb.any():
            continue
        left = mb & (xb < 1.0)
        sel = horiz & live & (yb[aa[:m]] >= r * LCELL) & (yb[aa[:m]] < (r + 1) * LCELL)
        hc = comp[sel].mean() * 1e6 if sel.any() else 0.0
        nsf = int((np.abs(sfl[mb]) > 1.0).sum())
        lbx = xb[left].min() if left.any() else -1
        if r % 2 == 0 or r >= 54:
            print("%3d | %6d | %10.0f | %9.0f | %8.2f | %12.3f | %4d"
                  % (r, nsf, np.abs(sfl[mb]).sum(), xs[mb].mean(), hc, lbx,
                     int(fix[mb].sum())))

    # сколько всего блоков касается стен по тесту
    rx = bs * 0.5
    touch = ((xb - rx <= 0.25 * LCELL) | (xb + rx >= COLS * LCELL - 0.25 * LCELL))
    print("\nblocks touching walls by test: %d / %d (fixed among them: %d)"
          % (int(touch.sum()), n, int((touch & (fix == 1)).sum())))
    print("min(bx-rx) over all = %.3f, max(bx+rx) = %.3f (walls at 0 and %.3f)"
          % ((xb - rx).min(), (xb + rx).max(), COLS * LCELL))


if __name__ == "__main__":
    main()
