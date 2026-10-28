import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_DIAG
"""Диагностика-2: по-строчные sideF/bSxx (симптом 2) + репро трещин (симптом 1).

Запуск:  C:\py312\python.exe _diag_cracks2.py
"""
import time

import render  # noqa: F401
from render import generate
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       bsz, bSxx, bSyy, sigVI, sideF, dbgBreakN, broken, bfixed,
                       nbond, COLS, ROWS, LCELL, BOND_JOINT)
from solver import step_physics, apply_rock, apply_gravity, count_intact, DT
import numpy as np

DEPTH = 600.0
K0 = 1.0
FRAMES = 600


def main():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = DEPTH
    P[None].k0 = K0

    generate(0, 0)
    print("diag2: tunnel blocks=%d bonds=%d depth=%g K0=%g" % (bcount[None], bondCount[None], DEPTH, K0))

    for f in range(FRAMES):
        step_physics(0, DT)
        if f % 100 == 0 or f == FRAMES - 1:
            print("  f=%3d intact=%d broken=%d dbg[tens,crush,shear]=%s"
                  % (f, count_intact(), broken[None], [dbgBreakN[k] for k in range(3)]))

    n = bcount[None]
    xb = bx.to_numpy()[:n]
    yb = by.to_numpy()[:n]
    xs = bSxx.to_numpy()[:n]
    ys = bSyy.to_numpy()[:n]
    sfl = sideF.to_numpy()[:n]
    fix = bfixed.to_numpy()[:n]
    sg = sigVI.to_numpy()

    print("\ndiag2: po-strochnyj profil (row: nfix, sum sideF, mean bSxx, mean bSyy, sigVI kraya):")
    for r in range(ROWS):
        m = (yb >= r * LCELL) & (yb < (r + 1) * LCELL)
        if not m.any():
            continue
        if r % 4 != 0 and r < ROWS - 8:
            continue
        print("   row %2d: n=%3d nfix=%d sideF=%9.0f bSxx=%9.0f bSyy=%9.0f sigVI(L/R)=%.0f/%.0f"
              % (r, int(m.sum()), int(fix[m].sum()), sfl[m].sum(),
                 xs[m].mean(), ys[m].mean(),
                 sg[r * COLS], sg[r * COLS + COLS - 1]))

    # разорванные связи: ориентация + позиция относительно выработки
    m = bondCount[None]
    ia = bondIntact.to_numpy()[:m]
    aa = bondA.to_numpy()[:m]
    bb = bondB.to_numpy()[:m]
    joints_v, joints_h = [], []
    for bd in range(m):
        if ia[bd] != BOND_JOINT:
            continue
        a, b = int(aa[bd]), int(bb[bd])
        dx, dy = abs(xb[b] - xb[a]), abs(yb[b] - yb[a])
        x0, y0 = 0.5 * (xb[a] + xb[b]) / LCELL, 0.5 * (yb[a] + yb[b]) / LCELL
        (joints_h if dx >= dy else joints_v).append((x0, y0))
    print("\ndiag2: joints: goriz.svyazi=%d (vert.treshini), vert.svyazi=%d (goriz.treshini)"
          % (len(joints_h), len(joints_v)))
    tx, ty = COLS * 0.5, ROWS * 0.6
    for name, arr in (("VERT svyazi (goriz treshini)", joints_v), ("GORIZ svyazi (vert treshini)", joints_h)):
        print("  %s:" % name)
        for x0, y0 in arr[:30]:
            print("    (%.1f, %.1f)  dist_to_tunnel_center=(%.1f, %.1f)" % (x0, y0, x0 - tx, y0 - ty))


if __name__ == "__main__":
    main()
