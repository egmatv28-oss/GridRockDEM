import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_DIAG
"""Верификация фиксов: симптом 2 (профиль bSxx) и симптом 1 (ориентация трещин).

Запуск:  C:\py312\python.exe _diag_verify.py
"""
import render  # noqa: F401
from render import generate
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       bSxx, dbgBreakN, broken, bfixed, COLS, ROWS, LCELL,
                       BOND_JOINT, BOND_INTACT)
from solver import step_physics, apply_rock, apply_gravity, count_intact, DT
import numpy as np


def main():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 600.0
    P[None].k0 = 1.0
    generate(0, 0)
    m0 = bondCount[None]
    ia0 = bondIntact.to_numpy()[:m0]
    for f in range(300):
        step_physics(0, DT)
    print("after 300 frames: broken=%d dbg[tens,crush,shear,bend]=%s"
          % (broken[None], [dbgBreakN[k] for k in range(4)]))

    n = bcount[None]
    yb = by.to_numpy()[:n]
    xs = bSxx.to_numpy()[:n]
    print("\n[symptom2] bSxx profile (MPa), expect no collapse at bottom:")
    for r in (0, 10, 20, 30, 44, 50, 54, 56, 58, 60, 61, 62, 63):
        m = (yb >= r * LCELL) & (yb < (r + 1) * LCELL)
        if m.any():
            print("   row %2d: %6.2f" % (r, xs[m].mean() / 1e6))

    # симптом 1: на глубине 1200 считаем новые разрывы по ориентации
    P[None].depth = 1200.0
    ia_prev = bondIntact.to_numpy()[:m0].copy()
    for f in range(300):
        step_physics(0, DT)
    print("\nafter +300 frames at depth=1200: broken=%d dbg=%s"
          % (broken[None], [dbgBreakN[k] for k in range(4)]))
    ia = bondIntact.to_numpy()[:m0]
    aa = bondA.to_numpy()[:m0]
    bb = bondB.to_numpy()[:m0]
    xb = bx.to_numpy()
    joints_v, joints_h = [], []
    for bd in range(m0):
        if ia_prev[bd] == BOND_INTACT and ia[bd] != BOND_INTACT:
            a, b = int(aa[bd]), int(bb[bd])
            dx, dy = abs(xb[b] - xb[a]), abs(yb[b] - yb[a])
            x0, y0 = 0.5 * (xb[a] + xb[b]) / LCELL, 0.5 * (yb[a] + yb[b]) / LCELL
            (joints_h if dx >= dy else joints_v).append((x0, y0))
    tx, ty = COLS * 0.5, ROWS * 0.6
    print("[symptom1] novye razryvy: VERT svyazi=%d (->goriz treshini), "
          "GORIZ svyazi=%d (->vert treshini)" % (len(joints_v), len(joints_h)))
    for name, arr in (("VERT svyazi (goriz treshini!)", joints_v),
                      ("GORIZ svyazi (vert treshini)", joints_h)):
        if not arr:
            continue
        print("  %s:" % name)
        for x0, y0 in arr[:15]:
            print("    (%.1f, %.1f) d=(%+.1f,%+.1f)" % (x0, y0, x0 - tx, y0 - ty))


if __name__ == "__main__":
    main()
