import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_DIAG
"""Диагностика трещин у выработки и бокового напряжения внизу.

Репро: выработка (tunnel), вертикальная нагрузка (depth), K0 как в GUI.
Печатает: счётчики обрывов по типам (dbgBreakN), список разорванных связей
с ориентацией/координатами, профили bSxx/bSyy/sigVI по высоте.

Запуск:  C:\py312\python.exe _diag_cracks.py
"""
import sys
import time

import render  # noqa: F401  (инициализирует taichi через physstate)
from render import generate  # noqa: F401
from gui import run_selftest  # noqa: F401
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       bsz, bSxx, bSyy, sigVI, dbgBreakN, broken, nbond, brest,
                       COLS, ROWS, LCELL, BOND_DEAD, BOND_JOINT)
from solver import step_physics, apply_rock, apply_gravity, count_intact, DT
import taichi as ti

DEPTH = 300.0   # м, как "большая глубина" слайдера
K0 = 1.0
FRAMES = 240    # 2 с модельного времени


def main():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = DEPTH
    P[None].k0 = K0

    t0 = time.perf_counter()
    generate(0, 0)
    print("diag: tunnel blocks=%d bonds=%d (%.1fs)" % (bcount[None], bondCount[None], time.perf_counter() - t0))

    nbi = count_intact()
    for f in range(FRAMES):
        step_physics(0, DT)
        if f % 60 == 0 or f == FRAMES - 1:
            print("  f=%3d intact=%d broken=%d dbg=%s" % (
                f, count_intact(), broken[None],
                [dbgBreakN[k] for k in range(4)]))
    print("diag: broken total=%d dbgBreakN[tens,crush,shear/bend,3]=%s"
          % (broken[None], [dbgBreakN[k] for k in range(4)]))

    # --- разорванные связи: где и какой ориентации ---
    n = bondCount[None]
    ia = bondIntact.to_numpy()[:n]
    aa = bondA.to_numpy()[:n]
    bb = bondB.to_numpy()[:n]
    xb = bx.to_numpy()
    yb = by.to_numpy()
    joints_v = []   # вертикальные связи (блок друг над другом) -> горизонт. трещина
    joints_h = []
    for bd in range(n):
        if ia[bd] != BOND_JOINT:
            continue
        a, b = int(aa[bd]), int(bb[bd])
        dx = abs(xb[b] - xb[a])
        dy = abs(yb[b] - yb[a])
        x0, y0 = 0.5 * (xb[a] + xb[b]), 0.5 * (yb[a] + yb[b])
        rec = (x0 / LCELL, y0 / LCELL, a, b)
        (joints_h if dx >= dy else joints_v).append(rec)
    print("diag: jointov %d: GORIZONTALNYH svyazey (vert.treshini)=%d, "
          "VERTIKALNYH svyazey (goriz.treshini)=%d"
          % (len(joints_h) + len(joints_v), len(joints_h), len(joints_v)))
    print("diag: vertikalnye svyazi (goriz.treshini) - pervye 20 (x,y v kletkah):")
    for r in joints_v[:20]:
        print("   (%.1f, %.1f) #%d-#%d" % r)
    print("diag: gorizontalnye svyazi (vert.treshini) - pervye 20:")
    for r in joints_h[:20]:
        print("   (%.1f, %.1f) #%d-#%d" % r)

    # --- профиль напряжений по строкам (средние по строке) ---
    xs = bSxx.to_numpy()[:bcount[None]]
    ys = bSyy.to_numpy()[:bcount[None]]
    sg = sigVI.to_numpy()
    yy = yb[:bcount[None]]
    xx = xb[:bcount[None]]
    print("diag: profil po strokam (row: bSxx sred, bSyy sred, sigVI kraya):")
    for r in range(0, ROWS, 8):
        m = (yy >= r * LCELL) & (yy < (r + 8) * LCELL)
        if not m.any():
            continue
        sxx = xs[m].mean()
        syy = ys[m].mean()
        sL = sg[r * COLS + 1]
        sR = sg[r * COLS + COLS - 2]
        sMid = sg[r * COLS + COLS // 2]
        print("   row %2d-%2d: bSxx=%9.0f  bSyy=%9.0f  sigVI(L/M/R)=%.0f/%.0f/%.0f"
              % (r, r + 7, sxx, syy, sL, sMid, sR))


if __name__ == "__main__":
    main()
