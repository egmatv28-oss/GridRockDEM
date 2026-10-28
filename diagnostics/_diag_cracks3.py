import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_DIAG
"""Диагностика-3: эксперименты.

E1: снять bfixed у строк 62-63 -> если bSxx выравнивается, причина просадки
    у основания - жёсткое основание, стягивающее боковое сжатие сдвигом связей.
E2: распределение |rel| и |shearF|/FsC по связям - калибровка порога изгиба.
E3: глубина 1500, длинный прогон - реальные разрывы, их ориентация и критерий.

Запуск:  C:\py312\python.exe _diag_cracks3.py [e1|e2|e3]
"""
import sys
import time

import render  # noqa: F401
from render import generate
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       brot, bsz, bSxx, bSyy, sigVI, sideF, dbgBreakN, broken,
                       bfixed, brest, bondNX, bondNY, COLS, ROWS, LCELL,
                       BOND_JOINT, BOND_INTACT)
from solver import step_physics, apply_rock, apply_gravity, count_intact, DT
import numpy as np


def profile(tag):
    n = bcount[None]
    yb = by.to_numpy()[:n]
    xs = bSxx.to_numpy()[:n]
    print("  [%s] bSxx profile:" % tag)
    for r in list(range(0, ROWS, 6)) + [56, 57, 58, 59, 60, 61, 62, 63]:
        m = (yb >= r * LCELL) & (yb < (r + 1) * LCELL)
        if m.any():
            print("    row %2d: bSxx=%9.0f" % (r, xs[m].mean()))


def e1():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 600.0
    P[None].k0 = 1.0
    generate(0, 0)
    print("e1: unfix rows 62-63 (theory: rigid base drains bSxx)")
    fix = bfixed.to_numpy()
    yb = by.to_numpy()
    for i in range(bcount[None]):
        if yb[i] >= 62 * LCELL:
            bfixed[i] = 0
    for f in range(240):
        step_physics(0, DT)
    profile("e1 unfixed base")


def e2():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 600.0
    P[None].k0 = 1.0
    generate(0, 0)
    for f in range(120):
        step_physics(0, DT)
    m = bondCount[None]
    ia = bondIntact.to_numpy()[:m]
    aa = bondA.to_numpy()[:m]
    bb = bondB.to_numpy()[:m]
    br = brot.to_numpy()
    rels = np.abs(br[bb[:m]] - br[aa[:m]])
    sel = ia[:m] == BOND_INTACT
    rels = rels[sel]
    print("e2: |rel| percentiles over %d intact bonds:" % len(rels))
    for p in (50, 90, 99, 99.9, 100):
        print("    p%-5s = %.5f rad" % (p, np.percentile(rels, p)))
    print("    current bend break limit = %.5f rad" % (2.0 * 4e6 / 1e10 * 3.0))


def e3():
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 1500.0
    P[None].k0 = 1.0
    generate(0, 0)
    m0 = bondCount[None]
    ia0 = bondIntact.to_numpy()[:m0]
    print("e3: depth=1500, %d frames" % frames)
    for f in range(frames):
        step_physics(0, DT)
        if f % max(1, frames // 6) == 0 or f == frames - 1:
            print("  f=%3d intact=%d broken=%d dbg=%s"
                  % (f, count_intact(), broken[None], [dbgBreakN[k] for k in range(3)]), flush=True)
    m = bondCount[None]
    ia = bondIntact.to_numpy()[:m]
    aa = bondA.to_numpy()[:m]
    bb = bondB.to_numpy()[:m]
    xb = bx.to_numpy()
    yb = by.to_numpy()
    joints_v, joints_h = [], []
    for bd in range(m):
        if ia0[bd] == BOND_INTACT and ia[bd] == BOND_JOINT:
            a, b = int(aa[bd]), int(bb[bd])
            dx, dy = abs(xb[b] - xb[a]), abs(yb[b] - yb[a])
            x0, y0 = 0.5 * (xb[a] + xb[b]) / LCELL, 0.5 * (yb[a] + yb[b]) / LCELL
            (joints_h if dx >= dy else joints_v).append((x0, y0))
    tx, ty = COLS * 0.5, ROWS * 0.6
    print("e3: novye razryvy: GORIZ svyazi=%d (->vert treshini), VERT svyazi=%d (->goriz treshini)"
          % (len(joints_h), len(joints_v)))
    for name, arr in (("VERT (goriz treshini)", joints_v), ("GORIZ (vert treshini)", joints_h)):
        print("  %s (x,y; otnositelno centra vyrabotki):" % name)
        for x0, y0 in arr[:25]:
            print("    (%.1f, %.1f)  d=(%+.1f, %+.1f)" % (x0, y0, x0 - tx, y0 - ty))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "e1"
    {"e1": e1, "e2": e2, "e3": e3}[mode]()
