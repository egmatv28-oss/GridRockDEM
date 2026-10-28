import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_DIAG
"""Диагностика-5: механизм просадки bSxx внизу (симптом 2).

A: профиль bSxx во времени (f=20/60/150/300) - сразу или растёт?
B: plastFlow=0 (нет пластичности) - остаётся просадка?
C: mu=0 (нет трения) - остаётся просадка?

Запуск:  C:\py312\python.exe _diag_cracks5.py [a|b|c]
"""
import sys

import render  # noqa: F401
from render import generate
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       bSxx, sideF, bfixed, brest, COLS, ROWS, LCELL)
from solver import step_physics, apply_rock, apply_gravity, DT
import numpy as np


def profile(tag):
    n = bcount[None]
    yb = by.to_numpy()[:n]
    xs = bSxx.to_numpy()[:n]
    vals = []
    for r in (0, 10, 20, 30, 44, 50, 54, 56, 58, 60, 61, 62):
        m = (yb >= r * LCELL) & (yb < (r + 1) * LCELL)
        vals.append(xs[m].mean() / 1e6 if m.any() else -1)
    print("  [%s] " % tag + " ".join("%.1f" % v for v in vals))


def run(tag, plast_flow, mu, frames):
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, mu)
    P[None].plastFlow = plast_flow
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 600.0
    P[None].k0 = 1.0
    generate(0, 0)
    checkpoints = {20, 60, 150, 300, frames - 1}
    for f in range(frames):
        step_physics(0, DT)
        if f in checkpoints:
            profile("f=%3d" % f)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    print("rows:      0  10  20  30  44  50  54  56  58  60  61  62   (bSxx, MPa)")
    if mode == "a":
        run("A default", 1.6, 0.62, 301)
    elif mode == "b":
        run("B plastFlow=0", 0.0, 0.62, 301)
    elif mode == "c":
        run("C mu=0", 1.6, 0.0, 301)
    elif mode == "d":
        run("D plastFlow=0 mu=0", 0.0, 0.0, 301)


if __name__ == "__main__":
    main()
