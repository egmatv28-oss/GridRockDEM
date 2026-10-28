import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
r"""Validation of the Coulomb friction coefficient.

A single loose block lies on the floor. A constant horizontal force F is
applied; the block starts to slide only when F exceeds mu*m*g. The measured
threshold is compared with mu*m*g (mu is P.mu from apply_rock).

Run:
    C:\py312\python.exe _verify_friction.py
"""
import os
os.environ.setdefault("GRIDROCKDEM_ARCH", "cpu")
import numpy as np
import taichi as ti  # noqa: F401

from physstate import (ROWS, bx, by, bvx, bvy, bmass, LCELL, DT,
                       add_block_func, refillOff, P)
import solver
from gui import clear_scene


@ti.kernel
def add_block(x: ti.f32, y: ti.f32):
    add_block_func(x, y, 1.0, 0)


@ti.kernel
def add_block_kick(x: ti.f32, y: ti.f32, vx0: ti.f32):
    i = add_block_func(x, y, 1.0, 0)
    if i >= 0:
        bvx[i] = vx0


def slide(F):
    solver.apply_gravity(1.0)
    solver.apply_rock(10.0, 4.0, 0.62)
    clear_scene()
    refillOff[None] = 1
    P[None].walls = 1
    add_block(10.5, ROWS - 0.5)
    m = float(bmass[0])
    x0 = float(bx[0] / LCELL)
    solver.set_ext_force(0, F, 0.0)
    for _ in range(300):
        solver.step_physics(0, DT)
    x1 = float(bx[0] / LCELL)
    solver.clear_ext_forces()
    return (x1 - x0) * LCELL, m


def wall_stick():
    """A block thrown at the left wall must NOT stick in mid-air."""
    solver.apply_gravity(1.0)
    solver.apply_rock(10.0, 4.0, 0.62)
    clear_scene()
    refillOff[None] = 1
    P[None].walls = 1
    add_block_kick(3.0, 20.5, -4.0)
    for _ in range(500):
        solver.step_physics(0, DT)
    x = float(bx[0] / LCELL)
    y = float(by[0] / LCELL)
    v = float((bvx[0] ** 2 + bvy[0] ** 2) ** 0.5)
    on_floor = y > ROWS - 2.0
    print("wall test: final x=%.2f y=%.2f |v|=%.3f  %s" %
          (x, y, v, "fell (ok)" if on_floor else "STUCK IN AIR"))
    return on_floor


def main():
    solver.apply_gravity(1.0)
    solver.apply_rock(10.0, 4.0, 0.62)
    mu = float(P[None].mu)
    # mass of a 1-cell block and the expected threshold are taken in slide()
    print("mu=%.2f  expected static threshold F=mu*m*g" % mu)
    forces = [2000.0, 3000.0, 3400.0, 3600.0, 3800.0, 4000.0, 4500.0]
    moved_at = None
    for F in forces:
        d, m = slide(F)
        exp = mu * m * 9.81
        moved = abs(d) > 0.05
        if moved and moved_at is None:
            moved_at = (F, exp)
        print("  F=%5.0f N  (mu*m*g=%.0f)  slide=%.3f m  %s" %
              (F, exp, d, "MOVED" if moved else "held"))
    print("measured threshold ~ %.0f N, expected %.0f N" %
          (moved_at[0] if moved_at else float('nan'),
           moved_at[1] if moved_at else float('nan')))
    wall_stick()


if __name__ == "__main__":
    main()
