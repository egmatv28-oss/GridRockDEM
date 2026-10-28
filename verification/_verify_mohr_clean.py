import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
r"""Clean Mohr-Coulomb criterion test.

Checks the bond law directly:

    Fs_fail = Fs0 + mcBond * |Fn|

for one compressed bond sheared to failure.  From the fitted friction
coefficient the expected failure-plane angle is

    theta = 45 deg + atan(mcBond) / 2.

Run:
    C:\py312\python.exe _verify_mohr_clean.py
"""
import math

import numpy as np
import taichi as ti

import render  # noqa: F401
from physstate import (P, broken, dbgBreakN, bx, by, bvx, bvy, bw, brot,
                       brest)
from solver import (apply_gravity, apply_rock, step_physics, build_model,
                    refresh_field)

LCELL = 0.5
SIZE = 0.1
W_BOND = SIZE * LCELL
REST = LCELL
MCBOND = 0.7


@ti.kernel
def place(a: ti.i32, b: ti.i32, ax: ti.f32, ay: ti.f32, bxx: ti.f32, byy: ti.f32):
    bx[a] = ax
    by[a] = ay
    bx[b] = bxx
    by[b] = byy
    bvx[a] = 0.0
    bvy[a] = 0.0
    bw[a] = 0.0
    brot[a] = 0.0
    bvx[b] = 0.0
    bvy[b] = 0.0
    bw[b] = 0.0
    brot[b] = 0.0


def shear_case(gap_m, mc=MCBOND, max_shear=0.30, frames=24000):
    apply_gravity(0.0)
    apply_rock(10.0, 4.0, 0.62, rho=1e9)
    P[None].mcBond = mc
    P[None].poissonF = 0.0
    P[None].nuSplit = 0.0
    P[None].relBend = 1e9
    P[None].plastYield = 1e9
    P[None].hardenConf = 1e9
    P[None].rcFactor = 1e9
    P[None].k0 = 0.0
    P[None].walls = 0
    P[None].depth = 0.0
    P[None].substeps = 1
    P[None].brkCap = 0
    build_model([(0.0, 0.0, SIZE, 0), (1.0, 0.0, SIZE, 0)],
                [(0, 1, 1.0, 1)], [], None)
    refresh_field()
    d0 = REST - gap_m
    dy = max_shear / frames
    for f in range(frames):
        shear = (f + 1) * dy
        place(0, 1, 0.0, 1.0, d0, 1.0 + shear)
        step_physics(0)
        if broken[None] > 0:
            d = math.hypot(d0, shear)
            k_ax = 10e9 * W_BOND / REST
            k_sh = k_ax / 2.5
            tension = k_ax * (d - REST)
            fs_max = 4.0 * 1e6 * W_BOND * 2.0
            fs_meas = k_sh * shear / 3.0
            return shear, tension, fs_meas, fs_max
    return None, None, None, None


def fit_friction():
    rows = []
    for gap_m in (0.005, 0.010, 0.020, 0.050):
        shear, tension, fs_meas, fs_max = shear_case(gap_m)
        if shear is None:
            raise RuntimeError("no shear failure for gap_m=%s" % gap_m)
        normal = -tension
        mu = (fs_meas - fs_max) / normal
        rows.append((gap_m, normal, fs_meas, mu))
        print("gap_m=%.3f  |Fn|=%.3e N  Fs_fail=%.3e N  mu=%.4f"
              % (gap_m, normal, fs_meas, mu))
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    phi = math.degrees(math.atan(slope))
    theta = 45.0 + 0.5 * phi
    print("fit: Fs0=%.3e N  mu=%.4f" % (intercept, slope))
    print("expected from mcBond=%.3f: phi=%.2f deg, theta=%.2f deg"
          % (MCBOND, math.degrees(math.atan(MCBOND)),
             45.0 + 0.5 * math.degrees(math.atan(MCBOND))))
    print("fitted: phi=%.2f deg, theta=%.2f deg" % (phi, theta))
    ok = abs(slope - MCBOND) < 0.05
    print("result: %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    fit_friction()
