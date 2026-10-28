import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""ВЕРИФИКАЦИЯ КРОВЛИ (упругий отклик выработки vs Кирш), с КОНТРОЛЕМ.

Проблема прямого замера: массив сжимается под собственным весом (~σ_v·h/E,
десятки мм), и это засоряет «сходимость» контура. Поэтому считаем ДВА опыта с
ОДИНАКОВОЙ нагрузкой:
  * solid   — сплошной прямоугольник (нет полости);
  * hole    — тот же прямоугольник с круглой полостью радиуса a.
Разность смещений Δu = u_hole − u_solid (блок в той же клетке) — чистое
влияние выработки. Радиальная составляющая Δu_r у контура сравнивается с
Киршем в гидростатическом поле (K0=1):

    σ_r = p(1−a²/r²), σ_θ = p(1+a²/r²),  u_r(a) = 2p·a/E (пл. напр.)
                                          = 2p·a/E·(1−ν²) (пл. деф., ν=0.25)

Запуск:  C:\\py312\\python.exe verification/_verify_roof.py
"""
import math
import sys

import numpy as np

import render  # noqa: F401
from physstate import (P, bcount, bx, by, bvx, bvy, refillOff, broken,
                       RHO_R, G_PHYS, LCELL, COLS, ROWS)
from solver import (apply_gravity, apply_rock, build_model, refresh_field,
                    step_physics, update_mass_rho)
from _verify_common import SIGMA_PER_M, make_rect_hole

E_ROCK = 10e9
NU = 0.25
HX = COLS // 2
HY = ROWS // 2
HR = 4     # радиус полости (клетки)


def run(hr, depth, k0, frames, substeps=192):
    blocks, bonds, idx = make_rect_hole(HX, HY, hr)
    build_model(blocks, bonds, [], (HX, HY, hr, hr) if hr > 0 else None)
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    update_mass_rho()
    P[None].walls = 1
    P[None].fixedRows = 2
    P[None].k0 = k0
    P[None].depth = depth
    P[None].substeps = substeps
    P[None].brkCap = 0
    refresh_field()
    refillOff[None] = 1
    for _ in range(frames):
        step_physics(0)
    x = bx.to_numpy(); y = by.to_numpy()
    vx = bvx.to_numpy(); vy = bvy.to_numpy()
    pos = {ij: (x[k], y[k]) for ij, k in idx.items()}
    vmax = float(np.max(np.hypot(vx[:bcount[None]], vy[:bcount[None]])))
    nb = int(broken[None])
    return pos, idx, vmax, nb


def main():
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    print("Domen: %dx%d, a=%d kletok (%.1f m)" % (COLS, ROWS, HR, HR * LCELL))
    print("Kontrol: ta zhe model BEZ polosti; delta u = u_hole - u_solid\n")
    for depth in (500.0, 1000.0, 1500.0):
        k0 = 1.0
        ps, isolid, vmax_s, nb_s = run(0, depth, k0, frames)
        ph, ihole, vmax_h, nb_h = run(HR, depth, k0, frames)
        a = HR * LCELL
        p = depth * SIGMA_PER_M
        u_ps = 2.0 * p * a / E_ROCK              # Krissh, ploskaya deformaciya (2pa/E)
        u_pe = 2.0 * p * a / E_ROCK * (1.0 - NU * NU)  # ploskaya deformaciya
        acc = {"top": [], "bottom": [], "left": [], "right": []}
        for (i, j), k in ihole.items():
            x0 = ps[(i, j)]              # sootv. blok v solid
            xh = ph[(i, j)]
            dux = xh[0] - x0[0]
            duy = xh[1] - x0[1]
            bx0 = i + 0.5 - HX           # kletki
            by0 = j + 0.5 - HY
            r = math.hypot(bx0, by0)
            if not (a / LCELL - 0.6 <= r <= a / LCELL + 1.4):
                continue
            radius = math.hypot(bx0, by0) * LCELL
            ur = -(dux * (bx0 * LCELL) + duy * (by0 * LCELL)) / radius
            if abs(by0) > abs(bx0):
                acc["top" if by0 < 0 else "bottom"].append(ur)
            else:
                acc["left" if bx0 < 0 else "right"].append(ur)

        def m(t):
            v = acc[t]
            return (sum(v) / len(v)) if v else float("nan")
        um = (m("top") + m("bottom") + m("left") + m("right")) / 4.0
        print("===== depth=%.0f  K0=%.1f =====" % (depth, k0))
        print("  p=%.2f MPa  teoriya: pl.deform. u=2pa/E=%.3f mm, pl.napr. %.3f mm"
              % (p / 1e6, u_ps * 1000, u_pe * 1000))
        print("  delta u_r (mm): krovlya=%.3f podoshva=%.3f bokL=%.3f bokR=%.3f  sred=%.3f"
              % (m("top") * 1000, m("bottom") * 1000, m("left") * 1000, m("right") * 1000,
                 um * 1000))
        print("  otnoshenie k teorii: pl.napr.=%.2f  pl.deform.=%.2f"
              % (um / (u_pe + 1e-30), um / (u_ps + 1e-30)))
        print("  max|v| solid=%.2e hole=%.2e  razryvy solid=%d hole=%d"
              % (vmax_s, vmax_h, nb_s, nb_h), flush=True)


if __name__ == "__main__":
    main()

