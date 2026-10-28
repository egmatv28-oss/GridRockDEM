import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""ВЕРИФИКАЦИЯ №2 — одноосное сжатие (UCS) прямоугольного образца.

Образец строится на CPU (build_model): w×h блоков-клеток, нижний ряд
закреплён (плита пресса), сверху — растущая нагрузка P.depth (вес столба),
с боков ничего (K0=0), гравитация выключена (чистое осевое нагружение).

Соотношение сторон как у керна в опытах (ISRM H/D = 2..2.5), но и выше:
    2:1  (w=10, h=20)
    3:1  (w=10, h=30)

Измеряем:
  * предел прочности на сжатие UCS = σ_v в момент первого разрыва связей,
    σ_v = P.depth · ρ·g  (нагрузка на верх = depth·LOAD_OVER / LCELL);
  * модуль Юнга E = σ/ε по линейному участку (ε — укорочение образца);
  * коэффициент Пуассона ν = −ε_попер/ε_прод (по уширению образца);
  * отношение UCS/σ_t (σ_t = rp — прочность на отрыв, вход модели).

Сравнение с реальной породой (песчаник): UCS 20..80 МПа, E 5..50 ГПа,
σ_t ≈ 0.05..0.1·UCS (отношение UCS/σ_t ≈ 10..20).

Запуск:  C:\\py312\\python.exe verification/_verify_ucs.py
"""
import numpy as np

import render  # noqa: F401
from physstate import P, bcount, bondCount, broken, bx, by, refillOff, dbgBreakN
from solver import apply_gravity, apply_rock, step_physics, DT, count_intact
from _verify_common import (make_rect, load_model, SIGMA_PER_M, LCELL,
                            intact_pair_set, new_cracks_pairs, crack_orientation,
                            ascii_map)


E_GPa = 10.0
RP_MPa = 4.0
MU = 0.62
SPECS = [("2:1", 10, 20), ("3:1", 10, 30)]


def build_specimen(w, h):
    x0 = (64 - w) // 2
    y0 = 2
    blocks, bonds, idx = make_rect(x0, y0, w, h, fixed_rows=1)
    load_model(blocks, bonds, cavity=None)
    refillOff[None] = 1
    return idx


def run_ucs(name, w, h, frames=1400,     dmax=3000.0, settle=60):
    apply_gravity(0.0)          # чистое осевое нагружение, без собственного веса
    apply_rock(E_GPa, RP_MPa, MU)
    P[None].k0 = 0.0
    P[None].walls = 0
    P[None].depth = 0.0
    idx = build_specimen(w, h)
    top = [idx[(i, 0)] for i in range(w)]
    low = [idx[(i, h - 1)] for i in range(w)]
    left = [idx[(0, j)] for j in range(1, h - 1)]
    right = [idx[(w - 1, j)] for j in range(1, h - 1)]
    dbg0 = [int(dbgBreakN[i]) for i in range(4)]
    start_pairs = intact_pair_set()

    for _ in range(settle):     # осадка без нагрузки
        step_physics(0, DT)

    xb = bx.to_numpy()
    yb = by.to_numpy()
    y0_ax = float(np.mean(yb[top]))
    w0_lat = float(np.mean(xb[right])) - float(np.mean(xb[left]))
    H0 = (h - 1) * LCELL
    W0 = (w - 1) * LCELL

    rec = []      # (sigma, eps_ax, eps_lat, broken)
    for f in range(frames):
        depth = dmax * (f + 1) / frames
        P[None].depth = depth
        step_physics(0, DT)
        if f % 5 == 0 or broken[None] > 0:
            xb = bx.to_numpy()
            yb = by.to_numpy()
            eps_ax = (float(np.mean(yb[top])) - y0_ax) / H0     # >0 = сжатие (укорочение)
            wl = float(np.mean(xb[right])) - float(np.mean(xb[left]))
            eps_lat = (wl - w0_lat) / W0                        # >0 = расширение вбок
            rec.append((depth * SIGMA_PER_M, eps_ax, eps_lat, broken[None]))
        if broken[None] > 0 and len(rec) > 40:
            break

    sigma = np.array([r[0] for r in rec])
    ea = np.array([r[1] for r in rec])
    el = np.array([r[2] for r in rec])
    br = np.array([r[3] for r in rec])
    ib = int(np.argmax(br > 0)) if (br > 0).any() else len(br) - 1
    ucs = sigma[ib]
    eps_f = ea[ib]

    # E и nu по линейному участку (10..60% от UCS)
    lo, hi = 0.1 * ucs, 0.6 * ucs
    m = (sigma >= lo) & (sigma <= hi) & (ea > 0)
    E_meas = np.nan
    nu_meas = np.nan
    if m.sum() >= 3:
        k, b = np.polyfit(ea[m], sigma[m], 1)
        E_meas = k
        nu_meas = np.mean(el[m] / ea[m])
    g = [int(dbgBreakN[i]) - dbg0[i] for i in range(4)]
    cracks = new_cracks_pairs(start_pairs)
    nh, nv = crack_orientation(cracks)
    y0c = 2 + 0.5
    def zone(y):
        f = (y - y0c) / h
        return "verh" if f < 0.34 else ("niz" if f > 0.66 else "ser")
    zc = {"verh": 0, "ser": 0, "niz": 0}
    for c in cracks:
        zc[zone(c[1])] += 1

    print("\n===== UCS: obrazets %s  (%dx%d blokov) =====" % (name, w, h))
    print("  blokov=%d svyazey=%d" % (bcount[None], bondCount[None]))
    print("  UCS = %.2f MPa  pri deformacii eps=%.4f%%  (pervyj razryv)"
          % (ucs / 1e6, 100 * eps_f))
    print("  E (izmerennyj) = %.2f GPa   (zadano %.1f GPa)   nu = %.2f"
          % (E_meas / 1e9, E_GPa, nu_meas))
    print("  otnoshenie UCS/sigma_t = %.1f   (sigma_t=rp=%.1f MPa)"
          % (ucs / 1e6 / RP_MPa, RP_MPa))
    print("  tipy razryvov dbg[tens,crush,shear,bend] = %s   vsego=%d"
          % (g, broken[None]))
    print("  treshchin=%d  goriz.svyazi(vert.tresh)=%d  vert.svyazi(goriz.tresh)=%d"
          "  zony verh/ser/niz=%d/%d/%d"
          % (len(cracks), nh, nv, zc['verh'], zc['ser'], zc['niz']))
    return dict(name=name, w=w, h=h, ucs=ucs, E=E_meas, nu=nu_meas, dbg=g,
                broken=broken[None], nh=nh, nv=nv)


def main():
    rows = [run_ucs(n, w, h) for (n, w, h) in SPECS]
    print("\n\n================= SVODKA UCS =================")
    print("obrazets  UCS(MPa)  E(GPa)  nu   UCS/sigma_t  razryvy[t,c,s,b]")
    for r in rows:
        print("%-7s  %8.2f  %6.2f  %.2f  %8.1f     %s"
              % (r['name'], r['ucs'] / 1e6, r['E'] / 1e9, r['nu'],
                 r['ucs'] / 1e6 / RP_MPa, r['dbg']))
    print("\nReal'nyj peschanik: UCS 20..80 MPa, E 5..50 GPa, UCS/sigma_t ~ 10..20.")
    print("Sravnenie otnosheniya UCS/sigma_t - klyuchevoj masshtabno-invariantnyj pokazatel'.")


if __name__ == "__main__":
    main()
