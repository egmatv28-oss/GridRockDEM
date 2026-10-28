import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""ВЕРИФИКАЦИЯ №3 — бразильский тест (раскалывание диска).

Круглый образец (диск) радиуса R загружается точечной (сосредоточенной)
сжимающей силой P по вертикальному диаметру: снизу — жёсткая опора на
центральный блок, сверху — внешняя сила efy=-P на центральный блок
(solver.set_ext_force, поле efx/efy в physstate).

В центре диска возникает РАСТЯЖЕНИЕ перпендикулярно линии нагрузки:

    σ_t = 2P / (π·D·t)      (D — диаметр, t — толщина, здесь t = 1 м)

Порог разрушения сравниваем с прочностью на отрыв материала (rp) и с
реальным бетоном: предел прочности на растяжение при раскалывании
обычного бетона 2..5 МПа, отрыв при изгибе ~3..6 МПа.

Печатает силу первого разрыва, σ_t, ориентацию/положение трещин и карту.

Запуск:  C:\\py312\\python.exe verification/_verify_brazil.py
"""
import math

import render  # noqa: F401
from physstate import P, bcount, bondCount, broken, bx, by, refillOff, dbgBreakN
from solver import (apply_gravity, apply_rock, step_physics, DT,
                    set_ext_force, clear_ext_forces, count_intact)
from _verify_common import (make_disc, load_model, new_cracks_pairs,
                            intact_pair_set, crack_orientation, ascii_map, LCELL)


R = 12                 # радиус диска, клетки
T = 1.0                # толщина (вне плоскости), м
RP_MPa = 4.0


def run_brazil(Pmax=3.0e8, frames=900, settle=50):
    cx = 32.0
    cy = 32.0
    apply_gravity(0.0)
    apply_rock(10.0, RP_MPa, 0.62)
    P[None].k0 = 0.0
    P[None].walls = 0
    P[None].depth = 0.0
    blocks, bonds, idx, top, bottom = make_disc(cx, cy, R, fixed_bottom=False)

    # нагрузку и опору распределяем по МАЛОЙ дуге (|i|<=cap, кровля/подошва):
    # точечная нагрузка даёт смятие у контакта раньше раскалывания. Доля
    # дуги ~ как у криволинейных щёк пресса в стандартном бразильском тесте.
    cap = 2
    top_cap = sorted(idx[(i, j)] for (i, j) in idx if j >= R - 2 and abs(i) <= cap)
    bot_cap = sorted(idx[(i, j)] for (i, j) in idx if j <= -R + 2 and abs(i) <= cap)
    for a in bot_cap:
        x_, y_, s_, _ = blocks[a]
        blocks[a] = (x_, y_, s_, 1)
    load_model(blocks, bonds, cavity=None)
    refillOff[None] = 1
    D = 2.0 * R * LCELL

    start_pairs = intact_pair_set()
    dbg0 = [int(dbgBreakN[i]) for i in range(4)]
    for _ in range(settle):
        step_physics(1, DT)

    Pfail = None
    rec = []
    q_each = 0.0
    for f in range(frames):
        Q = Pmax * (f + 1) / frames
        q_each = -Q / max(1, len(top_cap))
        for a in top_cap:
            set_ext_force(a, 0.0, q_each)
        step_physics(1, DT)
        if f % 10 == 0:
            rec.append((Q, broken[None]))
        if broken[None] > 0:
            Pfail = Q
            break
    clear_ext_forces()

    cracks = new_cracks_pairs(start_pairs)
    nh, nv = crack_orientation(cracks)
    g = [int(dbgBreakN[i]) - dbg0[i] for i in range(4)]

    print("\n===== BRAZIL'SKII TEST: disk R=%d kletok (D=%.1f m) =====" % (R, D))
    print("  nagruzka po duge: %d blokov sverkhu, %d snizu" % (len(top_cap), len(bot_cap)))
    print("  blokov=%d svyazey=%d" % (bcount[None], bondCount[None]))
    if Pfail is None:
        print("  razryva NE bylo do P=%.1f MN -> sigma_t > %.2f MPa"
              % (Pmax / 1e6, 2 * Pmax / (math.pi * D * T) / 1e6))
    else:
        st = 2.0 * Pfail / (math.pi * D * T)
        print("  P (pervyj razryv) = %.2f MN" % (Pfail / 1e6))
        print("  sigma_t = 2P/(pi D t) = %.2f MPa   (rp modeli = %.1f MPa)"
              % (st / 1e6, RP_MPa))
        print("  sigma_t / rp = %.2f" % (st / 1e6 / RP_MPa))
    print("  tipy razryvov dbg[tens,crush,shear,bend] = %s   vsego=%d"
          % (g, broken[None]))
    print("  treshchin=%d  goriz.svyazi=%d  vert.svyazi=%d" % (len(cracks), nh, nv))
    if cracks:
        xs = [c[0] for c in cracks]
        ys = [c[1] for c in cracks]
        print("  polozhenie treshchin: x %.1f..%.1f (centr=%.1f), y %.1f..%.1f (centr=%.1f)"
              % (min(xs), max(xs), cx, min(ys), max(ys), cy))
        print(ascii_map(cracks, cavity=None))
    return dict(Pfail=Pfail, D=D, cracks=len(cracks), nh=nh, nv=nv, dbg=g)


def main():
    run_brazil()


if __name__ == "__main__":
    main()
