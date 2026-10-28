import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""ВЕРИФИКАЦИЯ №1 — задача Кирша: круглая выработка на глубине.

Сцена «Выработка» (render.gen_tunnel) заполняет домен породой и вырезает
горизонтальную эллиптическую выработку. Сверху прикладывается вес породы
(P.depth), с боков — σ_h = K0·σ_v (P.k0, compute_side_load).

Аналитическое решение Кирша для контура круглой выработки (далёкие напряжения
σ_v вертикально, σ_h = K0·σ_v горизонтально; θ от бока):

    σ_θθ(θ) = σ_v · [ (1+K0) + 2(1−K0)·cos 2θ ]

    · бока (θ=0/180°):  σ_θθ = σ_v·(3 − K0)   — всегда сжатие
    · кровля/подошва:   σ_θθ = σ_v·(3K0 − 1)  — растяжение при K0 < 1/3

Проверяемые предсказания:
    K0 = 1.0  — всестороннее сжатие 2·σ_v, растяжения на контуре НЕТ;
    K0 = 0.5  — кровля/подошва 0.5·σ_v, бока 2.5·σ_v; растяжения нет;
    K0 = 0.0  — кровля/подошва −σ_v (РАСТЯЖЕНИЕ) -> отрыв в кровле/подошве;
                бока 3·σ_v — сильнейшее сжатие.

Печатает по каждому (K0, глубина): типы разрывов, число трещин, их
распределение по зонам и ориентацию, сравнивая с предсказанием Кирша.

Запуск:  C:\\py312\\python.exe verification/_verify_kirsch.py [frames]
"""
import sys

import render  # noqa: F401
from render import generate
from physstate import (P, bcount, broken, dbgBreakN, refillOff,
                       cavX, cavY, cavRX, cavRY)
from solver import apply_gravity, apply_rock, count_intact, diag_stats
from _verify_common import (intact_pair_set, new_cracks_pairs, crack_orientation,
                            ring_split, angle_hist, ascii_map, kirsch_hoop,
                            run_frames, SIGMA_PER_M)


CONFIGS = [1.0, 0.5, 0.0]
DEPTHS = [800.0, 1000.0]


def set_scene(k0, depth):
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].k0 = k0
    P[None].depth = depth
    generate(0, 0)
    refillOff[None] = 1   # без досыпки: не меняется число блоков и индексы


def one_case(k0, depth, frames, verbose=False):
    set_scene(k0, depth)
    cx, cy = cavX[None], cavY[None]
    rx, ry = cavRX[None], cavRY[None]
    start_pairs = intact_pair_set()
    brk0 = [dbgBreakN[k] for k in range(4)]
    nblocks0 = bcount[None]

    run_frames(0, frames)

    cracks = new_cracks_pairs(start_pairs)
    changed = nblocks0 != bcount[None]
    regions = ring_split(cracks, cx, cy, rx, ry)
    nh, nv = crack_orientation(cracks)
    sv = depth * SIGMA_PER_M
    side_k, crown_k = kirsch_hoop(k0, sv)
    rem = [dbgBreakN[k] - brk0[k] for k in range(4)]

    print("\n===== K0=%.2f  depth=%.0f m   (frames=%d) =====" % (k0, depth, frames))
    print("  sigma_v = %.2f MPa   Kirsch: bok=%.2f MPa, krovlya/podoshva=%.2f MPa%s"
          % (sv / 1e6, side_k / 1e6, crown_k / 1e6,
             "   <-- RASTYAZHENIE" if crown_k < 0 else ""))
    print("  predskazanie: %s" % (
        "rastyazheniya NET (razrushenie - sdvig/smyatie po krugu)"
        if crown_k >= 0 else
        "OTRYV v krovle/podoshve + szhatie po bokam"))
    print("  razryvy dbg[tens,crush,shear,bend] = %s   vsego=%d   celih svyazey=%d"
          % (rem, broken[None], count_intact()))
    dmax = diag_stats()
    print("  max smeschenie bloka = %.3f m %s" % (
        dmax, "   [!! MODEL RAZVALSA/NaN]" if (not dmax == dmax or dmax > 4.0) else ""))
    if changed:
        print("  [WARN] izmenilsya bcount (%d->%d): indeksy blokov sdvinulis, karty netochny"
              % (nblocks0, bcount[None]))
    print("  TRESCHIN: %d   goriz.svyazi(vert.tresch)=%d  vert.svyazi(goriz.tresch)=%d"
          % (len(cracks), nh, nv))
    print("  zony: krovlya=%d podoshva=%d bok_L=%d bok_R=%d | blizhneye pole(rho<2.5)=%d dalnee=%d"
          % (regions['top'], regions['bottom'], regions['side_l'], regions['side_r'],
             regions['near'], regions['far']))
    hist = angle_hist(cracks, cx, cy, rmax=2.5 * max(rx, ry), nbins=12)
    print("  uglovaya gistogramma (30 grad/bin, 0=bok, 90=krovlya): %s" % hist)
    if verbose and cracks:
        print(ascii_map(cracks, cavity=(cx, cy, rx, ry)))
    return dict(k0=k0, depth=depth, cracks=len(cracks), nh=nh, nv=nv,
                regions=regions, rem=rem)


def main():
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rows = []
    for dep in DEPTHS:
        for k0 in CONFIGS:
            verbose = (k0 == 0.0 and dep == DEPTHS[-1])
            rows.append(one_case(k0, dep, frames, verbose=verbose))
    print("\n\n================= SVODKA KIRSCH =================")
    print("K0     depth  tresch  goriz/vert  krovlya  podoshva  boka(L+R)  dbg[t,c,s,b]")
    for r in rows:
        g = r['regions']
        print("%.2f  %6.0f  %6d   %4d/%-4d   %6d  %7d   %6d      %s"
              % (r['k0'], r['depth'], r['cracks'], r['nh'], r['nv'],
                 g['top'], g['bottom'], g['side_l'] + g['side_r'], r['rem']))
    print("\nOzhidanie Kirsha: pri K0=0 otryvnye treschiny - v krovle/podoshve;")
    print("pri K0=0.5 i K0=1 rastyazheniya na konture net.")


if __name__ == "__main__":
    main()
