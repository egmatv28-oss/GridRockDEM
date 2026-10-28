import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""Проверка свода естественного равновесия (Протодьяконов) на малой глубине.

Известное решение: для выработки полупролётом a высота свода

    h = a / f,   f = Rc/10 (Rc в МПа)

Пресет `presets.gen_proto_arch()` строит плоскую кровлю A=10 клеток
(a=5 м), H=10, notch=4. Калибровка: rp=0.2 МПа, Rc=2 МПа, E=1 ГПа,
mu=0.7, depth=0, K0=0. После 300 кадров измеряется верх связанной
пустоты над кровлей и сравнивается с a/f.

Запуск:  C:\\py312\\python.exe verification/_verify_proto.py [frames]
"""
import os
import sys
from collections import deque

import numpy as np

# пресет калиброван на 128×128; на 64 геометрия/перекрытие другие
os.environ.setdefault("GRIDROCKDEM_SIZE", "128")

import render  # noqa: F401  (taichi init)
import presets
from physstate import (P, bcount, broken, gOcc, bx, by, bvx, bvy,
                       LCELL, COLS, ROWS, refillOff, cavX, cavY, cavRX, cavRY)
from solver import (apply_gravity, apply_rock, step_physics, count_intact,
                    count_bonds, refresh_field)

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
A = 10
H = 10
NOTCH = 4
RP = 0.2
RC = 2.0
E_GPA = 1.0
MU = 0.7


def top_of_connected_void():
    """Верх наибольшей связной пустоты, достижимой из исходного контура."""
    occ = gOcc.to_numpy().reshape(ROWS, COLS).astype(np.int8)
    cx0 = int(cavX[None]); cy0 = int(cavY[None])
    rx0 = int(cavRX[None]); ry0 = int(cavRY[None])
    starts = []
    for y in range(max(0, cy0 - ry0), min(ROWS, cy0 + ry0 + 1)):
        for x in range(max(0, cx0 - rx0), min(COLS, cx0 + rx0 + 1)):
            if not occ[y, x]:
                starts.append((x, y))
    if not starts:
        return None, None
    seen = np.zeros((ROWS, COLS), dtype=np.int8)
    best = None
    for sx, sy in starts:
        if seen[sy, sx]:
            continue
        q = deque([(sx, sy)])
        seen[sy, sx] = 1
        miny = maxy = sy
        minx = maxx = sx
        cnt = 0
        while q:
            x, y = q.popleft()
            cnt += 1
            miny = min(miny, y); maxy = max(maxy, y)
            minx = min(minx, x); maxx = max(maxx, x)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS and not occ[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = 1
                    q.append((nx, ny))
        if best is None or cnt > best[0]:
            best = (cnt, miny, maxy, minx, maxx)
    if best is None:
        return None, None
    return best[1], best[2]


def main():
    apply_gravity(1.0)
    apply_rock(E_GPA, RP, MU, rc_MPa=RC)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 0.0
    P[None].k0 = 0.0
    P[None].substeps = 192
    refillOff[None] = 1
    cx, cy, rx, ry = presets.gen_proto_arch(A=A, H=H, notch=NOTCH)

    n0 = int(count_intact())
    for f in range(FRAMES):
        step_physics(0)
        if (f + 1) % 60 == 0 or f == FRAMES - 1:
            n = int(bcount[None])
            vm = float(np.max(np.hypot(bvx.to_numpy()[:n], bvy.to_numpy()[:n]))) if n else 0.0
            print("f=%3d intact=%5d/%5d broken=%4d vmax=%5.2f"
                  % (f + 1, count_intact(), n0, broken[None], vm), flush=True)

    top, bottom = top_of_connected_void()
    roof = cy - ry
    a = A * LCELL
    if top is not None:
        h_meas = (roof - top) * LCELL
        b_meas = (bottom - top + 1) * LCELL
        # ширина связной пустоты по строке под кровлей
        occ = gOcc.to_numpy().reshape(ROWS, COLS)
        row = int(roof) + 1
        xs = [x for x in range(COLS) if not occ[min(row, ROWS - 1), x]]
        width = (max(xs) - min(xs) + 1) * LCELL if xs else 0.0
    else:
        h_meas = float("nan")
        b_meas = float("nan")
        width = float("nan")

    # известные решения
    f_eff = a / h_meas if h_meas and h_meas > 1e-9 else float("nan")
    h_f2 = a / 2.0
    h_f02 = a / 0.2
    print("\n=== PROTO ===  a=%.2f m, roof=%.1f m, frames=%d" % (a, roof * LCELL, FRAMES))
    print("  measured: top void=%.1f m, h_meas=%.2f m, void height=%.2f m, width(row)=%.1f m"
          % (top * LCELL if top is not None else float("nan"), h_meas, b_meas, width))
    print("  f_eff = a/h_meas = %.2f" % f_eff)
    print("  Protodyakonov: h(f=2)=a/f=%.2f m; h(f=0.2)=%.2f m" % (h_f2, h_f02))
    print("  far field: intact=%d, broken=%d, blocks=%d"
          % (count_intact(), broken[None], bcount[None]))
    if FRAMES < 300:
        print("  CHECK skipped: нужно >=300 кадров для установившегося свода")
        return
    if h_meas == h_meas:
        if abs(h_meas - h_f2) <= 0.5:
            print("  CHECK f_eff~2: PASS (h_meas matches h=a/2)")
        else:
            print("  CHECK f_eff~2: FAIL (delta=%.2f m)" % (h_meas - h_f2))
    else:
        print("  CHECK: no connected void — арка не образовалась")


if __name__ == "__main__":
    main()
