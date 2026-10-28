import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
r"""Smoke test for the cellular water model.

Checks:
1. a fresh domain is empty;
2. water falls under gravity and its mass is conserved;
3. solid cells keep water out;
4. a source adds water that reaches the lower half of the domain.

Run:
    C:\py312\python.exe _verify_water.py
"""
import numpy as np
import taichi as ti

from physstate import N, COLS, ROWS, srcX, srcY, srcCount
import water


@ti.kernel
def _fill_rect(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32):
    for y in range(y0, y1):
        for x in range(x0, x1):
            c = y * COLS + x
            if water.solid[c] == 0:
                water.fill[c] = 1.0


def mass():
    return float(water.fill.to_numpy()[:N].sum())


def com_y():
    m = water.fill.to_numpy()[:N]
    ys = np.arange(N) // COLS + 0.5
    s = m.sum()
    return float((m * ys).sum() / s) if s > 0 else 0.0


def run_empty():
    water.water_init_host()
    ok = mass() == 0.0
    print("empty: total=%.2f -> %s" % (mass(), "PASS" if ok else "FAIL"))
    return ok


def run_fall():
    water.water_init_host()
    _fill_rect(COLS // 2 - 3, 2, COLS // 2 + 3, 6)
    m0 = mass()
    c0 = com_y()
    for _ in range(100):
        water.water_step()
    m1 = mass()
    c1 = com_y()
    ok = abs(m1 - m0) / max(m0, 1e-9) < 1e-5 and c1 > c0 + 1.0
    print("fall: mass %.2f -> %.2f, y %.1f -> %.1f -> %s" %
          (m0, m1, c0, c1, "PASS" if ok else "FAIL"))
    return ok


def run_solid():
    water.water_init_host()
    yy = np.arange(N) // COLS
    arr = np.zeros(N, dtype=np.int32)
    arr[yy >= ROWS - 2] = 1
    water.solid.from_numpy(arr)
    _fill_rect(COLS // 2 - 3, 2, COLS // 2 + 3, 6)
    m0 = mass()
    for _ in range(200):
        water.water_step(update_solid=False)
    m1 = mass()
    inside = float(water.fill.to_numpy()[:N][yy >= ROWS - 2].sum())
    ok = abs(m1 - m0) / max(m0, 1e-9) < 1e-5 and inside < 1e-9
    print("solid: mass %.2f -> %.2f, inside floor=%.4f -> %s" %
          (m0, m1, inside, "PASS" if ok else "FAIL"))
    return ok


def run_displace():
    water.water_init_host()
    _fill_rect(COLS // 2 - 3, 4, COLS // 2 + 3, 8)
    m0 = mass()
    c = 5 * COLS + COLS // 2
    water.solid[c] = 1
    water.water_displace_solids()
    m1 = mass()
    ok = abs(m1 - m0) / max(m0, 1e-9) < 1e-5 and float(water.fill[c]) == 0.0
    print("displace: mass %.2f -> %.2f, solid cell fill=%.3f -> %s" %
          (m0, m1, float(water.fill[c]), "PASS" if ok else "FAIL"))
    return ok


def run_source():
    water.water_init_host()
    srcCount[None] = 1
    srcX[0] = COLS * 0.5
    srcY[0] = 3.0
    m_before = mass()
    for _ in range(120):
        water.water_emit_sources(10.0)
        water.water_step()
    m_after = mass()
    yy = np.arange(N) // COLS
    low = float(water.fill.to_numpy()[:N][yy > ROWS // 2].sum())
    ok = m_after > m_before + 10.0 and low > 1.0
    print("source: fill %.2f -> %.2f, lower half=%.2f -> %s" %
          (m_before, m_after, low, "PASS" if ok else "FAIL"))
    srcCount[None] = 0
    return ok


if __name__ == "__main__":
    ok = True
    ok = run_empty() and ok
    ok = run_fall() and ok
    ok = run_solid() and ok
    ok = run_displace() and ok
    ok = run_source() and ok
    print("result:", "PASS" if ok else "FAIL")
