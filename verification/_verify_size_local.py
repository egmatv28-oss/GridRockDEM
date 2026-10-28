import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
import math
import render  # noqa: F401
from physstate import (P, COLS, ROWS, bcount, bondCount, broken,
                       loadCur, refillOff, energyEl, bx, by, bvx, bvy)
from solver import apply_gravity, apply_rock, build_model, refresh_field, step_physics
from _verify_common import make_rect_hole

X0 = 12
Y0 = 12
W = 40
H = 40
TX = X0 + 20.0
TY = Y0 + 24.0
TRX = 7.0
TRY = 5.0


def run(frames=120):
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    blocks, bonds, idx = make_rect_hole(
        TX, TY, 0.0, x0=X0, y0=Y0, w=W, h=H, fixed_rows=2,
        rx=TRX, ry=TRY)
    build_model(blocks, bonds, [], (TX, TY, TRX, TRY))
    refillOff[None] = 1
    P[None].depth = 1500.0
    P[None].k0 = 1.0
    P[None].walls = 0
    P[None].fixedRows = 2
    P[None].substeps = 192
    P[None].brkCap = 0
    refresh_field()
    out = []
    for f in range(frames):
        step_physics(0)
        if f in (0, 10, 30, 60, 119):
            n = int(bcount[None])
            vx = bvx.to_numpy()[:n]
            vy = bvy.to_numpy()[:n]
            sp = (vx * vx + vy * vy) ** 0.5
            out.append((f, int(broken[None]), float(loadCur[None]), float(energyEl[None]), float(sp.max() if n else 0.0)))
    print("size=%dx%d local_blocks=%d bonds=%d" % (COLS, ROWS, bcount[None], bondCount[None]))
    for f, br, lc, ee, vmax in out:
        print("f=%3d broken=%4d loadCur=%.3e Eel=%.3e vmax=%.3e" % (f, br, lc, ee, vmax))


if __name__ == "__main__":
    run()
