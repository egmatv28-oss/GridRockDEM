import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
import math
import render  # noqa: F401
from physstate import (P, COLS, ROWS, LCELL, bcount, bondCount, broken,
                       loadCur, refillOff, energyTotal, energyEl, energyPot,
                       bx, by, bvx, bvy, bsz)
from solver import apply_gravity, apply_rock, step_physics
import presets


def run(frames=120):
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    P[None].fixedRows = 2
    P[None].depth = 1500.0
    P[None].k0 = 1.0
    P[None].walls = 1
    P[None].substeps = 192
    P[None].brkCap = 0
    refillOff[None] = 1
    presets.generate(0, 0, 0.0)
    stats = []
    for f in range(frames):
        step_physics(0)
        if f in (0, 10, 30, 60, 119):
            n = int(bcount[None])
            vx = bvx.to_numpy()[:n]
            vy = bvy.to_numpy()[:n]
            sp = (vx * vx + vy * vy) ** 0.5
            stats.append((f, int(broken[None]), float(loadCur[None]),
                          float(energyEl[None]), float(energyTotal[None]),
                          float(sp.max() if n else 0.0)))
    print("size=%dx%d blocks=%d bonds=%d" % (COLS, ROWS, bcount[None], bondCount[None]))
    for f, br, lc, ee, et, vmax in stats:
        print("f=%3d broken=%4d loadCur=%.3e Eel=%.3e Etot=%.3e vmax=%.3e" %
              (f, br, lc, ee, et, vmax))


if __name__ == "__main__":
    run()
