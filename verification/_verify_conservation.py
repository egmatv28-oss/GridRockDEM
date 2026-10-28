import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Conservation checks for the DEM pair laws.

The pair interactions (contact, bond, friction) must conserve linear and
angular momentum.  Global damping, sleep, the vmax clamp and the boundary
cleanup are explicit NUMERICAL sinks: they are disabled here to measure the
pair laws, and the vmax loss is reported separately.

Run:  C:\\py312\\python.exe verification/_verify_conservation.py
"""
import os
os.environ["GRIDROCKDEM_ARCH"] = "cpu"

import numpy as np  # noqa: E402
import render  # noqa: E402,F401
import solver  # noqa: E402
from physstate import (P, bcount, bmass, bvx, bvy, nbond, bondIntact, bondCount,
                       energyKin, energyTotal, energyFrac)  # noqa: E402
from solver import (build_model, apply_gravity, apply_rock, step_physics,
                    refresh_field, compute_energy, update_mass_rho)  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("OK" if ok else "FAIL", name, detail))
    if not ok:
        FAIL.append(name)


def setup(blocks, bonds=None, gravity=0.0, vmax=1e6, sleepv=0.0, gf=100.0):
    # isolate the pair laws from global numerical sinks
    solver.DAMP_FREE_TICK = 1.0
    solver.DAMP_SUPP_TICK = 1.0
    solver.DAMP_ROT_FREE_TICK = 1.0
    solver.DAMP_ROT_BOND_TICK = 1.0
    apply_rock(10.0, 4.0, 0.62)
    apply_gravity(gravity)
    build_model(blocks, bonds or [], [], None)
    update_mass_rho()
    P[None].walls = 0
    P[None].depth = 0.0
    P[None].k0 = 0.0
    P[None].fixedRows = 0
    P[None].substeps = 192
    P[None].vmax = vmax
    P[None].sleepV2 = sleepv
    P[None].Gf = gf
    P[None].tensConf = 0.0
    refresh_field()


def momentum():
    n = bcount[None]
    m = bmass.to_numpy()[:n]
    vx = bvx.to_numpy()[:n]
    vy = bvy.to_numpy()[:n]
    return float(np.sum(m * vx)), float(np.sum(m * vy))


def kinetic():
    n = bcount[None]
    m = bmass.to_numpy()[:n]
    vx = bvx.to_numpy()[:n]
    vy = bvy.to_numpy()[:n]
    return float(np.sum(0.5 * m * (vx * vx + vy * vy)))


print("== 1. CONTACT PAIR: p conserved, energy dissipates ==")
setup([(20.0, 10.0, 1.0, 0), (22.0, 10.0, 1.0, 0)])
bvx[0] = 2.0
p0 = momentum(); k0 = kinetic()
for _ in range(60):
    step_physics(1)
p1 = momentum(); k1 = kinetic()
rel = abs(p1[0] - p0[0]) / (abs(p0[0]) + 1e-30)
check("contact pair momentum", rel < 1e-5, "relerr=%.2e" % rel)
check("contact pair energy not created", k1 <= k0 * (1.0 + 1e-5),
      "K %.3e -> %.3e" % (k0, k1))

print("== 2. BOND BREAK (brittle + kick): p conserved ==")
setup([(20.0, 10.0, 1.0, 0), (21.0, 10.0, 1.0, 0)],
      bonds=[(0, 1, 1.0, 1)], gf=0.0)
bvx[0] = -20.0; bvx[1] = 20.0
p0 = momentum()
for _ in range(80):
    step_physics(1)
p1 = momentum()
broke = bondIntact.to_numpy()[:bondCount[None]][0] != 1
rel = abs(p1[0] - p0[0]) / (abs(p0[0]) + 1e-30)
check("bond broke", broke, "bondIntact=%s" % bondIntact.to_numpy()[:bondCount[None]])
check("bond-kick momentum", rel < 1e-5, "relerr=%.2e" % rel)

print("== 3. BOND BREAK (cohesive Gf): p conserved, Gf*w tracked ==")
setup([(20.0, 10.0, 1.0, 0), (21.0, 10.0, 1.0, 0)],
      bonds=[(0, 1, 1.0, 1)], gf=100.0)
bvx[0] = -20.0; bvx[1] = 20.0
p0 = momentum(); ef0 = float(energyFrac[None])
for _ in range(80):
    step_physics(1)
p1 = momentum(); ef1 = float(energyFrac[None])
broke = bondIntact.to_numpy()[:bondCount[None]][0] != 1
rel = abs(p1[0] - p0[0]) / (abs(p0[0]) + 1e-30)
check("cohesive bond broke", broke, "bondIntact=%s" % bondIntact.to_numpy()[:bondCount[None]])
check("cohesive momentum", rel < 1e-5, "relerr=%.2e" % rel)
check("Gf*w tracked", ef1 > ef0, "energyFrac %.3e -> %.3e" % (ef0, ef1))

print("== 4. VMAX CLAMP: explicit momentum/energy sink (by design) ==")
setup([(20.0, 10.0, 1.0, 0)], vmax=15.0)
bvx[0] = 25.0
p0 = momentum(); k0 = kinetic()
for _ in range(2):
    step_physics(1)
p1 = momentum(); k1 = kinetic()
check("vmax reduces p (expected sink)", p1[0] < p0[0] * 0.999,
      "p %.3e -> %.3e, K %.3e -> %.3e" % (p0[0], p1[0], k0, k1))

print()
print("ITOG:", "VSE PROVERKI PROYDENY" if not FAIL else ("PROVALY: " + ", ".join(FAIL)))
_sys.exit(1 if FAIL else 0)
