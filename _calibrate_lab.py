r"""Calibration card for lab tests: E, UCS, sigma_t, Poisson ratio.

Run:
    C:\py312\python.exe _calibrate_lab.py

The script uses the square 2:1/3:1 UCS specimens from _verify_ucs.py and
reports measured values against the sandstone targets. Poisson coupling is
enabled through P.poissonF and P.nuSplit in solver.apply_rock().
"""
import contextlib
import io

import _verify_ucs as v
from physstate import P

TARGET = {
    "E_GPa": 10.0,
    "UCS_MPa": 40.0,
    "ST_MPa": 4.0,
    "nu": 0.25,
}

CAL = {
    "E_GPa": 10.6,
    "RP_MPa": 7.9,
    "MU": 0.62,
    "nuSplit": 0.23,
    "poissonF": 1.0,
    "hardenConf": 1.0,
    "tensConf": 0.0,
    "Gf": 100.0,
}


def run_calibrated():
    v.E_GPa = CAL["E_GPa"]
    v.RP_MPa = CAL["RP_MPa"]
    v.MU = CAL["MU"]
    orig = v.apply_rock

    def apply_rock(E, rp, mu, rho=None, rc_MPa=None):
        orig(E, rp, mu, rho, rc_MPa)
        P[None].nuSplit = CAL["nuSplit"]
        P[None].poissonF = CAL["poissonF"]
        P[None].hardenConf = CAL["hardenConf"]
        P[None].tensConf = CAL["tensConf"]
        P[None].Gf = CAL["Gf"]

    v.apply_rock = apply_rock
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        rows.append(v.run_ucs("2:1", 10, 20, frames=1400))
        rows.append(v.run_ucs("3:1", 10, 30, frames=1400))
    return rows


def main():
    rows = run_calibrated()
    print("Calibration inputs: E=%.2f GPa, rp=%.2f MPa, nuSplit=%.2f, poissonF=%.2f"
          % (CAL["E_GPa"], CAL["RP_MPa"], CAL["nuSplit"], CAL["poissonF"]))
    print("Targets:            E=%.2f GPa, UCS=%.2f MPa, sigma_t=%.2f MPa, nu=%.2f"
          % (TARGET["E_GPa"], TARGET["UCS_MPa"], TARGET["ST_MPa"], TARGET["nu"]))
    print()
    print("spec  measured_E  target_E  measured_UCS  target_UCS  sigma_t  nu     UCS/sigma_t")
    for r in rows:
        ucs = r["ucs"] / 1e6
        E = r["E"] / 1e9
        st = CAL["RP_MPa"]
        print("%-4s  %10.2f  %8.2f  %12.2f  %10.2f  %7.2f  %.2f  %11.2f"
              % (r["name"], E, TARGET["E_GPa"], ucs, TARGET["UCS_MPa"],
                 st, r["nu"], ucs / st))
    print()
    print("Note: with Poisson coupling the model prioritizes nu; UCS/sigma_t is")
    print("lower than the lab target (~5.8 instead of 10) and needs a separate")
    print("strength calibration or a change in the bond failure law.")


if __name__ == "__main__":
    main()
