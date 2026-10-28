import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""Mohr-Coulomb failure-plane verification.

Reproduces the setup with a top row of bedrock blocks and two blocks
left non-bedrock.  The model is loaded vertically and the orientation of
the new bond breaks is compared with the Mohr-Coulomb prediction

    phi = atan(mcBond)
    theta = 45 deg + phi / 2

where theta is measured from the horizontal (0 deg = horizontal,
90 deg = vertical).  The script is diagnostic: it prints both the
observed and predicted angles and exits 0 even when no plane is found.

Usage:
    C:\\py312\\python.exe verification/_verify_mohr.py
    C:\\py312\\python.exe verification/_verify_mohr.py --mode free --frames 1800
    C:\\py312\\python.exe verification/_verify_mohr.py --mode open --k0 1
"""
import argparse
import math
import sys

import numpy as np

import render  # noqa: F401  (selects Taichi backend before solver import)
from physstate import (P, bcount, bbedrock, bx, by, broken, COLS, ROWS,
                       LCELL, refillOff, dbgBreakN)
from solver import (apply_gravity, apply_rock, step_physics, refresh_field,
                    count_intact)
from _verify_common import load_model, intact_pair_set, new_cracks_pairs

E_GPA = 10.0
RP_MPa = 4.0
MU = 0.62
W = COLS                 # specimen spans the full width so both side walls act
H = 60
X0 = 0
Y0 = 0
GAP = (COLS // 2 - 1, COLS // 2)   # two central blocks in the top row
MARGIN = 3               # ignore cracks on the top/bottom bedrock interfaces
SUBSTEPS = 64


def build_specimen(gap_mode):
    """Top row: bedrock except GAP; bottom row: bedrock."""
    idx = {}
    blocks = []
    for j in range(H):
        for i in range(W):
            if j == 0 and gap_mode == "open" and i in GAP:
                continue
            top_fixed = (j == 0 and i not in GAP)
            bottom_fixed = (j == H - 1)
            fixed = 1 if (top_fixed or bottom_fixed) else 0
            idx[(i, j)] = len(blocks)
            blocks.append((X0 + i + 0.5, Y0 + j + 0.5, 1.0, fixed))

    bonds = []
    for (i, j), a in idx.items():
        if (i + 1, j) in idx:
            bonds.append((a, idx[(i + 1, j)], 1.0, 1))
        if (i, j + 1) in idx:
            bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    return blocks, bonds, idx


def mark_bedrock(idx, gap_mode):
    for (i, j), k in idx.items():
        top_bedrock = (j == 0 and (gap_mode == "open" or i not in GAP))
        if top_bedrock or j == H - 1:
            bbedrock[k] = 1


def fit_plane(points, min_inliers=5, seed=12345):
    """Robust line fit to crack midpoints; returns (angle_deg, inliers, span)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(seed)
    best = None
    n = len(pts)
    samples = min(1200, max(300, 20 * n))
    for _ in range(samples):
        i, j = rng.choice(n, 2, replace=False)
        dx = pts[j, 0] - pts[i, 0]
        dy = pts[j, 1] - pts[i, 1]
        norm = math.hypot(dx, dy)
        if norm < 0.6:
            continue
        # distance to the candidate line
        dist = np.abs(dy * (pts[:, 0] - pts[i, 0])
                      - dx * (pts[:, 1] - pts[i, 1])) / norm
        inl = dist < 1.05
        cnt = int(inl.sum())
        if cnt < min_inliers:
            continue
        # reject a cluster with no real extent along the line
        q = pts[inl]
        t = (q[:, 0] - pts[i, 0]) * dx / norm + (q[:, 1] - pts[i, 1]) * dy / norm
        span = float(t.max() - t.min())
        if span < 3.0:
            continue
        score = (cnt, span)
        if best is None or score > best[0]:
            best = (score, i, j, inl)

    if best is None:
        return None
    _, _, _, inl = best
    q = pts[inl]
    c = q.mean(axis=0)
    cov = np.cov((q - c).T)
    evals, evecs = np.linalg.eigh(cov)
    tangent = evecs[:, int(np.argmax(evals))]
    angle = math.degrees(math.atan2(abs(tangent[1]), abs(tangent[0])))
    # extent of the selected points along the fitted line
    proj = (q - c) @ tangent
    return angle, int(inl.sum()), float(proj.max() - proj.min())


def estimator_self_test():
    """Sanity check: a synthetic 62.5-degree line must be recovered."""
    pts = []
    a = math.radians(62.5)
    for t in np.linspace(0.0, 20.0, 30):
        pts.append((10.0 + t * math.cos(a) + 0.02 * (t % 3),
                    5.0 + t * math.sin(a) - 0.02 * (t % 2)))
    fit = fit_plane(pts)
    ok = fit is not None and abs(fit[0] - 62.5) < 2.0
    return ok, fit


def run_mode(gap_mode, frames, dmax, k0, walls):
    blocks, bonds, idx = build_specimen(gap_mode)
    apply_gravity(0.0)
    apply_rock(E_GPA, RP_MPa, MU)
    load_model(blocks, bonds)
    mark_bedrock(idx, gap_mode)

    P[None].k0 = float(k0)
    P[None].walls = int(walls)
    P[None].depth = 0.0
    P[None].substeps = SUBSTEPS
    P[None].brkCap = 0
    refillOff[None] = 1
    refresh_field()

    for _ in range(40):
        step_physics(0)

    start = intact_pair_set()
    first_break = None
    for f in range(frames):
        P[None].depth = dmax * (f + 1) / frames
        step_physics(0)
        if first_break is None and broken[None] > 0:
            first_break = (f, float(P[None].depth), int(broken[None]))
        if broken[None] > 220:
            break

    cracks = new_cracks_pairs(start)
    # Cracks on the rigid top/bottom interfaces form horizontal rows that
    # dominate a naive fit; the Mohr-Coulomb band lives inside the body.
    inner = [(float(c[0]), float(c[1])) for c in cracks
             if Y0 + MARGIN < c[1] < Y0 + H - MARGIN]
    outer = [(float(c[0]), float(c[1])) for c in cracks
             if X0 + 1.0 < c[0] < X0 + W - 1.0
             and Y0 + 0.7 < c[1] < Y0 + H - 1.0]
    fit = fit_plane(inner)
    fit_outer = fit_plane(outer)
    return dict(
        mode=gap_mode, frames=f, depth=float(P[None].depth),
        broken=int(broken[None]), cracks=len(cracks), points=len(inner),
        points_outer=len(outer), first_break=first_break, fit=fit,
        fit_outer=fit_outer, intact=count_intact(),
        types=[int(dbgBreakN[i]) for i in range(4)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("free", "open", "both"), default="free")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--dmax", type=float, default=5000.0)
    ap.add_argument("--k0", type=float, default=1.0)
    ap.add_argument("--walls", type=int, default=1)
    args = ap.parse_args()

    phi = math.degrees(math.atan(0.7))
    theta = 45.0 + 0.5 * phi
    print("Mohr-Coulomb on bond: mcBond=0.700 -> phi=%.2f deg" % phi)
    print("Predicted failure plane: theta = 45 + phi/2 = %.2f deg "
          "from horizontal" % theta)
    ok, fit = estimator_self_test()
    print("Estimator self-test: %s%s"
          % ("OK" if ok else "FAIL",
             "" if fit is None else "  (%.2f deg, n=%d)" % (fit[0], fit[1])))
    print("")

    modes = ("free", "open") if args.mode == "both" else (args.mode,)
    rows = []
    for mode in modes:
        r = run_mode(mode, args.frames, args.dmax, args.k0, args.walls)
        rows.append(r)
        print("===== mode=%s =====" % mode)
        print("  specimen: %dx%d blocks, top bedrock row, GAP=%s" % (W, H, GAP))
        print("  blocks=%d bonds=%d intact=%d broken=%d cracks=%d"
              % (bcount[None], r["intact"] + r["broken"], r["intact"],
                 r["broken"], r["cracks"]))
        if r["first_break"] is None:
            print("  first break: none")
        else:
            print("  first break: frame=%d depth=%.0f m broken=%d"
                  % r["first_break"])
        print("  break counters [tens,crush,shear,bend] = %s" % r["types"])
        if r["fit_outer"] is not None:
            print("  all cracks (incl. interfaces): %.2f deg, inliers=%d, span=%.1f"
                  % (r["fit_outer"][0], r["fit_outer"][1], r["fit_outer"][2]))
        if r["fit"] is None:
            print("  observed plane: NOT FOUND (points=%d)" % r["points"])
            print("  result: FAIL - no coherent failure plane")
        else:
            ang, n, span = r["fit"]
            d = ang - theta
            print("  observed plane: %.2f deg from horizontal "
                  "(inliers=%d, span=%.1f cells, points=%d)"
                  % (ang, n, span, r["points"]))
            print("  difference: %+.2f deg  (tolerance +/-10 deg)" % d)
            if abs(d) <= 10.0:
                print("  result: PASS")
            else:
                print("  result: FAIL - plane is not at the MC angle")
        print("")

    print("Summary:")
    for r in rows:
        if r["fit"] is None:
            print("  %-4s observed=NOT FOUND expected=%.1f -> FAIL"
                  % (r["mode"], theta))
        else:
            print("  %-4s observed=%.1f expected=%.1f diff=%+.1f -> %s"
                  % (r["mode"], r["fit"][0], theta, r["fit"][0] - theta,
                     "PASS" if abs(r["fit"][0] - theta) <= 10.0 else "FAIL"))


if __name__ == "__main__":
    main()
