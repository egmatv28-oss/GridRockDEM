import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
r"""Horizontal shear-pair load test.

Checks that P.loadX creates a shear pair:
top exposed movable block gets +loadX, bottom exposed movable block gets -loadX.
Bedrock screens and hidden blocks get no horizontal load.

Run:
    C:\py312\python.exe _verify_loadx.py
"""
import render  # noqa: F401
from physstate import (P, bbedrock, bfixed, loadCur, loadXCur, ovf, ovfX,
                       refillOff)
from solver import (apply_gravity, apply_rock, build_model, compute_top_load,
                    refresh_field)

W = 8
H = 6
TOP_HOLES = (2, 5)
BOT_HOLES = (1, 6)


def build():
    blocks = []
    idx = {}
    for j in range(H):
        for i in range(W):
            idx[(i, j)] = len(blocks)
            blocks.append((i + 0.5, j + 0.5, 1.0, 0))
    bonds = []
    for (i, j), a in idx.items():
        if i + 1 < W:
            bonds.append((a, idx[(i + 1, j)], 1.0, 1))
        if j + 1 < H:
            bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    build_model(blocks, bonds, [], None)
    for i in range(W):
        if i not in TOP_HOLES:
            bbedrock[idx[(i, 0)]] = 3
            bfixed[idx[(i, 0)]] = 1
        if i not in BOT_HOLES:
            bbedrock[idx[(i, H - 1)]] = 3
            bfixed[idx[(i, H - 1)]] = 1
    return idx


def run_case(load_x, depth):
    apply_gravity(0.0)
    apply_rock(10.0, 4.0, 0.62)
    idx = build()
    refillOff[None] = 1
    P[None].k0 = 0.0
    P[None].walls = 0
    P[None].depth = depth
    P[None].loadX = load_x
    P[None].substeps = 8
    refresh_field()
    loadCur[None] = depth * 24500.0
    loadXCur[None] = load_x
    compute_top_load(0)
    top = [idx[(i, 0)] for i in TOP_HOLES]
    bot = [idx[(i, H - 1)] for i in BOT_HOLES]
    screen = [idx[(i, 0)] for i in range(W) if i not in TOP_HOLES]
    screen += [idx[(i, H - 1)] for i in range(W) if i not in BOT_HOLES]
    hidden = [idx[(i, 1)] for i in range(W) if i not in TOP_HOLES]
    hidden += [idx[(i, H - 2)] for i in range(W) if i not in BOT_HOLES]
    top_x = [float(ovfX[k]) for k in top]
    bot_x = [float(ovfX[k]) for k in bot]
    scr_x = [float(ovfX[k]) for k in screen]
    hid_x = [float(ovfX[k]) for k in hidden]
    ok = True
    if load_x != 0.0:
        ok = ok and all(abs(v - load_x) < 1.0 for v in top_x)
        ok = ok and all(abs(v + load_x) < 1.0 for v in bot_x)
        ok = ok and all(abs(v) < 1.0 for v in scr_x + hid_x)
    else:
        ok = ok and all(abs(v) < 1.0 for v in top_x + bot_x + scr_x + hid_x)
    if depth != 0.0:
        ok = ok and all(float(ovf[k]) > 0.0 for k in top)
        ok = ok and all(float(ovf[k]) < 1.0 for k in screen + hidden + bot)
    else:
        ok = ok and all(abs(float(ovf[k])) < 1.0 for k in top + bot + screen + hidden)
    print("loadX=%+.0f depth=%.0f top=%s bot=%s hidden=%s screen=%s -> %s"
          % (load_x, depth, top_x, bot_x, hid_x, scr_x, "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ok = True
    ok = run_case(1.0e6, 0.0) and ok
    ok = run_case(-1.0e6, 0.0) and ok
    ok = run_case(0.0, 100.0) and ok
    print("result:", "PASS" if ok else "FAIL")
