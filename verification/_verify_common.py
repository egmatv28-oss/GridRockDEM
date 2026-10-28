import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""Общие утилиты верификации DEM-модели (Kirsch / UCS / бразильский тест).

Модели строятся на CPU и грузятся через solver.build_model — решатель не
привязан ни к сцене, ни к размеру блока. Здесь же собраны:
  * построение прямоугольного образца и круглого диска;
  * поиск трещин (связи, целые в начале и не целые после прогона);
  * ASCII-карта массива с трещинами;
  * ориентация/положение трещин и угловая гистограмма вокруг выработки.

Запуск любого скрипта:  C:\\py312\\python.exe verification/_verify_<тест>.py
"""
import math

import numpy as np

import render  # noqa: F401  (инициализирует taichi, единая точка входа)
from physstate import (P, bcount, bondCount, bondIntact, bondA, bondB, bx, by,
                       bsz, gOcc, COLS, ROWS, LCELL, RHO_R, G_PHYS,
                       BOND_INTACT, BOND_JOINT, BOND_DEAD)
from solver import build_model, refresh_field, count_intact, step_physics, DT

# far-field вертикальное напряжение на 1 м глубины, Па/м  (sigma_v = h * SIGMA_PER_M)
SIGMA_PER_M = RHO_R * G_PHYS
# вес столба породы на 1 м глубины (Н на колонку шириной 1 клетку) — как в LOAD_OVER
LOAD_PER_M = RHO_R * G_PHYS * LCELL


# ------------------------------------------------------------------ построение моделей
def make_rect(x0, y0, w, h, fixed_rows=1):
    """Прямоугольный образец w×h клеток (верхний левый угол (x0,y0)).
    Возвращает (blocks, bonds, index[(i,j)]->idx, top_idx)."""
    idx = {}
    blocks = []
    for j in range(h):
        for i in range(w):
            fixed = 1 if j >= h - fixed_rows else 0
            idx[(i, j)] = len(blocks)
            blocks.append((x0 + i + 0.5, y0 + j + 0.5, 1.0, fixed))
    bonds = []
    for j in range(h):
        for i in range(w):
            a = idx[(i, j)]
            if i + 1 < w:
                bonds.append((a, idx[(i + 1, j)], 1.0, 1))
            if j + 1 < h:
                bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    return blocks, bonds, idx


def make_rect_hole(cx, cy, hr, x0=0, y0=0, w=None, h=None, fixed_rows=2,
                   rx=None, ry=None):
    """Прямоугольный образец w×h с эллиптической полостью (cx,cy,rx,ry).
    По умолчанию rx=ry=hr (круг). Стандартный builder для верификаций
    кровли/размерного эффекта."""
    if w is None:
        w = COLS
    if h is None:
        h = ROWS
    if rx is None:
        rx = hr
    if ry is None:
        ry = hr
    idx = {}
    blocks = []
    for j in range(h):
        for i in range(w):
            if rx > 0 and ry > 0 and ((x0 + i + 0.5 - cx) / rx) ** 2 + ((y0 + j + 0.5 - cy) / ry) ** 2 < 1.0:
                continue
            fixed = 1 if j >= h - fixed_rows else 0
            idx[(i, j)] = len(blocks)
            blocks.append((x0 + i + 0.5, y0 + j + 0.5, 1.0, fixed))
    bonds = []
    for (i, j), a in idx.items():
        if (i + 1, j) in idx:
            bonds.append((a, idx[(i + 1, j)], 1.0, 1))
        if (i, j + 1) in idx:
            bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    return blocks, bonds, idx


def make_disc(cx, cy, R, fixed_bottom=True):
    """Круглый диск радиуса R клеток с центром (cx, cy).
    Возвращает (blocks, bonds, index, top_idx, bottom_idx)."""
    idx = {}
    blocks = []
    for j in range(-R, R + 1):
        for i in range(-R, R + 1):
            if i * i + j * j <= R * R:
                idx[(i, j)] = len(blocks)
                blocks.append((cx + i + 0.5, cy + j + 0.5, 1.0, 0))
    bonds = []
    for (i, j), a in list(idx.items()):
        if (i + 1, j) in idx:
            bonds.append((a, idx[(i + 1, j)], 1.0, 1))
        if (i, j + 1) in idx:
            bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    top = bottom = -1
    for (i, j), a in idx.items():
        if j == R and i == 0:
            top = a
        if j == -R and i == 0:
            bottom = a
    if fixed_bottom and bottom >= 0:
        blocks[bottom] = (blocks[bottom][0], blocks[bottom][1], 1.0, 1)
    return blocks, bonds, idx, top, bottom


def load_model(blocks, bonds, cavity=None):
    build_model(blocks, bonds, [], cavity)
    refresh_field()
    return bcount[None], bondCount[None]


# ------------------------------------------------------------------ трещины
def snapshot_intact():
    """Снимок состояния связей: массив статусов + число связей."""
    n = bondCount[None]
    return bondIntact.to_numpy()[:n].copy(), n


def new_cracks(prev, n0):
    """Список трещин: связи, целые в снимке и уже не целые сейчас.
    Каждая запись — (x0, y0, dx, dy, a, b) в КЛЕТКАХ (x0,y0 — середина связи,
    (dx,dy) — ось связи). Швы, которые были целыми при загрузке, не считаются."""
    n = bondCount[None]
    ia = bondIntact.to_numpy()[:n]
    aa = bondA.to_numpy()[:n]
    bb = bondB.to_numpy()[:n]
    xb = bx.to_numpy()
    yb = by.to_numpy()
    out = []
    m = min(n, n0)
    for bd in range(m):
        if prev[bd] == BOND_INTACT and ia[bd] != BOND_INTACT:
            a = int(aa[bd])
            b = int(bb[bd])
            x0 = 0.5 * (xb[a] + xb[b]) / LCELL
            y0 = 0.5 * (yb[a] + yb[b]) / LCELL
            dx = (xb[b] - xb[a]) / LCELL
            dy = (yb[b] - yb[a]) / LCELL
            out.append((x0, y0, dx, dy, a, b))
    return out


def intact_pair_set():
    """Множество целых пар (a,b), a<b — устойчиво к переуплотнению слотов связей
    (compact_bonds сдвигает индексы слотов, но НЕ индексы блоков)."""
    n = bondCount[None]
    ia = bondIntact.to_numpy()[:n]
    aa = bondA.to_numpy()[:n]
    bb = bondB.to_numpy()[:n]
    s = set()
    for bd in range(n):
        if ia[bd] == BOND_INTACT:
            a = int(aa[bd])
            b = int(bb[bd])
            s.add((a, b) if a < b else (b, a))
    return s


def new_cracks_pairs(start_pairs):
    """Трещины = целые пары из start_pairs, которых больше нет среди целых."""
    broken = start_pairs - intact_pair_set()
    xb = bx.to_numpy()
    yb = by.to_numpy()
    out = []
    for (a, b) in broken:
        x0 = 0.5 * (xb[a] + xb[b]) / LCELL
        y0 = 0.5 * (yb[a] + yb[b]) / LCELL
        dx = (xb[b] - xb[a]) / LCELL
        dy = (yb[b] - yb[a]) / LCELL
        if not (math.isfinite(x0) and math.isfinite(y0)):
            continue
        out.append((x0, y0, dx, dy, a, b))
    return out


def crack_orientation(cracks):
    """(n_horiz, n_vert): связь с |dx|>=|dy| -> ГОРИЗОНТАЛЬНАЯ (трещина вертикальная),
    иначе ВЕРТИКАЛЬНАЯ (трещина горизонтальная)."""
    nh = nv = 0
    for (_, _, dx, dy, _, _) in cracks:
        if abs(dx) >= abs(dy):
            nh += 1
        else:
            nv += 1
    return nh, nv


def ascii_map(cracks=None, cavity=None, step=1, mark_rock=True):
    """ASCII-карта массива. '#' порода, ' ' пусто, '*' трещина, 'O' полость."""
    occ = gOcc.to_numpy().reshape(ROWS, COLS)
    grid = [[' '] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if occ[r, c]:
                grid[r][c] = '#' if mark_rock else '.'
    if cavity is not None:
        cx, cy, rx, ry = cavity
        for r in range(ROWS):
            for c in range(COLS):
                dx = (c + 0.5 - cx) / rx
                dy = (r + 0.5 - cy) / ry
                if dx * dx + dy * dy < 1.0:
                    grid[r][c] = 'O'
    if cracks:
        for (x0, y0, _, _, _, _) in cracks:
            c = int(x0)
            r = int(y0)
            if 0 <= r < ROWS and 0 <= c < COLS:
                grid[r][c] = '*'
    lines = []
    for r in range(0, ROWS, step):
        lines.append(''.join(grid[r][c] for c in range(0, COLS, step)))
    return '\n'.join(lines)


def angle_hist(cracks, cx, cy, rmax=None, nbins=12):
    """Гистограмма углов трещин вокруг (cx,cy) (клетки). 0° = вправо (бок),
    90° = вверх (кровля), 180° = влево, 270° = вниз (подошва)."""
    hist = [0] * nbins
    for (x0, y0, _, _, _, _) in cracks:
        dx = x0 - cx
        dy = y0 - cy
        r = math.hypot(dx, dy)
        if not math.isfinite(r):
            continue
        if rmax is not None and r > rmax:
            continue
        a = math.degrees(math.atan2(dy, dx)) % 360.0
        b = int(a / (360.0 / nbins)) % nbins
        hist[b] += 1
    return hist


def ring_split(cracks, cx, cy, rcx, rcy):
    """Разбивка трещин вокруг выработки по зонам Кирша.
    Нормализованный радиус rho = sqrt(((x-cx)/rcx)^2+((y-cy)/rcy)^2).
    Возвращает счётчики по зонам: кровля/подошва (|dy|>|dx|), бока (|dx|>=|dy|),
    ближнее поле (rho<2.5) и дальнее."""
    top = bot = side_l = side_r = 0
    near = far = 0
    for (x0, y0, _, _, _, _) in cracks:
        dx = x0 - cx
        dy = y0 - cy
        rho = math.hypot(dx / rcx, dy / rcy) if (rcx > 0 and rcy > 0) else 0.0
        if rho < 2.5:
            near += 1
        else:
            far += 1
        if abs(dy) > abs(dx):
            if dy > 0:
                top += 1
            else:
                bot += 1
        else:
            if dx > 0:
                side_r += 1
            else:
                side_l += 1
    return dict(top=top, bottom=bot, side_l=side_l, side_r=side_r, near=near, far=far)


def kirsch_hoop(ko, sv):
    """Пиковые тангенциальные напряжения Кирша на контуре круглой выработки,
    σ_θθ = σv·[(1+K0) ± 2(1−K0)], знак + у боков, − у кровли/подошвы.
    Возвращает (side, crown) в Па: crown<0 — растяжение."""
    side = sv * ((1.0 + ko) + 2.0 * (1.0 - ko))
    crown = sv * ((1.0 + ko) - 2.0 * (1.0 - ko))
    return side, crown


def run_frames(mode, frames, dt=None):
    for _ in range(frames):
        step_physics(mode, DT if dt is None else dt)
