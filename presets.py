"""ГЕОМАССИВ/2D · DEM — пресеты сцен и схемы (presets)

Генераторы готовых сцен (выработка, склон, мозаика, наклонные слои, образцы
UCS и бразильского теста), а также сохранение/загрузка модели в JSON (./scenes).
Модель передаётся в решатель через add_block_func/add_bond_func или build_model.
"""

import io
import json
import math
import os
import random

import numpy as np
import taichi as ti

from solver import *
from solver import _load_bedrock
from water import water_inject_func

# текущая ориентация СОБРАННОЙ модели к горизонту (град, 0 = как сгенерирована).
# generate() обнуляет её; GU видело её через presets.model_angle (см. gui.py).
model_angle = 0.0

# =================================================== модель: служебные поля генерации
# Позиции блоков/воды/пыли в решателе — в МЕТРАХ (см. solver). Чтобы нарисовать их
# в пикселях (CELL пикселей на клетку), нужен масштаб PX = CELL / LCELL.
PX = CELL / LCELL

# (сетка занятости клеток и пометки разрезов используются только при построении
# модели на регулярной сетке; в физике их нет)
cutR = ti.field(ti.i32, N)
cutD = ti.field(ti.i32, N)

@ti.func
def reset_scene_fields():
    for c in range(N):
        r_filled[c] = 0
        r_block[c] = -1
        cutR[c] = 0
        cutD[c] = 0


@ti.func
def link_grid_bonds():
    c = 0
    while c < N:
        a = r_block[c]
        if a >= 0:
            if (c % COLS) < COLS - 1 and r_block[c + 1] >= 0:
                add_bond_func(a, r_block[c + 1], 1.0, 1 if cutR[c] == 0 else BOND_JOINT)
            if (c // COLS) < ROWS - 1 and r_block[c + COLS] >= 0:
                add_bond_func(a, r_block[c + COLS], 1.0, 1 if cutD[c] == 0 else BOND_JOINT)
        c += 1


@ti.func
def in_ellipse(x: ti.f32, y: ti.f32, tx: ti.f32, ty: ti.f32,
               rx: ti.f32, ry: ti.f32) -> ti.i32:
    dx = (x - tx) / rx
    dy = (y - ty) / ry
    return 1 if dx * dx + dy * dy < 1.0 else 0

# полилиния трещины (2 клика -> A* -> разрыв связей вдоль пути), в клетках
pathX = ti.field(ti.f32, 1024)
pathY = ti.field(ti.f32, 1024)
pathN = ti.field(ti.i32, ())
bondPath = ti.field(ti.i32, 1024)   # индексы целых связей по пути трещины
bondN = ti.field(ti.i32, ())        # сколько связей в bondPath
noisePts = ti.field(ti.f32, 128)
n1Arr = ti.field(ti.f32, COLS)
n2Arr = ti.field(ti.f32, COLS)
hgtArr = ti.field(ti.f32, COLS)
r_filled = ti.field(ti.i32, N)
r_block = ti.field(ti.i32, N)

# ================================================================== генерация модели
# ================================================================== генерация модели
@ti.func
def cut_between(ax: ti.i32, ay: ti.i32, bxx: ti.i32, byy: ti.i32):
    ok = (ax >= 0 and ay >= 0 and bxx >= 0 and byy >= 0
          and ax < COLS and ay < ROWS and bxx < COLS and byy < ROWS)
    if ok:
        dx = bxx - ax
        dy = byy - ay
        if dy == 0 and ti.abs(dx) == 1:
            cutR[ay * COLS + ti.min(ax, bxx)] = 1
        elif dx == 0 and ti.abs(dy) == 1:
            cutD[ti.min(ay, byy) * COLS + ax] = 1
        elif ti.abs(dx) == 1 and ti.abs(dy) == 1:
            x0 = ti.min(ax, bxx)
            y0 = ti.min(ay, byy)
            cutR[y0 * COLS + x0] = 1
            cutD[y0 * COLS + x0] = 1


@ti.func
def noise_line(step: ti.f32):
    m = int(ti.ceil(COLS / step)) + 2
    i = 0
    while i < m:
        noisePts[i] = ti.random()
        i += 1
    x = 0
    while x < COLS:
        t = x / step
        i0 = int(t)
        f = t - i0
        u = (1.0 - ti.cos(f * PI)) * 0.5
        hgtArr[x] = noisePts[i0] * (1.0 - u) + noisePts[i0 + 1] * u
        x += 1


@ti.kernel
def gen_tunnel(weak: ti.i32):
    """Выработка КРУГ: фиксированный радиус 10 клеток (диаметр 20), центр домена."""
    reset_scene_fields()
    tx = COLS * 0.5
    ty = ROWS * 0.5
    # Радиус 10 клеток = 5 м при LCELL 0.5 м. На 128×128 это даёт EDZ
    # ~2.8 клетки и дальнее поле до границы ~5.4·a; на 64×64 — уже 2.2·a,
    # поэтому для количественной аналитики 64 не хватает (см. project.md).
    trx = 10.0
    try_ = trx                      # круг, а не эллипс
    cavX[None] = tx
    cavY[None] = ty
    cavRX[None] = trx
    cavRY[None] = try_
    y = 0
    while y < ROWS:
        x = 0
        while x < COLS:
            cell = y * COLS + x
            fixed = 1
            if y < ROWS - 2:
                if in_ellipse(x + 0.5, y + 0.5, tx, ty, trx, try_) == 1:
                    x += 1
                    continue
                fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            i = add_block_func(x + 0.5, y + 0.5, 1.0, fixed)
            if i >= 0:
                r_filled[cell] = 1
                r_block[cell] = i
            x += 1
        y += 1
    if weak == 1:
        b = 0
        while b < 2:
            base = 13 if b == 0 else 33
            ph = ti.random() * 6.0
            x = 6
            while x < COLS - 6:
                yb = base + int(ti.floor(ti.sin(x * 0.22 + ph) * 1.5 + 0.5))
                if yb > 1 and yb < ROWS - 3:
                    cutD[yb * COLS + x] = 1
                x += 1
            b += 1
        jn = 0
        while jn < 3:
            jx = int(8.0 + ti.random() * (COLS - 16))
            ang = PI * 0.5 + (ti.random() - 0.5) * 0.5
            cx2 = jx
            cy2 = 1
            fx = jx + 0.5
            fy = 1.5
            s = 0
            while s < 40:
                fx += ti.cos(ang) * 0.9
                fy += ti.sin(ang) * 0.9
                ang += (ti.random() - 0.5) * 0.3
                nx = int(fx)
                ny = int(fy)
                if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 2:
                    break
                if nx != cx2 or ny != cy2:
                    cut_between(cx2, cy2, nx, ny)
                    cx2 = nx
                    cy2 = ny
                s += 1
            jn += 1
    link_grid_bonds()


def gen_proto_arch(A=10, H=10, notch=4):
    """Пресет «свод Протодьяконова» — плоская кровля на малой глубине.

    A — половина пролёта (клетки), H — высота выработки. Кровля плоская,
    в центре — небольшой начальный вывал (notch), который запускает
    образование свода. Это НЕ глубокая выработка: top load (depth) = 0,
    K0 = 0, работает только собственный вес.

    Геометрия подобрана под A=10 (a=5 м), H=10, notch=4: при
    rp=0.2 МПа, Rc=2 МПа, E=1 ГПа, mu=0.7 после 300 кадров остаётся
    устойчивый свод высотой ~5 клеток (2.5 м), дальнее поле цело.
    Для сравнения: h=a/f; при f=2 (Rc=20 МПа) h=2.5 м — совпадает;
    номинальное Rc=2 МПа (f=0.2) дало бы h=25 м и в этой bonded-модели
    не воспроизводится (нужны трещины/ослабленные плоскости).
    """
    tx = COLS * 0.5
    roof = ROWS * 0.5 - H * 0.5
    floor = roof + H
    blocks = []
    idx = {}
    for j in range(ROWS):
        for i in range(COLS):
            x = i + 0.5
            y = j + 0.5
            if tx - A <= x < tx + A and roof <= y < floor:
                continue
            if notch > 0 and abs(x - tx) < notch * 0.5 and roof - 1 <= y < roof:
                continue
            fixed = 1 if j >= ROWS - P[None].fixedRows else 0
            idx[(i, j)] = len(blocks)
            blocks.append((x, y, 1.0, fixed))
    bonds = []
    for (i, j), a in idx.items():
        if (i + 1, j) in idx:
            bonds.append((a, idx[(i + 1, j)], 1.0, 1))
        if (i, j + 1) in idx:
            bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    cx = tx
    cy = roof + H * 0.5
    build_model(blocks, bonds, [], (cx, cy, A, H * 0.5))
    update_mass_rho()
    loadCur[None] = 0.0
    loadXCur[None] = 0.0
    refresh_field()
    print("proto_arch: blocks=%d bonds=%d A=%d H=%d notch=%d" %
          (bcount[None], bondCount[None], A, H, notch))
    return cx, cy, A, H * 0.5


@ti.kernel
def gen_standard(weak: ti.i32):
    """Выработка СТАНДАРТ (неглубокая шахта): полукруглая арка сверху,
    вертикальные прямые стены и горизонтальный плоский пол."""
    reset_scene_fields()
    tx = COLS * 0.5
    floor_y = float(ROWS) * 0.72
    # Фиксированные размеры выработки (клетки), не зависят от размера домена.
    hw = 9.0                        # половина ширины
    wh = 6.0                        # высота прямых стен
    arch_cy = floor_y - wh          # пята арки (верх стен)
    # габаритный эллипс для fill_pct/воды (сама форма — арка+прямоугольник)
    cavX[None] = tx
    cavY[None] = floor_y - 0.5 * (wh + hw)
    cavRX[None] = hw + 1.0
    cavRY[None] = 0.5 * (wh + hw) + 1.0
    y = 0
    while y < ROWS:
        x = 0
        while x < COLS:
            cell = y * COLS + x
            xf = x + 0.5
            yc = y + 0.5
            fixed = 1
            if y < ROWS - 2:
                inside = 0
                # прямоугольная часть: отвесные стены + плоский пол
                if xf >= tx - hw and xf <= tx + hw and yc >= arch_cy and yc <= floor_y:
                    inside = 1
                # полукруглый свод только НАД пятой арки (ниже — стены/пол,
                # иначе круг вырезал бы скруглённое дно)
                dxa = xf - tx
                dya = yc - arch_cy
                if yc <= arch_cy and dxa * dxa + dya * dya <= hw * hw:
                    inside = 1
                if inside == 1:
                    x += 1
                    continue
                fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            i = add_block_func(x + 0.5, y + 0.5, 1.0, fixed)
            if i >= 0:
                r_filled[cell] = 1
                r_block[cell] = i
            x += 1
        y += 1
    if weak == 1:
        b = 0
        while b < 2:
            base = int(ROWS * 0.18) if b == 0 else int(ROWS * 0.38)
            ph = ti.random() * 6.0
            x = 6
            while x < COLS - 6:
                yb = base + int(ti.floor(ti.sin(x * 0.22 + ph) * 1.5 + 0.5))
                if yb > 1 and yb < ROWS - 3:
                    cutD[yb * COLS + x] = 1
                x += 1
            b += 1
    link_grid_bonds()


@ti.kernel
def gen_slope(weak: ti.i32):
    reset_scene_fields()
    noise_line(11.0)
    for x in range(COLS):
        n1Arr[x] = hgtArr[x]
    noise_line(4.0)
    for x in range(COLS):
        n2Arr[x] = hgtArr[x]
    for x in range(COLS):
        c1 = x - COLS * 0.44
        c2 = x - COLS * 0.8
        m = 17.0 * ti.exp(-c1 * c1 / (2.0 * 15.0 * 15.0)) + 8.0 * ti.exp(-c2 * c2 / (2.0 * 8.0 * 8.0))
        hgtArr[x] = ROWS * 0.66 - m - n1Arr[x] * 5.0 - n2Arr[x] * 2.2
    cavX[None] = COLS * (0.4 + ti.random() * 0.12)
    cavY[None] = ROWS * 0.74
    cavRX[None] = 5.0 + ti.random() * 2.5
    cavRY[None] = 2.8 + ti.random() * 1.4
    y = 0
    while y < ROWS:
        x = 0
        while x < COLS:
            cell = y * COLS + x
            fixed = 1
            if y < ROWS - 2:
                cutoff = ti.max(4, int(hgtArr[x] + 0.5))
                if y < cutoff:
                    x += 1
                    continue
                if in_ellipse(x + 0.5, y + 0.5, cavX[None], cavY[None], cavRX[None], cavRY[None]) == 1:
                    x += 1
                    continue
                fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            i = add_block_func(x + 0.5, y + 0.5, 1.0, fixed)
            if i >= 0:
                r_filled[cell] = 1
                r_block[cell] = i
            x += 1
        y += 1
    if weak == 1:
        nj = 4 + int(ti.random() * 3)
        jn = 0
        while jn < nj:
            jx = int(4.0 + ti.random() * (COLS - 8))
            jy = 0
            while jy < ROWS - 1 and r_filled[jy * COLS + jx] == 0:
                jy += 1
            ang = PI * 0.5 + (ti.random() - 0.5) * 1.1
            cx2 = jx
            cy2 = jy
            fx = jx + 0.5
            fy = jy + 0.5
            ln = int(7.0 + ti.random() * 11)
            s = 0
            while s < ln:
                fx += ti.cos(ang) * 0.9
                fy += ti.sin(ang) * 0.9
                ang += (ti.random() - 0.5) * 0.55
                nx = int(fx)
                ny = int(fy)
                if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 2 or r_filled[ny * COLS + nx] == 0:
                    break
                if nx != cx2 or ny != cy2:
                    cut_between(cx2, cy2, nx, ny)
                    cx2 = nx
                    cy2 = ny
                s += 1
            jn += 1
    link_grid_bonds()
    k = 0
    while k < 240:
        a = ti.random() * 2.0 * PI
        rr = ti.sqrt(ti.random())
        water_inject_func(
            cavX[None] + ti.cos(a) * rr * cavRX[None] * 0.85,
            cavY[None] + ti.sin(a) * rr * cavRY[None] * 0.6 + cavRY[None] * 0.25,
            0.02,
        )
        k += 1


@ti.kernel
def gen_solid():
    """Полностью заполненный массив породой (монолит без полостей)."""
    reset_scene_fields()
    c = 0
    while c < N:
        fixed = 1 if c // COLS >= ROWS - P[None].fixedRows else 0
        i = add_block_func((c % COLS) + 0.5, (c // COLS) + 0.5, 1.0, fixed)
        if i >= 0:
            r_filled[c] = 1
            r_block[c] = i
        c += 1
    link_grid_bonds()


# ================================================================== наклонные слои
# НаклонНАЯ решётка: u — вдоль слоя, v — поперёк. Мировые координаты центра блока:
#   x = COLS/2 + u·cosθ − v·sinθ,  y = ROWS/2 + u·sinθ + v·cosθ   (клетки, y вниз)
# Слои и ослабленные плоскости идут ПОД УГЛОМ θ к горизонтали, блоки повёрнуты
# (brot=θ) — OBB-контакт это поддерживает. Гравитация и давление вышележащих
# пород (ovf/sigVI решателя) остаются ВЕРТИКАЛЬНЫМИ, как в реальном массиве:
# в наклонных слоях вес столба даёт вертикальное σ_v, а связи работают на
# сдвиг/отрыв вдоль наклонных швов — распределение сил другое, и оно возникает
# из геометрии честно, без поправки "вручную".
LATN = 160          # сторона решётки (u,v ∈ [−80, 80) — с запасом на диагональ)
LATH = LATN // 2
latB = ti.field(ti.i32, LATN * LATN)


@ti.kernel
def gen_tilted(weak: ti.i32, ct: ti.f32, st: ti.f32, cav: ti.i32):
    """Массив с НАКЛОННЫМИ слоями (θ = atan2(st, ct)), опция: полость выработки.

    weak=1 — две ослабленные плоскости ПО СЛОЯМ (v = WV1/WV2: связи поперёк слоя
    в этих полосах создаются сразу порванными, intact=0 — наклонные швы, вдоль
    которых массив может съезжать). cav=1 — эллиптическая выработка (мировые
    координаты, как в gen_tunnel: горизонтальная выработка в наклонных слоях)."""
    reset_scene_fields()
    for c in range(LATN * LATN):
        latB[c] = -1
    tx = COLS * 0.5
    ty = ROWS * 0.5
    trx = 10.0
    try_ = trx                      # круглая выработка и в наклонных слоях
    cavX[None] = tx
    cavY[None] = ty
    cavRX[None] = trx
    cavRY[None] = try_
    tilt = ti.atan2(st, ct)
    ox0 = COLS * 0.5
    oy0 = ROWS * 0.5
    WV1 = -6        # ослабленная плоскость 1 (номер полосы v, связь v→v+1 рвётся)
    WV2 = 7         # ослабленная плоскость 2
    v = -LATH + 2
    while v < LATH - 2:
        u = -LATH + 2
        while u < LATH - 2:
            x = ox0 + u * ct - v * st
            y = oy0 + u * st + v * ct
            ok = 1
            if x < 0.75 or x > COLS - 0.75 or y < 0.75 or y > ROWS - 0.5:
                ok = 0
            if ok == 1 and cav == 1:
                dx = (x - tx) / trx
                dy = (y - ty) / try_
                if dx * dx + dy * dy < 1.08:
                    ok = 0
            if ok == 1:
                fixed = 1 if (y >= ROWS - 2.0 or y >= ROWS - P[None].fixedRows) else 0
                i = add_block_func(x, y, 1.0, fixed)
                if i >= 0:
                    brot[i] = tilt      # блок повёрнут вдоль слоя
                    latB[(u + LATH) * LATN + (v + LATH)] = i
                    cx2 = int(x)
                    cy2 = int(y)
                    if cx2 >= 0 and cx2 < COLS and cy2 >= 0 and cy2 < ROWS:
                        c2 = cy2 * COLS + cx2
                        if r_block[c2] < 0:
                            r_block[c2] = i
                            r_filled[c2] = 1
            u += 1
        v += 1
    # связи вдоль (u+1) и поперёк (v+1) слоя; в ослабленных полосах — рваные.
    # Индексные переменные vv/uu — НЕ v/u: taichi запрещает переиспользовать
    # имена уже объявленных в ядре переменных как переменные циклов.
    for vv in range(-LATH + 2, LATH - 3):
        for uu in range(-LATH + 2, LATH - 3):
            a = latB[(uu + LATH) * LATN + (vv + LATH)]
            if a >= 0:
                b = latB[(uu + 1 + LATH) * LATN + (vv + LATH)]
                if b >= 0:
                    add_bond_func(a, b, 1.0, 1)
                b = latB[(uu + LATH) * LATN + (vv + 1 + LATH)]
                if b >= 0:
                    intact = 1
                    if weak == 1 and (vv == WV1 or vv == WV2):
                        intact = BOND_JOINT
                    add_bond_func(a, b, 1.0, intact)


def build_bigblocks():
    """Сцена с блоками ПРОИЗВОЛЬНОГО размера (мозаика 1..3 клетки).

    Модель строится на CPU (списки) и передаётся в решатель через build_model —
    демонстрация, что решатель не привязан ни к сетке, ни к размеру блока.
    """
    cx = COLS * 0.5
    cy = ROWS * 0.6
    rx = 7.0
    ry = 5.0
    occ = [[0] * COLS for _ in range(ROWS)]
    for y in range(ROWS):
        for x in range(COLS):
            dx = (x + 0.5 - cx) / rx
            dy = (y + 0.5 - cy) / ry
            if dx * dx + dy * dy < 1.0:
                occ[y][x] = -1   # полость выработки
    blocks = []
    for y in range(ROWS):
        for x in range(COLS):
            if occ[y][x] != 0:
                continue
            maxs = min(3, COLS - x, ROWS - y)
            r = random.random()
            s = 3 if r < 0.2 else (2 if r < 0.55 else 1)
            if s > maxs:
                s = maxs
            fits = True
            for yy in range(y, y + s):
                for xx in range(x, x + s):
                    if occ[yy][xx] != 0:
                        fits = False
                        break
                if not fits:
                    break
            if not fits:
                occ[y][x] = 1
                continue
            fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            idx = len(blocks)
            blocks.append((x + s / 2.0, y + s / 2.0, float(s), fixed))
            for yy in range(y, y + s):
                for xx in range(x, x + s):
                    occ[yy][xx] = idx + 1
    bonds = []
    seen = set()
    for y in range(ROWS):
        for x in range(COLS):
            b = occ[y][x]
            if b <= 0:
                continue
            a0 = b - 1
            for (xx, yy) in ((x + 1, y), (x, y + 1)):
                if xx >= COLS or yy >= ROWS:
                    continue
                c = occ[yy][xx]
                if c <= 0 or c == b:
                    continue
                a1 = c - 1
                if (a0, a1) in seen or (a1, a0) in seen:
                    continue
                dx = blocks[a1][0] - blocks[a0][0]
                dy = blocks[a1][1] - blocks[a0][1]
                # длина покоя — ИСТИННОЕ расстояние между центрами (в клетках,
                # add_bond_func переведёт в метры): max-ось занижала его у пар
                # «большой+малый» блок (смещение центров диагональное), связь
                # создавалась НАТЯНУТОЙ и рвалась лишний раз.
                rest = math.sqrt(dx * dx + dy * dy)
                bonds.append((a0, a1, rest, 1))
                seen.add((a0, a1))
    water = []
    for _ in range(200):
        a = random.uniform(0.0, 2.0 * math.pi)
        rr = math.sqrt(random.random())
        water.append((cx + math.cos(a) * rr * rx * 0.85,
                      cy + math.sin(a) * rr * ry * 0.6 + ry * 0.25,
                      0.02, math.pi * 0.5))
    build_model(blocks, bonds, water, (cx, cy, rx, ry))


def gen_ucs(w=10, h=20, fixed_rows=1):
    """Пресет ОДНООСНОГО СЖАТИЯ (UCS): монолитный образец w×h блоков, нижний
    ряд закреплён (плита пресса). Нагрузка подаётся сверху слайдером «порода над
    схемой» (mode 0), K0=0, границы/досыпка выключены. Возвращает границы
    образца в клетках (x0, y0, x1, y1) — для наведения камеры."""
    x0 = (COLS - w) // 2
    y0 = 2
    blocks = []
    idx = {}
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
    build_model(blocks, bonds, [], None)
    refillOff[None] = 1
    refresh_field()
    return x0, y0, x0 + w, y0 + h


def gen_brazil(R=12, cap=2):
    """Пресет БРАЗИЛЬСКОГО ТЕСТА: круглый диск радиуса R клеток. Нагрузка и опора
    приложены по малой центральной дуге (|i|<=cap) — чтобы не смять края у
    контакта. Возвращает ((cx, cy, R), список индексов верхних блоков)."""
    cx = COLS * 0.5
    cy = ROWS * 0.5
    blocks = []
    idx = {}
    for j in range(-R, R + 1):
        for i in range(-R, R + 1):
            if i * i + j * j <= R * R:
                idx[(i, j)] = len(blocks)
                blocks.append((cx + i + 0.5, cy + j + 0.5, 1.0, 0))
    top_cap = sorted(idx[(i, j)] for (i, j) in idx if j >= R - 2 and abs(i) <= cap)
    bot_cap = sorted(idx[(i, j)] for (i, j) in idx if j <= -R + 2 and abs(i) <= cap)
    for a in bot_cap:
        x_, y_, s_, _ = blocks[a]
        blocks[a] = (x_, y_, s_, 1)
    bonds = []
    for (i, j), a in idx.items():
        if (i + 1, j) in idx:
            bonds.append((a, idx[(i + 1, j)], 1.0, 1))
        if (i, j + 1) in idx:
            bonds.append((a, idx[(i, j + 1)], 1.0, 1))
    build_model(blocks, bonds, [], None)
    refillOff[None] = 1
    refresh_field()
    return (cx, cy, float(R)), top_cap


def gen_beam(w=48, h=4, sup=2):
    """Пресет БАЛКИ НА ДВУХ ОПОРАХ: горизонтальная балка w×h клеток, нижний
    ряд закреплён ТОЛЬКО у концов (две опоры по sup клеток). Свободная
    середина прогибается; нагрузка — узкой площадкой в центре верхнего ряда
    (прикладывается в GUI, test_kind="beam"). Возвращает bbox образца в
    клетках (x0, y0, x1, y1) и индексы блоков площадки."""
    w = max(8, min(int(w), COLS - 8))
    h = max(3, min(int(h), max(3, ROWS // 8)))
    sup = max(1, min(int(sup), w // 4))
    x0 = (COLS - w) // 2
    y0 = max(2, ROWS // 8)
    blocks = []
    idx = {}
    for j in range(h):
        for i in range(w):
            # опоры — нижний ряд у левого и правого торца балки
            fixed = 1 if (j == h - 1 and (i < sup or i >= w - sup)) else 0
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
    build_model(blocks, bonds, [], None)
    refillOff[None] = 1
    refresh_field()
    # нагрузка — площадкой ~w/5 клеток в центре: на 3 блоках DEM давал
    # локальное продавливание, а равномерно по всему верху балка почти не
    # гнулась. Для гибкой балки (w/h ~12) площадка даёт изгибную трещину снизу.
    c = w // 2
    half = max(1, w // 10)
    top = [idx[(i, 0)] for i in range(max(0, c - half), min(w, c + half + 1))]
    return (x0, y0, x0 + w, y0 + h), top


def gen_buckling(w=2, h=50, fixed_rows=1):
    """Пресет ПРОДОЛЬНОГО ИЗГИБА (устойчивость стойки): вертикальная стойка
    w×h клеток, нижний ряд закреплён. Осевая нагрузка — сверху, малый боковой
    толчок задаётся в GUI (seed): без него идеально симметричная стойка может
    не выпучиться. Возвращает bbox в клетках и индексы блоков верхнего ряда."""
    w = max(2, min(int(w), max(2, COLS // 8)))
    h = max(20, min(int(h), ROWS - 8))
    x0 = (COLS - w) // 2
    y0 = max(2, (ROWS - h) // 2)
    blocks = []
    idx = {}
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
    build_model(blocks, bonds, [], None)
    refillOff[None] = 1
    refresh_field()
    top = [idx[(i, 0)] for i in range(w)]
    return (x0, y0, x0 + w, y0 + h), top


def generate(mode, weak, tilt_deg=0.0):
    """Построить модель сцены и передать её в решатель.

    mode 0 — выработка-круг (единичные блоки), mode 1 — склон,
    mode 2 — мозаика произвольных размеров (build_model),
    mode 3 — выработка-стандарт (арка + прямые стены и пол).
    weak — ослабленные плоскости.
    tilt_deg — НАКЛОН СЛОЁВ к горизонтали (градусы, + = падение вправо):
    применяется к сцене «выработка» и к «Заполнить массив»; гравитация и
    давление вышележащих пород остаются вертикальными (см. gen_tilted)."""
    global model_angle
    model_angle = 0.0      # новая сцена — ориентация "как сгенерирована"
    P[None].tiltC = 1.0    # сброс наклона модели (локальные оси = мировые)
    P[None].tiltS = 0.0
    if mode == 2:
        build_bigblocks()
    else:
        reset_all()
        refillOff[None] = 1   # новая сцена — досыпка сверху ОТКЛЮЧЕНА (по умолчанию;
                              # включается кнопкой "Dosypka" в GUI). Спавн шёл в
                              # занятые клетки без проверки -> "толчки" и трещины.
        if mode == 0:
            if abs(tilt_deg) > 0.5:
                t = math.radians(tilt_deg)
                gen_tilted(weak, math.cos(t), math.sin(t), 1)
            else:
                gen_tunnel(weak)
        elif mode == 3:
            gen_standard(weak)
        else:
            gen_slope(weak)
    # Боковое давление задаётся прямой силой sideF (compute_side_load в solver),
    # престресса в модели нет. Перегруз loadCur набирается ПЛАВНО в compute_top_load_fused
    # (LOAD_TIME: линейно 0 -> ровно 100% за LOAD_TIME с внутреннего времени).
    loadCur[None] = 0.0
    loadXCur[None] = 0.0
    refresh_field()


@ti.kernel
def rebuild_occ():
    for c in range(N):
        r_block[c] = -1
    for i in range(bcount[None]):
        # Полный AABB блока (с учётом поворота), а не только центральная клетка:
        # иначе блок, сдвинувшийся или повёрнутый со "своей" клетки, накрывает
        # соседние, кисть штампует новый блок ВПЕРЕКРЫСТ ему - гигантское
        # проникновение (kконтакта=1e9) вышибает старые блоки ("отлетают"),
        # рвёт связи ("ломают") и прилипает к их боковым граням ("висят").
        r = block_prj(i)
        x0 = clamp_cell(bx[i] - r + 1e-4)
        x1 = clamp_cell(bx[i] + r - 1e-4)
        y0 = row_cell(by[i] - r + 1e-4)
        y1 = row_cell(by[i] + r - 1e-4)
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                r_block[cy * COLS + cx] = i

# ------------------------------------------------------------------ сохранение/загрузка схемы
SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")


@ti.kernel
def _apply_brot(vals: ti.types.ndarray()):
    for i in range(vals.shape[0]):
        if i < bcount[None]:
            brot[i] = vals[i]


def _scene_params():
    """Набор настраиваемых параметров породы/решателя — как в GUI (см. gui.py)."""
    return {
        "rho": float(P[None].rho), "E": float(P[None].E), "mu": float(P[None].mu),
        "rp": float(P[None].rp), "mcBond": float(P[None].mcBond),
        "nuSplit": float(P[None].nuSplit), "relBend": float(P[None].relBend),
        "plastYield": float(P[None].plastYield), "plastLimit": float(P[None].plastLimit),
        "plastFlow": float(P[None].plastFlow), "creepRecover": float(P[None].creepRecover),
        "creepDev": float(P[None].creepDev), "hardenConf": float(P[None].hardenConf),
        "tensConf": float(P[None].tensConf), "Gf": float(P[None].Gf),
        "k0": float(P[None].k0), "g": float(P[None].g),
        "depth": float(P[None].depth), "loadX": float(P[None].loadX), "head": float(P[None].head),
        "fixedRows": int(P[None].fixedRows), "walls": int(P[None].walls),
        "substeps": int(P[None].substeps), "kick": float(P[None].kick),
        "splitN": float(P[None].splitN), "stabC": float(P[None].stabC),
        "vmax": float(P[None].vmax), "lateralFloor": float(P[None].lateralFloor),
        "sleepV2": float(P[None].sleepV2), "sleepA2": float(P[None].sleepA2),
        "rollMu": float(P[None].rollMu), "rockLift": float(P[None].rockLift),
        "brkCap": int(P[None].brkCap), "dmgMax": float(P[None].dmgMax),
        "dmgDecay": float(P[None].dmgDecay),
        "rc": float(P[None].rp * P[None].rcFactor),
        "rcFactor": float(P[None].rcFactor),
        "fsFactor": float(P[None].fsFactor), "cn": float(P[None].cn),
        "cnc": float(P[None].cnc),
        "soften": int(P[None].soften), "softResidual": float(P[None].softResidual),
        "absorb": int(P[None].absorb), "absorbWidth": float(P[None].absorbWidth),
        "absorbCoef": float(P[None].absorbCoef), "absorbTop": int(P[None].absorbTop),
    }


def scene_path(name):
    """Путь к файлу схемы по имени слота (без расширения)."""
    return os.path.join(SCENES_DIR, name + ".json")


def list_scenes():
    """Отсортированный список имён сохранённых схем (без расширения .json)."""
    if not os.path.isdir(SCENES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SCENES_DIR) if f.endswith(".json"))


def save_scene(path, mode=0, weak=0, tilt_deg=0.0, camera=None, test=None):
    """Сохранить текущую модель в JSON: блоки, связи, вода, параметры, камера."""
    nb = int(bcount[None])
    nbd = int(bondCount[None])
    xb = bx.to_numpy(); yb = by.to_numpy(); s = bsz.to_numpy(); fx = bfixed.to_numpy()
    bb = bbedrock.to_numpy()
    br = brot.to_numpy()
    A = bondA.to_numpy(); B = bondB.to_numpy(); R = brest.to_numpy(); I = bondIntact.to_numpy()
    blocks = [[float(xb[i] / LCELL), float(yb[i] / LCELL), float(s[i] / LCELL),
               int(fx[i]), float(br[i]), int(bb[i])] for i in range(nb)]
    bonds = []
    for bd in range(nbd):
        if int(I[bd]) == BOND_DEAD:
            continue
        bonds.append([int(A[bd]), int(B[bd]), float(R[bd] / LCELL),
                      1 if int(I[bd]) == BOND_INTACT else 0])
    water = []
    cav = None
    if cavRX[None] > 0.0:
        cav = [float(cavX[None]), float(cavY[None]), float(cavRX[None]), float(cavRY[None])]
    data = {
        "version": 1, "mode": int(mode), "weak": int(weak), "tilt_deg": float(tilt_deg),
        "camera": list(camera) if camera else None, "test": test,
        "params": _scene_params(), "cavity": cav,
        "blocks": blocks, "bonds": bonds, "water": water,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return len(blocks), len(bonds)


def load_scene(path):
    """Загрузить схему из JSON в решатель. Возвращает dict-заголовок (mode/weak/...)."""
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    p = data.get("params", {})
    if p:
        P[None].rho = float(p.get("rho", RHO_R))
        P[None].E = float(p.get("E", 1e10))
        P[None].mu = float(p.get("mu", 0.62))
        P[None].rp = float(p.get("rp", 4.0))
        P[None].mcBond = float(p.get("mcBond", 0.7))
        P[None].nuSplit = float(p.get("nuSplit", 0.25))
        P[None].relBend = float(p.get("relBend", 8e-4))
        P[None].plastYield = float(p.get("plastYield", 0.5))
        P[None].plastLimit = float(p.get("plastLimit", 0.32))
        P[None].plastFlow = float(p.get("plastFlow", 1.6))
        P[None].creepRecover = float(p.get("creepRecover", 0.6))
        P[None].creepDev = float(p.get("creepDev", 0.3))
        P[None].hardenConf = float(p.get("hardenConf", 1.0))
        P[None].tensConf = float(p.get("tensConf", 0.0))
        P[None].Gf = float(p.get("Gf", 100.0))
        P[None].k0 = float(p.get("k0", K0_LATERAL))
        P[None].depth = float(p.get("depth", 0.0))
        P[None].loadX = float(p.get("loadX", 0.0))
        P[None].head = float(p.get("head", 0.0))
        P[None].fixedRows = int(p.get("fixedRows", 2))
        P[None].walls = int(p.get("walls", 1))
        P[None].substeps = int(p.get("substeps", 384))
        P[None].kick = float(p.get("kick", 0.14))
        P[None].splitN = float(p.get("splitN", 0.7))
        P[None].stabC = float(p.get("stabC", STAB_C))
        P[None].vmax = float(p.get("vmax", VMAX))
        P[None].lateralFloor = float(p.get("lateralFloor", LATERAL_FLOOR))
        P[None].sleepV2 = float(p.get("sleepV2", SLEEP_V2))
        P[None].sleepA2 = float(p.get("sleepA2", SLEEP_A2))
        P[None].rollMu = float(p.get("rollMu", ROLL_MU))
        P[None].rockLift = float(p.get("rockLift", ROCK_LIFT))
        P[None].brkCap = int(p.get("brkCap", 24))
        P[None].dmgMax = float(p.get("dmgMax", 6.0))
        P[None].dmgDecay = float(p.get("dmgDecay", 0.85))
        if "rc" in p:
            P[None].rcFactor = (float(p["rc"])
                                / max(float(P[None].rp), 1e-9))
        else:
            P[None].rcFactor = float(p.get("rcFactor", 10.0))
        P[None].fsFactor = float(p.get("fsFactor", 2.0))
        P[None].cn = float(p.get("cn", 0.5))
        P[None].cnc = float(p.get("cnc", 0.6))
        P[None].soften = int(p.get("soften", 0))
        P[None].softResidual = float(p.get("softResidual", 0.15))
        P[None].absorb = int(p.get("absorb", 0))
        P[None].absorbWidth = float(p.get("absorbWidth", 8.0))
        P[None].absorbCoef = float(p.get("absorbCoef", 500.0))
        P[None].absorbTop = int(p.get("absorbTop", 0))
    raw = data.get("blocks", [])
    blocks = [(float(b[0]), float(b[1]), float(b[2]), int(b[3])) for b in raw]
    brots = [float(b[4]) if len(b) > 4 else 0.0 for b in raw]
    bedrock = [int(b[5]) if len(b) > 5 else 0 for b in raw]
    bonds = [(int(b[0]), int(b[1]), float(b[2]), int(b[3])) for b in data.get("bonds", [])]
    water = [(float(w[0]), float(w[1]), float(w[2]), float(w[3])) for w in data.get("water", [])]
    cav = data.get("cavity")
    cavity = (cav[0], cav[1], cav[2], cav[3]) if cav else None
    build_model(blocks, bonds, water, cavity)
    if bedrock:
        _load_bedrock(np.asarray(bedrock, dtype=np.int32))
    update_mass_rho()
    if brots:
        _apply_brot(np.asarray(brots, dtype="float32"))
    refresh_field()
    global model_angle
    model_angle = 0.0
    P[None].tiltC = 1.0    # схема хранится в ориентации "как сгенерирована"
    P[None].tiltS = 0.0
    return data
