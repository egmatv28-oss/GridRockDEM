"""ГЕОМАССИВ/2D · DEM — интерфейс и главный цикл (gui)

Панель управления, инструменты редактирования (порода/трещина/источник/ластик),
две группы настроек (solver и poroda) и главный цикл окна run_gui. Точки входа —
GridRockDem64.py / GridRockDem128.py (через _bootstrap.py).
"""

import argparse
import heapq
import math
import os
import time

import taichi as ti

import render
import presets
import recorder as rec_mod
from solver import *
from render import *
from presets import *
from render import _CAM, _user32, _app_focused

# текущая ориентация модели (град) живёт в presets.model_angle: generate()
# сбрасывает её при построении новой сцены.

# ================================================================== инструменты
@ti.kernel
def stamp_rock(x: ti.f32, y: ti.f32, r: ti.f32):
    cx = int(x)
    cy = int(y)
    rr = int(ti.ceil(r))
    oy = -rr
    while oy <= rr:
        ox = -rr
        while ox <= rr:
            ax = cx + ox
            ay = cy + oy
            if ax >= 0 and ax < COLS and ay >= 0 and ay < ROWS - 1:
                if ox * ox + oy * oy <= r * r:
                    cell = ay * COLS + ax
                    if r_block[cell] < 0:
                        fixed = 1 if ay >= ROWS - P[None].fixedRows else 0
                        i = add_block_func(ax + 0.5, ay + 0.5, 1.0, fixed)
                        if i >= 0:
                            # Кисть "Rock" - материал как в массиве: полные связи
                            # с соседями (так задают конструкции: балки, плиты,
                            # перекрытия держатся связями). Как в исходном коммите.
                            if ax > 0 and r_block[cell - 1] >= 0:
                                add_bond_func(r_block[cell - 1], i, 1.0, 1)
                            if ax < COLS - 1 and r_block[cell + 1] >= 0:
                                add_bond_func(i, r_block[cell + 1], 1.0, 1)
                            if ay > 0 and r_block[cell - COLS] >= 0:
                                add_bond_func(r_block[cell - COLS], i, 1.0, 1)
                            if ay < ROWS - 1 and r_block[cell + COLS] >= 0:
                                add_bond_func(i, r_block[cell + COLS], 1.0, 1)
                            r_block[cell] = i
            ox += 1
        oy += 1


@ti.kernel
def stamp_bedrock(x: ti.f32, y: ti.f32, r: ti.f32):
    cx = int(x)
    cy = int(y)
    rr = int(ti.ceil(r))
    oy = -rr
    while oy <= rr:
        ox = -rr
        while ox <= rr:
            ax = cx + ox
            ay = cy + oy
            if ax >= 0 and ax < COLS and ay >= 0 and ay < ROWS - 1:
                if ox * ox + oy * oy <= r * r:
                    cell = ay * COLS + ax
                    b = r_block[cell]
                    if b < 0:
                        i = add_block_func(ax + 0.5, ay + 0.5, 1.0, 1)
                        if i >= 0:
                            bbedrock[i] = 3
                            if ax > 0 and r_block[cell - 1] >= 0:
                                add_bond_func(r_block[cell - 1], i, 1.0, 1)
                            if ax < COLS - 1 and r_block[cell + 1] >= 0:
                                add_bond_func(i, r_block[cell + 1], 1.0, 1)
                            if ay > 0 and r_block[cell - COLS] >= 0:
                                add_bond_func(r_block[cell - COLS], i, 1.0, 1)
                            if ay < ROWS - 1 and r_block[cell + COLS] >= 0:
                                add_bond_func(i, r_block[cell + COLS], 1.0, 1)
                            r_block[cell] = i
            ox += 1
        oy += 1


@ti.kernel
def stamp_unbedrock(x: ti.f32, y: ti.f32, r: ti.f32):
    r2 = r * r * LCELL * LCELL
    i = bcount[None] - 1
    while i >= 0:
        if bbedrock[i] != 0:
            dx = bx[i] - x * LCELL
            dy = by[i] - y * LCELL
            if dx * dx + dy * dy <= r2:
                if bbedrock[i] == 3:
                    remove_block_func(i)
                else:
                    if bbedrock[i] == 1:
                        bfixed[i] = 0
                    else:
                        bfixed[i] = 1
                    bbedrock[i] = 0
                    bvx[i] = 0.0
                    bvy[i] = 0.0
                    bw[i] = 0.0
        i -= 1


@ti.kernel
def stamp_erase(x: ti.f32, y: ti.f32, r: ti.f32):
    # x, y, r — в клетках; позиции блоков bx/by — в метрах
    i = bcount[None] - 1
    while i >= 0:
        if bfixed[i] == 0 and bbedrock[i] == 0:
            dx = bx[i] - x * LCELL
            dy = by[i] - y * LCELL
            if dx * dx + dy * dy < r * r * LCELL * LCELL:
                remove_block_func(i)
        i -= 1


@ti.kernel
def crack_at(x: ti.f32, y: ti.f32, r: ti.f32):
    # x, y, r — в клетках; позиции блоков bx/by — в метрах
    r2 = r * r * LCELL * LCELL
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            mx = (bx[bondA[bd]] + bx[bondB[bd]]) * 0.5 - x * LCELL
            my = (by[bondA[bd]] + by[bondB[bd]]) * 0.5 - y * LCELL
            if mx * mx + my * my < r2:
                a = bondA[bd]
                b = bondB[bd]
                bondIntact[bd] = BOND_JOINT
                bondDmg[bd] = P[None].dmgMax
                clear_bond_bit(a, b)
                ti.atomic_add(broken[None], 1)
                breakCrush[None] = 0
                dx = bx[b] - bx[a]
                dy = by[b] - by[a]
                d = ti.sqrt(dx * dx + dy * dy)
                kx = 0.15
                ky = 0.0
                if d > 1e-9:
                    kx = dx / d * 0.15
                    ky = dy / d * 0.15
                bvx[a] -= kx
                bvy[a] -= ky
                bvx[b] += kx
                bvy[b] += ky


@ti.kernel
def crack_break_path():
    """Разрыв ровно по цепочке связей bondPath (индексы целых связей).

    Путь собирается в crack_line по графу связей: каждая ступенька пути —
    уже реальная связь, поэтому разрывы идут сплошной цепочкой без дыр на
    любых углах, размерах блоков и разрезах слоёв."""
    n = bondN[None]
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            for i in range(n):
                if bondPath[i] == bd:
                    bondIntact[bd] = BOND_JOINT
                    bondDmg[bd] = P[None].dmgMax
                    clear_bond_bit(bondA[bd], bondB[bd])
                    ti.atomic_add(broken[None], 1)
                    breakCrush[None] = 0
                    a = bondA[bd]
                    b = bondB[bd]
                    dx = bx[b] - bx[a]
                    dy = by[b] - by[a]
                    d = ti.sqrt(dx * dx + dy * dy)
                    kx = 0.15
                    ky = 0.0
                    if d > 1e-9:
                        kx = dx / d * 0.15
                        ky = dy / d * 0.15
                    bvx[a] -= kx
                    bvy[a] -= ky
                    bvx[b] += kx
                    bvy[b] += ky
                    break


@ti.kernel
def clear_scene():
    """Оставить только закреплённые блоки, связи пересобрать по геометрии."""
    n = 0
    i = 0
    while i < bcount[None]:
        if bfixed[i] == 1 or bbedrock[i] != 0:
            if i != n:
                copy_block_state(i, n)
                reset_block_runtime(n)
            n += 1
        i += 1
    bcount[None] = n
    bondCount[None] = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            ri = bsz[i] * 0.5
            rj = bsz[j] * 0.5
            dx = bx[j] - bx[i]
            dy = by[j] - by[i]
            axx = ti.abs(dx)
            ayy = ti.abs(dy)
            rs = ri + rj
            # rest — в КЛЕТКАХ (add_bond_func умножает на LCELL): метры / LCELL
            if axx > rs - 0.05 * LCELL and axx < rs + 0.05 * LCELL and ayy < ti.min(ri, rj):
                add_bond_func(i, j, ti.sqrt(dx * dx + dy * dy) / LCELL, 1)
            elif ayy > rs - 0.05 * LCELL and ayy < rs + 0.05 * LCELL and axx < ti.min(ri, rj):
                add_bond_func(i, j, ti.sqrt(dx * dx + dy * dy) / LCELL, 1)
            j += 1
        i += 1
        dustCount[None] = 0
    broken[None] = 0
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
        fricS2[s] = 0.0
        fricF2[s] = 0


def do_stroke(mx, my, last_mx, last_my, tool, brush):
    dx = mx - last_mx
    dy = my - last_my
    dist = math.hypot(dx, dy)
    steps = max(1, int(math.ceil(dist / 0.35)))
    r_rock = brush / 2.0 + 0.4
    if tool in (0, 4):
        rebuild_occ()
    for s in range(1, steps + 1):
        t = s / steps
        x = last_mx + dx * t
        y = last_my + dy * t
        if tool == 0:
            stamp_rock(x, y, r_rock)
        elif tool == 4:
            stamp_bedrock(x, y, r_rock)
        elif tool == 3:
            stamp_erase(x, y, r_rock)
    return mx, my

def crack_line(x0, y0, x1, y1):
    """Трещина от точки к точке — НЕПРЕРЫВНЫЙ шов ПРИМЕРНО ПО НАРИСОВАННОЙ ЛИНИИ.

    Путь идёт по ЛИНИЯМ СЕТКИ (границам блоков, целые координаты) от угла
    к углу: «шла трещина, пришла к краю — и у неё был выбор, куда дальше».
    Каждый единичный шаг по сетке — ровно одна грань (одна связь), которую
    трещина рвёт; подряд идущие грани стыкуются углами, поэтому шов
    НЕПРЕРЫВЕН на любом угле: горизонталь/вертикаль — сплошная линия вдоль
    границы, диагональ — лесенкой. «Прыжков сквозь блоки» не бывает: путь
    физически не может зайти в тело блока (только по границам).

    Шов идёт ТОЛЬКО в коридоре вдоль нарисованной линии (отступ не больше
    CRACK_CORR клеток) — трещина не уползает от линии, например вдоль стенки
    пустоты. Концы тоже НЕ привязываются к ближайшей точке: старт и цель
    берутся как углы, ближайшие к началу и концу линии ВДОЛЬ неё, поэтому
    клик по пустоте продолжает шов там, где линия встречает породу, а не
    тащит его к случайному углу. Если цель недостижима (пустота поперёк
    линии), шов рвётся вдоль линии с обеих сторон пустоты. A* по углам:
    стоимость шага = длина + 2.0 * отступ середины грани от линии.
    Углы (целые координаты, в клетках) пишутся в pathX/pathY.
    Модель должна быть в ориентации «как сгенерирована» (как и раньше)."""
    A = bondA.to_numpy()
    B = bondB.to_numpy()
    INT = bondIntact.to_numpy()
    bxv = bx.to_numpy()
    byv = by.to_numpy()
    bszv = bsz.to_numpy()
    nb = int(bondCount[None])
    if nb == 0:
        return 0
    # карта сегментов сетки: каждая целая связь даёт ключи единичных
    # отрезков её грани (грань большого блока = несколько сегментов).
    # Ключ: (0, ix, iy) — горизонтальный отрезок (ix,iy)-(ix+1,iy) [грань y=iy];
    #       (1, ix, iy) — вертикальный  (ix,iy)-(ix,iy+1) [грань x=ix].
    seg2bond = {}
    for bd in range(nb):
        if INT[bd] != 1:
            continue
        a = int(A[bd])
        b = int(B[bd])
        fx = (bxv[a] + bxv[b]) * 0.5 / LCELL
        fy = (byv[a] + byv[b]) * 0.5 / LCELL
        ax0 = bxv[a] / LCELL - bszv[a] / (2.0 * LCELL)
        ax1 = bxv[a] / LCELL + bszv[a] / (2.0 * LCELL)
        ay0 = byv[a] / LCELL - bszv[a] / (2.0 * LCELL)
        ay1 = byv[a] / LCELL + bszv[a] / (2.0 * LCELL)
        bx0 = bxv[b] / LCELL - bszv[b] / (2.0 * LCELL)
        bx1 = bxv[b] / LCELL + bszv[b] / (2.0 * LCELL)
        by0 = byv[b] / LCELL - bszv[b] / (2.0 * LCELL)
        by1 = byv[b] / LCELL + bszv[b] / (2.0 * LCELL)
        if abs(bxv[a] - bxv[b]) < abs(byv[a] - byv[b]):
            # связь вертикальная -> грань горизонтальная (y = iy, x-диапазон)
            iy = int(round(fy))
            x0i = int(round(max(ax0, bx0)))
            x1i = int(round(min(ax1, bx1)))
            for ix in range(x0i, x1i):
                seg2bond[(0, ix, iy)] = bd
        else:
            # связь горизонтальная -> грань вертикальная (x = ix, y-диапазон)
            ix = int(round(fx))
            y0i = int(round(max(ay0, by0)))
            y1i = int(round(min(ay1, by1)))
            for iy in range(y0i, y1i):
                seg2bond[(1, ix, iy)] = bd
    if not seg2bond:
        return 0
    # направление нарисованной линии
    ldx = x1 - x0
    ldy = y1 - y0
    l2 = ldx * ldx + ldy * ldy
    LLEN = math.sqrt(l2) if l2 > 1e-9 else 0.0
    if l2 < 1e-9:
        l2 = 1e-9

    def dev(px, py):
        # отступ точки от ОТРЕЗКА нарисованной линии (не от прямой!)
        t = ((px - x0) * ldx + (py - y0) * ldy) / l2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return math.hypot(x0 + t * ldx - px, y0 + t * ldy - py)

    def alng(px, py):
        # позиция точки ВДОЛЬ линии (в клетках от 1-й точки)
        t = ((px - x0) * ldx + (py - y0) * ldy) / l2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return t * LLEN

    def seg_mid(o, ix, iy):
        return (ix + 0.5, iy) if o == 0 else (ix, iy + 0.5)

    # коридор вдоль линии: шов идёт ТОЛЬКО по граням, проходящим не дальше
    # CRACK_CORR клеток от нарисованной линии — трещина не уползает от неё
    # (раньше A* мог обойти пустоту вдоль стенки далеко от линии)
    CRACK_CORR = 2.0
    sok = {}
    for (o, ix, iy), bd in seg2bond.items():
        mx_, my_ = seg_mid(o, ix, iy)
        if dev(mx_, my_) <= CRACK_CORR:
            sok[(o, ix, iy)] = bd
    if not sok:
        return 0

    # углы коридора (детерминированный порядок)
    corners = set()
    for (o, ix, iy) in sok:
        corners.add((ix, iy))
        if o == 0:
            corners.add((ix + 1, iy))
        else:
            corners.add((ix, iy + 1))
    clist = sorted(corners)

    # концы шва — углы, ближайшие к НАЧАЛУ и КОНЦУ линии ВДОЛЬ неё
    # (раньше брался просто ближайший к клику угол — тот мог стоять далеко
    # от линии, и весь шов уходил в сторону)
    start = None
    sbest = 1e18
    goal = None
    gbest = 1e18
    for cxy in clist:
        dv = dev(cxy[0], cxy[1])
        al = alng(cxy[0], cxy[1])
        s = al + 2.0 * dv
        if s < sbest:
            sbest = s
            start = cxy
        s = (LLEN - al) + 2.0 * dv
        if s < gbest:
            gbest = s
            goal = cxy
    if start is None or goal is None:
        return 0

    if start == goal and LLEN >= 0.75:
        # цель не должна схлопываться в старт: берём самый дальний угол
        # вдоль линии (иначе короткий шов превращался в одну грань «в точке»)
        best2 = None
        bs2 = -1e18
        for cxy in clist:
            if cxy == start:
                continue
            s = alng(cxy[0], cxy[1]) - 2.0 * dev(cxy[0], cxy[1])
            if s > bs2:
                bs2 = s
                best2 = cxy
        if best2 is not None:
            goal = best2

    def seg_cost(o, ix, iy):
        mx_, my_ = seg_mid(o, ix, iy)
        return 1.0 + 2.0 * dev(mx_, my_)

    # соседние углы: из (cx, cy) — четыре возможных шага по целым граням
    def neigh(cx, cy):
        if (0, cx, cy) in sok:
            yield (cx + 1, cy), sok[(0, cx, cy)], seg_cost(0, cx, cy)
        if (0, cx - 1, cy) in sok:
            yield (cx - 1, cy), sok[(0, cx - 1, cy)], seg_cost(0, cx - 1, cy)
        if (1, cx, cy) in sok:
            yield (cx, cy + 1), sok[(1, cx, cy)], seg_cost(1, cx, cy)
        if (1, cx, cy - 1) in sok:
            yield (cx, cy - 1), sok[(1, cx, cy - 1)], seg_cost(1, cx, cy - 1)

    if start == goal:
        # очень короткий клик: рвём одну ближайшую к ЛИНИИ грань
        best_bd = -1
        bestd = 1e18
        mxl = (x0 + x1) * 0.5
        myl = (y0 + y1) * 0.5
        for (o, ix, iy), bidx in sok.items():
            mx_, my_ = seg_mid(o, ix, iy)
            d = dev(mx_, my_) + 0.25 * math.hypot(mx_ - mxl, my_ - myl)
            if d < bestd:
                bestd = d
                best_bd = bidx
        if best_bd < 0:
            return 0
        bondPath[0] = best_bd
        pathX[0] = x0
        pathY[0] = y0
        pathN[None] = 1
        bondN[None] = 1
        b0 = broken[None]
        crack_break_path()
        return int(broken[None] - b0)

    def astar(s, g):
        # A* по углам коридора; эвристика — евклидово до угла цели
        # (допустимо: шаг длиной 1 стоит >= 1)
        dist = {s: 0.0}
        prev = {s: None}
        pseg = {s: -1}
        open_h = []
        heapq.heappush(open_h, (math.hypot(g[0] - s[0], g[1] - s[1]), 0.0, s))
        found = False
        # цель может оказаться за пустотой/разрывом: помним угол, дальше
        # всех ушедший ВДОЛЬ линии к цели, и обрываемся на нём
        gt = alng(g[0], g[1])
        best_cur = s
        best_sc = abs(alng(s[0], s[1]) - gt) + 2.0 * dev(s[0], s[1])
        while open_h:
            f, gc, cur = heapq.heappop(open_h)
            if cur == g:
                found = True
                break
            if gc > dist.get(cur, 1e18):
                continue
            sc = abs(alng(cur[0], cur[1]) - gt) + 2.0 * dev(cur[0], cur[1])
            if sc < best_sc:
                best_sc = sc
                best_cur = cur
            for nxt, bidx, cst in neigh(cur[0], cur[1]):
                ng = gc + cst
                if ng < dist.get(nxt, 1e18):
                    dist[nxt] = ng
                    prev[nxt] = cur
                    pseg[nxt] = bidx
                    h = math.hypot(g[0] - nxt[0], g[1] - nxt[1])
                    heapq.heappush(open_h, (ng + h, ng, nxt))
        end = g if found else best_cur
        pts = []
        bonds = []
        cur = end
        while cur is not None:
            pts.append(cur)
            bidx = pseg[cur]
            if bidx >= 0:
                bonds.append(bidx)
            cur = prev[cur]
        pts.reverse()
        bonds.reverse()
        return pts, bonds, found

    pts, bonds, found = astar(start, goal)
    if not found:
        # пустота поперёк линии: рвём шов вдоль линии с ОБЕИХ сторон
        # пустоты, а не обходим её далеко от линии
        pts2, bonds2, _ = astar(goal, start)
        seenb = set(bonds)
        for bd in bonds2:
            if bd not in seenb:
                seenb.add(bd)
                bonds.append(bd)
        pts = pts + pts2
    for i, bd in enumerate(bonds):
        if i >= 1024:
            break
        bondPath[i] = bd
    for i, (px, py) in enumerate(pts):
        if i >= 1024:
            break
        pathX[i] = px
        pathY[i] = py
    pathN[None] = min(len(pts), 1024)
    bondN[None] = min(len(bonds), 1024)
    b0 = broken[None]
    crack_break_path()
    return int(broken[None] - b0)

# ================================================================== selftest / GUI
def run_selftest(frames=120):
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    apply_water(12.0)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].substeps = 48
    P[None].depth = 60.0

    t0 = time.perf_counter()
    generate(0, 0)
    t_gen = time.perf_counter() - t0
    nbi = count_intact()
    print("selftest: generated mode=tunnel blocks=%d bonds=%d intact=%d (%.2fs)" % (bcount[None], bondCount[None], nbi, t_gen))

    t0 = time.perf_counter()
    for f in range(frames):
        step_physics(0, DT)
        if f < 20 or f % 20 == 0:
            print(
                "  f=%d intact=%d broken=%d maxdisp=%.3f maxst=%.3f blocks=%d"
                % (f, count_intact(), broken[None], diag_stats(), diag_strain(), bcount[None])
            )
    t_phys = time.perf_counter() - t0
    fill = fill_pct()
    print(
        "selftest: %d frames in %.2fs (%.1f fps) blocks=%d broken=%d fill=%.0f%%"
        % (frames, t_phys, frames / t_phys, bcount[None], broken[None], fill)
    )
    print("selftest: internal simulation time = %.3f s" % simT[None])
    ek, er, ep, ee, et = energy_report()
    cex, cey, cem = center_of_mass_report()
    print("selftest: energy J: kin=%.3e rot=%.3e pot=%.3e elast=%.3e total=%.3e" % (ek, er, ep, ee, et))
    print("selftest: center of mass: x=%.3f m, y=%.3f m, mass=%.3f kg" % (cex, cey, cem))
    if broken[None] > 0:
        print("selftest: WARNING — bonds broke during settle (broken=%d)" % broken[None])

    t0 = time.perf_counter()
    crack_at(COLS * 0.5, ROWS * 0.2, 2.5)
    nb1 = count_intact()
    n0 = bcount[None]
    stamp_rock(COLS * 0.5, ROWS * 0.6, 4.0)
    nb2 = bcount[None] - n0
    stamp_erase(COLS * 0.5, ROWS * 0.6, 4.0)
    nb3 = n0 - bcount[None]
    for f in range(10):
        step_physics(0, DT)
    clear_scene()
    nb4 = bcount[None]
    t_tools = time.perf_counter() - t0
    print(
        "selftest: tools crack->%d bonds, stamp->+%d blocks, erase->-%d, clear->%d blocks (%.2fs)"
        % (nb1, nb2, nb3, nb4, t_tools)
    )

    t0 = time.perf_counter()
    generate(0, 0)
    place_source(float(cavX[None]), float(cavY[None]))
    for f in range(60):
        step_physics(0, DT)
    fp = fill_pct()
    nwater = int(water_fluid_count())
    t_wat = time.perf_counter() - t0
    print("selftest: water cells=%d fill=%.0f%% broken=%d (%.2fs)" %
          (nwater, fp, broken[None], t_wat))
    print("selftest: WARNING — water not created" if nwater == 0 else "")

    # мозаика произвольных размеров
    t0 = time.perf_counter()
    generate(2, 0)
    ms = P[None].maxSize
    for f in range(20):
        step_physics(2, DT)
    t_big = time.perf_counter() - t0
    print(
        "selftest: big blocks n=%d maxSize=%d intact=%d broken=%d (%.2fs)"
        % (bcount[None], ms, count_intact(), broken[None], t_big)
    )
    if broken[None] > bcount[None] // 4:
        print("selftest: WARNING — big blocks collapsed (broken=%d)" % broken[None])

    # рендер одного кадра (без окна)
    t0 = time.perf_counter()
    render_base()
    render_triangles(0.0, 1)
    render_water()
    stress_peaks()
    render_blocks(1)
    render_joints()
    render_bonds(1, 1)
    render_cracks()
    render_dust()
    render_sources(0.0)
    render_mouse(-999.0, 0.0, 2.0, 0)
    hud_gather()
    render_stress_legend(hudF[HUD_RP], hudF[HUD_RP] * hudF[HUD_RCFACTOR])
    t_render = time.perf_counter() - t0
    print("selftest: one render pass in %.1f ms" % (t_render * 1000.0))
    print("selftest OK")


def run_gui():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    apply_water(12.0)
    P[None].substeps = SUBSTEPS   # подшагов на тик (слайдер "настройки решателя");
    # НЕ опускать ниже ~96: при грубом dt приповерхностный выброс от верхней
    # нагрузки (depth) у свободной кровли уходит в растяжение -> отрыв верхнего
    # слоя и "волна" по поверхности (на 128×128 особенно). SUBSTEPS=192 кратен
    # шагу слайдера (x64), поэтому начальное значение согласовано со слайдером.
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].depth = 60.0
    refillOff[None] = 1   # досыпка сверху ОТКЛЮЧЕНА по умолчанию (тумблер "Dosypka")

    mode = 0
    tool = 0
    brush = 2
    paused = True
    recorder = None            # FrameRecorder или None
    record_next = 0.0          # следующее модельное время кадра
    record_start = 0.0         # модельное время старта записи
    record_fps = 30.0          # 30 кадров/с: 1 с симуляции = 1 с GIF
    record_max_s = 10.0        # максимум 10 с записи (300 кадров)
    weak = 0
    tilt_deg = 0        # наклон слоёв к горизонтали, градусы (+ = падение вправо)
    show_bonds = 1
    E_GPa = 10
    rho = RHO_R
    scene_names = {0: "Circular working", 1: "Slope", 2: "Large blocks",
                   3: "Standard working"}
    # активный верификационный пресет: "none" | "ucs" | "brazil".
    # Kirsch — это обычная сцена «vyrabotka», поэтому test_kind остаётся "none".
    test_kind = "none"
    test_label = ""
    brazil_top = []          # индексы верхних (нагружаемых) блоков диска
    brazil_P = 0.0           # текущая суммарная нагрузка, Н
    brazil_rate = 2.5e5      # прирост нагрузки за тик, Н
    brazil_D = 12.0          # диаметр диска, м
    pt_top = []              # балка/изгиб: блоки, к которым приложена нагрузка
    pt_P = 0.0               # текущая суммарная нагрузка, Н
    pt_rate = 5.0e4          # прирост нагрузки за тик, Н
    pt_fx = 0.0              # боковой seed (суммарный, Н) для выпучивания

    # без vsync: окно не ограничивает цикл 60 Гц, тики идут так быстро,
    # как позволяет CPU (см. step_physics: следующий тик — после полного расчёта)
    window = ti.ui.Window("GridRockDem", (W, H), vsync=False)
    canvas = window.get_canvas()
    reset_camera()   # стартовый кадр: вся область с запасом (барьер + нагрузка)
    gui = window.get_gui()

    generate(0, 0)

    last = time.perf_counter()
    fps = 60.0
    hud_t = 0.0
    last_mx = -1.0
    last_my = -1.0
    crack_p0 = None            # первая точка трещины (клетки), None = ждём клика
    flash_msg = "Click START to begin simulation"
    flash_wall = 0.0           # реальное время (time.perf_counter) для всплывающих сообщений
    last_broken = 0
    last_hud_broken = 0
    cam_drag = False          # зажата средняя кнопка — таскаем камеру
    cam_prev = (0.0, 0.0)     # курсор прошлого кадра (для дельты драга)
    ang_focus = 1             # какое поле угла принимает цифры: 0 = слои, 1 = модель (Tab)
    buf_sloi = ""             # буфер набора угла слоёв, град
    buf_model = ""            # буфер набора угла модели, град
    show_rock = False         # группа "настройки породы" (материал) развёрнута
    show_solver = False       # группа "настройки решателя" (время/устойчивость) развёрнута
    scene_slots = ["scene1", "scene2", "scene3", "scene4", "scene5",
                   "scene6", "scene7", "scene8", "scene9"]  # слоты сохранений (JSON)
    scene_idx = 0             # выбранный слот сохранения/загрузки
    legend_on = False         # легенда напряжений (кнопка под START)
    hud_cache = None          # кэш снимка HUD (hudF.to_numpy() — синхронизация!)
    hud_refresh = -1.0        # когда обновляли кэш (perf_counter)
    freeze_t = -1.0           # диагностика зависших блоков (в консоль, раз в 3 с)

    def preset_rock(E=10.0, rp=4.0, rc=40.0, mu=0.62):
        nonlocal E_GPa, rho, weak, tilt_deg, show_rock, show_solver
        E_GPa = int(round(E))
        rho = RHO_R
        weak = 0
        tilt_deg = 0.0
        presets.model_angle = 0.0
        P[None].tiltC = 1.0
        P[None].tiltS = 0.0
        apply_rock(float(E), rp, mu, rho, rc_MPa=rc)
        # Все solver-параметры, которые не задаёт apply_rock, сбрасываем к
        # дефолтам: иначе Fault-slip/Strainburst зависели бы от того, какие
        # слайдеры пользователь двигал до нажатия пресета.
        P[None].kick = 0.14
        P[None].splitN = 0.7
        P[None].tensConf = 0.0
        P[None].Gf = 100.0
        P[None].stabC = STAB_C
        P[None].vmax = VMAX
        P[None].lateralFloor = LATERAL_FLOOR
        P[None].sleepV2 = SLEEP_V2
        P[None].sleepA2 = SLEEP_A2
        P[None].rollMu = ROLL_MU
        P[None].rockLift = ROCK_LIFT
        P[None].soften = 0
        P[None].softResidual = 0.15
        P[None].absorb = 0
        P[None].absorbWidth = 8.0
        P[None].absorbCoef = 500.0
        P[None].absorbTop = 0
        P[None].loadX = 0.0
        loadCur[None] = 0.0
        loadXCur[None] = 0.0
        show_rock = False
        show_solver = True

    while window.running:
        now = time.perf_counter()
        dt_real = min(0.05, now - last)
        last = now
        # угол обзора (для инверсии ввода курсора) — синхронизируем ДО обработки
        # ввода, чтобы клик/зум попадали в повёрнутую (экранную) позицию.
        render.set_view(presets.model_angle)
        # снимок HUD одним ядром + одним копированием НЕ ЧАЩЕ 5 Гц (вместо
        # каждого кадра): to_numpy() синхронизирует CPU<->GPU и сливает очередь
        # ядер, из-за чего GPU регулярно «простаивает» между мелкими лаунчами.
        # Слайдерам GUI свежесть 0.2 с не важна, время тика ведётся локально.
        if hud_cache is None or now - hud_refresh > 0.2:
            hud_cache = hud_snapshot()
            hud_refresh = now
        hud = hud_cache
        t_sim = hud[HUD_SIMT]     # внутреннее время симуляции (сек, дробное)
        # раз в 3 с сканируем "зависшие" блоки и печатаем в консоль, КТО их держит
        if now - freeze_t > 3.0:
            freeze_t = now
            if not paused:
                frozen_report()

        try:
            # один проход по ВСЕМ событиям (колесо — не типа PRESS, отдельный
            # фильтр его теряет). Отпускание кнопки/клавиши отличаем от нажатия
            # через window.is_pressed(k): у события отпуска is_pressed == False.
            # символы числа (цифры/минус/точка) — опросом клавиатуры: события
            # ti.ui их не приносят (см. poll_type_chars)
            for ch in poll_type_chars():
                if ang_focus == 0:
                    if len(buf_sloi) < 8:
                        buf_sloi += ch
                else:
                    if len(buf_model) < 8:
                        buf_model += ch
            for e in window.get_events():
                k = e.key
                if k == "Wheel" or k == "WHEEL":
                    # колесо — зум к точке под курсором (delta[1] > 0 = к нам)
                    dy = 0.0
                    try:
                        d = e.delta
                        dy = d[1] if isinstance(d, (tuple, list)) else float(d)
                    except Exception:
                        dy = 0.0
                    if dy != 0.0:
                        pos = window.get_cursor_pos()
                        f = 1.15 if dy > 0 else 1.0 / 1.15
                        zoom_at(pos, f)
                    continue
                if k == ti.ui.TAB:
                    # Tab — какое поле угла получает цифры (сами символы приходят
                    # из poll_type_chars: события ti.ui цифры/минус/точку не дают)
                    ang_focus = 1 - ang_focus
                    continue
                if k == ti.ui.RETURN:
                    if ang_focus == 0:
                        tv = parse_deg(buf_sloi)
                        if tv is None or tv < -45.0 or tv > 45.0:
                            flash_msg = "Layer angle: enter a number from -45 to 45"
                        else:
                            tilt_deg = tv
                            srcCount[None] = 0
                            generate(mode, weak, tilt_deg)
                            presets.model_angle = 0.0
                            flash_msg = ("Layer dip %g deg: layers are inclined, "
                                         "gravity and top load are vertical" % tilt_deg)
                            buf_sloi = ""
                        flash_wall = now
                    else:
                        tv = parse_deg(buf_model)
                        if tv is None or tv < -90.0 or tv > 90.0:
                            flash_msg = "Model angle: enter a number from -90 to 90"
                        else:
                            presets.model_angle = tv
                            P[None].tiltC = math.cos(math.radians(tv))
                            P[None].tiltS = math.sin(math.radians(tv))
                            flash_msg = ("Model tilt %g deg: gravity and load "
                                         "vertical, view rotated" % tv)
                            buf_model = ""
                        flash_wall = now
                    continue
                if k == ti.ui.ESCAPE:
                    if crack_p0 is not None:
                        crack_p0 = None
                        flash_msg = "Crack cancelled"
                        flash_wall = now
                    elif buf_sloi or buf_model:
                        buf_sloi = ""
                        buf_model = ""
                        flash_msg = "Angle input cancelled"
                        flash_wall = now
                    else:
                        window.running = False
                    continue
                if k == ti.ui.BACKSPACE:
                    if ang_focus == 0:
                        buf_sloi = buf_sloi[:-1]
                    else:
                        buf_model = buf_model[:-1]
                    continue
                if k not in (ti.ui.SPACE, "c", "p", "P", ti.ui.LMB, ti.ui.RMB, ti.ui.MMB):
                    continue
                if not window.is_pressed(k):
                    continue
                if k == ti.ui.SPACE:
                    paused = not paused
                elif k == "c":
                    reset_camera()
                    flash_msg = "Camera reset"
                    flash_wall = now
                elif k in ("p", "P"):
                    render.panel_visible = not render.panel_visible
                    flash_msg = ("Panel hidden - press P" if not render.panel_visible
                                 else "Panel shown")
                    flash_wall = now
                elif k == ti.ui.LMB:
                    pos = window.get_cursor_pos()
                    mx, my = cursor_to_cells(pos)
                    if panel_zone(pos):
                        if tool == 1 and crack_p0 is not None:
                            flash_msg = "Click under the panel is ignored (P - panel)"
                            flash_wall = now
                        continue
                    if tool == 2:
                        flash_msg = place_source(mx, my)
                        flash_wall = now
                    elif tool == 1:
                        # трещина задаётся ДВУМЯ кликами (как в AutoCAD):
                        # 1-й — точка, между кликами линия следует за курсором,
                        # 2-й — A* находит путь примерно по линии и рвёт связи
                        if crack_p0 is None:
                            crack_p0 = (mx, my)
                            flash_msg = "Crack: first point set, click second (Esc - cancel)"
                            flash_wall = now
                        else:
                            nb = crack_line(crack_p0[0], crack_p0[1], mx, my)
                            crack_p0 = None
                            flash_msg = "Crack: bonds broken %d" % nb
                            flash_wall = now
                    else:
                        last_mx = mx
                        last_my = my
                        last_mx, last_my = do_stroke(mx, my, last_mx, last_my, tool, brush)
                elif k == ti.ui.RMB:
                    pos = window.get_cursor_pos()
                    mx, my = cursor_to_cells(pos)
                    if tool == 4:
                        if not panel_zone(pos):
                            stamp_unbedrock(mx, my, brush / 2.0 + 0.4)
                            flash_msg = "Bedrock cleared"
                            flash_wall = now
                    elif remove_source(mx, my):
                        flash_msg = "Water source removed"
                        flash_wall = now
                elif k == ti.ui.MMB:
                    # запасной старт драга по событию (основной - опрос VK_MBUTTON)
                    pos = window.get_cursor_pos()
                    if not panel_zone(pos):
                        cam_drag = True
                        cam_prev = pos
        except Exception:
            import traceback
            traceback.print_exc()
            flash_msg = "Input error - see console"
            flash_wall = now

        if window.is_pressed(ti.ui.LMB):
            pos = window.get_cursor_pos()
            if not panel_zone(pos):
                mx, my = cursor_to_cells(pos)
                if tool not in (1, 2) and last_mx >= 0:
                    last_mx, last_my = do_stroke(mx, my, last_mx, last_my, tool, brush)

        # перетаскивание камеры СКМ (колесо мыши): состояние кнопки опросом
        # Win32 VK_MBUTTON - события ti.ui для MMB ненадёжны (как и для цифр),
        # а window.is_pressed(MMB) не работает в GGUI. Дельта - по курсору.
        mmb = (_user32.GetAsyncKeyState(0x04) & 0x8000) != 0 and _app_focused()
        if mmb:
            pos = window.get_cursor_pos()
            if not cam_drag:
                if not panel_zone(pos):
                    cam_drag = True
                    cam_prev = pos
            else:
                cam_write(_CAM[0] + (pos[0] - cam_prev[0]) * W,
                          _CAM[1] - (pos[1] - cam_prev[1]) * H,
                          _CAM[2])
                cam_prev = pos
        else:
            cam_drag = False

        # стрелки <- -> ^ v: панорамирование камеры (аналог перетаскивания СКМ)
        pan = 14.0
        ax = 0.0
        ay = 0.0
        if window.is_pressed(ti.ui.LEFT):
            ax += pan
        if window.is_pressed(ti.ui.RIGHT):
            ax -= pan
        if window.is_pressed(ti.ui.UP):
            ay += pan
        if window.is_pressed(ti.ui.DOWN):
            ay -= pan
        if ax != 0.0 or ay != 0.0:
            cam_write(_CAM[0] + ax, _CAM[1] + ay, _CAM[2])

        # зум numpad +/-: якорь - курсор над канвой, иначе центр окна
        zs = poll_zoom_keys(now)
        if zs != 0:
            pos = window.get_cursor_pos()
            if panel_zone(pos):
                pos = (0.5, 0.5)
            zoom_at(pos, 1.18 if zs > 0 else 1.0 / 1.18)

        # ---- GUI
        if render.panel_visible:
            gui.begin("Upravlenie", 0.65, 0.02, 0.34, 0.96)
            if gui.button("Hide panel (P)"):
                render.panel_visible = False
            # кнопка старт/стоп: символы ▶/❚❚ не рисуются шрифтом, используем ASCII.
            start_lbl = "> START" if paused else "|| STOP"
            if gui.button(start_lbl):
                paused = not paused
                flash_msg = "Simulation running" if not paused else "Paused"
                flash_wall = now
            if gui.button("Record: OFF" if recorder is None else "Record: ON"):
                if recorder is None:
                    stamp = time.strftime("rec_%Y%m%d_%H%M%S")
                    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "recordings", stamp)
                    recorder = rec_mod.FrameRecorder(
                        out, fps=record_fps, width=W, height=H,
                        tick_hz=1.0 / TICK,
                        meta={"scene": test_label or scene_names[mode],
                              "mode": int(mode), "depth": float(P[None].depth),
                              "k0": float(P[None].k0),
                              "record_max_s": record_max_s})
                    record_next = t_sim
                    record_start = t_sim
                    flash_msg = "Record -> %s (max 10 s, 30 fps)" % out
                else:
                    out = recorder.close()
                    recorder = None
                    flash_msg = "Record saved: %s (video.gif)" % out
                flash_wall = now
            if recorder is not None:
                gui.text("REC %d frames" % recorder.frames)
            # счётчик приложенной верхней нагрузки относительно планируемой
            # (перегруз набирается линейно: loadCur -> depth*LOAD_OVER за LOAD_TIME, см. LOAD_TIME)
            _ltgt = P[None].depth * LOAD_OVER if (mode == 0 or mode == 3) else 0.0
            if _ltgt > 1.0:
                _lp = 100.0 * loadCur[None] / _ltgt
                if _lp > 100.0:
                    _lp = 100.0
                gui.text("Load: %.1f%%" % _lp)
            else:
                gui.text("Load: -")
            # под START — переключатель легенды напряжений
            if gui.button("Hide legend" if legend_on else "Show legend"):
                legend_on = not legend_on
            if legend_on:
                pkT = hud[HUD_PEAKT] if hud[HUD_PEAKT] > 1e-4 else 1.0
                pkC = hud[HUD_PEAKC] if hud[HUD_PEAKC] > 1e-4 else 1.0
                gui.text("Legend: 10 color bands, 0..stress peak")
                gui.text("Red scale - tension: 0..%.1f MPa" % (hud[HUD_RP] * pkT))
                gui.text("Blue scale - compression: 0..%.1f MPa" % (hud[HUD_RP] * hud[HUD_RCFACTOR] * pkC))
            gui.text("Simulation time: %.3f s" % t_sim)
            if not test_label:
                gui.text("Scene: %s" % scene_names[mode])
            else:
                gui.text("TEST: %s" % test_label)
                if test_kind == "brazil":
                    st = 2.0 * brazil_P / (math.pi * max(1e-6, brazil_D)) / 1e6
                    gui.text("P=%.1f MN  sigma_t=%.2f MPa (Rp=%d MPa)"
                             % (brazil_P / 1e6, st, int(hud[HUD_RP])))
                elif test_kind == "ucs":
                    gui.text("sigma_v=%.1f MPa  (slider 'Rock above the scheme')"
                             % (hud[HUD_DEPTH] * hud[HUD_RHO] * G_PHYS / 1e6))
                elif test_kind in ("beam", "buckling"):
                    gui.text("P=%.2f MN  (load ramps automatically)" % (pt_P / 1e6))
            gui.text("Layer angle: %g deg" % tilt_deg)
            gui.text("Rock: E=%d GPa, Rp=%d MPa, Rc=%d MPa" % (E_GPa, int(round(hud[HUD_RP])), int(round(hud[HUD_RP] * hud[HUD_RCFACTOR]))))
            gui.text("Water: %.2f MPa (H=%g m)" % (RHO_W * G_PHYS * hud[HUD_HEAD] / 1e6, hud[HUD_HEAD]))
            gui.text("Overburden pressure 100%%: %.2f MPa" % (hud[HUD_DEPTH] * hud[HUD_RHO] * G_PHYS / 1e6))
            gui.text("LMB - tool / place source * RMB - remove")
            gui.text("MMB - pan * Numpad +/- zoom * P - panel")
            gui.text("Crack: 2 clicks (line), Esc - cancel")
            gui.text("Bedrock: LMB - create, RMB - clear")
            if gui.button("Rock"):
                tool = 0
                crack_p0 = None
            if gui.button("Bedrock"):
                tool = 4
                crack_p0 = None
            if gui.button("Crack"):
                tool = 1
                crack_p0 = None
            if gui.button("Water source"):
                tool = 2
                crack_p0 = None
            if gui.button("Eraser"):
                tool = 3
                crack_p0 = None
            brush = gui.slider_int("Brush", brush, 1, 6)
            # ---- две группы настроек: материал (порода) и решатель ----
            # Обе разом не раскрываются: панель остаётся компактной.
            if gui.button("ROCK SETTINGS"):
                show_rock = True
                show_solver = False
            if gui.button("SOLVER SETTINGS"):
                show_solver = True
                show_rock = False

            if show_rock:
                gui.text("--- ROCK (material) ---")
                rp_cur = int(round(hud[HUD_RP]))
                rc_cur = int(round(hud[HUD_RP] * hud[HUD_RCFACTOR]))
                mu_cur = int(round(hud[HUD_MU] * 100))
                E_new = gui.slider_int("Young's modulus E, GPa", E_GPa, 1, 40)
                rp_new = gui.slider_int("Rp (tension) MPa", rp_cur, 1, 30)
                rc_new = gui.slider_int("Rc (compression) MPa", rc_cur, 1, 200)
                mu_new = gui.slider_int("Friction mu %", mu_cur, 0, 100)
                if (E_new != E_GPa or rp_new != rp_cur or rc_new != rc_cur
                        or mu_new != mu_cur):
                    E_GPa = E_new
                    P[None].E = E_new * 1e9
                    P[None].rp = float(rp_new)
                    P[None].rcFactor = float(rc_new) / max(float(rp_new), 1e-9)
                    P[None].mu = mu_new / 100.0
                    P[None].relBend = (2.0 * float(rp_new) * 1e6
                                       / (float(E_new) * 1e9))
                rho_new = gui.slider_int("Rock density, kg/m3", int(rho), 1000, 3500)
                if rho_new != rho:
                    rho = float(rho_new)
                    P[None].rho = rho
                    update_mass_rho()
                    apply_water(hud[HUD_HEAD])
                cur = P[None].mcBond
                v = gui.slider_float("Joint tan(phi) (mcBond)", cur, 0.0, 1.5)
                if v != cur:
                    P[None].mcBond = v
                cur = P[None].nuSplit
                v = gui.slider_float("Poisson splitting", cur, 0.0, 0.5)
                if v != cur:
                    P[None].nuSplit = v
                cur = P[None].plastYield
                v = gui.slider_float("Plastic yield (fraction of Rp)", cur, 0.0, 1.0)
                if v != cur:
                    P[None].plastYield = v
                cur = P[None].plastLimit
                v = gui.slider_float("Plastic limit (strain)", cur, 0.0, 1.0)
                if v != cur:
                    P[None].plastLimit = v
                cur = P[None].plastFlow
                v = gui.slider_float("Plastic flow rate, 1/s", cur, 0.0, 5.0)
                if v != cur:
                    P[None].plastFlow = v
                cur = P[None].relBend
                v = gui.slider_float("Bending limit (relBend)", cur, 1e-4, 5e-2)
                if v != cur:
                    P[None].relBend = v
                cur = P[None].fsFactor
                v = gui.slider_float("Shear strength / Rp", cur, 0.1, 10.0)
                if v != cur:
                    P[None].fsFactor = v
                cur = P[None].cn
                v = gui.slider_float("Bond damping cn", cur, 0.0, 2.0)
                if v != cur:
                    P[None].cn = v
                cur = P[None].cnc
                v = gui.slider_float("Contact damping cnc", cur, 0.0, 2.0)
                if v != cur:
                    P[None].cnc = v
                cur = P[None].dmgDecay
                v = gui.slider_float("Damage decay", cur, 0.0, 1.0)
                if v != cur:
                    P[None].dmgDecay = v

            if show_solver:
                gui.text("--- SOLVER (time/stability/load) ---")
                # Глубина и K0 — вверху: это главные параметры нагружения.
                depth_new = gui.slider_int("Rock depth, m", int(hud[HUD_DEPTH]), 0, 3000)
                if depth_new != int(hud[HUD_DEPTH]):
                    P[None].depth = depth_new
                k0_new = gui.slider_float("K0 (lateral)", P[None].k0, 0.0, 2.0)
                if k0_new != P[None].k0:
                    P[None].k0 = k0_new
                loadx_kn = gui.slider_float("Shear force X, kN (+top/-bottom)",
                                            P[None].loadX / 1000.0, -10000.0, 10000.0)
                if abs(loadx_kn * 1000.0 - P[None].loadX) > 1e-3:
                    P[None].loadX = loadx_kn * 1000.0
                if test_kind in ("beam", "buckling"):
                    cur = pt_rate / 1e3
                    v = gui.slider_float("Specimen loading rate, kN/tick",
                                         cur, 0.5, 50.0)
                    if abs(v - cur) > 1e-6:
                        pt_rate = v * 1e3
                if test_kind == "buckling":
                    cur = pt_fx / 1e3
                    v = gui.slider_float("Lateral seed, kN", cur, 0.0, 100.0)
                    if abs(v - cur) > 1e-6:
                        pt_fx = v * 1e3
                if test_kind == "brazil":
                    cur = brazil_rate / 1e3
                    v = gui.slider_float("Brazil loading rate, kN/tick",
                                         cur, 10.0, 1000.0)
                    if abs(v - cur) > 1e-6:
                        brazil_rate = v * 1e3
                cur = P[None].substeps if P[None].substeps > 0 else 384
                ns = gui.slider_int("Substeps/tick (x64)", cur // 64, 1, 32)
                ns = max(64, min(2048, ns * 64))
                if ns != cur:
                    P[None].substeps = ns
                cur = P[None].kick
                v = gui.slider_float("Fracture kick", cur, 0.0, 0.5)
                if v != cur:
                    P[None].kick = v
                cur = P[None].splitN
                v = gui.slider_float("Kick normal fraction", cur, 0.0, 1.0)
                if v != cur:
                    P[None].splitN = v
                cur = P[None].stabC
                v = gui.slider_float("Stiffness cap (STAB_C)", cur, 0.05, 1.0)
                if v != cur:
                    P[None].stabC = v
                cur = P[None].vmax
                v = gui.slider_float("Max velocity, m/s", cur, 1.0, 100.0)
                if v != cur:
                    P[None].vmax = v
                cur = P[None].lateralFloor
                v = gui.slider_float("Lateral pressure", cur, 0.0, 1.0)
                if v != cur:
                    P[None].lateralFloor = v
                cur = P[None].rollMu
                v = gui.slider_float("Rolling resistance", cur, 0.0, 1.0)
                if v != cur:
                    P[None].rollMu = v
                cur = P[None].rockLift
                v = gui.slider_float("Edge moment (rockLift)", cur, 0.0, 2.0)
                if v != cur:
                    P[None].rockLift = v
                fr_new = gui.slider_int("Fixed rows", int(hud[HUD_FIXEDROWS]), 1, 4)
                if fr_new != int(hud[HUD_FIXEDROWS]):
                    P[None].fixedRows = fr_new
                bc_new = gui.slider_int("Break limit/tick", int(hud[HUD_BRKCAP]), 1, 200)
                if bc_new != int(hud[HUD_BRKCAP]):
                    P[None].brkCap = bc_new
                dm_new = gui.slider_int("Damage threshold, ticks", int(hud[HUD_DMGMAX]), 1, 60)
                if dm_new != int(hud[HUD_DMGMAX]):
                    P[None].dmgMax = dm_new
                walls = gui.checkbox("Side walls", hud[HUD_WALLS] == 1)
                if walls != (hud[HUD_WALLS] == 1):
                    P[None].walls = 1 if walls else 0
                soft_on = gui.checkbox("Strain-softening", P[None].soften == 1)
                if soft_on != (P[None].soften == 1):
                    P[None].soften = 1 if soft_on else 0
                if P[None].soften == 1:
                    cur = P[None].softResidual
                    v = gui.slider_float("Residual strength (soft)", cur, 0.0, 0.8)
                    if v != cur:
                        P[None].softResidual = v
                abs_on = gui.checkbox("Absorbing boundaries", P[None].absorb == 1)
                if abs_on != (P[None].absorb == 1):
                    P[None].absorb = 1 if abs_on else 0
                if P[None].absorb == 1:
                    cur = P[None].absorbWidth
                    v = gui.slider_float("Layer thickness, cells", cur, 2.0, 24.0)
                    if v != cur:
                        P[None].absorbWidth = v
                    cur = P[None].absorbCoef
                    v = gui.slider_float("Layer damping, 1/s", cur, 50.0, 3000.0)
                    if v != cur:
                        P[None].absorbCoef = v
                    top_on = gui.checkbox("Top boundary (absorb)", P[None].absorbTop == 1)
                    if top_on != (P[None].absorbTop == 1):
                        P[None].absorbTop = 1 if top_on else 0
                gv = gui.slider_int("Gravity x0.1g", int(hud[HUD_G] / G_PHYS * 10.0), 0, 20)
                if gv != int(hud[HUD_G] / G_PHYS * 10.0):
                    apply_gravity(gv / 10.0)
                # угол модели к горизонту: блоки НЕ двигаются. Мировые гравитация и
                # нагрузки прикладываются проекциями на локальные оси (см.
                # interactions.forces_reset), а вид поворачивается (render.rotate_view):
                # на экране модель с доменом наклонена, а g/нагрузка — вертикальны.
                cur = presets.model_angle
                v = gui.slider_float("Dip angle, deg", cur, -45.0, 45.0)
                if abs(v - cur) > 1e-6:
                    presets.model_angle = v
                    P[None].tiltC = math.cos(math.radians(v))
                    P[None].tiltS = math.sin(math.radians(v))

            weak_new = gui.checkbox("Weak planes", weak == 1)
            if weak_new != weak:
                weak = 1 if weak_new else 0
                srcCount[None] = 0
                generate(mode, weak, tilt_deg)
                flash_msg = "Model %s: bonds %d" % ("weakened" if weak else "monolithic", count_intact())
                flash_wall = now
            # угол модели — поворот СОБРАННОЙ модели; угол слоёв — перегенерация
            # с наклонными слоями; в обоих случаях сила тяжести остаётся вертикальной.
            # угол — просто СТРОКИ набора числа (кнопка не нужна): цифры/знак/точка
            # пишутся в активное поле сразу (символы читает poll_type_chars — события
            # ti.ui цифры не приносят), Tab — переключить поле, Enter — применить,
            # BackSpace — удалить, Esc — очистить буферы.
            gui.text("Model angle, deg (now %g):" % presets.model_angle)
            gui.text("  [ %s ]%s   Tab - field, Enter - apply" % (buf_model, "_" if ang_focus == 1 else ""))
            gui.text("Layer angle, deg (now %g):" % tilt_deg)
            gui.text("  [ %s ]%s   rebuilds scene, g vertical" % (buf_sloi, "_" if ang_focus == 0 else ""))
            gui.text("--- PRESETS ---")
            if gui.button("Clean scene"):
                clear_scene()
                presets.model_angle = 0.0
                P[None].tiltC = 1.0
                P[None].tiltS = 0.0
                clear_ext_forces()
                refillOff[None] = 1
                srcCount[None] = 0
                test_kind = "none"
                test_label = ""
                flash_msg = "Clean scene"
                flash_wall = now
            if gui.button("Circular working"):
                if mode != 0 or test_kind != "none" or test_label:
                    mode = 0
                    test_kind = "none"
                    test_label = ""
                    clear_ext_forces()
                    apply_gravity(1.0)
                    refillOff[None] = 1
                    srcCount[None] = 0
                    generate(mode, weak, tilt_deg)
            if gui.button("Standard working"):
                if mode != 3 or test_kind != "none" or test_label:
                    mode = 3
                    test_kind = "none"
                    test_label = ""
                    clear_ext_forces()
                    apply_gravity(1.0)
                    refillOff[None] = 1
                    srcCount[None] = 0
                    generate(mode, weak, tilt_deg)
            if gui.button("Protodyakonov: shallow arch"):
                # Пресет для проверки свода естественного равновесия:
                # плоская кровля A=10 (a=5 м), H=10, notch=4, малая глубина
                # (top load=0), K0=0. Калибровка: rp=0.2 МПа, Rc=2 МПа,
                # E=1 ГПа, mu=0.7 -> устойчивый свод ~5 клеток (2.5 м).
                mode = 0
                test_kind = "none"
                test_label = "Protodyakonov f_eff=2 (a=5m, h=2.5m)"
                preset_rock(E=1.0, rp=0.2, rc=2.0, mu=0.7)
                clear_ext_forces()
                P[None].walls = 1
                P[None].fixedRows = 2
                P[None].depth = 0.0
                P[None].k0 = 0.0
                P[None].substeps = max(SUBSTEPS, 192)
                apply_gravity(1.0)
                refillOff[None] = 1
                srcCount[None] = 0
                cx0, cy0, rx0, ry0 = gen_proto_arch()
                focus_region(cx0 - 24.0, cy0 - 22.0, cx0 + 24.0, cy0 + 22.0)
                paused = False
                flash_msg = "Protodyakonov: a=5 m, h=a/f=2.5 m — observe the arch"
                flash_wall = now
            if gui.button("Strainburst"):
                if COLS < 128:
                    flash_msg = "Impact: run GridRockDem128.py"
                else:
                    mode = 0
                    test_kind = "none"
                    test_label = "Strainburst 2000 m (soft+absorb)"
                    preset_rock(E=10.0, rp=4.0, rc=40.0, mu=0.62)
                    clear_ext_forces()
                    P[None].walls = 1
                    P[None].fixedRows = 2
                    P[None].depth = 2000.0
                    P[None].k0 = 1.0
                    P[None].substeps = max(SUBSTEPS, 192)
                    P[None].soften = 1
                    P[None].softResidual = 0.15
                    P[None].absorb = 1
                    P[None].absorbWidth = 8.0
                    P[None].absorbCoef = 2000.0
                    P[None].absorbTop = 1
                    apply_gravity(1.0)
                    refillOff[None] = 1
                    srcCount[None] = 0
                    generate(mode, weak, tilt_deg)
                    focus_region(cavX[None] - 28.0, cavY[None] - 24.0,
                                 cavX[None] + 28.0, cavY[None] + 24.0)
                    paused = False
                    flash_msg = "Strainburst 2000 m: soften+sponge, check localization"
                flash_wall = now
            if gui.button("Fault-slip"):
                if COLS < 128:
                    flash_msg = "Fault-slip: run GridRockDem128.py"
                else:
                    mode = 0
                    test_kind = "none"
                    test_label = "Fault-slip: R=10, L=4.8R, 16deg, K0=0.5, mu=0.50, tanphi=0.40"
                    preset_rock(E=10.0, rp=4.0, rc=40.0, mu=0.50)
                    P[None].mcBond = 0.40
                    clear_ext_forces()
                    P[None].walls = 1
                    P[None].fixedRows = 2
                    P[None].depth = 1500.0
                    P[None].k0 = 0.5
                    P[None].substeps = max(SUBSTEPS, 192)
                    P[None].soften = 1
                    P[None].softResidual = 0.15
                    P[None].absorb = 1
                    P[None].absorbWidth = 8.0
                    P[None].absorbCoef = 2000.0
                    P[None].absorbTop = 1
                    apply_gravity(1.0)
                    refillOff[None] = 1
                    srcCount[None] = 0
                    generate(mode, weak, tilt_deg)
                    # используем существующий инструмент трещин (A* по граням),
                    # а не отдельную реализацию разлома
                    fx0 = cavX[None] - 2.4 * 10.0
                    fy0 = cavY[None] - 3.2 * 10.0
                    fx1 = cavX[None] + 2.4 * 10.0
                    fy1 = cavY[None] - 1.8 * 10.0
                    nf = crack_line(fx0, fy0, fx1, fy1)
                    focus_region(cavX[None] - 36.0, cavY[None] - 30.0,
                                 cavX[None] + 36.0, cavY[None] + 30.0)
                    paused = False
                    flash_msg = ("Fault-slip: fault above the working, K0=0.5, "
                                 "bonds broken %d" % nf)
                flash_wall = now
            if gui.button("Collapse arch (128)"):
                # Готовый опыт: 128×128 на закритической глубине. При 1533 м
                # свод над выработкой теряет устойчивость и обрушается
                # (64×64 при той же глубине ещё стоит — размерный эффект).
                # Камера сразу наведена на выработку и кровлю, чтобы было
                # видно, как формируется свод обрушения.
                if COLS < 128:
                    flash_msg = "Arch: run GridRockDem128.py (current %dx%d)" % (COLS, ROWS)
                else:
                    mode = 0
                    test_kind = "none"
                    test_label = "Collapse arch 128"
                    preset_rock()
                    clear_ext_forces()
                    P[None].walls = 1
                    P[None].fixedRows = 2
                    P[None].depth = 1533.0
                    P[None].k0 = 1.0
                    P[None].substeps = max(SUBSTEPS, 192)
                    apply_gravity(1.0)
                    refillOff[None] = 1
                    srcCount[None] = 0
                    generate(mode, weak, tilt_deg)
                    focus_region(cavX[None] - 22.0, cavY[None] - 30.0,
                                 cavX[None] + 22.0, cavY[None] + 8.0)
                    paused = False
                    flash_msg = ("Arch 128: %.0f m, K0=%.1f — observe the arch"
                                 % (P[None].depth, P[None].k0))
                flash_wall = now
            if gui.button("Fill model"):
                srcCount[None] = 0
                test_kind = "none"
                test_label = ""
                clear_ext_forces()
                apply_gravity(1.0)
                reset_all()
                if abs(tilt_deg) > 0.5:
                    t = math.radians(tilt_deg)
                    gen_tilted(weak, math.cos(t), math.sin(t), 0)
                else:
                    gen_solid()
                flash_msg = "Model filled: blocks %d" % bcount[None]
                flash_wall = now
            if gui.button("Regenerate"):
                srcCount[None] = 0
                test_kind = "none"
                test_label = ""
                clear_ext_forces()
                apply_gravity(1.0)
                refillOff[None] = 1
                generate(mode, weak, tilt_deg)
                flash_msg = "Model: bonds %d" % count_intact()
                flash_wall = now
            if gui.button("Refill: %s" % ("ON" if refillOff[None] == 0 else "OFF")):
                refillOff[None] = 1 - refillOff[None]
                flash_msg = "Top refill: %s" % ("ON" if refillOff[None] == 0 else "OFF")
                flash_wall = now
            # ---- пресеты ВЕРИФИКАЦИИ (Кирш, UCS, бразильский, балка, изгиб) ----
            # Kirsch — обычная сцена выработки с K0/глубиной; остальные — образцы
            # (gen_ucs/gen_brazil/gen_beam/gen_buckling).
            gui.text("--- VERIFICATION ---")
            if gui.button("Kirsch: K0=0, depth 1000 m"):
                mode = 0
                test_kind = "none"
                test_label = "Kirsch K0=0 @1000m"
                preset_rock()
                clear_ext_forces()
                P[None].walls = 1
                P[None].fixedRows = 2
                P[None].k0 = 0.0
                P[None].depth = 1000.0
                P[None].substeps = max(SUBSTEPS, 192)
                apply_gravity(1.0)
                refillOff[None] = 1
                srcCount[None] = 0
                generate(0, weak, tilt_deg)
                reset_camera()
                flash_msg = "Kirsch K0=0: any fractures? START"
                flash_wall = now
            if gui.button("Kirsch: K0=0.5, 1000 m"):
                mode = 0
                test_kind = "none"
                test_label = "Kirsch K0=0.5 @1000m"
                preset_rock()
                clear_ext_forces()
                P[None].walls = 1
                P[None].fixedRows = 2
                P[None].k0 = 0.5
                P[None].depth = 1000.0
                P[None].substeps = max(SUBSTEPS, 192)
                apply_gravity(1.0)
                refillOff[None] = 1
                srcCount[None] = 0
                generate(0, weak, tilt_deg)
                reset_camera()
                flash_msg = "Kirsch K0=0.5: no fractures expected"
                flash_wall = now
            if gui.button("Kirsch: K0=1, 1000 m"):
                mode = 0
                test_kind = "none"
                test_label = "Kirsch K0=1 @1000m"
                preset_rock()
                clear_ext_forces()
                P[None].walls = 1
                P[None].fixedRows = 2
                P[None].k0 = 1.0
                P[None].depth = 1000.0
                P[None].substeps = max(SUBSTEPS, 192)
                apply_gravity(1.0)
                refillOff[None] = 1
                srcCount[None] = 0
                generate(0, weak, tilt_deg)
                reset_camera()
                flash_msg = "Kirsch K0=1: hydrostatic, no fractures"
                flash_wall = now
            if gui.button("UCS (uniaxial compression) 2:1"):
                mode = 0
                test_kind = "ucs"
                test_label = "UCS 10x20 (2:1)"
                preset_rock()
                clear_ext_forces()
                apply_gravity(0.0)
                refillOff[None] = 1
                P[None].walls = 0
                P[None].k0 = 0.0
                P[None].depth = 100.0
                P[None].substeps = max(SUBSTEPS, 192)
                srcCount[None] = 0
                x0, y0, x1, y1 = gen_ucs(10, 20)
                focus_region(x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0)
                flash_msg = "UCS: initial depth 100 m, slider nearby"
                flash_wall = now
            if gui.button("UCS (uniaxial compression) 3:1"):
                mode = 0
                test_kind = "ucs"
                test_label = "UCS 10x30 (3:1)"
                preset_rock()
                clear_ext_forces()
                apply_gravity(0.0)
                refillOff[None] = 1
                P[None].walls = 0
                P[None].k0 = 0.0
                P[None].depth = 100.0
                P[None].substeps = max(SUBSTEPS, 192)
                srcCount[None] = 0
                x0, y0, x1, y1 = gen_ucs(10, 30)
                focus_region(x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0)
                flash_msg = "UCS: initial depth 100 m, slider nearby"
                flash_wall = now
            if gui.button("Brazil (disc splitting)"):
                mode = 1
                test_kind = "brazil"
                preset_rock()
                brazil_P = 0.0
                brazil_rate = 2.5e5
                clear_ext_forces()
                apply_gravity(0.0)
                refillOff[None] = 1
                P[None].walls = 0
                P[None].k0 = 0.0
                P[None].depth = 0.0
                P[None].substeps = max(SUBSTEPS, 192)
                srcCount[None] = 0
                (bcx, bcy, bR), brazil_top = gen_brazil(12)
                brazil_D = 2.0 * bR * LCELL
                test_label = "Brazil disk D=%.0f m" % brazil_D
                focus_region(bcx - bR - 2.0, bcy - bR - 2.0, bcx + bR + 2.0, bcy + bR + 2.0)
                flash_msg = "Brazil: load ramps automatically — watch the crack"
                flash_wall = now
            if gui.button("Beam on two supports"):
                mode = 1
                test_kind = "beam"
                preset_rock()
                test_label = "Beam on two supports"
                pt_P = 0.0
                pt_rate = 5.0e3
                pt_fx = 0.0
                clear_ext_forces()
                apply_gravity(0.0)
                refillOff[None] = 1
                P[None].walls = 0
                P[None].k0 = 0.0
                P[None].depth = 0.0
                P[None].substeps = max(SUBSTEPS, 192)
                srcCount[None] = 0
                (x0, y0, x1, y1), pt_top = gen_beam(48, 4)
                focus_region(x0 - 2.0, y0 - 3.0, x1 + 2.0, y1 + 3.0)
                flash_msg = "Beam: START - center load, supports at ends"
                flash_wall = now
            if gui.button("Buckling (column)"):
                mode = 1
                test_kind = "buckling"
                preset_rock()
                test_label = "Buckling"
                pt_P = 0.0
                pt_rate = 1.0e4
                pt_fx = 2.0e4
                clear_ext_forces()
                apply_gravity(0.0)
                refillOff[None] = 1
                P[None].walls = 0
                P[None].k0 = 0.0
                P[None].depth = 0.0
                P[None].substeps = max(SUBSTEPS, 192)
                srcCount[None] = 0
                (x0, y0, x1, y1), pt_top = gen_buckling(2, 50)
                focus_region(x0 - 6.0, y0 - 2.0, x1 + 6.0, y1 + 2.0)
                flash_msg = "Buckling: START - speed and seed in the solver panel"
                flash_wall = now
            # ---- сохранение/загрузка схем (JSON в ./scenes) ----
            # Текстовый ввод в GGUI недоступен, поэтому слоты нумерованные,
            # выбор — кнопками < >; имя файла = <slot>.json.
            gui.text("--- SCENES (JSON in ./scenes) ---")
            if gui.button("<"):
                scene_idx = (scene_idx - 1) % len(scene_slots)
            if gui.button(">"):
                scene_idx = (scene_idx + 1) % len(scene_slots)
            sname = scene_slots[scene_idx]
            sfile = presets.scene_path(sname)
            gui.text("Slot: %s %s" % (sname, "[exists]" if os.path.exists(sfile) else "[-]"))
            if gui.button("Save"):
                try:
                    nb, nbd = presets.save_scene(sfile, mode=mode, weak=weak,
                                                 tilt_deg=tilt_deg, camera=list(_CAM),
                                                 test=test_label)
                    flash_msg = "Saved %s: blocks %d, bonds %d" % (sname, nb, nbd)
                except Exception as e:
                    flash_msg = "Save error: %s" % e
                flash_wall = now
            if gui.button("Load") and os.path.exists(sfile):
                try:
                    d = presets.load_scene(sfile)
                    mode = int(d.get("mode", 0))
                    weak = int(d.get("weak", 0))
                    tilt_deg = float(d.get("tilt_deg", 0.0))
                    test_label = d.get("test") or ""
                    test_kind = "none"
                    brazil_top = []
                    brazil_P = 0.0
                    pt_top = []
                    pt_P = 0.0
                    E_GPa = max(1, int(round(P[None].E / 1e9)))
                    rho = float(P[None].rho)
                    show_solver = True
                    show_rock = False
                    srcCount[None] = 0
                    clear_ext_forces()
                    cam = d.get("camera")
                    if cam:
                        cam_write(cam[0], cam[1], cam[2])
                    gn = float((d.get("params") or {}).get("g", G_PHYS))
                    apply_gravity(gn / G_PHYS)
                    refillOff[None] = 1
                    flash_msg = "Loaded %s" % sname
                except Exception as e:
                    flash_msg = "Load error: %s" % e
                flash_wall = now
            if gui.button("Delete") and os.path.exists(sfile):
                try:
                    os.remove(sfile)
                    flash_msg = "Deleted %s" % sname
                except Exception as e:
                    flash_msg = "Delete error: %s" % e
                flash_wall = now
            show_bonds = 1 if gui.checkbox("Show bonds", show_bonds == 1) else 0
            head_new = gui.slider_int("Water head H, m", int(hud[HUD_HEAD]), 1, 40)
            if head_new != int(hud[HUD_HEAD]):
                apply_water(float(head_new))
            if gui.button("Drain water"):
                water_init_host()
                filledFlag[None] = 0
                flash_msg = "Water drained"
                flash_wall = now
            if gui.button("Remove sources"):
                srcCount[None] = 0
                flash_msg = "Sources removed"
                flash_wall = now
            fp = hud[HUD_FILL]
            gui.text("fps %.0f * blocks %d" % (fps, int(hud[HUD_BCOUNT])))
            gui.text("Water: cells %d * bonds %d" % (int(hud[HUD_PCOUNT]), int(hud[HUD_INTACT])))
            gui.text("fractures %d * fill %.0f%%" % (int(hud[HUD_BROKEN]), fp))
            gui.text("Energy: E=%.1e J (K=%.1e, P=%.1e)" % (hud[HUD_ETOT], hud[HUD_EKIN], hud[HUD_EPOT]))
            gui.text("Center of mass: x=%.2f, y=%.2f m, mass=%.0f kg" % (hud[HUD_COMX], hud[HUD_COMY], hud[HUD_COMASS]))
            # flash-сообщения — строкой внутри панели (без оверлея поверх канвы)
            if flash_msg and (now - flash_wall < 2.0):
                gui.text(flash_msg)
            gui.end()

        # ---- симуляция
        # Один тик за итерацию окна (без vsync цикл крутится на полной скорости
        # CPU). Каждый тик = фиксированный шаг внутреннего времени TICK; следующий
        # тик начинается только после полного расчёта предыдущего (локальная
        # физика решателя, см. step_physics).
        if not paused:
            # бразильский тест: нагрузка растёт каждый тик вдоль малой дуги сверху
            if test_kind == "brazil" and brazil_top:
                brazil_P += brazil_rate
                q = -brazil_P / len(brazil_top)
                for a in brazil_top:
                    set_ext_force(a, 0.0, q)
            # балка/продольный изгиб: нагружаем верхние блоки вниз (+y) и
            # добавляем малый боковой seed (для выпучивания стойки)
            elif test_kind in ("beam", "buckling") and pt_top:
                pt_P += pt_rate
                q = pt_P / len(pt_top)
                fxl = pt_fx / len(pt_top)
                for a in pt_top:
                    set_ext_force(a, fxl, q)
            step_physics(mode)
            t_sim += TICK
            hud[HUD_SIMT] = t_sim
            dust_step()
            dust_compact()
            if int(hud[HUD_BROKEN]) != last_broken:
                if int(hud[HUD_BROKEN]) % 8 == 1:
                    flash_msg = "Rock crushing (Rc)" if hud[HUD_CRUSH] == 1 else "Bond tension failure (Rp)"
                    flash_wall = now
                last_broken = int(hud[HUD_BROKEN])
            if hud[HUD_FILLFLAG] == 0 and fp >= 95:
                filledFlag[None] = 1
                flash_msg = "Working filled with water"
                flash_wall = now
        else:
            if last_broken != int(hud[HUD_BROKEN]):
                last_broken = int(hud[HUD_BROKEN])

        # ---- отрисовка
        render_base()
        render_triangles(t_sim, 1 if ((mode == 0 or mode == 3) and hud[HUD_DEPTH] > 0
                                      and test_kind != "ucs") else 0)
        # вода рисуется ДО блоков: блоки её перекрывают, а не тонируются ею
        render_water()
        stress_peaks()   # пики напряжений — динамическая шкала легенды
        render_blocks(1 if legend_on else 0)
        render_joints()
        render_bonds(show_bonds, 1 if legend_on else 0)
        render_cracks()
        render_dust()
        render_sources(t_sim)

        pos = window.get_cursor_pos()
        if not panel_zone(pos):
            mx, my = cursor_to_cells(pos)
            render_mouse(mx, my, brush, 1 if tool == 2 else 0)
            if tool == 1 and crack_p0 is not None:
                # резиновая линия трещины от 1-й точки до курсора
                render_rubber(crack_p0[0], crack_p0[1], mx, my)
        else:
            render_mouse(-999.0, 0.0, brush, 0)

        # ---- поворот ВИДА: модель+домен наклонены на экране, g/нагрузка вертикальны
        # (физика — в локальных осях, см. forces_reset). Крутим ПОСЛЕ кисти/курсора
        # (они рисуются в реальных экранных позициях) и ДО легенды (HUD не крутится).
        if abs(presets.model_angle) > 1e-6:
            k = PX * render._CAM[2]
            scx = COLS * LCELL * 0.5 * k + render._CAM[0]
            scy = (H - 1) - (ROWS * LCELL * 0.5 * k + render._CAM[1])
            render.rot_view(math.radians(presets.model_angle), scx, scy)
            render.paste_rot()

        if legend_on:
            render_stress_legend(hud[HUD_RP], hud[HUD_RP] * hud[HUD_RCFACTOR])
        if recorder is not None and t_sim + 1e-9 >= record_next:
            try:
                recorder.capture(img.to_numpy(), t_sim)
                record_next += 1.0 / record_fps
                # Жёсткий лимит: 30 fps × 10 c = 300 кадров. Закрываем сами,
                # чтобы GIF не рос без ограничения.
                if (recorder.frames >= int(record_fps * record_max_s)
                        or t_sim - record_start >= record_max_s - 1e-9):
                    out = recorder.close()
                    recorder = None
                    flash_msg = "Record saved (10 s / 30 fps): %s" % out
                    flash_wall = now
            except Exception as e:
                try:
                    recorder.close()
                except Exception:
                    pass
                recorder = None
                flash_msg = "Record error: %s" % e
                flash_wall = now
        canvas.set_image(img)
        window.show()

        fps += (1.0 / max(1e-6, dt_real) - fps) * 0.06
        if now - hud_t > 0.25:
            hud_t = now

    if recorder is not None:
        try:
            recorder.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="GridRockDem - 2D DEM on Taichi")
    ap.add_argument("--selftest", action="store_true", help="run physics without a window")
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()
    if args.selftest:
        run_selftest(args.frames)
    else:
        run_gui()


if __name__ == "__main__":
    main()



