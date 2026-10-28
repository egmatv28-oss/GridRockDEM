"""ГЕОМАССИВ/2D · DEM — визуализация (render)

Камера и ввод, примитивы рисования и ядра отрисовки сцены, анимация пыли.
Здесь НЕТ генерации сцен (см. presets.py) и НЕТ панели/инструментов (см. gui.py).
Модель и физика — solver.py/physstate.py. Точки входа — GridRockDem64.py / GridRockDem128.py.
"""

import math

import taichi as ti
import water

from solver import *

img = ti.Vector.field(3, ti.f32, (W, H))
img_rot = ti.Vector.field(3, ti.f32, (W, H))   # буфер повёрнутого вида

# камера: CAMX/CAMY — смещение картинки в пикселях окна, CAMZ — масштаб
# (во сколько раз увеличен рисунок; 1.0 = как раньше). Хранятся в полях,
# чтобы их видели GPU-ядра отрисовки.
CAMX = ti.field(ti.f32, ())
CAMY = ti.field(ti.f32, ())
CAMZ = ti.field(ti.f32, ())
CAMX[None] = 0.0
CAMY[None] = 0.0
CAMZ[None] = 1.0
CAM_ZMIN = 0.25
CAM_ZMAX = 8.0

# python-зеркало камеры: хост-код (курсор/зум/пан) работает с ним, а не читает
# поля — каждый read скалярного поля из Python это синхронизация CPU<->GPU.
# GPU-ядра рендера по-прежнему читают поля CAMX/CAMY/CAMZ.
_CAM = [0.0, 0.0, 1.0]
panel_visible = True   # панель управления (клавиша P / кнопка) — зона ввода 0.71

# угол обзора модели (град): вся сцена (модель+домен) на экране поворачивается на
# этот угол вокруг центра домена. Физика при этом в ЛОКАЛЬНЫХ осях (модель осевая,
# силы — проекции на локальные оси), поэтому гравитация/нагрузки выглядят
# вертикальными, а модель — наклонённой. Хост-зеркало нужно для инверсии ввода.
_view_deg = [0.0]


def set_view(deg):
    _view_deg[0] = deg


@ti.kernel
def rot_view(ang: ti.f32, scx: ti.f32, scy: ti.f32):
    """Повернуть готовую сцену img на ang вокруг точки (scx, scy) в img_rot.

    Мировая точка экранной проекции p берётся из источника Sc + R(ang)·(p - Sc),
    где Sc — экранная позиция центра домена. Это в точности «поворот всего»
    (модель+домен+сетка+пыль) на экране на угол ang."""
    c = ti.cos(ang)
    s = ti.sin(ang)
    for X, Y in img:
        dx = X - scx
        dy = Y - scy
        sx = scx + c * dx - s * dy
        sy = scy + s * dx + c * dy
        ix = int(sx + 0.5)
        iy = int(sy + 0.5)
        if ix >= 0 and ix < W and iy >= 0 and iy < H:
            img_rot[X, Y] = img[ix, iy]
        else:
            img_rot[X, Y] = BG


@ti.kernel
def paste_rot():
    """Вернуть повёрнутый буфер в img (чтобы дальше рисовать HUD-легенду)."""
    for X, Y in img:
        img[X, Y] = img_rot[X, Y]


def cam_write(ox, oy, oz):
    _CAM[0] = ox
    _CAM[1] = oy
    _CAM[2] = oz
    CAMX[None] = ox
    CAMY[None] = oy
    CAMZ[None] = oz


# =================================================== ввод символов с клавиатуры
# Taichi GGUI (button_id_to_name в taichi/ui/utils/utils.h) отдаёт события
# ТОЛЬКО для букв и спецклавиш (Shift/Return/Escape/BackSpace/Space/стрелки):
# GLFW-коды цифр, минуса и точки кидают std::runtime_error и событие ТЕРЯЕТСЯ
# в window_base.cpp. То есть цифры через window.get_events() не приходят
# НИКОГДА — старый «ввод числа угла» был мёртв по этой причине. Символы
# читаем напрямую из состояния клавиатуры (user32.GetAsyncKeyState), ловя
# фронт нажатия up->down (авто-повтор ОС не дублирует ввод). Опрос идёт
# только когда окно GridRockDem активно.
import ctypes

_user32 = ctypes.windll.user32
_user32.GetActiveWindow.restype = ctypes.c_void_p
_user32.GetForegroundWindow.restype = ctypes.c_void_p
_user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
_vk_chars = {}
for _d in range(10):
    _vk_chars[0x30 + _d] = str(_d)     # основной ряд цифр
    _vk_chars[0x60 + _d] = str(_d)     # numpad
_vk_chars[0xBD] = "-"                  # VK_OEM_MINUS
_vk_chars[0xBE] = "."                  # VK_OEM_PERIOD
_vk_chars[0xBC] = "."                  # VK_OEM_COMMA (запятая = десятичная точка)
_vk_chars[0x6E] = "."                  # VK_DECIMAL (numpad)
_vk_prev = set()
_our_hwnd = None
WIN_TITLE = "GridRockDem"


def _app_focused():
    """Активно ли окно GridRockDem: опрос клавиш — только для своего окна."""
    global _our_hwnd
    if _our_hwnd is None:
        _our_hwnd = _user32.GetActiveWindow()
        if not _our_hwnd:
            hwnd = _user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            if hwnd:
                _user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value.startswith(WIN_TITLE):
                _our_hwnd = hwnd
        if not _our_hwnd:
            _our_hwnd = -1   # окно не нашли — работаем без проверки фокуса
    if _our_hwnd == -1:
        return True
    return _user32.GetForegroundWindow() == _our_hwnd


def poll_type_chars():
    """Символы для набора числа: цифры, '-', '.' (',' и numpad-точка = '.').

    Список символов по фронтам нажатия (up->down); пусто, если окно не активно."""
    out = []
    if not _app_focused():
        _vk_prev.clear()
        return out
    for vk, ch in _vk_chars.items():
        down = (_user32.GetAsyncKeyState(vk) & 0x8000) != 0
        was = vk in _vk_prev
        if down and not was:
            out.append(ch)
        if down:
            _vk_prev.add(vk)
        else:
            _vk_prev.discard(vk)
    return out


# numpad +/- — зум камеры; события ti.ui их не приносят (как цифры), поэтому
# опросом GetAsyncKeyState с автоповтором ~8 Гц при удержании.
_VK_ADD = 0x6B
_VK_SUB = 0x6D
_zoom_next = {0x6B: 0.0, 0x6D: 0.0}


def poll_zoom_keys(now):
    """+1/-1/0 — шаг зума за кадр (numpad +/-), с автоповтором при удержании."""
    if not _app_focused():
        return 0
    step = 0
    for vk, sgn in ((_VK_ADD, 1), (_VK_SUB, -1)):
        down = (_user32.GetAsyncKeyState(vk) & 0x8000) != 0
        if not down:
            _zoom_next[vk] = 0.0
        elif now >= _zoom_next[vk]:
            step += sgn
            _zoom_next[vk] = now + 0.125
    return step


def parse_deg(buf):
    try:
        return float(buf.replace(",", "."))
    except ValueError:
        return None

# ================================================================== отрисовка
BG = ti.Vector([0.0431, 0.0667, 0.0941])
GRID = ti.Vector([0.49, 0.608, 0.765])
AMBER = ti.Vector([0.961, 0.647, 0.141])
CYAN = ti.Vector([0.239, 0.784, 1.0])
FIXED = ti.Vector([0.169, 0.212, 0.267])
CRACK = ti.Vector([0.72, 0.32, 0.9])   # яркий разрыв: тёмный терялся на фоне
ORANGE = ti.Vector([0.96, 0.30, 0.14])
BLUE = ti.Vector([0.22, 0.62, 0.96])
PURPLE = ti.Vector([0.72, 0.32, 0.9])
JOINT = ti.Vector([0.34, 0.41, 0.48])  # ослабленный шов (не разрыв) — приглушённый
WHITE = ti.Vector([1.0, 1.0, 1.0])

# --- палитра шкал напряжений: по 10 ступеней на шкалу ---------------------
# Растяжение — тёплая шкала (бордовый -> красный -> оранжевый -> жёлтый),
# сжатие — холодная (тёмно-синий -> синий -> голубой -> белёсый). Яркость
# ступеней растёт равномерно: соседние квадраты легенды различаются чётко,
# а две шкалы мгновенно различаются по оттенку. Цвета ТОЧНЫЕ (без сглаженной
# интерполяции): квадрат легенды, блок и линия связи одного интервала всегда
# одного цвета.
BAND_RGB = (
    (   # шкала растяжения (0)
        (122, 30, 45), (158, 36, 42), (190, 42, 38), (216, 55, 32),
        (234, 74, 26), (244, 100, 24), (250, 130, 30), (253, 162, 42),
        (255, 195, 62), (255, 226, 110),
    ),
    (   # шкала сжатия (1)
        (28, 48, 112), (32, 72, 156), (36, 98, 194), (42, 124, 220),
        (52, 150, 236), (72, 172, 243), (100, 192, 247), (136, 210, 250),
        (176, 226, 251), (216, 241, 253),
    ),
)
bandCol = ti.field(ti.f32, (2, 10, 3))
for _bi in range(2):
    for _k in range(10):
        for _c in range(3):
            bandCol[_bi, _k, _c] = BAND_RGB[_bi][_k][_c] / 255.0


@ti.func
def fill_rect(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, col: ti.template()):
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 >= W:
        x1 = W - 1
    if y1 >= H:
        y1 = H - 1
    X = x0
    while X <= x1:
        Y = y0
        while Y <= y1:
            img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def fill_rot_rect(cx: ti.f32, cy: ti.f32, ang: ti.f32, half: ti.f32, col: ti.template()):
    # заливка квадрата со стороной 2*half, повёрнутого на ang вокруг (cx, cy)
    # ang в соглашении физики: brot>0 = по часовой на экране (мир y вниз),
    # буфер же Y вверх, поэтому тест R(+ang)·d даёт фигуру R(-ang) — по часовой
    c = ti.cos(ang)
    s = ti.sin(ang)
    # половинки повёрнутого квадрата в проекциях на оси
    hx = half * (ti.abs(c) + ti.abs(s))
    hy = hx
    X = ti.max(0, int(cx - hx))
    while X <= ti.min(W - 1, int(cx + hx)):
        Y = ti.max(0, int(cy - hy))
        while Y <= ti.min(H - 1, int(cy + hy)):
            dx = X - cx
            dy = Y - cy
            ux = c * dx - s * dy
            uy = s * dx + c * dy
            if ti.abs(ux) <= half and ti.abs(uy) <= half:
                img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def dist_to_seg(px: ti.f32, py: ti.f32, x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32) -> ti.f32:
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    d = 0.0
    if l2 < 1e-9:
        d = ti.sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1))
    else:
        t = ((px - x1) * dx + (py - y1) * dy) / l2
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        cx = x1 + t * dx
        cy = y1 + t * dy
        d = ti.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
    return d


@ti.func
def draw_line(x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32, col: ti.template(), w: ti.f32):
    lo_x = int(ti.min(x1, x2) - w)
    hi_x = int(ti.max(x1, x2) + w)
    lo_y = int(ti.min(y1, y2) - w)
    hi_y = int(ti.max(y1, y2) + w)
    X = ti.max(0, lo_x)
    while X <= ti.min(W - 1, hi_x):
        Y = ti.max(0, lo_y)
        while Y <= ti.min(H - 1, hi_y):
            d = dist_to_seg(X + 0.5, Y + 0.5, x1, y1, x2, y2)
            if d <= w * 0.5 + 0.5:
                img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def draw_circle_outline(cx: ti.f32, cy: ti.f32, r: ti.f32, col: ti.template(), w: ti.f32):
    X = int(cx - r - w)
    while X <= int(cx + r + w):
        Y = int(cy - r - w)
        while Y <= int(cy + r + w):
            d = ti.sqrt((X - cx) * (X - cx) + (Y - cy) * (Y - cy))
            if ti.abs(d - r) <= w * 0.5 + 0.5:
                if X >= 0 and X < W and Y >= 0 and Y < H:
                    img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def fill_diamond(cx: ti.f32, cy: ti.f32, rad: ti.f32, col: ti.template()):
    X = int(cx - rad)
    while X <= int(cx + rad):
        Y = int(cy - rad)
        while Y <= int(cy + rad):
            if ti.abs(X - cx) + ti.abs(Y - cy) <= rad:
                if X >= 0 and X < W and Y >= 0 and Y < H:
                    img[X, Y] = col
            Y += 1
        X += 1


@ti.kernel
def render_base():
    z = CAMZ[None]
    k = PX * z
    ox = CAMX[None]
    oy = CAMY[None]
    d = 0.5 / k
    for X in range(W):
        for Y in range(H):
            col = BG
            wx = (X + 0.5 - ox) / k
            wy = (H - 1.5 - Y - oy) / k
            if ti.abs(wx - ti.round(wx)) < d or ti.abs(wy - ti.round(wy)) < d:
                a = 0.045
                col = col * (1.0 - a) + GRID * a
            img[X, Y] = col


@ti.kernel
def render_triangles(t: ti.f32, on: ti.i32):
    if on == 1:
        a = 0.35 + 0.2 * ti.sin(t * 3.0)
        col = BG * (1.0 - a) + AMBER * a
        zc = CAMZ[None]
        oxc = CAMX[None]
        x = 2
        while x < COLS:
            # стрелки нагрузки рисуем над ОН-СКРИН шириной домена (камера может
            # быть отдалена: у 128 z ~0.45, у 64 ~0.9)
            X = int((x + 0.5) * CELL * zc + oxc)
            bob = ti.sin(t * 3.0 + x) * 1.5
            y = int(4.0 + bob)
            while y <= int(12.0 + bob):
                tt = (y - (4.0 + bob)) / (12.0 + bob - (4.0 + bob))
                half = 4.0 * (1.0 - tt)
                px = int(X - half)
                while px <= int(X + half):
                    if px >= 0 and px < W and y >= 0 and y < H:
                        img[px, H - 1 - y] = col
                    px += 1
                y += 1
            x += 4


@ti.func
def stress_band(q: ti.f32, pk: ti.f32) -> ti.i32:
    """Номер цветового интервала 0..9 для напряжения q (нормировано на прочность)
    при пике шкалы pk: шкала разбита на 10 равных интервалов 0..pk."""
    qq = 0.0
    if pk > 1e-6:
        qq = q / pk
    if qq > 1.0:
        qq = 1.0
    k = int(qq * 10.0)
    if k > 9:
        k = 9
    return k


@ti.func
def legend_band_color(scale: ti.i32, k: ti.i32):
    """Цвет интервала k (0..9) шкалы: scale 0 — растяжение (тёплая шкала),
    1 — сжатие (холодная). Тёмные цвета соответствуют большим напряжениям,
    светлые — малым. Точный цвет из палитры BAND_RGB — один и тот же
    для квадрата легенды, блока и линии связи своего интервала."""
    kk = 9 - k
    return ti.Vector([bandCol[scale, kk, 0], bandCol[scale, kk, 1], bandCol[scale, kk, 2]])


@ti.func
def ramp_color(scale: ti.i32, t: ti.f32):
    """Непрерывный цвет шкалы для t в 0..1: кусочно-линейно по ступеням палитры
    (для заливки блоков — градиент без «ступенек», те же цвета, что у легенды)."""
    tt = t
    if tt < 0.0:
        tt = 0.0
    if tt > 1.0:
        tt = 1.0
    x = tt * 9.0
    k0 = int(x)
    if k0 > 8:
        k0 = 8
    f = x - k0
    c0 = legend_band_color(scale, k0)
    c1 = legend_band_color(scale, k0 + 1)
    return c0 + (c1 - c0) * f


@ti.kernel
def stress_peaks():
    """Пики напряжений по сцене для динамической шкалы легенды (нормировано на
    прочность: 0 — растяжение, 1 — сжатие). Пик растёт мгновенно, спадает
    медленно — значения на шкале меняются плавно, без дёрганья."""
    stressPeakRaw[0] = 0.0
    stressPeakRaw[1] = 0.0
    for i in range(bcount[None]):
        if bbedrock[i] != 0 or bfixed[i] == 1:
            continue
        ti.atomic_max(stressPeakRaw[0], bstress[i])
        ti.atomic_max(stressPeakRaw[1], bstressC[i])
    for k in range(2):
        if stressPeakRaw[k] > stressPeak[k]:
            stressPeak[k] = stressPeakRaw[k]
        else:
            stressPeak[k] = stressPeak[k] * 0.97 + stressPeakRaw[k] * 0.03


@ti.kernel
def render_blocks(show: ti.i32):
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for i in range(bcount[None]):
        ri = bsz[i] * PX * z * 0.5
        px = int(bsz[i] * PX * z)
        X = int(bx[i] * PX * z + ox - ri)
        Y = H - 1 - int(by[i] * PX * z + oy + ri)
        if bbedrock[i] != 0 or bfixed[i] == 1:
            fill_rect(X, Y, X + px - 1, Y + px - 1, FIXED)
            diag = FIXED * (1.0 - 0.35) + AMBER * 0.35
            draw_line(X + 3, Y + 3, X + px - 3, Y + px - 3, diag, 1.0)
            strip = FIXED * (1.0 - 0.14) + AMBER * 0.14
            fill_rect(X, Y + px - 3, X + px - 1, Y + px - 1, strip)
        else:
            s = bshade[i]
            r = 112.0 + s * 26.0
            g = 121.0 + s * 24.0
            b = 134.0 + s * 22.0
            # Закраска блоков по напряжениям (сжатие/растяжение) убрана:
            # блоки всегда рисуются телом (серый тон + шум bshade).
            # Напряжения показывают связи (render_bonds) и легенда шкал.
            col = ti.Vector([r / 255.0, g / 255.0, b / 255.0])
            if ti.abs(brot[i]) > 0.02:
                fill_rot_rect(bx[i] * PX * z + ox, H - 1 - (by[i] * PX * z + oy), brot[i], ri, col)
            else:
                fill_rect(X, Y, X + px - 1, Y + px - 1, col)
                hl = col * (1.0 - 0.07) + WHITE * 0.07
                fill_rect(X, Y + px - 2, X + px - 1, Y + px - 1, hl)


# =================================================== легенда напряжений (шрифт 3x5)
# ti.ui не умеет рисовать текст в канвасе — цифры шкал рисуем пикселями:
# глиф 3x5 (маска 15 бит), масштаб s пикселей на точку.

@ti.func
def digit_mask(d: ti.i32) -> ti.i32:
    """Маска глифа 3x5 (15 бит, старший бит — левый верхний пиксель).
    0..9 — цифры, 10 — точка, 11 — минус, 12..14 — буквы М, П, а (МПа)."""
    m = 0
    if d == 0:
        m = 0b111101101101111
    elif d == 1:
        m = 0b010110010010111
    elif d == 2:
        m = 0b111001111100111
    elif d == 3:
        m = 0b111001111001111
    elif d == 4:
        m = 0b101101111001001
    elif d == 5:
        m = 0b111100111001111
    elif d == 6:
        m = 0b111100111101111
    elif d == 7:
        m = 0b111001001001001
    elif d == 8:
        m = 0b111101111101111
    elif d == 9:
        m = 0b111101111001111
    elif d == 10:
        m = 0b000000000000010
    elif d == 11:
        m = 0b000001110000000
    elif d == 12:
        m = 0b101111111101101   # М
    elif d == 13:
        m = 0b111101111100100   # P
    else:
        m = 0b010101111101101   # а (А-образный)
    return m


@ti.func
def draw_digit(x: ti.i32, y: ti.i32, d: ti.i32, s: ti.i32, col: ti.template()):
    """Один глиф 3x5 масштаба s; y — НИЗ глифа. Канвас img на экране идёт
    снизу вверх (img[.,0] — низ экрана), поэтому строки глифа рисуются вверх
    от y — текст на экране читается, а не перевёрнут."""
    m = digit_mask(d)
    for py in range(5):
        for px in range(3):
            if ((m >> (14 - (py * 3 + px))) & 1) == 1:
                fill_rect(x + px * s, y + (4 - py) * s,
                          x + px * s + s - 1, y + (4 - py) * s + s - 1, col)


@ti.func
def draw_num(x: ti.i32, y: ti.i32, val: ti.f32, s: ti.i32, col: ti.template(), dec: ti.i32):
    """Число (МПа): целая часть + dec знаков после точки (0..2), y — низ цифр.
    Значение округляется до dec знаков, хвостовые нули при dec=1 не пишутся."""
    v = val
    x0 = x
    if v < 0.0:
        draw_digit(x0, y, 11, s, col)
        x0 += 4 * s
        v = -v
    scale = 1
    if dec == 1:
        scale = 10
    elif dec == 2:
        scale = 100
    num = int(v * float(scale) + 0.5)   # значение в долях 1/scale
    whole = num // scale
    frac = num - whole * scale
    if whole >= 100:
        draw_digit(x0, y, (whole // 100) % 10, s, col)
        x0 += 4 * s
    if whole >= 10:
        draw_digit(x0, y, (whole // 10) % 10, s, col)
        x0 += 4 * s
    draw_digit(x0, y, whole % 10, s, col)
    x0 += 4 * s
    if dec == 1 and frac != 0:
        draw_digit(x0, y, 10, s, col)
        x0 += 2 * s
        draw_digit(x0, y, frac, s, col)
    elif dec == 2:
        draw_digit(x0, y, 10, s, col)
        x0 += 2 * s
        draw_digit(x0, y, (frac // 10) % 10, s, col)
        x0 += 4 * s
        draw_digit(x0, y, frac % 10, s, col)


@ti.kernel
def render_stress_legend(rp: ti.f32, rc: ti.f32):
    """Динамическая легенда напряжений (левый верхний угол экрана): по 10
    цветных квадратов на шкалу — красная (растяжение) и синяя (сжатие),
    значения в МПа от 0 до ПИКА напряжений в сцене (stressPeak), поэтому
    числа меняются по мере роста/спада напряжений. Цвет квадрата —
    legend_band_color: ровно тот же, что у линий связей своего интервала
    (render_bonds). Каждый квадрат подписан верхней границей интервала.
    Канвас: y ВВЕРХ (img[.,0] — низ экрана) — вся геометрия снизу вверх,
    глифы рисуются от нижнего края, чтобы текст читался, а не был вверх ногами."""
    s = 3                      # масштаб шрифта (пикселей в точке)
    n = 10                     # квадратов (цветовых интервалов) на шкалу
    sw_w = 18                  # квадрат: ширина
    sw_h = 20                  # квадрат: высота
    stack_h = n * sw_h
    x_t = 26                   # шкала растяжения (красная)
    x_c = 168                  # шкала сжатия (синяя)
    y_base = H - 280           # низ стопки квадратов (y канваса вверх)
    # подложка: числа читаются на любом фоне сцены
    fill_rect(x_t - 10, y_base - 24, x_c + 116, y_base + stack_h + 30,
              ti.Vector([0.05, 0.06, 0.08]))
    for bi in range(2):
        xb = x_t if bi == 0 else x_c
        pk = stressPeak[0] if bi == 0 else stressPeak[1]
        vmax = rp if bi == 0 else rc
        if pk < 1e-4:
            pk = 1.0            # напряжений ещё нет — шкала 0..предел прочности
        vmax = vmax * pk
        # над шкалой — единицы измерения: МПа (глифы 12, 13, 14)
        wcol = ti.Vector([1.0, 1.0, 1.0])
        draw_digit(xb, y_base + stack_h + 6, 12, s, wcol)
        draw_digit(xb + 4 * s, y_base + stack_h + 6, 13, s, wcol)
        draw_digit(xb + 8 * s, y_base + stack_h + 6, 14, s, wcol)
        for k in range(n):
            col = legend_band_color(bi, k)
            y0 = y_base + k * sw_h
            fill_rect(xb, y0, xb + sw_w - 1, y0 + sw_h - 2, col)
            # подпись: верхняя граница интервала (МПа). Точность подбирается так,
            # чтобы значения не сливались и не «прыгали» после округления:
            # <10 МПа — два знака (0.45, 0.89...), 10..100 — один (4.5, 9, 13.5...),
            # >=100 — целые (15, 30...). Раньше всё >=10 округлялось до целых,
            # и шкала сжатия (часто 10..100 МПа) давала рваный ряд 5, 9, 14, 18...
            dec = 2
            if vmax >= 10.0:
                dec = 1
            if vmax >= 100.0:
                dec = 0
            draw_num(xb + sw_w + 6, y0 + (sw_h - 5 * s) // 2,
                     vmax * float(k + 1) / float(n), s, wcol, dec)
        # ноль — под стопкой квадратов
        draw_num(xb + sw_w + 6, y_base - 17, 0.0, s, wcol, 0)


@ti.kernel
def render_bonds(show: ti.i32, scale: ti.i32):
    """Линии напряжений по связям: растяжение — красная шкала, сжатие — синяя.
    При scale=1 (легенда включена) цвет линии — цвет интервала
    legend_band_color её напряжения, ровно как квадрат на шкале легенды;
    иначе обычные оранжевый/синий."""
    if show == 1:
        z = CAMZ[None]
        ox = CAMX[None]
        oy = CAMY[None]
        for bd in range(bondCount[None]):
            # Только ЦЕЛЫЕ связи. Разорванная связь становится BOND_JOINT
            # (сжатие-только) и рисуется как трещина/шов в render_cracks;
            # раньше условие "!= BOND_DEAD" рисовало её линией напряжения,
            # и у обрушенных блоков связь выглядела целой.
            if bondIntact[bd] == BOND_INTACT:
                a = bondA[bd]
                b = bondB[bd]
                if bbedrock[a] != 0 or bbedrock[b] != 0 or bfixed[a] == 1 or bfixed[b] == 1:
                    continue
                r = bondR[bd]
                tphy = r
                cphy = -r * 0.25 * P[None].rcFactor if r < 0.0 else 0.0
                col = ORANGE
                if tphy > 0.12 or cphy > 0.12:
                    if tphy >= cphy:
                        col = ORANGE
                        if scale == 1:
                            col = legend_band_color(0, stress_band(r, stressPeak[0]))
                    else:
                        col = BLUE
                        if scale == 1:
                            col = legend_band_color(1, stress_band(-r * 0.25, stressPeak[1]))
                    # На мелком масштабе (128) 2-пиксельные линии забивают
                    # картинку: связи рисуем тоньше, швы остаются читаемыми.
                    bw = 2.0 if z > 0.6 else 1.0
                    draw_line(bx[a] * PX * z + ox, H - 1 - (by[a] * PX * z + oy),
                              bx[b] * PX * z + ox, H - 1 - (by[b] * PX * z + oy), col, bw)


@ti.kernel
def render_joints():
    # швы между связанными блоками (линия по общему ребру, поперёк связи)
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for bd in range(bondCount[None]):
        if bondIntact[bd] != BOND_INTACT:
            continue
        a = bondA[bd]
        b = bondB[bd]
        mx = (bx[a] + bx[b]) * 0.5
        my = (by[a] + by[b]) * 0.5
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        d = ti.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            continue
        ux = dx / d
        uy = dy / d
        w = ti.min(bsz[a], bsz[b]) * 0.5
        # нормаль связи В БУФЕРЕ (мир y вниз, буфер Y = H-1-y вверх):
        # связь в буфере (ux, -uy), её перпендикуляр (uy, ux). Раньше перпендикуляр
        # брался в мировых осях (-uy, ux) и рисовался без переворота Y — засечка
        # зеркалилась и на повёрнутой модели стояла поперёк связи с ошибкой 2*theta
        # (ровно как квадраты блоков до фикса fill_rot_rect).
        px = uy * w * PX * z
        py = ux * w * PX * z
        cx0 = mx * PX * z + ox
        cy0 = H - 1 - (my * PX * z + oy)
        draw_line(cx0 - px, cy0 - py, cx0 + px, cy0 + py, JOINT, 1.0)


@ti.kernel
def render_cracks():
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for bd in range(bondCount[None]):
        if bondIntact[bd] == BOND_INTACT:
            continue
        if bondIntact[bd] == BOND_DEAD:
            continue        # связь удалена — трещины тут нет
        # Разорванная связь тоже становится BOND_JOINT (см. fracture_bond), но у
        # неё bondDmg > 0. Поэтому: dmg>0 — ЯРКАЯ трещина; dmg==0 — ненадрванный
        # ослабленный шов (приглушённый, не путать с разрывом).
        ccol = CRACK
        if bondDmg[bd] <= 0.0:
            ccol = JOINT
        a = bondA[bd]
        b = bondB[bd]
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        dd = dx * dx + dy * dy
        if dd > 1.7 * LCELL * LCELL:
            continue
        d = ti.sqrt(dd)
        mx = (bx[a] + bx[b]) * 0.5 * PX * z + ox
        my = H - 1 - ((by[a] + by[b]) * 0.5 * PX * z + oy)
        j = bondJ[bd] * PX * z
        half = ti.min(bsz[a], bsz[b]) * PX * z * 0.5
        # оси буфера (мир y вниз -> Y вверх): вдоль связи (ux, -uy), поперёк (uy, ux).
        ux = dx / d
        uy = dy / d
        tx = ux
        ty = -uy
        nx = uy
        ny = ux
        # линия ШВА: вся грань от угла до угла. Путь трщины идёт по сетке
        # границ блоков (см. crack_line) - подряд идущие разрывы стыкуются
        # углами, поэтому шов читается сплошной линией на любом угле.
        draw_line(mx - half * nx, my - half * ny,
                  mx + half * nx, my + half * ny, ccol, 2.0)
        # полюс скольжения: короткая засечка ВДОЛЬ связи со смещением
        if ti.abs(j) > 0.05:
            draw_line(mx + 0.05 * j * tx, my + 0.05 * j * ty,
                      mx + 0.75 * j * tx, my + 0.75 * j * ty, ccol, 1.0)


@ti.kernel
def render_water():
    if P[None].lbmOn == 1:
        z = CAMZ[None]
        ox = CAMX[None]
        oy = CAMY[None]
        for c in range(N):
            fr = water.fill[c]
            if water.solid[c] == 0 and fr > 0.04:
                x = c % COLS
                y = c // COLS
                fc = ti.min(fr, 1.0)
                sp = ti.sqrt(water.ux[c] * water.ux[c] + water.uy[c] * water.uy[c])
                a = 0.20 + 0.55 * fc + 0.20 * ti.min(sp * 0.15, 1.0) * fc
                if a > 0.90:
                    a = 0.90
                col = CYAN * a + BG * (1.0 - a)
                X = int(x * CELL * z + ox)
                Y = H - 1 - int(y * CELL * z + oy)
                fill_rect(X, Y, X + max(1, int(CELL * z)) - 1,
                          Y + max(1, int(CELL * z)) - 1, col)


@ti.kernel
def render_dust():
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for i in range(dustCount[None]):
        if i >= MAXDUST:
            continue
        a = dustLife[i] * 0.5
        c = ti.Vector([0.745, 0.784, 0.843])
        if dustCol[i] == 1:
            c = ti.Vector([0.471, 0.667, 0.784])
        cc = BG * (1.0 - a) + c * a
        cx = int(dustX[i] * PX * z + ox) - 2
        cy = H - 1 - int(dustY[i] * PX * z + oy) + 2
        fill_rect(cx, cy, cx + 3, cy + 3, cc)


@ti.kernel
def render_sources(t: ti.f32):
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for s in range(srcCount[None]):
        X = srcX[s] * LCELL * PX * z + ox
        Y = H - 1 - (srcY[s] * LCELL * PX * z + oy)
        pulse = (t * 1.1 + srcPh[s]) % 1.0
        r = (0.35 + pulse * 0.95) * CELL * z
        draw_circle_outline(X, Y, r, CYAN * (0.55 * (1.0 - pulse)) + BG * (1.0 - 0.55 * (1.0 - pulse)), 2.0)
        fill_diamond(X, Y, 5.0 * z, CYAN)


@ti.kernel
def render_mouse(mx: ti.f32, my: ti.f32, brush: ti.f32, is_water: ti.i32):
    if mx > -900.0:
        z = CAMZ[None]
        X = mx * CELL * z + CAMX[None]
        Y = H - 1 - (my * CELL * z + CAMY[None])
        if is_water == 1:
            draw_circle_outline(X, Y, 8.0 * z, CYAN * 0.7, 1.5)
        else:
            r = (brush / 2.0 + 0.3) * CELL * z
            draw_circle_outline(X, Y, r, AMBER * 0.7, 1.5)


@ti.kernel
def render_rubber(x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32):
    # резиновая линия трещины: от первой точки до курсора (как в AutoCAD)
    z = CAMZ[None]
    X0 = x0 * CELL * z + CAMX[None]
    Y0 = H - 1 - (y0 * CELL * z + CAMY[None])
    X1 = x1 * CELL * z + CAMX[None]
    Y1 = H - 1 - (y1 * CELL * z + CAMY[None])
    draw_line(X0, Y0, X1, Y1, AMBER, 1.0)
    # крестик на первой точке
    draw_line(X0 - 6, Y0, X0 + 6, Y0, AMBER, 1.0)
    draw_line(X0, Y0 - 6, X0, Y0 + 6, AMBER, 1.0)


@ti.kernel
def dust_step():
    for i in range(dustCount[None]):
        if i >= MAXDUST:
            continue
        dustX[i] += dustVX[i]
        dustY[i] += dustVY[i]
        dustVY[i] += 0.004 * LCELL
        dustLife[i] -= 0.04


@ti.kernel
def dust_compact():
    w = 0
    i = 0
    while i < dustCount[None]:
        if i < MAXDUST and dustLife[i] > 0.0:
            dustX[w] = dustX[i]
            dustY[w] = dustY[i]
            dustVX[w] = dustVX[i]
            dustVY[w] = dustVY[i]
            dustLife[w] = dustLife[i]
            dustCol[w] = dustCol[i]
            w += 1
        i += 1
    dustCount[None] = w


# ================================================================== ввод
# курсор окна (нормализованные координаты) -> координаты КЛЕТОК модели
def cursor_to_cells(pos):
    k = CELL * _CAM[2]
    sx = pos[0] * W
    sy = pos[1] * H
    mx = (sx - _CAM[0]) / k
    my = (H - 1.5 - sy - _CAM[1]) / k
    # Повёрнутый вид: экранные координаты дают положение ТОЧКИ НА ЭКРАНЕ, а физика
    # живёт в локальных (неповёрнутых) клетках. Инвертируем поворот обзора:
    # world = Cc + R(-θ)·(screen_cells - Cc), R(-θ) = [[c, s], [-s, c]].
    ang = math.radians(_view_deg[0])
    if ang != 0.0:
        c = math.cos(ang)
        s = math.sin(ang)
        ccx = COLS * 0.5
        ccy = ROWS * 0.5
        dx = mx - ccx
        dy = my - ccy
        mx = ccx + c * dx + s * dy
        my = ccy - s * dx + c * dy
    return mx, my


def panel_zone(pos):
    """Клик под панелью (правые 29% окна) — только когда панель ВИДИМА.

    Раньше проверка pos[0] > 0.71 стояла жёстко в пяти местах и блокировала
    кисть/курсор даже при скрытой панели."""
    return panel_visible and pos[0] > 0.64


# зум мышью: масштабируется к точке под курсором, чтобы она осталась на месте
def zoom_at(pos, factor):
    z0 = _CAM[2]
    z1 = min(CAM_ZMAX, max(CAM_ZMIN, z0 * factor))
    if z1 == z0:
        return
    sx = pos[0] * W
    sy = pos[1] * H
    # при повёрнутом виде точка под курсором на экране — ПОСЛЕ поворота; камера же
    # работает ДО него. Инвертируем поворот, чтобы фиксировать ту же мировую точку.
    ang = math.radians(_view_deg[0])
    if ang != 0.0:
        c = math.cos(ang)
        s = math.sin(ang)
        k0 = PX * z0
        scx = COLS * LCELL * 0.5 * k0 + _CAM[0]
        scy = (H - 1) - (ROWS * LCELL * 0.5 * k0 + _CAM[1])
        dx = sx - scx
        dy = sy - scy
        sx = scx + c * dx + s * dy
        sy = scy - s * dx + c * dy
    wx = (sx - _CAM[0]) / (PX * z0)
    wy = (H - 1.5 - sy - _CAM[1]) / (PX * z0)
    k = PX * z1
    cam_write(sx - wx * k, H - 1.5 - sy - wy * k, z1)


def reset_camera():
    # Вписываем ВСЮ сетку и ЦЕНТРИРУЕМ её в окне: центр модели = центр экрана.
    # Размер окна фиксирован (physstate.W/H), поэтому 64 и 128 получают свой
    # масштаб автоматически: 128 больше не вылезает за экран, а его трещины
    # остаются толщиной 2-3 px и видны. Панель управления (0.72W) может
    # перекрывать правый край — её можно скрыть клавишей P.
    mw = COLS * CELL
    mh = ROWS * CELL
    avw = W * (0.64 if panel_visible else 0.94)   # панель скрыта — можно шире
    z = min(avw / mw, H * 0.93 / mh)
    z = min(CAM_ZMAX, max(CAM_ZMIN, z))
    ox = (W - mw * z) * 0.5
    oy = (H - mh * z) * 0.5
    cam_write(ox, oy, z)


def focus_region(x0, y0, x1, y1):
    """Навести камеру на область (клетки) и ЦЕНТРИРОВАТЬ её в окне: центр
    образца = центр экрана (для UCS, бразильского, балки и стойки).
    Координаты — в КЛЕТКАХ, поэтому экранный масштаб CELL px/клетку (а не PX:
    PX — это пиксели на МЕТР; раньше из-за путаницы образец уезжал вбок)."""
    wcx = max(1.0, (x1 - x0)) * CELL
    hcy = max(1.0, (y1 - y0)) * CELL
    avw = W * (0.36 if panel_visible else 0.86)   # не заезжать под панель
    z = min(CAM_ZMAX, max(CAM_ZMIN, min(avw / wcx, H * 0.86 / hcy)))
    wx = (x0 + x1) * 0.5
    wy = (y0 + y1) * 0.5
    ox = W * 0.5 - wx * CELL * z
    oy = H - 1.5 - H * 0.5 - wy * CELL * z
    cam_write(ox, oy, z)

# генераторы сцен живут в presets.py; реэкспорт — совместимость (from render import generate)
from presets import *  # noqa: E402,F401,F403
