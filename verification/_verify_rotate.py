import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # ROOT_FOR_VERIFY
"""Проверка «угла к горизонту» (эквивалентная система + повёрнутый вид).

Модель ОСЕВАЯ: блоки не поворачиваются, локальные оси = мировые. Мировой наклон
учитывается (1) проекциями мировой гравитации/нагрузок на локальные оси
(interactions.forces_reset) и (2) поворотом ВИДА (render.rot_view). Поэтому:

  A. локальная гравитация = g·(sinθ, cosθ); в МИРОВЫХ осях (R_app(θ)·local)
     она даёт строго вертикаль вниз (0, g) — проверяем по движению блока;
  B. модель устойчива при наклоне (не «взрывается» проекциями сил);
  C. вид/ввод согласованы (round-trip курсор↔клетки при любом угле);
  D. боковые нагрузки стоят на ОСЕВЫХ границах домена (не на повёрнутых).
"""

import math
import os
import sys

project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project)
os.environ["GRIDROCKDEM_ARCH"] = "cpu"

import numpy as np  # noqa: E402
import render  # noqa: E402,F401
import presets  # noqa: E402
from solver import (apply_gravity, apply_rock, build_model, step_physics,
                    reset_all, count_intact, update_mass_rho,
                    compute_energy, refresh_field)  # noqa: E402
from physstate import (P, bx, by, bvx, bvy, bcount, sideF, ovf, energyPot,
                       LCELL, COLS, ROWS)  # noqa: E402

STEPS = int(os.environ.get("STEPS", "10"))
FAIL = []


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("OK" if ok else "FAIL", name, detail))
    if not ok:
        FAIL.append(name)


def set_tilt(deg):
    presets.model_angle = deg
    P[None].tiltC = math.cos(math.radians(deg))
    P[None].tiltS = math.sin(math.radians(deg))


# ------------------------------------------------------------------ (A) гравитация
print("== (A) gravitaciya ostayotsya MIROVOY (vertikal, +y vniz) ==")
for deg in (35.0, -40.0):
    reset_all()
    apply_rock(10.0, 4.0, 0.62)
    build_model([(32.0, 5.0, 1.0, 0)], [], [], None)
    update_mass_rho()
    P[None].walls = 0
    P[None].depth = 0.0
    P[None].k0 = 0.0
    P[None].substeps = 64
    apply_gravity(1.0)
    set_tilt(deg)
    for _ in range(20):
        step_physics(1)
    vx = float(bvx[0]); vy = float(bvy[0])       # ЛОКАЛЬНАЯ скорость блока
    c = math.cos(math.radians(deg)); s = math.sin(math.radians(deg))
    wx = c * vx - s * vy                          # world = R_app(θ)·local
    wy = s * vx + c * vy
    check("theta=%+.0f: mirovaya skorost vertikalna (|wx|<<|wy|)" % deg,
          abs(wx) < 0.02 * abs(wy) + 1e-6, "wx=%+.4f wy=%+.4f" % (wx, wy))
    check("theta=%+.0f: blok padaet vniz (wy>0)" % deg, wy > 0.01,
          "wy=%+.3f m/s" % wy)

# ------------------------------------------- (A2) знак потенциальной энергии
print("== (A2) potencial'naya energiya pri padenii ==")
reset_all()
apply_rock(10.0, 4.0, 0.62)
build_model([(32.0, 5.0, 1.0, 0)], [], [], None)
update_mass_rho()
P[None].walls = 0
P[None].depth = 0.0
P[None].k0 = 0.0
P[None].substeps = 64
apply_gravity(1.0)
compute_energy()
pot0 = float(energyPot[None])
y0 = float(by[0])
for _ in range(20):
    step_physics(1)
compute_energy()
pot1 = float(energyPot[None])
y1 = float(by[0])
check("blok padaet (y rastet)", y1 > y0 + 1e-4, "y %.3f -> %.3f" % (y0, y1))
check("potencial UMEN'SHAETSYA pri padenii", pot1 < pot0,
      "P %.3e -> %.3e" % (pot0, pot1))

# ------------------------------------------------ (A3) улёт вверх удаляется
print("== (A3) ulet vverh udalyaetsya, a ne derzhitsya klampom ==")
reset_all()
apply_rock(10.0, 4.0, 0.62)
build_model([(32.0, 0.5, 1.0, 0)], [], [], None)
update_mass_rho()
P[None].walls = 0
P[None].depth = 1.0e-6   # нагруженная модель: верх — нагрузочная граница
P[None].k0 = 0.0
P[None].substeps = 64
apply_gravity(0.0)
refresh_field()
bvy[0] = -5.0
for _ in range(48):
    step_physics(1)
check("blok udalen sverhu", bcount[None] == 0,
      "bcount=%d" % bcount[None])

# ---------------------------------------------------------------- (B) устойчивость
print("== (B) ustoychivost modeli pri naklone (bez obvala) ==")
for deg in (0.0, 10.0, 20.0, 45.0):
    reset_all()
    apply_rock(10.0, 4.0, 0.62)
    apply_gravity(1.0)
    P[None].substeps = 32
    presets.generate(0, 0)
    P[None].walls = 1
    P[None].fixedRows = 2
    P[None].k0 = 0.5
    P[None].depth = 400.0
    set_tilt(deg)
    n0 = int(count_intact())
    seq = []
    for _ in range(STEPS):
        step_physics(0)
        seq.append(int(count_intact()))
    keep = min(seq) / max(1, n0)
    check("theta=%+.0f: model ne obvalilas (sohraneno >=90%%)" % deg,
          keep >= 0.90, "intact %d->%d (%.0f%%), seq=%s"
          % (n0, seq[-1], 100.0 * keep, seq))

# ---------------------------------------------------- (C) вид/ввод согласованы
print("== (C) soglasovannost vida i vvoda (round-trip) ==")


def world_to_pos(mx, my, deg):
    z = render._CAM[2]
    k = render.CELL * z
    sx = mx * k + render._CAM[0]
    sy = (render.H - 1) - (my * k + render._CAM[1])
    ang = math.radians(deg)
    if ang != 0.0:
        cc = math.cos(ang); ss = math.sin(ang)
        kk = render.PX * z
        scx = COLS * LCELL * 0.5 * kk + render._CAM[0]
        scy = (render.H - 1) - (ROWS * LCELL * 0.5 * kk + render._CAM[1])
        dx = sx - scx; dy = sy - scy
        sx = scx + cc * dx + ss * dy
        sy = scy - ss * dx + cc * dy
    return sx / render.W, sy / render.H


for deg in (0.0, 15.0, -25.0, 40.0):
    render.set_view(deg)
    worst = 0.0
    for (mx, my) in [(5, 5), (32, 32), (58, 10), (10, 55), (32, 3)]:
        rx, ry = render.cursor_to_cells(world_to_pos(mx, my, deg))
        worst = max(worst, abs(rx - mx), abs(ry - my))
    check("theta=%+.0f: kurser<->kletki tochno" % deg, worst < 0.1,
          "max oshibka=%.4f kletki" % worst)

# ----------------------------------------------- (D) нагрузки на ОСЕВЫХ границах
print("== (D) bokovye nagruzki na osevyh granicah ==")
reset_all()
apply_rock(10.0, 4.0, 0.62)
apply_gravity(1.0)
P[None].substeps = 32
presets.generate(0, 0)
P[None].walls = 1
P[None].fixedRows = 2
P[None].k0 = 0.5
P[None].depth = 400.0
set_tilt(20.0)
for _ in range(30):
    step_physics(0)
nb = int(bcount[None])
sv = sideF.to_numpy()[:nb]
ov = ovf.to_numpy()[:nb]
xs = bx.to_numpy()[:nb]
ys = by.to_numpy()[:nb]
bad = 0
for i in range(nb):
    if sv[i] == 0.0:
        continue
    # follower-load: +sideF только на ЛЕВОМ крайнем блоке строки,
    # -sideF — только на ПРАВОМ. Это свойство важнее близости к кромке:
    # после сжатия крайний блок может отойти на несколько клеток.
    row = int(ys[i] / LCELL)
    row_x = [xs[j] for j in range(nb) if int(ys[j] / LCELL) == row]
    if not row_x:
        bad += 1
        continue
    # один тик лага: sideF считается до подшагов, а позиции проверяются
    # после них; допускаем смещение до 2 клеток за тик.
    tol = 2.0 * LCELL
    if sv[i] > 0.0 and xs[i] > min(row_x) + tol:
        bad += 1
    if sv[i] < 0.0 and xs[i] < max(row_x) - tol:
        bad += 1
check("sideF na krajnih blokah svoih strok", bad == 0, "narusheniy=%d" % bad)
check("est bokovaya nagruzka (sideF!=0)", int((sv != 0.0).sum()) > 0,
      "blokov=%d" % int((sv != 0.0).sum()))
check("est verhnyaya nagruzka (ovf!=0)", int((ov != 0.0).sum()) > 0,
      "blokov=%d" % int((ov != 0.0).sum()))

print()
print("ITOG:", "VSE PROVERKI PROYDENY" if not FAIL else ("PROVALY: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)

