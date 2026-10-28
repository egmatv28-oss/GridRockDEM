r"""Mass-conserving cellular water for the 2D DEM mine model.

Each non-solid cell stores the amount of water ``fill`` in [0, 1]. One
``water_step`` moves water downwards first (gravity) and then sideways
(because a higher column pushes into a lower one). The transfer is computed
as explicit fluxes and then applied symmetrically, so the total amount of
water is conserved exactly (up to float rounding) and the scheme cannot blow
up.

Sources add water to a cell; the water then pours down, spreads along the
floor and pools inside the working. ``ux/uy`` are derived from the fluxes in
physical units (m/s) for the drag coupling to the DEM blocks.

Run the smoke test with:
    C:\py312\python.exe _verify_water.py
"""
import taichi as ti

from physstate import (COLS, ROWS, N, LCELL, TICK, bcount, bx, by, bvx, bvy,
                       bsz, clamp_cell, row_cell, gOcc, wfx, wfy, srcX, srcY,
                       srcCount)

EPS = 1.0e-5
SRC_RATE = 0.35          # water added by one source emit
DOWN = 0
RIGHT = 1
LEFT = 2

fill = ti.field(ti.f32, N)
fill_tmp = ti.field(ti.f32, N)
flux = ti.field(ti.f32, shape=(N, 3))
solid = ti.field(ti.i32, N)
ux = ti.field(ti.f32, N)
uy = ti.field(ti.f32, N)
pressure = ti.field(ti.f32, N)
rho = ti.field(ti.f32, N)


@ti.func
def flat(x: ti.i32, y: ti.i32) -> ti.i32:
    return y * COLS + x


@ti.func
def cell_x(x: ti.f32) -> ti.i32:
    c = int(x)
    if c < 0:
        c = 0
    elif c >= COLS:
        c = COLS - 1
    return c


@ti.func
def cell_y(y: ti.f32) -> ti.i32:
    c = int(y)
    if c < 0:
        c = 0
    elif c >= ROWS:
        c = ROWS - 1
    return c


@ti.kernel
def water_solid_from_occupancy():
    for c in range(N):
        solid[c] = 1 if gOcc[c] != 0 else 0


@ti.kernel
def water_displace_solids():
    """Water trapped in a cell that just became solid is pushed to a free
    neighbour (up first, so a falling block squeezes water upwards/sideways)."""
    for c in range(N):
        if solid[c] == 1 and fill[c] > EPS:
            x = c % COLS
            y = c // COLS
            amt = fill[c]
            fill[c] = 0.0
            t = -1
            if y > 0:
                n = flat(x, y - 1)
                if solid[n] == 0:
                    t = n
            if t < 0 and x > 0:
                n = flat(x - 1, y)
                if solid[n] == 0:
                    t = n
            if t < 0 and x + 1 < COLS:
                n = flat(x + 1, y)
                if solid[n] == 0:
                    t = n
            if t < 0 and y + 1 < ROWS:
                n = flat(x, y + 1)
                if solid[n] == 0:
                    t = n
            if t >= 0:
                ti.atomic_add(fill[t], amt)


@ti.kernel
def water_init(rho0: ti.f32):
    for c in range(N):
        fill[c] = 0.0
        fill_tmp[c] = 0.0
        solid[c] = 0
        ux[c] = 0.0
        uy[c] = 0.0
        pressure[c] = 0.0
        rho[c] = 0.0
        for d in ti.static(range(3)):
            flux[c, d] = 0.0


@ti.kernel
def water_flux():
    for c in range(N):
        x = c % COLS
        y = c // COLS
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        if solid[c] == 0 and fill[c] > EPS:
            rem = fill[c]
            # 1) gravity: most of the water pours straight down, a little is
            #    kept for the sides so a falling stream fans out instead of
            #    becoming a one-cell-wide column
            if y + 1 < ROWS:
                n = flat(x, y + 1)
                if solid[n] == 0:
                    cap = 1.0 - fill[n]
                    if cap > 0.0:
                        f0 = ti.min(rem * 0.75, cap)
                        rem -= f0
            # 2) spread sideways into lower columns, half of the rest each way
            half = rem * 0.5
            if x + 1 < COLS:
                n = flat(x + 1, y)
                if solid[n] == 0:
                    cap = 1.0 - fill[n]
                    diff = fill[c] - fill[n]
                    if cap > 0.0 and diff > 0.0:
                        f1 = ti.min(half, ti.min(cap, diff * 0.5))
                        rem -= f1
            if x > 0:
                n = flat(x - 1, y)
                if solid[n] == 0:
                    cap = 1.0 - fill[n]
                    diff = fill[c] - fill[n]
                    if cap > 0.0 and diff > 0.0:
                        f2 = ti.min(rem, ti.min(cap, diff * 0.5))
                        rem -= f2
        flux[c, DOWN] = f0
        flux[c, RIGHT] = f1
        flux[c, LEFT] = f2


@ti.kernel
def water_apply():
    for c in range(N):
        x = c % COLS
        y = c // COLS
        if solid[c] == 1:
            # keep it here: water_displace_solids() pushes it out on the next
            # step instead of silently deleting the volume
            fill_tmp[c] = fill[c]
        else:
            out = flux[c, DOWN] + flux[c, RIGHT] + flux[c, LEFT]
            inflow = 0.0
            if y > 0:
                n = flat(x, y - 1)
                if solid[n] == 0:
                    inflow += flux[n, DOWN]
            if x > 0:
                n = flat(x - 1, y)
                if solid[n] == 0:
                    inflow += flux[n, RIGHT]
            if x + 1 < COLS:
                n = flat(x + 1, y)
                if solid[n] == 0:
                    inflow += flux[n, LEFT]
            v = fill[c] - out + inflow
            # No upper clamp: when several neighbours pour into the same cell
            # at once it may temporarily hold more than one cell volume. That
            # excess drains down/sideways on the next step, so mass is exact.
            if v < 0.0:
                v = 0.0
            fill_tmp[c] = v


@ti.kernel
def water_copy_and_velocity():
    k = LCELL / TICK
    for c in range(N):
        fill[c] = fill_tmp[c]
        if solid[c] == 1 or fill[c] <= EPS:
            ux[c] = 0.0
            uy[c] = 0.0
            pressure[c] = 0.0
            rho[c] = 0.0
        else:
            # horizontal flow from the side flux, vertical from the down flux
            vx = (flux[c, RIGHT] - flux[c, LEFT]) * k
            vy = flux[c, DOWN] * k
            ux[c] = vx
            uy[c] = vy
            rho[c] = fill[c]
            pressure[c] = 0.0


def water_step(tau=0.8, body_x=0.0, body_y=1.0, update_solid=True):
    """One water step. ``tau``/``body_*`` are kept for API compatibility."""
    if update_solid:
        water_solid_from_occupancy()
        water_displace_solids()
    water_flux()
    water_apply()
    water_copy_and_velocity()


@ti.kernel
def water_apply_forces(drag: ti.f32, pressure_scale: ti.f32):
    for i in range(bcount[None]):
        wfx[i] = 0.0
        wfy[i] = 0.0
        cx = clamp_cell(bx[i])
        cy = row_cell(by[i])
        c = flat(cx, cy)
        if solid[c] == 0 and fill[c] > 0.05:
            w = fill[c]
            wfx[i] += drag * (ux[c] - bvx[i]) * bsz[i] * w
            wfy[i] += drag * (uy[c] - bvy[i]) * bsz[i] * w


def water_init_host(rho0=1.0):
    water_init(rho0)
    water_solid_from_occupancy()


@ti.func
def water_inject_func(x: ti.f32, y: ti.f32, v0: ti.f32):
    cx = cell_x(x)
    cy = cell_y(y)
    c = flat(cx, cy)
    if solid[c] == 0:
        fill[c] = 1.0


@ti.kernel
def water_inject(x: ti.f32, y: ti.f32, v0: ti.f32):
    water_inject_func(x, y, v0)


@ti.kernel
def water_emit_sources(v0: ti.f32):
    for s in range(srcCount[None]):
        cx = cell_x(srcX[s])
        cy = cell_y(srcY[s])
        c = flat(cx, cy)
        if solid[c] == 0:
            fill[c] = ti.min(1.0, fill[c] + SRC_RATE)


@ti.func
def water_fill_pct_func(cx: ti.f32, cy: ti.f32, rx: ti.f32, ry: ti.f32) -> ti.f32:
    n = 0.0
    cap = 0.0
    for c in range(N):
        x = c % COLS + 0.5
        y = c // COLS + 0.5
        dx = (x - cx) / rx
        dy = (y - cy) / ry
        if dx * dx + dy * dy < 1.0:
            cap += 1.0
            if solid[c] == 0 and fill[c] > 0.5:
                n += 1.0
    p = 0.0
    if cap > 0.5:
        p = n / cap * 100.0
        if p > 100.0:
            p = 100.0
    return p


@ti.kernel
def water_fill_pct(cx: ti.f32, cy: ti.f32, rx: ti.f32, ry: ti.f32) -> ti.f32:
    return water_fill_pct_func(cx, cy, rx, ry)


@ti.func
def water_fluid_count_func() -> ti.f32:
    n = 0.0
    for c in range(N):
        if solid[c] == 0 and fill[c] > EPS:
            n += 1.0
    return n


@ti.kernel
def water_fluid_count() -> ti.f32:
    return water_fluid_count_func()
