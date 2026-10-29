"""ГЕОМАССИВ/2D · DEM — расчёт взаимодействий между блоками (interactions)

Парные взаимодействия дискретно-элементной модели (DEM). Каждая функция ПРИНИМАЕТ
текущее состояние (поля из physstate) и ВОЗВРАЩАЕТ результат в силовые поля
(bfx/bfy/btor); оркестратор тиков (solver.py) затем читает эти силы и интегрирует.

Ядра расчёта — @ti.func (forces_*), чтобы решатель мог вызывать их ПО ОТДЕЛЬНОСТИ
(phys_reset / phys_bonds / phys_contact / phys_rock) или ОДНИМ слитым ядром
(phys_forces — все силы; phys_substep в solver.py — весь подшаг, включая
интегрирование) — меньше лаунчей на подшаг, меньше накладных расходов CPU и
выше загрузка GPU. Порядок силовых проходов: zero -> contact -> reset -> bonds:
  forces_zero    — обнуление аккумуляторов сил (bfx/bfy/btor);
  forces_contact — точные контакты квадратов (SAT) + трение + конфайнмент трещин
                   (заодно собирает флаги граней faceL/faceR/crack* — до чтения
                   в forces_reset);
  forces_reset   — базовые силы: гравитация, вода, перегрузка, боковой распор;
  forces_bonds   — силы упругих связей (нормаль/сдвиг/изгиб, пластика, энергия);
  forces_rock    — качание на полу и тормоз перекатывания (вызывается из
                   phys_integrate — момент нужен непосредственно перед интегрированием).
Каждый проход выполняется параллельно по ВСЕМ блокам сразу — физика равномерна,
ни один блок не обходится проходкой.
"""

import taichi as ti

from physstate import *

# ================================================================== DEM физика
@ti.func
def forces_zero():
    """Обнуление аккумуляторов сил подшага — строго ДО всех проходов сил.

    Все проходы сил (контакт/связи/распор/качание) только НАКАПЛИВАЮТ сюда
    (atomic_add +=), поэтому слитое ядро подшага считает их в любом порядке,
    не опасаясь затирания (раньше forces_reset делал ПРИСВАИВАНИЕ bfx=wfx и
    стёр бы всё, что набрал контакт)."""
    for i in range(bcount[None]):
        bfx[i] = 0.0
        bfy[i] = 0.0
        btor[i] = 0.0


@ti.func
def forces_reset():
    g = P[None].g
    tc = P[None].tiltC      # cos(θ): проекция на локальные оси
    ts = P[None].tiltS      # sin(θ)
    for i in range(bcount[None]):
        # Модель ОСЕВАЯ (блоки не поворачиваются). Мировые силы прикладываем как
        # их проекции на локальные оси: мировая вертикаль (0,·) -> (·ts, ·tc),
        # мировая горизонталь (·,0) -> (·tc, -·ts). Складывая, получаем:
        bfx[i] += wfx[i] + efx[i] + sideF[i] * tc + ovf[i] * ts + ovfX[i] * tc
        bfy[i] += wfy[i] + efy[i] + ovf[i] * tc - sideF[i] * ts - ovfX[i] * ts
        if bfixed[i] == 0:
            # вес — мировая вертикаль вниз -> проекции на локальные оси
            bfx[i] += bmass[i] * g * ts
            bfy[i] += bmass[i] * g * tc
    # Силовой "распор рыхляка" (толкание блоков через конфаймент: th по sigma_v)
    # УБРАН - он выдавливал одиночные обломки к стенам/соседям и они зависали
    # на одной и той же высоте. Боковое давление массива - ПРЯМАЯ сила sideF
    # (compute_side_load: σ_h = K0·σ_v от границ домена внутрь, растёт с
    # глубиной), престресса в модели нет.


@ti.func
def forces_bonds(dt: ti.f32):
    E = P[None].E
    zeta = P[None].cn
    for bd in range(bondCount[None]):
        a = bondA[bd]
        b = bondB[bd]
        # Связь ГРАНИЦЫ модели (с закреплённым блоком — дно/опора): это не порода,
        # крип/разрывы тут не копим, силы и моменты считаем как обычно.
        bfixBond = bfixed[a] + bfixed[b]
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        d = ti.sqrt(dx * dx + dy * dy)
        # боковое давление на свободных гранях (σ_h = K0·σ_v) считается в phys_reset
        # для каждого блока отдельно — см. там.
        if bondIntact[bd] == BOND_DEAD:
            continue
        if d < 1e-9:
            continue
        ux = dx / d
        uy = dy / d
        rvx = bvx[b] - bvx[a]
        rvy = bvy[b] - bvy[a]
        vn = rvx * ux + rvy * uy
        # --- геометрия контакта (СИ) ---
        w = bsz[a]
        if bsz[b] < w:
            w = bsz[b]          # ширина шва = меньший блок, м
        ma = bmass[a]
        mb = bmass[b]
        mred = ma * mb / (ma + mb + 1e-9)
        k_ax = E * w / brest[bd]    # жёсткость связи, Н/м
        # предел устойчивости явной схемы (см. STAB_C): k ≤ C·m_red/dt²
        kstab = P[None].stabC * mred / (dt * dt)
        if k_ax > kstab:
            k_ax = kstab
        Ftmax = P[None].rp * P[None].ftScale * w    # Н
        Fcmax = Ftmax * P[None].rcFactor
        Fsmax = Ftmax * P[None].fsFactor
        # --- strain-softening: до обрыва прочность падает с накопленным
        # криповым повреждением bondCreep. sfac=1 в начале, sfac=softResidual
        # при bondCreep=plastLimit·brest. Это даёт post-peak ветвь вместо
        # бинарного "целая/шов" и локализует разрушение у контура.
        if P[None].soften == 1:
            lim = P[None].plastLimit * brest[bd]
            df = 0.0
            if lim > 1e-9:
                df = bondCreep[bd] / lim
                if df > 1.0:
                    df = 1.0
            sres = P[None].softResidual
            if sres < 0.0:
                sres = 0.0
            elif sres > 1.0:
                sres = 1.0
            sfac = sres + (1.0 - sres) * (1.0 - df)
            # Разупрочняем растяжение и сдвиг; СЖАТИЕ не трогаем: иначе
            # softening сам ослабляет несущий скелет и на большой глубине
            # даёт глобальную компакционную лавину вместо локального удара.
            Ftmax *= sfac
            Fsmax *= sfac
        # --- предел демпфера связи (ЯВНАЯ схема) ---
        # Dashpot 2ζ√(k·m_red) распределяется по всем связям блока: у блока с
        # n связями суммарное гашение даёт Δv ≈ -2·n·cd·dt/m·v (коэффициент 2 -
        # шахматная мода соседей в противофазе). При ζ>~0.35 и n=4 это выходит
        # за |Δv|>|v| за подшаг - демпфер превращается в антидемпфер и массив
        # начинает самопроизвольно звенеть ("скачущие напряжения"). Cap как у
        # crot: cd ≤ 0.25·min(m_a/n_a, m_b/n_b)/dt - перелёт исключён.
        na_ = nbond[a] if nbond[a] > 1 else 1
        nb_ = nbond[b] if nbond[b] > 1 else 1
        cdCap = 0.25 * ti.min(ma / na_, mb / nb_) / dt
        cd = 2.0 * zeta * ti.sqrt(k_ax * mred)
        if cd > cdCap:
            cd = cdCap

        if bondIntact[bd] == BOND_JOINT:
            # Шов держит сжатие ТОЛЬКО у блоков, ещё входящих в связный
            # массив. У полностью оторванного обломка (nbond==0) шов не
            # должен его ловить/удерживать: иначе отлетевший при разрыве
            # блок «зависает» на шве и крутится. Контакт/трение при
            # фактическом касании остаются.
            if nbond[a] == 0 or nbond[b] == 0:
                bondR[bd] = 0.0
                bondE[bd] = 0.0
                continue
            strain = d - brest[bd]
            Fn = -k_ax * strain         # > 0 = сжатие шва
            if Fn < 0.0:
                Fn = 0.0
            cd_j = cd                   # см. cap cd выше
            Fn -= cd_j * vn             # демпфер смыкания
            if Fn < 0.0:
                Fn = 0.0                # шов не тянет: суммарное усилие ≥ 0
            ti.atomic_add(bfx[b], Fn * ux)
            ti.atomic_add(bfy[b], Fn * uy)
            ti.atomic_add(bfx[a], -Fn * ux)
            ti.atomic_add(bfy[a], -Fn * uy)
            # маркеры напряжений шва: сжатие — в bstressC/легенду, bondR как у
            # сжатой связи (render_bonds рисует швы тем же кодом)
            if strain < 0.0:
                bondR[bd] = (k_ax * strain) / (Fcmax * 0.25)
                ti.atomic_max(bstressC[a], (-k_ax * strain) / Fcmax)
                ti.atomic_max(bstressC[b], (-k_ax * strain) / Fcmax)
            else:
                bondR[bd] = 0.0
            bondE[bd] = 0.0
            continue
        strain = d - brest[bd]
        tension = k_ax * strain          # упругое осевое усилие (до пластики)
        yieldT = Ftmax * P[None].plastYield
        # предел по СЖАТИЮ - Друкер-Прагер по РЕАЛЬНОМУ сжатию: σ3 = min главных
        # сжатий из bSxx/bSyy (пересчитываются каждый подшаг в bond_stress_field).
        # Раньше σ_conf брался теоретически (k0·sigVI) и не видел фактического
        # горизонтального обжатия: при K0=0 запаса не было, при K0>0 боковое
        # давление не усиливало связь на смятие. Теперь обжатие от sideF реально
        # повышает yieldC, как в настоящем Мора-Кулоне/Друкер-Прагере.
        cx = 0.25 * (bSxx[a] + bSxx[b])
        cy = 0.25 * (bSyy[a] + bSyy[b])
        sx = cx / LCELL
        sy = cy / LCELL
        sigma_conf = sx if sx < sy else sy   # sigma3
        # Геостатический ПОЛ конфайнмента: у свободной грани (бортик выработки)
        # прокси bSxx/bSyy обваливается - у блока нет соседа в пустоту, и его
        # сумма сжатий занижена. Без пола DP мгновенно ослаблял вертикальные
        # связи борта -> крип сжатия рвал их -> ГОРИЗОНТАЛЬНЫЕ трещины вдоль
        # выработки. Пол - часть реального обжатия массива вокруг точки.
        sv = sigVI[row_cell(0.5 * (by[a] + by[b])) * COLS
                   + clamp_cell(0.5 * (bx[a] + bx[b]))] / LCELL
        geo = P[None].k0 * sv
        if geo > sv:
            geo = sv
        if sigma_conf < 0.5 * geo:
            sigma_conf = 0.5 * geo
        # Растяжение НЕ упрочняется обжатием: σ3 входит только в сжатие/сдвиг.
        # Иначе горизонтальное обжатие «держит» вертикальную растянутую связь,
        # и в своде остаются висящие блоки без растяжения-разрыва.
        yieldC = Fcmax * P[None].plastYield + P[None].mcBond * sigma_conf * w
        # --- пуассоново раскалывание: сжатие σ_c растягивает связи ПЕРПЕНДИКУЛЯРНО
        # ему (в реальном массиве это даёт неоднородность; в идеальной решётке
        # поперечное растяжение не накапливается само). Трещины идут ВДОЛЬ сжатия —
        # осевое раскалывание/слоение, как в бетоне и вокруг выработки, а не
        # "блины" поперёк нагрузки. σ_c — локальное главное сжатие из
        # bond_stress_field (направление само поворачивается у контура выработки);
        # собственное обжатие связи гасит разрыв (net tension = tension + split).
        # Скалывание — по ДЕВИАТОРУ, а не по абсолютному сжатию: под равномерным
        # обжатием (k0≈1 в поле) поперечного растяжения НЕТ, оно возникает лишь
        # там, где одно главное сжатие ослаблено (свободная грань выработки) —
        # т.е. ∝ (σ_max − σ_min). Прежний split ∝ σ_v рос с глубиной по ВСЕЙ модели
        # и с ~300 м рвал всё разом (не горный удар, а тотальное размельчение). Теперь
        # в поле split≈0, трещины только у контура; обжатие связи по-прежнему гасит
        # разрыв (net tension = tension + split).
        acy = ti.abs(cy)
        acx = ti.abs(cx)
        split = 0.0
        if acy >= acx:
            split = P[None].nuSplit * (acy - acx) * ux * ux
        else:
            split = P[None].nuSplit * (acx - acy) * uy * uy
        # Девиаторный гейт крипа-УПЛОТНЕНИЯ (ветка ovC): fac=0 при относительном
        # девиаторе |σ1−σ3|/σ_mean ниже creepDev, 1 при вдвое большем. Под
        # изотропным обжатием (k1≈k0) связь НЕ течёт; у свободной грани девиатор
        # большой — уплотнение идёт как раньше. (Растяжение/сдвиг не гейтятся.)
        dmean = 0.5 * (acy + acx)
        creepFac = 1.0
        if dmean > 1e-9:
            creepFac = ti.abs(acy - acx) / (P[None].creepDev * dmean)
            if creepFac > 1.0:
                creepFac = 1.0
        tens_eff = tension + (1.0 - P[None].poissonF) * split   # только критерий растяжения; силы не трогаем
        anyOver = 0
        # --- пластическое течение (крип): фаза А — СРАЗУ ТОЛЬКО ОЦЕНКА.
        # Фильтруются САМЫ осевые усилия: усилие связи (bondOver) и пуассоново
        # раскалывающее слагаемое (bondSplitAvg), окно ~0.15 с; по СРЕДНИМ
        # считаются нормальные перегрузки и аварийное растяжение. Фильтр сил,
        # а не «max перегрузок»: при дрожании решётки (ω·dt≈0.5) полупериоды
        # сжатия и растяжения по очереди дают положительный max — рецификация
        # и ложный крип/обрывы от частоты; по средним силам симметричное
        # дрожание вокруг безопасной статики даёт ровно ноль, а удержанная
        # перегрузка проходит и рвёт в прежнем темпе. Связи границы не текут.
        ovT = 0.0
        ovC = 0.0
        favg = tension
        if bfixBond == 0:
            bondOver[bd] += (tension - bondOver[bd]) * (dt / (0.15 + dt))
            bondSplitAvg[bd] += (split - bondSplitAvg[bd]) * (dt / (0.15 + dt))
            favg = bondOver[bd]
            # Крип гоняется по ЧИСТОМУ осевому усилию favg, БЕЗ split: пуассоново
            # раскалывание (split ∝ σ_v) добавляло фиктивное растяжение во все
            # горизонтальные связи, и с глубины ~300 м крип полз по всей модели,
            # давая вертикальные трещины везде. Split остаётся только в хрупком
            # критерии ниже (tens_eff > 2.5·Ftmax ≈ Rc) — трещины от реальной
            # прочности, а не от медленного расползания.
            ovT = (favg - yieldT) / Ftmax   # <0 = запас до отрыва
            ovC = ((-favg) - yieldC) / Fcmax      # <0 = запас до смятия
        # --- пересчёт упругих величин ПОСЛЕ пластики (Баг 1/3): сила и энергия
        # должны считаться по АКТУАЛЬНОМУ остаточной деформации (d-brest), иначе
        # каждый подшаг прикладывается старая/большая сила с резким последующим
        # падением -> осцилляция и ложные полосы пластики
        strain = d - brest[bd]
        tension = k_ax * strain
        tension -= P[None].poissonF * split
        # --- cohesive mode I (Gf): линейное разупрочнение по раскрытию ---
        # Fpeak — та же эффективная прочность, что у хрупкого критерия
        # (2.5·Ftmax, плюс старое tensConf при A/B). δ0 = Fpeak/k_ax;
        # δf = δ0 + 2·Gf·w/Fpeak. bondOpen — НЕОБРАТИМОЕ макс. раскрытие.
        Fpeak = Ftmax * 2.5
        if P[None].tensConf != 0.0:
            Fpeak += P[None].tensConf * P[None].hardenConf * sigma_conf * w
        delta0 = Fpeak / (k_ax + 1e-9)
        deltaF = delta0 + 2.0 * P[None].Gf * w / (Fpeak + 1e-9)
        if strain > bondOpen[bd]:
            bondOpen[bd] = strain
        cohD = 0.0
        if deltaF > delta0:
            cohD = (bondOpen[bd] - delta0) / (deltaF - delta0)
            if cohD < 0.0:
                cohD = 0.0
            elif cohD > 1.0:
                cohD = 1.0
        # Честный Мора-Кулон на связи: сдвиговая прочность шва зависит от нормального
        # напряжения по шву (tension>0 — растяжение, tension<0 — обжатие).
        # τ_s = Fs0 + mcBond·σ_n: обжатие добавляет прочности на сдвиг, растяжение — отнимает.
        # floor: даже сильно растянутый шов сохраняет остаточное сцепление (не делит на 0).
        FsC = Fsmax - P[None].mcBond * tension
        if FsC < Ftmax * 0.2:
            FsC = Ftmax * 0.2
        # сдвиг поперёк НАЧАЛЬНОГО направления связи (bondNX/NY) — это правильная
        # опорная система склеенного шва: сдвиг меряется относительно исходного
        # контакта, а не текущей оси (иначе off становится тождественно нулю и
        # связь теряет сдвиговую жёсткость). Ложный сдвиг при пластической вытяжке
        # НЕ копится, потому что strain уже пересчитан по актуальному brest.
        px = -bondNY[bd]
        py = bondNX[bd]
        ux0 = bondNX[bd]
        uy0 = bondNY[bd]
        # упругий сдвиг поперёк НАЧАЛЬНОГО направления связи: off_eff = (off - bondSlip)
        # — пластическое проскальзывание УМЕНЬШАЕТ упругий сдвиг (как длина покоя для
        # нормали), поэтому блок фактически СЪЕЗЖАЕТ по шву, а не висит на упругой
        # деформации. Направление slip совпадает со знаком текущего off.
        # ТОЧКА ШВА — середина связи (общая грань блоков), а не ЦТ блока: точки
        # приложения сдвинуты от ЦТ каждого блока на hArm = d/2, поэтому вращение
        # блоков даёт вклад в ПОПЕРЕЧНЫЙ сдвиг шва: off -= hArm·(θ_a+θ_b),
        # vs -= hArm·(ω_a+ω_b). Без этого поля смещений u(y) и v(y) решётки
        # РАЗВЯЗАНЫ: вертикальная нагрузка несётся только сдвигом горизонтальных
        # связей, осевые силы верх/низ балки не развиваются — изгибающего
        # напряжения нет, балка ведёт себя как «мембрана».
        hArm = 0.5 * w * (ti.abs(ux0) + ti.abs(uy0))
        ca = ti.cos(brot[a])
        sa = ti.sin(brot[a])
        cb = ti.cos(brot[b])
        sb = ti.sin(brot[b])
        rax = hArm * (ux0 * ca - uy0 * sa)
        ray = hArm * (ux0 * sa + uy0 * ca)
        rbx = -hArm * (ux0 * cb - uy0 * sb)
        rby = -hArm * (ux0 * sb + uy0 * cb)
        off = ((bx[b] + rbx) - (bx[a] + rax)) * px + ((by[b] + rby) - (by[a] + ray)) * py
        offEff = off - bondSlip[bd]
        vs = ((bvx[b] - bw[b] * rby) - (bvx[a] - bw[a] * ray)) * px \
           + ((bvy[b] + bw[b] * rbx) - (bvy[a] + bw[a] * rax)) * py
        k_sh = k_ax / (2.0 * (1.0 + 0.25))    # G = E / 2(1+ν)
        cds = 2.0 * zeta * ti.sqrt(k_sh * mred)
        if cds > cdCap:
            cds = cdCap
        Fs = -k_sh * offEff - cds * vs
        shearF = k_sh * offEff
        rel = brot[b] - brot[a]
        # фильтр разности поворотов для ИЗГИБНОГО критерия (окно 0.15 с, как у
        # сил): мгновенный rel за один подшаг выдаёт пики дребезга (ω·dt≈0.5),
        # и связь на свободной поверхности рвалась «на ровном месте» -> лавина.
        # По среднему реальный изгиб (растущий плавно) проходит, а шум гасится.
        bondBendAvg[bd] += (rel - bondBendAvg[bd]) * (dt / (0.15 + dt))
        rw = bw[b] - bw[a]
        krot = k_ax * w * w / 12.0             # Н·м/рад
        Ia = bmass[a] * bsz[a] * bsz[a] / 6.0
        Ib = bmass[b] * bsz[b] * bsz[b] / 6.0
        Ired = Ia * Ib / (Ia + Ib + 1e-9)
        # демпфер поворота связи (ζ≈0.8 для пары). ОГРАНИЧЕНИЕ ПО СТАБИЛЬНОСТИ:
        # в решётке блок участвует в nbond связях, и в шахматной моде вращения
        # (соседние блоки крутятся в противофазе) каждый демпфер даёт на блок
        # 2·c·ω, поэтому суммарное торможение доходит до 8·c. Без капа оно
        # превышает предел явной схемы 2·I/dt и решётка входит в период-2 дребезг
        # на клампе скорости (±10 рад/с). Кап: 2·crot·nbond·dt < 2·I  => crot <
        # I/(nbond·dt), берём с запасом 0.5 (тот же стиль, что STAB_C для k).
        crot = 1.6 * ti.sqrt(krot * Ired)
        cCap = 0.5 * ti.min(Ia / na_, Ib / nb_) / dt
        if crot > cCap:
            crot = cCap
        # Cohesive damage снижает ТОЛЬКО растяжение и сдвиг; сжатие не трогаем.
        tensionF = tension
        if tensionF > 0.0:
            tensionF *= (1.0 - cohD)
        FsF = Fs * (1.0 - cohD)
        Fn = -tensionF - cd * vn
        Fx = Fn * ux + FsF * px
        Fy = Fn * uy + FsF * py
        # Мгновенная упругая энергия связи (нормаль + сдвиг + изгиб), Дж:
        eN = 0.5 * k_ax * strain * strain
        if strain > 0.0:
            eN *= (1.0 - cohD)
        bondE[bd] = eN + 0.5 * k_sh * offEff * offEff + 0.5 * krot * rel * rel
        ti.atomic_add(bfx[b], Fx)
        ti.atomic_add(bfy[b], Fy)
        ti.atomic_add(bfx[a], -Fx)
        ti.atomic_add(bfy[a], -Fy)
        # момент от приложения силы в ТОЧКЕ ШВА (r = ±hArm·u от ЦТ каждого блока):
        # τ = r×F, для пары τ_a = τ_b = -hArm·(u×F). Нормальная составляющая (вдоль u)
        # момента не даёт — только сдвиговая. Эквивалентно градиенту потенциала по
        # углам (см. off -= hArm·(θ_a+θ_b) выше). Без него изгибающий момент шва
        # несётся ложным рычагом силы, приложенной в ЦТ, и решётка не гнётся.
        tlv_a = -(rax * Fy - ray * Fx)
        tlv_b = (rbx * Fy - rby * Fx)
        ti.atomic_add(btor[a], tlv_a)
        ti.atomic_add(btor[b], tlv_b)
        # Момент связи: tor - момент, приложенный к блоку A (b - противоположный).
        # ЗНАК: восстанавливающий. Потенциал изгиба U = ½·krot·rel² (rel = brot[b]-brot[a])
        # даёт τ_a = -∂U/∂brot[a] = +krot·rel, τ_b = -krot·rel; демпфер тянет блоки
        # к равенству угловых скоростей: +c·(bw[b]-bw[a]) на A. Раньше стояло
        # «tor = -krot·rel», что при применении btor[a]+=tor / btor[b]-=tor давало
        # АНТИ-восстанавливающий момент: rel разгонялся, связи рвались по
        # перегибу (|rel| > 3·relBend), а решётка раскручивалась в шахматном
        # порядке (±bw до клампа). Баг был скрыт, пока контакт не давал блокам
        # момент (brot связанных блоков оставался 0 и krot вообще не работал).
        tor = krot * rel + crot * rw
        ti.atomic_add(btor[a], tor)
        ti.atomic_add(btor[b], -tor)
        # --- сдвиг: мгновенная перегрузка участвует только как направление/вклад
        # при УДЕРЖАННОЙ нормальной перегрузке (ovF>0) — иначе дрожание
        # сдвига копил бы |slip| от частоты колебаний.
        yieldS = FsC * P[None].plastYield
        ovS = 0.0
        if bfixBond == 0:
            ovS = (ti.abs(shearF) - yieldS) / FsC
        # --- фаза Б: текучесть по СРЕДНЕЙ (отфильтрованной) перегрузке ---
        ovmax = ovT
        if ovC > ovmax:
            ovmax = ovC
        ovF = ovmax
        if bfixBond == 0 and ovF > 0.0:
            anyOver = 1
            if ovT > 0.0:
                # медленное пластическое течение (реальная вытяжка, саморазгрузка):
                grow = P[None].plastFlow * ovF * ovF * dt * brest[bd] * 0.1
                if grow > strain * 0.9:
                    grow = strain * 0.9    # не даём длине покоя перескочить деформацию
                brest[bd] += grow           # растяжение: блоки отдаляются, связь растёт
                bondPlast[bd] += grow
            elif ovC > 0.0:
                grow = P[None].plastFlow * ovF * ovF * dt * brest[bd] * 0.1 * creepFac
                if grow > -strain * 0.9:
                    grow = -strain * 0.9
                brest[bd] -= grow           # сжатие: блоки сближаются (уплотнение)
                bondPlast[bd] -= grow       # компакция < 0: это НЕ вытяжка и НЕ разрыв
            if ovS > 0.0:
                # пластическое проскальзывание по шву: bondSlip растёт, снижая offEff —
                # блок реально съезжает в направлении сдвига, разгружая связь (и себя)
                sgn = 1.0 if shearF > 0.0 else -1.0
                slipG = P[None].plastFlow * ovF * ovF * dt * brest[bd] * 0.1
                if slipG > ti.abs(off) * 0.9:
                    slipG = ti.abs(off) * 0.9
                bondSlip[bd] += sgn * slipG
            # БЫСТРОЕ накопление ПОВРЕЖДЕНИЯ (отдельно от вытяжки): удержанный
            # перегруз должен рвать за ~2 секунды, а не за 400. Иначе связь
            # бесконечно "течёт" (сама себя разгружает ростом brest), не рвётся,
            # и фронт текучести ползёт вверх по кромке вместо обрушения в свод.
            # Накопление в МЕТРАХ (× brest): порог plastLimit·brest — масштабно
            # и по размеру блока инвариантен (время до разрыва не зависит от LCELL).
            # Повреждение копят только РАСТЯЖЕНИЕ и СДВИГ (ovD = max(ovT, ovS)).
            # Сжатие уплотняет (brest), но не копит разрыв: иначе вертикальные
            # связи борта выработки (концентрация 3*sigma_v по Киршу) рвались
            # крипом сжатия и давали ГОРИЗОНТАЛЬНЫЕ трещины вдоль выработки.
            # Разрушение при сжатии идёт split-механизмом (net tension) —
            # трещины ВДОЛЬ нагрузки, как в реальной породе.
            ovD = ovT if ovT > ovS else ovS
            if ovD > 0.0:
                # strain-softening копит повреждение только там, где есть
                # девиатор (у контура). В гидростатическом дальнем поле
                # creepFac≈0, иначе softening ослаблял бы весь массив.
                gate = creepFac if P[None].soften == 1 else 1.0
                bondCreep[bd] += ovD * dt * 0.5 * brest[bd] * gate
        # --- предел суммарной пластической вытяжки И проскальзывания: исчерпав
        # пластичность (и запас деформации), связь больше не течёт — только рвётся.
        # Без этого кромка сцепления текла бы бесконечно, давая всё тот же ползущий
        # фронт. 0.15 = 15% от длины покоя: блок может "отойти"/"съехать" на 15%
        # (раньше было 6% — запас слишком мал и связи рвались слишком рано,
        # не дав блоку реально возобновиться).
        # bondPlast > 0 — только ВЫТЯЖКА рвёт (компакция < 0 — уплотнение, не разрыв);
        # bondSlip — по модулю: сдвиг любого знака одинаково разрушителен.
        if bfixBond == 0 and (bondPlast[bd] > 0.15 * brest[bd] or ti.abs(bondSlip[bd]) > 0.15 * brest[bd]):
            anyOver = 1
            bondCreep[bd] = P[None].plastLimit * brest[bd]   # исчерпана пластичность -> рвём
        # залечивание: если ни один критерий не превышен — крип возвращается,
        # кратковременный выброс не рвёт связь
        if anyOver == 0:
            # strain-softening необратим: накопленное повреждение не лечится
            if P[None].soften == 0:
                bondCreep[bd] *= ti.exp(-P[None].creepRecover * dt)
                if bondCreep[bd] < 1e-6:
                    bondCreep[bd] = 0.0
        # маркеры текущей связи (фиолетовый — идёт крип)
        if anyOver == 1:
            bondFlow[bd] = 1
            bondDmg[bd] = P[None].dmgMax
        # маркеры напряжений
        if tension > 0.0:
            r = tension / Ftmax
            bondR[bd] = r
            ti.atomic_max(bstress[a], r)
            ti.atomic_max(bstress[b], r)
        else:
            bondR[bd] = tension / (Fcmax * 0.25)
            c = -tension / Fcmax
            ti.atomic_max(bstressC[a], c)
            ti.atomic_max(bstressC[b], c)
        # --- обрыв ---
        # (а) аварийный ХРУПКИЙ разрыв при значительном перегрузе: это останавливает
        # фронт локальным разрывом и передаёт нагрузку в свод. Окно 0.9–1.25·Ftmax —
        # чистое пластическое течение; выше — хрупкий обрыв.
        # (б) иначе — только при УДЕРЖАННОМ криповом повреждении (плавное оседание).
        overE = 0
        crushE = 0
        # Мгновенный разрыв - по СЫРОМУ усилию (пик удара/волны), а не по
        # фильтрованному tens_eff_f (bondOver, окно 0.15 с): кратковременный
        # удар раньше отфильтровывался и не мог оборвать связь - сброшенные
        # сверху блоки отскакивали, а зависший кусок висел вечно. Фильтр
        # оставлен для плазт. течения/ползучести (там нужны НАКОПЛЕННЫЕ
        # перегрузки, см. ovT/ovC).
        # Растяжение — mode I по СВОЕЙ прочности: σn > 2.5·ft (запас на
        # динамический пик), БЕЗ добавки hardenConf·σ3·w. tensConf=1 включает
        # старое поведение для A/B (по умолчанию 0). Сжатие — отдельный
        # критерий ниже и по-прежнему упрочняется обжатием (Друкер-Прагер).
        # При Gf>0 хрупкий порог — только аварийный (2·Fpeak): обычное
        # растяжение проходит через softening до d=1. Иначе критерий
        # tens_eff>Fpeak срабатывал бы сразу на пике и «съедал» softening.
        tensThr = Fpeak * 2.0 if P[None].Gf > 0.0 else Fpeak
        if tens_eff > tensThr or strain > 0.24 * LCELL or cohD >= 1.0:
            overE = 1
            dbgBreakN[0] += 1
        elif (-tension) > Fcmax * 2.5 + P[None].hardenConf * sigma_conf * w:
            # Смятие упрочняется обжатием (Друкер-Прагер): в гидростатическом
            # поле σ3 велик, и дальний массив не должен рассыпаться от
            # динамической волны. У свободной грани σ3 мал — там смятие идёт
            # как прежде.
            overE = 1
            crushE = 1
            dbgBreakN[1] += 1
        elif ti.abs(shearF) > FsC * 3.0:
            overE = 1
            dbgBreakN[2] += 1
        elif ti.abs(bondBendAvg[bd]) > P[None].relBend * (brest[bd] / w) * 10.0:
            # изгиб: деформация крайнего волокна ε = |rel|·(w/2)/brest > ε_t = σ_t/E
            # (см. apply_rock). Критерий по ФИЛЬТРУ rel (bondBendAvg, окно 0.15 с):
            # мгновенный rel давал за подшаг пики дребезга и рвал поверхностные
            # связи на старте; множитель 10 остаётся запасом на остаточный шум DEM.
            overE = 1
            dbgBreakN[3] += 1
        shell = 0
        if brkShell[a] == 1 or brkShell[b] == 1:
            shell = 1
        cohesive = 1 if cohD >= 1.0 else 0
        if overE == 1 and bfixBond == 0 and shell == 0:
            if try_fracture(bd, a, b, crushE):
                fracture_bond(bd, a, b, crushE, strain, off, rel, cohesive)
        elif bfixBond == 0 and shell == 0 and anyOver == 1 and bondCreep[bd] >= P[None].plastLimit * brest[bd]:
            crush = 0
            if tension < -1e-6:
                crush = 1
            if try_fracture(bd, a, b, crush):
                fracture_bond(bd, a, b, crush, strain, off, rel, 0)


# ================================================================== геометрия контакта квадратов
# Точный контакт двух ПОВЁРНУТЫХ квадратов (OBB): SAT по 4 осям + клиппинг граней.
# Даёт нормаль, глубину и РЕАЛЬНЫЕ точки контакта (1 или 2) с глубиной каждой.
# Никаких "раздутых проекций": углы зацепляются только там, где квадраты
# действительно перекрываются — фантомных пружин под повёрнутым блоком нет.

_mat3f = ti.types.matrix(3, 3, ti.f32)


@ti.func
def sat_squares(cix: ti.f32, ciy: ti.f32, ri: ti.f32, cti: ti.f32, sti: ti.f32,
                cjx: ti.f32, cjy: ti.f32, rj: ti.f32, ctj: ti.f32, stj: ti.f32) -> _mat3f:
    """ Separating Axis Theorem для двух квадратов.
    Возврат M: [0,0..1] — нормаль (единичная, от i к j), [0,2] — глубина проникновения,
    [1,0] — 1 если перекрытие есть, [1,1] — 1 если опорная грань у блока i (иначе у j)."""
    dx = cjx - cix
    dy = cjy - ciy
    vix = -sti
    viy = cti
    vjx = -stj
    vjy = ctj
    best = 1e9
    nx = 1.0
    ny = 0.0
    refI = 1.0
    hit = 1.0
    a = 0
    while a < 4 and hit > 0.5:
        ax = 0.0
        ay = 0.0
        if a == 0:
            ax = cti
            ay = sti
        elif a == 1:
            ax = vix
            ay = viy
        elif a == 2:
            ax = ctj
            ay = stj
        else:
            ax = vjx
            ay = vjy
        # полупроекция квадрата на ось: r·(|u·a| + |v·a|)
        ra = ri * (ti.abs(cti * ax + sti * ay) + ti.abs(vix * ax + viy * ay))
        rb = rj * (ti.abs(ctj * ax + stj * ay) + ti.abs(vjx * ax + vjy * ay))
        dc = dx * ax + dy * ay
        ov = ra + rb - ti.abs(dc)
        if ov <= 0.0:
            hit = 0.0    # разделяющая ось найдена — контакта нет
        elif ov < best:
            best = ov
            sgn = 1.0
            if dc < 0.0:
                sgn = -1.0
            nx = ax * sgn    # нормаль — от i к j
            ny = ay * sgn
            refI = 1.0
            if a >= 2:
                refI = 0.0
        a += 1
    return ti.Matrix([[nx, ny, best], [hit, refI, 0.0], [0.0, 0.0, 0.0]])


@ti.func
def clip_contact_points(cix: ti.f32, ciy: ti.f32, ri: ti.f32, cti: ti.f32, sti: ti.f32,
                        cjx: ti.f32, cjy: ti.f32, rj: ti.f32, ctj: ti.f32, stj: ti.f32,
                        nx: ti.f32, ny: ti.f32, refI: ti.f32) -> _mat3f:
    """Клиппинг падающей грани об опорную (Sutherland-Hodgman, 2 плоскости).
    Возврат M: строка0 = (p0x, p0y, δ0), строка1 = (p1x, p1y, δ1),
    строка2 = (число точек, ширина контакта W, 0). δ — глубина каждой точки."""
    # опорная грань: у блока refI её внешняя нормаль = n (от i к j), иначе −n
    nrx = nx
    nry = ny
    crx = cix
    cry = ciy
    rr = ri
    qix = cjx
    qiy = cjy
    rq = rj
    if refI < 0.5:
        nrx = -nx
        nry = -ny
        crx = cjx
        cry = cjy
        rr = rj
        qix = cix
        qiy = ciy
        rq = ri
    mx = -nry
    my = nrx
    # опорная грань: центр fc = cr + rr·nref, концы fc ± rr·m
    fcx = crx + rr * nrx
    fcy = cry + rr * nry
    # падающая грань второго квадрата (внешняя нормаль ≈ −nref): ic ± rq·m
    icx = qix - rq * nrx
    icy = qiy - rq * nry
    b0x = icx - rq * mx
    b0y = icy - rq * my
    b1x = icx + rq * mx
    b1y = icy + rq * my
    # клиппинг отрезка b0→b1 по полосе m·p ∈ [lo, hi]
    lo = (fcx * mx + fcy * my) - rr
    hi = lo + 2.0 * rr
    s0 = b0x * mx + b0y * my
    s1 = b1x * mx + b1y * my
    t0 = 0.0
    t1 = 1.0
    ds = s1 - s0
    ok = 1.0
    if ti.abs(ds) < 1e-12:
        if s0 < lo or s0 > hi:
            ok = 0.0
    else:
        ta = (lo - s0) / ds
        tb = (hi - s0) / ds
        if ta > tb:
            tmp = ta
            ta = tb
            tb = tmp
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            ok = 0.0
    count = 0.0
    wid = 0.0
    p0x = 0.0
    p0y = 0.0
    d0 = 0.0
    p1x = 0.0
    p1y = 0.0
    d1 = 0.0
    if ok > 0.5:
        dRef = fcx * nrx + fcy * nry    # плоскость опорной грани
        qax = b0x + (b1x - b0x) * t0
        qay = b0y + (b1y - b0y) * t0
        qbx = b0x + (b1x - b0x) * t1
        qby = b0y + (b1y - b0y) * t1
        da = dRef - (qax * nrx + qay * nry)
        db = dRef - (qbx * nrx + qby * nry)
        if da > 1e-9 and db > 1e-9:
            count = 2.0
            p0x = qax
            p0y = qay
            d0 = da
            p1x = qbx
            p1y = qby
            d1 = db
            wid = ti.sqrt((qbx - qax) * (qbx - qax) + (qby - qay) * (qby - qay))
        elif da > 1e-9:
            count = 1.0
            p0x = qax
            p0y = qay
            d0 = da
        elif db > 1e-9:
            count = 1.0
            p0x = qbx
            p0y = qby
            d0 = db
    return ti.Matrix([[p0x, p0y, d0], [p1x, p1y, d1], [count, wid, 0.0]])


@ti.func
def contact_fn(i: ti.i32, j: ti.i32, px: ti.f32, py: ti.f32, dp: ti.f32,
               nx: ti.f32, ny: ti.f32, kpt: ti.f32, cd: ti.f32) -> ti.f32:
    """Нормальная сила одной точки контакта: пружина (kpt·δ) + демпфер по скорости
    СБЛИЖЕНИЯ ТОЧЕК с учётом вращения обоих блоков (ω×r). Отскок гасится: Fn ≥ 0."""
    rix = px - bx[i]
    riy = py - by[i]
    rjx = px - bx[j]
    rjy = py - by[j]
    # скорость точки на теле: v + ω×r, где ω×r = ω·(−ry, rx)
    vrx = (bvx[j] - bw[j] * rjy) - (bvx[i] - bw[i] * riy)
    vry = (bvy[j] + bw[j] * rjx) - (bvy[i] + bw[i] * rix)
    vsep = vrx * nx + vry * ny      # < 0 — сближение
    Fn = kpt * dp - cd * vsep
    if Fn < 0.0:
        Fn = 0.0
    return Fn


@ti.func
def apply_contact_point(i: ti.i32, j: ti.i32, fkey: ti.i32, pt: ti.i32,
                        px: ti.f32, py: ti.f32,
                        nx: ti.f32, ny: ti.f32,
                        Fn: ti.f32, FnF: ti.f32, kpt: ti.f32, dt: ti.f32):
    """Приложить силу одной точки контакта: нормаль Fn (распор + момент вокруг ЦТ),
    опора/копилка интервала поддержки, кулоновское трение FnF в той же точке.

    Момент от нормали — р × F: у блока, вставшего КРАЕМ на опору, точка контакта
    уходит от вертикали под ЦТ, и вес реально опрокидывает его (никакой пружины
    "вверх из центра"). FnF — нормаль для трения (может быть выше Fn за счёт
    конфайнмента закрытой трещины)."""
    rix = px - bx[i]
    riy = py - by[i]
    rjx = px - bx[j]
    rjy = py - by[j]
    if Fn > 0.0:
        fxx = Fn * nx
        fyy = Fn * ny
        ti.atomic_add(bfx[i], -fxx)
        ti.atomic_add(bfy[i], -fyy)
        ti.atomic_add(bfx[j], fxx)
        ti.atomic_add(bfy[j], fyy)
        ti.atomic_max(cnck[i], Fn)
        ti.atomic_max(cnck[j], Fn)
        # τ = r × F прикладывается ВСЕГДА, независимо от связей блока:
        # сила в точке контакта -> сила в ЦТ + момент вокруг ЦТ. Пропуск момента
        # для связанных блоков (nbond>0) замораживал поворот ЛЮБОГО составного
        # тела: единственный внешний источник вращения - эксцентриситет контакта
        # (гравитация идёт в ЦТ, моменты связей krot отвечают лишь на ОТНОСИТЕЛЬНЫЙ
        # поворот блоков и при нулевых brot дают ноль). Балка-составное тело не
        # перевешивала и не опрокидывалась, а в её связях не развивались осевые
        # напряжения изгиба (brot оставался ровно 0 -> без натяжения верх/низ).
        ti.atomic_add(btor[i], Fn * (riy * nx - rix * ny))
        ti.atomic_add(btor[j], Fn * (rjx * ny - rjy * nx))
        # интервал опоры: контакт толкает блок ВВЕРХ (сила против +y)
        if ny > 0.01:
            ti.atomic_min(supLo[i], px)
            ti.atomic_max(supHi[i], px)
        elif ny < -0.01:
            ti.atomic_min(supLo[j], px)
            ti.atomic_max(supHi[j], px)
    tx = -ny
    ty = nx
    # относительная касательная скорость в точке (с учётом вращения)
    vrx = (bvx[j] - bw[j] * rjy) - (bvx[i] - bw[i] * riy)
    vry = (bvy[j] + bw[j] * rjx) - (bvy[i] + bw[i] * rix)
    vt = vrx * tx + vry * ty
    friction_vec(i, j, fkey, vt, FnF, dt, tx, ty, px, py, kpt, pt)


@ti.func
def forces_contact(dt: ti.f32):
    k_c = P[None].E * 0.1   # жёсткость контакта полной ширины (Н/м), k = E·A/L
    # динамический радиус поиска в клетках: описанная окружность квадрата r·√2
    # + запас на PROX (пре-контакт для phys_rock)
    RR = int(ti.ceil(P[None].maxSize * 1.5))
    # минимум 2 клетки: при maxSize=0 радиус был 0 и блоки «видели» друг друга
    # только после проникновения на полклетки — контакт срабатывал с огромным
    # перехлёстом и выбрасывал блок (диагностированные «выстрелы»).
    if RR < 2:
        RR = 2
    # флаги граней собираются ЗДЕСЬ (размер-осведомлённо, каждый подшаг) — не в
    # клеточном поле gOcc. По умолчанию грань свободна (1), трещины/опоры нет (0);
    # нюанс-init в начале каждого витка параллельного цикла обязателен: ниже флаги
    # только ОБНУЛЯЮТСЯ для занятых сторон.
    for i in range(bcount[None]):
        faceL[i] = 1
        faceR[i] = 1
        crackL[i] = 0
        crackR[i] = 0
        supp[i] = 0
        comIn[i] = 0
        inContact[i] = 0
        cnck[i] = 0.0
        supLo[i] = 1e9
        supHi[i] = -1e9
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        ri = bsz[i] * 0.5          # ПОЛОВИНА СТОРОНЫ квадрата (без раздутия проекций)
        cir = ri * 1.41421356      # описанный радиус: предел досягаемости углов
        cti = ti.cos(brot[i])
        sti = ti.sin(brot[i])
        rx = block_prj(i)          # только для size-aware флагов граней ниже
        # радиус поиска — динамический (±RR клеток): без статических 13×13 итераций
        # на блок (MAXNBR), которые почти все отсеивались фильтром — минус ~85%
        # холостых проходов хэша на подшаг
        for oy in range(-RR, RR + 1):
            yy = cyi + oy
            if yy >= 0 and yy < ROWS:
                for ox in range(-RR, RR + 1):
                    xx = cxi + ox
                    if xx >= 0 and xx < COLS:
                        j = hhead[yy * COLS + xx]
                        while j != -1:
                            if j > i:
                                dx = bx[j] - bx[i]
                                dy = by[j] - by[i]
                                rj = bsz[j] * 0.5
                                rxj = block_prj(j)
                                rs = rx + rxj
                                lat = rs
                                GAP = 0.25 * lat
                                # ---- геометрические флаги (size-aware), для ВСЕХ пар ----
                                # тот же "этаж" (без вертикального смещения больше лат*0.95):
                                if ti.abs(dy) < lat * 0.95:
                                    # j правее i в пределах lat+GAP
                                    if dx > 0.0 and dx < lat + GAP:
                                        faceR[i] = 0
                                        faceL[j] = 0
                                        if is_has_bit(i, j) == 0:
                                            # внутренняя трещина: сосед есть, записи
                                            # связи нет вообще (не целая и не шов) —
                                            # только такие пары получают распор рыхляка
                                            crackR[i] = 1
                                            crackL[j] = 1
                                    elif dx < 0.0 and -dx < lat + GAP:
                                        faceL[i] = 0
                                        faceR[j] = 0
                                        if is_has_bit(i, j) == 0:
                                            crackL[i] = 1
                                            crackR[j] = 1
                                if not (bfixed[i] == 1 and bfixed[j] == 1):
                                    key = bkey(i, j)
                                    if is_bonded_bit(i, j) == 0:
                                        axx = ti.abs(dx)
                                        ayy = ti.abs(dy)
                                        rsum = cir + rj * 1.41421356
                                        # близость (не только перекрытие): блок, зависший
                                        # над кучей, уже попадает под сопротивление перекатыванию
                                        PROX = 0.7 * LCELL
                                        if axx < rsum + PROX and ayy < rsum + PROX:
                                            inContact[i] = 1
                                            inContact[j] = 1
                                        if axx < rsum and ayy < rsum:
                                            ctj = ti.cos(brot[j])
                                            stj = ti.sin(brot[j])
                                            sat = sat_squares(bx[i], by[i], ri, cti, sti,
                                                              bx[j], by[j], rj, ctj, stj)
                                            if sat[1, 0] > 0.5:
                                                nx = sat[0, 0]
                                                ny = sat[0, 1]
                                                pts = clip_contact_points(
                                                    bx[i], by[i], ri, cti, sti,
                                                    bx[j], by[j], rj, ctj, stj,
                                                    nx, ny, sat[1, 1])
                                                npts = int(pts[2, 0])
                                                if npts > 0:
                                                    inContact[i] = 1
                                                    inContact[j] = 1
                                                    # --- ПЛОЩАДЬ КОНТАКТА (ширина в 2D) ---
                                                    # жёсткость пропорциональна ширине пятна:
                                                    # грань-к-грани — жёстко, угол-к-грани —
                                                    # мягко (0.2). Сила по-прежнему приложена
                                                    # В ТОЧКАХ пятна — момент от ЦТ честный.
                                                    W = pts[2, 1]
                                                    sAvg = 0.5 * (bsz[i] + bsz[j])
                                                    wFrac = W / sAvg
                                                    if npts == 1 or wFrac < 0.2:
                                                        wFrac = 0.2
                                                    if wFrac > 1.0:
                                                        wFrac = 1.0
                                                    fp = npts * 1.0
                                                    kpt = k_c * wFrac / fp
                                                    mred = bmass[i] * bmass[j] / (bmass[i] + bmass[j] + 1e-9)
                                                    # предел устойчивости (см. STAB_C)
                                                    kstab = P[None].stabC * mred / (dt * dt)
                                                    if kpt > kstab:
                                                        kpt = kstab
                                                    # ζ контакта: перекритическое для пары
                                                    # свободных блоков (рухляк не отскакивает),
                                                    # обычное — если хоть один на связи
                                                    zc = P[None].cnc
                                                    if nbond[i] == 0 and nbond[j] == 0:
                                                        zc = P[None].cncR
                                                    cd = 2.0 * zc * ti.sqrt(kpt * mred)
                                                    # страховка от глубокого проникновения
                                                    # (быстрый удар): δ ≤ 30% меньшего блока
                                                    dMax = 0.3 * ti.min(bsz[i], bsz[j])
                                                    d0 = ti.min(pts[0, 2], dMax)
                                                    d1 = 0.0
                                                    p0x = pts[0, 0]
                                                    p0y = pts[0, 1]
                                                    p1x = 0.0
                                                    p1y = 0.0
                                                    Fn0 = contact_fn(i, j, p0x, p0y, d0, nx, ny, kpt, cd)
                                                    Fn1 = 0.0
                                                    if npts == 2:
                                                        p1x = pts[1, 0]
                                                        p1y = pts[1, 1]
                                                        d1 = ti.min(pts[1, 2], dMax)
                                                        Fn1 = contact_fn(i, j, p1x, p1y, d1, nx, ny, kpt, cd)
                                                    # --- конфайнмент ПО ЗАКРЫТИЮ (аналогия сжатого стержня) ---
                                                    # Вертикальная трещина = вертикальная грань:
                                                    # как только берега сомкнулись, сквозь закрытую
                                                    # трещину передаётся горизонтальное сжатие
                                                    # σ_h = K0·σ_v — как у целого массива. Это
                                                    # поднимает только нормаль ТРЕНИЯ (сдвиг
                                                    # берегов), распор остаётся в phys_reset.
                                                    FnF0 = Fn0
                                                    FnF1 = Fn1
                                                    # Трение по "замкнутой трещине" (нормаль трения =
                                                    # конфайменмент Fconf, а не реальная сила контакта) -
                                                    # ТОЛЬКО для СЛАБЫХ ПЛОСКОСТЕЙ: пара соединена ШВОМ
                                                    # (BOND_JOINT: несёт сжатие, но не сдвиг - трение шва =
                                                    # mu*sigma_n). Критерий: есть запись связи, но она НЕ
                                                    # полная (is_has && !is_bonded). Раньше условием был
                                                    # просто "контакт вертикальный" - тогда ЛЮБОЕ касание
                                                    # боковых граней получало трение mu*sigma_v*K0 даже при
                                                    # нулевой силе контакта: обломки цеплялись за боковые
                                                    # грани и висели в воздухе (cnck=0N, вес гасил трение).
                                                    # Дополнительно требуем, чтобы хотя бы
                                                    # один блок ещё входил в связный массив
                                                    # (nbond>0). Иначе два УЖЕ ОТОРВАННЫХ
                                                    # обломка, у которых остался только шов,
                                                    # получали конфайнмент-трение от sigVI
                                                    # («клей») и висели цепочками в воздухе.
                                                    if ti.abs(nx) > ti.abs(ny) \
                                                            and is_has_bit(i, j) == 1 \
                                                            and is_bonded_bit(i, j) == 0 \
                                                            and nbond[i] > 0 and nbond[j] > 0:
                                                        sv_i = sigVI[row_cell(by[i]) * COLS + clamp_cell(bx[i])]
                                                        sv_j = sigVI[row_cell(by[j]) * COLS + clamp_cell(bx[j])]
                                                        sv = sv_i if sv_i > sv_j else sv_j
                                                        coef = P[None].k0
                                                        # closure — ЛИНЕЙНО от раскрытия, а не ступень:
                                                        # (δ+tol)/tol давало closure=1 уже при δ=1e-9,
                                                        # т.е. едва касающийся шов получал полный σ_h
                                                        # и трение-«клей». Теперь полная передача только
                                                        # при δ ≥ CONTACT_CLOSE_TOL.
                                                        closure = sat[0, 2] / CONTACT_CLOSE_TOL
                                                        if closure < 0.0:
                                                            closure = 0.0
                                                        elif closure > 1.0:
                                                            closure = 1.0
                                                        # нормаль сжатия закрытой трещины: σ_h·h = coef·sv·n
                                                        # (n — высота трещины в клетках = меньший блок); для n=1
                                                        # как раньше, для крупных — масштаб по грани, а не «на клетку»
                                                        hCrack = ti.min(bsz[i], bsz[j]) / LCELL
                                                        Fconf = sv * coef * closure * hCrack
                                                        FnTot = Fn0 + Fn1
                                                        # Fconf только УСИЛИВАЕТ уже существующую
                                                        # нормальную силу трения; при FnTot=0 он не
                                                        # создаёт «клей» из ничего (иначе оторванные
                                                        # кластеры висят на закрытой трещине).
                                                        if FnTot > 1e-6 and Fconf > FnTot:
                                                            fsc = Fconf / FnTot
                                                            FnF0 = Fn0 * fsc
                                                            FnF1 = Fn1 * fsc
                                                    apply_contact_point(i, j, key, 0,
                                                                        p0x, p0y, nx, ny,
                                                                        Fn0, FnF0, kpt, dt)
                                                    if npts == 2:
                                                        apply_contact_point(i, j, key, 1,
                                                                            p1x, p1y, nx, ny,
                                                                            Fn1, FnF1, kpt, dt)
                            j = hnext[j]
    # --- итог по опоре: supp/comIn из РЕАЛЬНЫХ контактов (а не из "сосед где-то снизу") ---
    # Пол — тоже опора (его реакция в phys_integrate приложена к центру, точек нет):
    # у лежащего на полу блока ЦТ всегда над опорой.
    for i in range(bcount[None]):
        if supLo[i] <= supHi[i]:
            supp[i] = 1
        rxp = block_prj(i)
        if by[i] >= ROWS * LCELL - rxp - 0.03 * LCELL:
            supp[i] = 1
            comIn[i] = 1
        elif supp[i] == 1:
            m = 0.05 * bsz[i]
            if bx[i] >= supLo[i] - m and bx[i] <= supHi[i] + m:
                comIn[i] = 1


@ti.func
def forces_rock():
    """Перекатывание свободного блока + тормоз вращения.

    Контакты блок-блок (phys_contact) теперь ТОЧНЫЕ: нормальная сила приложена
    в реальных точках пятна контакта, и её эксцентриситет относительно ЦТ сам
    даёт опрокидывающий/возвращающий момент — блок на краю опоры ЗАВАЛИВАЕТСЯ,
    блок на грани оседает плоско. Явный момент перекатывания нужен только для
    ПОЛА: пол — не блок, его реакция приложена к центру (кламп в phys_integrate)
    и момента не даёт. Для блока, лежащего НА ПОЛУ (supp==0):

      φ = brot, свёрнутый к ближайшей грани (−45°..45°);
      τ_rock = −ROCK_LIFT·m·g·r·(cosφ − sin|φ|)·tanh(10φ):
        честный момент тяжести вокруг нижнего ребра: максимум при малом φ
        (куб возвращается на грань — ЦТ опускается), НОЛЬ на ребре (45°) —
        неустойчивое равновесие, куб падает на следующую грань. tanh(10φ)
        гладко выключает момент у плоской посадки (φ=0 — опоры ребром нет).

    τ_roll = −μ_r·Fn·r·tanh(bw·2): сопротивление перекатыванию (тормоз вечного
    вращения). μ_r ОСЛАБЛЯЕТСЯ (×0.15), когда ЦТ вышел за интервал опоры
    (comIn==0): такой блок должен завалиться, и тормоз не может удерживать
    его кривым — иначе растут "пружинящие" столбы из блоков.

    На блоки внутри массива (nbond>0) не действует — их держат связи."""
    g = P[None].g
    half = PI * 0.5          # 90° — период симметрии квадрата
    for i in range(bcount[None]):
        if bfixed[i] == 1:
            continue
        if nbond[i] > 0:
            continue
        r = bsz[i] * 0.5
        m = bmass[i]
        rx = block_prj(i)
        on_floor = by[i] >= ROWS * LCELL - rx - 0.30 * LCELL
        near_ceil = by[i] <= rx + 0.30 * LCELL
        if supp[i] == 0 and inContact[i] == 0 and on_floor == 0 and near_ceil == 0:
            continue
        w = m * g * r         # масштаб: вес × рычаг полуграня (Н·м)
        if on_floor == 1:
            # качание по полу: сворачиваем угол к ближайшей грани, q ∈ (−0.5, 0.5]
            q = brot[i] / half
            q = q - ti.floor(q)
            if q > 0.5:
                q -= 1.0
            phi = q * half        # φ ∈ (−π/4, π/4]
            arm = ti.cos(phi) - ti.sin(ti.abs(phi))   # плечо ЦТ от нижнего ребра
            ti.atomic_add(btor[i], -P[None].rockLift * w * arm * ti.tanh(phi * 10.0))
        # тормоз перекатывания: сила — реальная нормаль контакта (или минимум —
        # собственный вес, если блок опирается/у пола); неустойчивый блок (ЦТ вне
        # опоры) тормозится слабо — он обязан завалиться
        fn_roll = cnck[i]
        if supp[i] == 1 or on_floor == 1:
            if fn_roll < w:
                fn_roll = w
        mu_r = P[None].rollMu
        if comIn[i] == 0:
            mu_r = P[None].rollMu * 0.15
        ti.atomic_add(btor[i], -mu_r * fn_roll * r * ti.tanh(bw[i] * 2.0))


# ================================================================== ядра-обёртки
# Отдельные ядра сохранены для совместимости (тесты/отладка); рабочий путь
# решателя — phys_substep в solver.py (весь подшаг одним лаунчем).

@ti.kernel
def phys_reset():
    forces_zero()
    forces_reset()


@ti.kernel
def phys_bonds(dt: ti.f32):
    forces_bonds(dt)


@ti.kernel
def phys_contact(dt: ti.f32):
    forces_contact(dt)


@ti.kernel
def phys_rock():
    forces_rock()


@ti.kernel
def phys_forces(dt: ti.f32):
    """Все силы подшага ОДНИМ ядром (без интегрирования).

    Порядок: zero -> contact -> reset -> bonds. Контакт идёт ПЕРВЫМ из
    накопителей сил, потому что он пишет флаги граней (faceL/faceR/crack*),
    которые читает forces_reset: при старом порядке (reset первым) флаги
    запаздывали на подшаг. Проходы параллельные, между ними implicit-барьер."""
    forces_zero()
    forces_contact(dt)
    forces_reset()
    forces_bonds(dt)

