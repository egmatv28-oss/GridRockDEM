"""ГЕОМАССИВ/2D · DEM — решатель: тик времени и оркестрация физики (solver)

Главный модуль физического движка. Его задача — чтобы физика работала РАВНОМЕРНО
и для всех блоков одновременно (плоскопараллельно), с соблюдением закона
сохранения энергии.

Каждый ТИК (step_physics) строит единый распорядок, одинаковый для всех блоков:
  1) проходка состояния — топология, напряжённое поле, контактный хэш, связи;
   2) передача на расчёт — силы взаимодействий считаются в interactions.py
      ОДНИМ слитым ядром phys_forces (reset + bonds + contact) за подшаг;
  3) приём и обработка  — phys_integrate читает полученные силы (bfx/bfy/btor),
     интегрирует скорости/позиции, демпфирует, держит блоки в домене;
  4) сводка интегралов  — compute_energy() (полная энергия КАЖДОГО блока и всей
     системы, закон сохранения энергии) и center_of_mass() (центр тяжести)
     после каждого тика.

Решатель не знает про сцены, инструменты и GUI: модель строится СНАРУЖИ
(render.py) и загружается через build_model() либо примитивы
add_block_func / add_bond_func. Состояние — в physstate.py.
Точки входа — GridRockDem64.py / GridRockDem128.py.
"""

import math
import time

import numpy as np

import taichi as ti

from physstate import *
from interactions import *
from water import (water_step, water_apply_forces, water_init_host,
                   water_emit_sources, water_fill_pct, water_inject,
                   water_inject_func, water_fill_pct_func,
                   water_fluid_count_func, water_fluid_count)

def apply_gravity(scale):
    P[None].g = G_PHYS * scale


def apply_rock(E_GPa, rp_MPa, mu, rho=None, rc_MPa=None):
    E = E_GPa * 1e9                      # модуль Юнга, Н/м2
    P[None].E = E
    if rho is not None:
        P[None].rho = rho
        update_mass_rho()
    elif P[None].rho == 0.0:
        P[None].rho = RHO_R
    # ζ — относительное демпфирование (доля критического) пары блоков:
    # оно же используется и для связей, и для контактов (см. phys_bonds/phys_contact).
    P[None].cn = 0.5     # ζ связи: гасит колебания при жёсткой связи (в критич. единицах).
                         # 0.28 оставляло заметные "скачки" напряжений по массиву
                         # (упругие волны в связной решётке звонят); 0.5 - около
                         # половины критического демпфирования, звон гасится за
                         # несколько периодов, квазистатика не меняется.
    P[None].cnc = 0.6    # ζ КОНТАКТА блоков: заметно гасит подпрыгивание обломков.
                         # Больше брать НЕЛЬЗЯ: эмпирически ζ≈0.8 и выше дают взрывной
                         # каскад при росте нагрузки, а полное критическое 2·√(k/m)
                         # рвёт связи уже при усадке.
    P[None].cncR = 1.1   # ζ контакта ДВУХ СВОБОДНЫХ блоков (оба nbond==0): перекритическое
                         # демпфирование — падающий обломок не отскакивает от рухляка.
                         # На связанный массив не влияет (там ζ=cnc).
    P[None].rp = rp_MPa
    P[None].mu = mu
    P[None].ftScale = FT_SIGMA
    if rc_MPa is None:
        rc_MPa = 10.0 * rp_MPa
    P[None].rcFactor = float(rc_MPa) / max(float(rp_MPa), 1e-9)
    P[None].fsFactor = 2.0
    # Честный Мора-Кулон на связи: сдвиговая прочность шва зависит от нормального
    # напряжения по шву (tension>0 — растяжение, tension<0 — обжатие).
    # τ_s = Fs0 + mcBond·σ_n, где σ_n = tension. Обжатие (σ_n<0) добавляет
    # прочности на сдвиг, растяжение (σ_n>0) — отнимает. mcBond — тангенс угла
    # внутреннего трения по шву (реалистичный ~tan 35°≈0.7).
    P[None].mcBond = 0.7
    # пуассоново раскалывание: доля сжатия σ_c, добавляемая как растяжение в
    # связи ПЕРПЕНДИКУЛЯРНЫЕ σ_c (см. forces_bonds) — даёт трещины ВДОЛЬ сжатия
    P[None].poissonF = 1.0   # enable Poisson eigenforce coupling (calibrated)
    P[None].nuSplit = 0.21
    # Крип-уплотнение включается только при ДЕВИАТОРЕ |σ1−σ3|/σ_mean > creepDev.
    # Под изотропным обжатием (k1≈k0, дальнее поле) связь не течёт; у грани
    # выработки девиатор большой — уплотнение идёт. creepDev=0.3 ≈ 15° от
    # гидростатики: заметно ниже концентрации у контура (σ_θ=2σ_r) и сильно
    # выше остаточного дрожания решётки в дальнем поле.
    P[None].creepDev = 0.3
    # Упрочнение хрупкого растяжения/split от ОБЖАТИЯ (confinement hardening).
    # Порог обрыва по растяжению = 2.5·Ftmax + hardenConf·σ3·w (σ3 — реальное мин.
    # главное сжатие, Pa; σ3·w — сила обжатия на грань связи). Под изотропным
    # обжимом k0≈1 дальнее поле имеет σ3 ~ σ_v, порода НЕ рассыпается от обжима;
    # у свободной стенки выработки σ3 мал (пол 0.5·geo) — связи рвутся как прежде,
    # так что разрушение локализуется у контура. UCS (k0=0, σ3≈0) не затрагивается.
    P[None].hardenConf = 1.0
    P[None].k0 = K0_LATERAL
    # критерий ИЗГИБА - по деформации крайнего волокна, а не фиксированный угол:
    # ε_волокна = |rel|·(w/2)/brest > ε_t = σ_t/E. Отсюда базовый поворот шва
    # relBend = 2·ε_t (при brest≈w). Хрупкий порог в forces_bonds:
    # |rel| > relBend·(brest/w)·3.0 (3.0 - запас на шум DEM).
    # Раньше стоял магический 0.15 рад (~22% деформации на волокне) - связи
    # гнулись как резина, и трещиноватые колонны после раскалывания текли
    # "макарониной" вместо излома. При дефолтах (E=10 ГПа, rp=4 МПа):
    # relBend ≈ 8e-4 рад - колонна гнётся на 1-2% и ЛОМАЕТСЯ (kink-полосы).
    P[None].relBend = 2.0 * (rp_MPa * 1e6) / E
    P[None].dmgMax = 6.0
    P[None].dmgDecay = 0.85   # быстрее "забывает" перегрузку: дальние связи не рвутся, обвал не расползается
    P[None].brkCap = 24
    # Пластическое течение (крип) связей — как в реальной породе: выше предела
    # текучести связь медленно "течёт" (остаточная длина растёт), продолжая
    # передавать усилие и оставаясь герметичной; обрыв наступает после накопления
    # предельной пластической деформации (plastLimit).
    P[None].plastYield = 0.5   # крип (пластическая стадия) начинается с ~50% прочности —
                            # раньше (0.65), чтобы перегруз расползался постепенно, а не рвался
    P[None].plastLimit = 0.32  # обрыв при ~32% накопленной криповой деформации (было 0.24) —
                                # больше запаса до разрыва, связь дольше течёт и перераспределяет
    P[None].plastFlow = 1.6     # скорость течения (ещё быстрее), самоперераспределение успевает
    P[None].creepRecover = 0.6  # медленнее залечивание: накопленный крип дольше держится,
                                # без резких скачков, и трещина растёт равномерно


def apply_water(head_m):
    P[None].head = head_m
@ti.func
def forces_integrate(dt: ti.f32, dfree: ti.f32, dsupp: ti.f32, drotFree: ti.f32, drotBond: ti.f32):
    """Интегрирование подшага: скорости из сил, демпфирование, вращение, позиции.

    Бывшее тело ядра phys_integrate, вынесено @ti.func, чтобы phys_substep
    выполнял его в ОДНОМ лаунче с силами. Перед ним идёт forces_rock: его
    моментам нужны supp/comIn/cnck, которые только что собрал контакт подшага."""
    walls = P[None].walls
    tc = P[None].tiltC   # cos(θ): нормальная (к полу) доля веса на наклоне
    for i in range(bcount[None]):
        rx = block_prj(i)   # проекция повёрнутого квадрата: угол достигает пола/стен раньше круга
        if bbedrock[i] != 0:
            bvx[i] = 0.0
            bvy[i] = 0.0
            bw[i] = 0.0
            continue
        if bfixed[i] == 1:
            # закреплено по y и вращению, но НЕ по x: основание - это продолжение
            # массива вниз (окно в породу), оно должно деформироваться по
            # горизонтали вместе со всеми. Жёсткое основание без x-подвижности
            # давало слой Сен-Венана во всю высоту: боковое сжатие стягивалось
            # вниз через сдвиг связей, и σ_h у дна падало на порядок (симптом 2).
            bvy[i] = 0.0
            bw[i] = 0.0
            bvx[i] += bfx[i] / bmass[i] * dt
            bvx[i] *= dsupp
            bx[i] += bvx[i] * dt
            if walls == 1:
                if bx[i] < rx:
                    bx[i] = rx
                    if bvx[i] < 0.0:
                        bvx[i] = 0.0
                elif bx[i] > COLS * LCELL - rx:
                    bx[i] = COLS * LCELL - rx
                    if bvx[i] > 0.0:
                        bvx[i] = 0.0
            continue
        # --- трение свободных блоков о жёсткие границы домена (пол/стены) ---
        # У обломка, лежащего прямо на дне, нет блока-соседа снизу, поэтому пара
        # (friction_apply) к нему не применяется; без этого пол — идеально гладкий,
        # и свободные обломки беспрепятственно раскатываются по поверхности.
        # Нормаль = собственный вес + вес столба над ним (sigVI — Н на клетку).
        kfric = P[None].E
        # явная схема: кап трения чтобы не "прилипало" (см. STAB_C)
        kstabF = P[None].stabC * bmass[i] / (dt * dt)
        if kfric > kstabF:
            kfric = kstabF
        # Нормальная сила трения о границы. Для СВЯЗАННОГО массива - геостатика
        # (sigVI). Для рыхлого обломка (nbond==0) - только его ФАКТИЧЕСКАЯ
        # нагрузка: вес + контактное давление сверху (cnck). Поле sigVI
        # пропускает нагрузку через пустоты (carry в sigvi_field не сбрасывается
        # на дырках), поэтому одинокий обломок в воздухе получал sigVI = вес всей
        # вышележащей породы (loadCur): mu*nF ~ 4.6e5 Н против веса 6.1e3 Н -
        # трение намертво "приклеивало" обломки к стенкам/полу, и они висели в
        # воздухе (в т.ч. целыми кластерами на таком якоре).
        nF = bmass[i] * P[None].g * tc + sigVI[row_cell(by[i]) * COLS + clamp_cell(bx[i])]
        if nbond[i] == 0:
            nF = bmass[i] * P[None].g * tc + cnck[i]
        if nF < bmass[i] * P[None].g * tc:
            nF = bmass[i] * P[None].g * tc
        # Нормаль трения о СТЕНКУ — только реальное контактное сжатие cnck.
        # sideF на КРАЙНЕМ блоке строки направлена ВНУТРЬ домена, т.е. ОТ
        # этой стены, и не является реакцией стены: |sideF| давал свободному
        # блоку у стены фиктивное вертикальное трение и «приклеивал» его.
        nFw = cnck[i]
        if by[i] <= rx + 0.02 * LCELL or by[i] >= (ROWS * LCELL - rx - 0.02 * LCELL):
            friction_against(i, bvx[i], nF, dt, -(i + 1), kfric, 0)
        # трение о БОКОВЫЕ стены — только для рухляка (nbond==0): связанный массив
        # несёт вес связями вниз, а не висит на искусственных границах домена
        # (иначе трение стены забирает часть веса столба и σ_v в поле занижается,
        # а при плавном наборе перегруза режим "зависания" ещё сильнее)
        if walls == 1 and nbond[i] == 0:
            if bx[i] <= rx + 0.02 * LCELL or bx[i] >= (COLS * LCELL - rx - 0.02 * LCELL):
                friction_against(i, bvy[i], nFw, dt, -(i + 1) - FRIC_SLOTS, kfric, 1)
        bvx[i] += bfx[i] / bmass[i] * dt
        bvy[i] += bfy[i] / bmass[i] * dt
        # Связанные и опирающиеся блоки гасятся сильно (устойчивость массива,
        # оседание обломков). Блок, летящий в пустоте — связей нет и снизу нет опоры
        # (supp==0, геометрия из phys_contact) — почти не гасится: обрушение выходит
        # динамичным, а не "вязким". Как только блок садится на опору, демпфирование
        # возвращается.
        # dfree/dsupp/drot* — множители демпфирования ЗА ПОДШАГ: базовые константы
        # (DAMP_*_TICK) задуманы на тик, но здесь применяются SUBSTEPS раз за тик,
        # поэтому шагом раньше пересчитаны как base^(dt/TICK). Иначе 384-кратное
        # применение давало бы v *= 0.9995^384 ≈ 0.825 за тик и искусственную
        # терминальную скорость падения ~3.4 клетки/с вместо реального свободного
        # падения (обломки "ползли" вниз много секунд).
        if nbond[i] == 0 and supp[i] == 0:
            bvx[i] *= dfree
            bvy[i] *= dfree
        else:
            bvx[i] *= dsupp
            bvy[i] *= dsupp
        bw[i] += btor[i] / (bmass[i] * bsz[i] * bsz[i] / 6.0) * dt
        if nbond[i] > 0:
            bw[i] *= drotBond
        else:
            bw[i] *= drotFree
        if ti.abs(bw[i]) < 2e-5:
            bw[i] = 0.0
        if bw[i] > 10.0:
            bw[i] = 10.0
        elif bw[i] < -10.0:
            bw[i] = -10.0
        brot[i] += bw[i] * dt
        v2 = bvx[i] * bvx[i] + bvy[i] * bvy[i]
        if v2 > P[None].vmax * P[None].vmax:
            s = P[None].vmax / ti.sqrt(v2)
            bvx[i] *= s
            bvy[i] *= s
            ti.atomic_add(vmaxHit[None], 1)
        # --- поглощающий слой у искусственных границ (side/bottom) ---
        # Не «жёсткая» неотражающая граница, а sponge: скорость гасится
        # экспоненциально в слое толщиной absorbWidth. Нужен, чтобы волна
        # от выработки не возвращалась от боков/подошвы и не превращала
        # локальный strainburst в глобальную лавину.
        if P[None].absorb == 1:
            aw = P[None].absorbWidth * LCELL
            if aw > 1e-9:
                dl = bx[i]
                dr = COLS * LCELL - bx[i]
                db = ROWS * LCELL - by[i]
                dd = ti.min(dl, dr)
                if db < dd:
                    dd = db
                if P[None].absorbTop == 1:
                    dtp = by[i]
                    if dtp < dd:
                        dd = dtp
                if dd < aw:
                    wgt = 1.0 - dd / aw
                    fac = ti.exp(-P[None].absorbCoef * dt * wgt * wgt)
                    bvx[i] *= fac
                    bvy[i] *= fac
        # Сон ОТКЛЮЧЁН: он не экономит вычисления (силы считаются для всех
        # блоков), но обнуляет скорость и «замораживает» обломки/висящие
        # блоки. Парные демпферы связей/контактов гасят движение без заморозки.
        bx[i] += bvx[i] * dt
        by[i] += bvy[i] * dt
        # Вязко-поглощающие границы (dashpot) + кламп по границам домена.
        # Модель ОСЕВАЯ: локальные оси совпадают с границами домена, пол и стены —
        # горизонтальны/вертикальны в системе решателя. Мировой наклон учитывается
        # проекциями сил (forces_reset) и поворотом ВИДА (render.rotate_view).
        damp = 2.0 * P[None].cnc * dt * ti.sqrt(P[None].E / bmass[i])
        if damp > 1.0:
            damp = 1.0
        if walls == 1:
            if bx[i] < rx:
                bx[i] = rx
                if bvx[i] < 0.0:
                    bvx[i] *= 1.0 - damp
            elif bx[i] > COLS * LCELL - rx:
                bx[i] = COLS * LCELL - rx
                if bvx[i] > 0.0:
                    bvx[i] *= 1.0 - damp
        # Верх — нагрузочная граница только у нагруженной модели (depth>0):
        # улетевший выше блок удаляет cleanup_fallen, а не держит кламп (иначе
        # он становится topCol и получает весь ovf). У ненагруженной сцены
        # (склон/откос, depth=0) верх — обычная крышка домена, кламп остаётся.
        if P[None].depth <= 0.0 and nbond[i] > 0 and by[i] < rx:
            by[i] = rx
            if bvy[i] < 0.0:
                bvy[i] *= 1.0 - damp
        if by[i] > ROWS * LCELL - rx:
            by[i] = ROWS * LCELL - rx
            if bvy[i] > 0.0:
                bvy[i] *= 1.0 - damp


@ti.kernel
def phys_integrate(dt: ti.f32, dfree: ti.f32, dsupp: ti.f32, drotFree: ti.f32, drotBond: ti.f32):
    """Ядро-обёртка (совместимость/отладка): качание + интегрирование."""
    forces_rock()
    forces_integrate(dt, dfree, dsupp, drotFree, drotBond)


@ti.kernel
def phys_substep(dt: ti.f32, dfree: ti.f32, dsupp: ti.f32, drotFree: ti.f32, drotBond: ti.f32):
    """Весь подшаг ЦЕЛИКОМ ОДНИМ лаунчем: поле напряжений -> силы -> качание ->
    интегрирование. Один запуск ядра на подшаг вместо двух (исторически пяти):
    при 384 подшагах это сотни лишних сабмитов в секунду — они держали CPU в
    очереди драйвера, а GPU простаивал между маленькими ядрами.

    Порядок внутри: sigvi_field (свежий sigVI) -> bond_stress_field (прокси
    главных сжатий для раскалывания) -> forces_zero (обнуление
    аккумуляторов) -> forces_contact (контакты + флаги граней) -> forces_reset
    (базовые силы читают свежие флаги) -> forces_bonds (связи) -> forces_rock
    (моменты по свежим supp/comIn/cnck) -> forces_integrate."""
    sigvi_field()
    bond_stress_field()
    forces_zero()
    forces_contact(dt)
    forces_reset()
    forces_bonds(dt)
    forces_rock()
    forces_integrate(dt, dfree, dsupp, drotFree, drotBond)


@ti.kernel
def count_bonds():
    for i in range(bcount[None]):
        nbond[i] = 0
        hasL[i] = 0
        hasR[i] = 0
    # Чистим только достижимые слова: биты ставятся индексами блоков < bcount,
    # значит слово j>>5 < nw. Параллельно по строкам bcount, слова nw — serial.
    # Плоский range(bcount*nw) на домене 128 = ~18.9 млн итераций = больше лимита
    # диспетчера Vulkan (~65535 групп × block_dim ≈ 8.4 млн) -> "Dispatch error:
    # not_supported". Параллельных итераций здесь ровно bcount (≤ MAXB = 24576).
    nw = (bcount[None] + 31) >> 5
    for i in range(bcount[None]):
        for j in range(nw):
            bondMap[i, j] = 0
            hasMap[i, j] = 0
    for bd in range(bondCount[None]):
        st = bondIntact[bd]
        if st == BOND_DEAD:
            continue
        a = bondA[bd]
        b = bondB[bd]
        # hasMap: у пары есть запись связи (целая ИЛИ шов) — для флагов трещин
        # crackL/crackR: трещиной считается только пара БЕЗ записи, иначе распор
        # рыхляка в forces_reset сложится с усилием сжимаемого шва (двойной счёт)
        ti.atomic_or(hasMap[a, b >> 5], 1 << (b & 31))
        ti.atomic_or(hasMap[b, a >> 5], 1 << (a & 31))
        if st == BOND_INTACT:
            ti.atomic_add(nbond[a], 1)
            ti.atomic_add(nbond[b], 1)
            ti.atomic_or(bondMap[a, b >> 5], 1 << (b & 31))
            ti.atomic_or(bondMap[b, a >> 5], 1 << (a & 31))
            # боковая связь: ось связи ближе к горизонтали -> у левого блока
            # правая грань связана (hasR), у правого — левая грань (hasL)
            if ti.abs(bx[b] - bx[a]) >= ti.abs(by[b] - by[a]):
                if bx[b] >= bx[a]:
                    ti.atomic_max(hasR[a], 1)
                    ti.atomic_max(hasL[b], 1)
                else:
                    ti.atomic_max(hasL[a], 1)
                    ti.atomic_max(hasR[b], 1)


@ti.kernel
def clear_block_hash():
    for c in range(N):
        hhead[c] = -1


@ti.kernel
def insert_block_hash(decay: ti.i32):
    # ВАЖНО: вставка в цепочку ячейки СЕРИЙНАЯ. При параллельной вставке два
    # блока в одной клетке (куча, столб) одновременно читают hhead[c] и один
    # теряется из хэша: у него пропадают контакты — блок проваливается или
    # «висит», а при возврате в хэш из-за глубокого перехлёста получает огромный
    # импульс (симптом «выстрелы» из рыхляка). ti.atomic_exchange в 1.7.4 нет,
    # поэтому вставку делаем одним потоком: O(bcount) и только при перестройке.
    # Ядра разделены (очистка / вставка), чтобы между ними был гарантированный
    # барьер: иначе очистка hhead могла ещё идти параллельно вставке.
    ti.loop_config(serialize=True)
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        c = cyi * COLS + cxi
        hnext[i] = hhead[c]
        hhead[c] = i
        if decay == 1:
            bstress[i] *= 0.88
            bstressC[i] *= 0.92


def build_hash():
    clear_block_hash()
    insert_block_hash(1)


def build_hash_now():
    clear_block_hash()
    insert_block_hash(0)


@ti.kernel
def fric_reset():
    """Полный сброс таблицы памяти трения (open-addressing хэш).

    Слоты освобождаются только при удалении блока, поэтому со временем таблица
    заполняется устаревшими ключами (пары, которые уже не контактируют, но и не
    удалены). Периодический сброс переиспользует ёмкость. Память трения и так
    «забывается» за FRIC_FORGET_T (3 тика), так что сброс раз в десятки тиков
    поведение не меняет."""
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
        fricS2[s] = 0.0
        fricF2[s] = 0


@ti.kernel
def cleanup_fallen():
    i = bcount[None] - 1
    while i >= 0:
        top_gone = by[i] < -2 * LCELL
        if bfixed[i] == 0 and (by[i] > (ROWS + 6) * LCELL or top_gone
                               or bx[i] < -4 * LCELL or bx[i] > (COLS + 4) * LCELL):
            remove_block_func(i)
        i -= 1


# ------------------------------------------------------------------ верхняя нагрузка
@ti.func
def find_top_blocks():
    """Верхний блок каждой колонки за O(bcount) атомарным min вместо
    O(COLS*bcount) вложенного скана (вложенный for на GPU сериализуется).
    topCol/topYArr заполняются в compute_top_load_fused и переиспользуются
    в refill_top_overburden того же тика. При равных y выбирается любой из
    равных (раньше - первый по индексу; на физику не влияет)."""
    for c in range(COLS):
        topCol[c] = -1
        topYArr[c] = 1e9
        loadCol[c] = -1
        loadColBot[c] = -1
        botCol[c] = -1
        botYArr[c] = -1e9
    for i in range(bcount[None]):
        c = clamp_cell(bx[i])
        ti.atomic_min(topYArr[c], by[i])
        ti.atomic_max(botYArr[c], by[i])
    for i in range(bcount[None]):
        c = clamp_cell(bx[i])
        if by[i] <= topYArr[c]:
            topCol[c] = i
        if by[i] >= botYArr[c]:
            botCol[c] = i
    for c in range(COLS):
        i = topCol[c]
        if i >= 0 and bbedrock[i] == 0 and bfixed[i] == 0:
            loadCol[c] = i
        i = botCol[c]
        if i >= 0 and bbedrock[i] == 0 and bfixed[i] == 0:
            loadColBot[c] = i


@ti.func
def ramp_scalar(cur: ti.f32, target: ti.f32, rate: ti.f32) -> ti.f32:
    x = cur
    if x < target:
        x += rate
        if x > target:
            x = target
    elif x > target:
        x -= rate
        if x < target:
            x = target
    return x


@ti.kernel
def compute_top_load_fused(mode: ti.i32):
    for i in range(bcount[None]):
        ovf[i] = 0.0
        ovfX[i] = 0.0
    tunnel = 1 if (mode == 0 or mode == 3) else 0
    target = P[None].depth * LOAD_OVER if tunnel == 1 else 0.0
    targetX = P[None].loadX if tunnel == 1 else 0.0
    ref = target
    if loadCur[None] > ref:
        ref = loadCur[None]
    loadCur[None] = ramp_scalar(loadCur[None], target, ref * (TICK / LOAD_TIME))
    refX = ti.abs(targetX)
    if ti.abs(loadXCur[None]) > refX:
        refX = ti.abs(loadXCur[None])
    loadXCur[None] = ramp_scalar(loadXCur[None], targetX, refX * (TICK / LOAD_TIME))
    find_top_blocks()
    f = loadCur[None]
    fx = loadXCur[None]
    for c in range(COLS):
        i = loadCol[c]
        if i >= 0:
            ti.atomic_add(ovf[i], f)
            ti.atomic_add(ovfX[i], fx)
        i = loadColBot[c]
        if i >= 0:
            ti.atomic_add(ovfX[i], -fx)

@ti.kernel
def find_side_blocks():
    """Крайние блоки КАЖДОЙ строки — текущая боковая граница модели.

    Раньше sideF получали только блоки в полосе 0.25·LCELL от НЕПОДВИЖНОЙ
    кромки домена. При сжатии массива крайний блок уходит внутрь дальше этой
    полосы (на 1500 м это доли клетки), и боковое давление на него
    ИСЧЕЗАЛО: сумма |sideF| падала с 4.8e9 до 2.4e9 (и локально до 0.05e9),
    т.е. модель на мгновение разгружалась с одной стороны — удар, лавина.
    Теперь давление прикладывается к фактическому крайнему блоку строки и
    следует за деформацией (follower load), как верхний перегруз
    (find_top_blocks → ovf)."""
    for r in range(ROWS):
        li = -1
        ri = -1
        lx = 1e30
        rx = -1e30
        for i in range(bcount[None]):
            # Боковая нагрузка — на КРАЙ СВЯЗНОГО МАССИВА, а не на
            # одиночный свободный обломок у стены: иначе он получает
            # многомегатонную боковую силу и «прилипает» к краю/соседям.
            if nbond[i] == 0:
                continue
            if row_cell(by[i]) == r:
                if bx[i] < lx:
                    lx = bx[i]
                    li = i
                if bx[i] > rx:
                    rx = bx[i]
                    ri = i
        sideLIdx[r] = li
        sideRIdx[r] = ri


@ti.kernel
def compute_side_load_fused(mode: ti.i32):
    """Боковое давление ПРЯМОЙ силой: σ_h = K0·σ_v жмёт на модель с боков
    (от границ домена внутрь) - аналог верхнего перегруза ovf, но по
    горизонтали. Сила растёт с глубиной автоматически: σ_v = вес столба
    породы над точкой + перегруз (sigVI, обновляется каждый подшаг).
    Престресса в модели НЕТ - боковое обжатие передаётся реальными силами
    через связи/контакты, как и вертикальное.
    F = σ_h · высота грани = K0·(sv/LCELL)·bsz. Работает в режиме шахты
    (mode==0), как и верхняя нагрузка; при K0=0 силы нет.
    Нагрузка — на КРАЙНИЕ блоки строк (sideLIdx/sideRIdx), т.е. на
    деформируемую границу, а не на фиксированную полосу у кромки домена."""
    for i in range(bcount[None]):
        sideF[i] = 0.0
    if mode == 0 or mode == 3:
        k0 = P[None].k0
        if k0 > 0.0:
            for r in range(ROWS):
                i = sideLIdx[r]
                if i >= 0:
                    sv = sigVI[r * COLS + clamp_cell(bx[i])]
                    if sv > 0.0:
                        # блок на ЛЕВОЙ границе -> давит вправо, в массив
                        sideF[i] += k0 * sv * (bsz[i] / LCELL)
                j = sideRIdx[r]
                if j >= 0:
                    sv = sigVI[r * COLS + clamp_cell(bx[j])]
                    if sv > 0.0:
                        # блок на ПРАВОЙ границе -> давит влево
                        sideF[j] -= k0 * sv * (bsz[j] / LCELL)


@ti.kernel
def refill_top_overburden(mode: ti.i32):
    """Досыпка обломков сверху (модель "кучи породы над выработкой").

    Каждый тик проверяем верх колонок: если верхняя клетка колонки пуста
    (кровля просела/обрушилась), кладём туда новый ОДИНОЧНЫЙ обломок. Он НЕ
    связывается со старыми блоками (пришёл сверху), но если в этом же тике
    досыпаны обломки в соседние колонки — они связываются между собой (если
    рядом). Так обрушение сверху просто приносит новые блоки вниз, как кучу
    породы в шахте."""
    on = 0
    if (mode == 0 or mode == 3) and refillOff[None] == 0 and bcount[None] < REFILL_MAXB:
        on = 1
    # 1. находим колонки с пустой верхней клеткой — ПАРАЛЛЕЛЬНО по колонкам.
    #    (раньше весь проход был сплошным while на ОДНОМ потоке GPU:
    #    O(COLS*blocks) серийно = ~32 мс/тик в одиночном ядре — главная
    #    статья времени GPU при «простаивающем» виде. ВАЖНО: for должен быть
    #    на ВЕРХНЕМ уровне ядра — цикл внутри if тайчи демотирует в serial.
    #    Спавн и связывание ниже остаются серийными: add_block_func не атомарен
    #    по bcount, но там всего O(COLS) работы.)
    #    Скан верха колонок переиспользуется из compute_top_load_fused
    #    (topCol/topYArr того же тика) — он уже посчитан за O(bcount).
    for col in range(COLS):
        newRubble[col] = -1
        if on == 1:
            best = topCol[col]
            bestY = topYArr[col]
            # верхняя клетка (y=0) свободна, если в колонке вообще нет блока,
            # либо самый верхний блок опустился ниже неё (координаты в м).
            # Новый блок центра y=0.5 клетки, его нижняя грань = LCELL, поэтому
            # нужен зазор: bestY - bsz/2 >= LCELL. Без учёта размера блок
            # спавнился ПЕРЕКРЫТЫМ с верхним и контакт выбрасывал его вверх
            # (симптом «обломки выстреливают» при включённой досыпке).
            # -2 = «нужен спавн», реальный индекс заполнит серийный проход ниже
            if best < 0 or bestY >= LCELL + 0.5 * bsz[best]:
                newRubble[col] = -2
    # 2. досыпаем блоки в колонки с пустой верхней клеткой (серийно)
    c = 0
    while c < COLS:
        if on == 1 and newRubble[c] == -2:
            j = add_block_func(c + 0.5, 0.5, 1.0, 0)
            if j >= 0:
                newRubble[c] = j
            else:
                newRubble[c] = -1
        c += 1
    # 3. Связи между новыми обломками НЕ создаём: досыпка — это сыпучий
    #    материал. Связанный «ковёр» из блоков одного тика висел в воздухе
    #    и выглядел как зависшие блоки.


def compute_top_load(mode):
    compute_top_load_fused(mode)
    refill_top_overburden(mode)


def compute_side_load(mode):
    if mode == 0 or mode == 3:
        find_side_blocks()
    compute_side_load_fused(mode)


# ================================================================== флюид
def fill_pct():
    return float(water_fill_pct(cavX[None], cavY[None], cavRX[None], cavRY[None]))


# ================================================================== диагностика
@ti.kernel
def count_intact() -> ti.i32:
    n = 0
    for i in range(bondCount[None]):
        if bondIntact[i] == 1:
            n += 1
    return n


@ti.kernel
def diag_stats() -> ti.f32:
    mx = 0.0
    for i in range(bcount[None]):
        dx = bx[i] - hx[i]
        dy = by[i] - hy[i]
        d = ti.sqrt(dx * dx + dy * dy)
        if d > mx:
            mx = d
    return mx


@ti.kernel
def diag_strain() -> ti.f32:
    dmax[None] = 0.0
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            a = bondA[bd]
            b = bondB[bd]
            dx = bx[b] - bx[a]
            dy = by[b] - by[a]
            st = ti.abs(ti.sqrt(dx * dx + dy * dy) - brest[bd])
            ti.atomic_max(dmax[None], st)
    return dmax[None]


# ================================================================== загрузка модели
@ti.kernel
def reset_all():
    """Полный сброс состояния к пустой модели (конфигурация не трогается)."""
    bcount[None] = 0
    bondCount[None] = 0
    dustCount[None] = 0
    broken[None] = 0
    loadCur[None] = 0.0
    loadXCur[None] = 0.0
    filledFlag[None] = 0
    srcCount[None] = 0
    simT[None] = 0.0
    compactT[None] = 0.0
    frameBrk[None] = 0
    vmaxHit[None] = 0
    energyFrac[None] = 0.0
    breakCrush[None] = 0
    dmax[None] = 0.0
    P[None].maxSize = 1
    for c in range(N):
        hhead[c] = -1
        gOcc[c] = 0
        sigVI[c] = 0.0
        holdL[c] = 0.0
        holdR[c] = 0.0
    for c in range(COLS):
        topCol[c] = -1
        topYArr[c] = 1e9
        loadCol[c] = -1
        loadColBot[c] = -1
        botCol[c] = -1
        botYArr[c] = -1e9
        newRubble[c] = -1
        sideLIdx[c] = -1
        sideRIdx[c] = -1
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
        fricS2[s] = 0.0
        fricF2[s] = 0
    for i in range(MAXB):
        bfixed[i] = 0
        brkShell[i] = 0
        nbond[i] = 0
        hasL[i] = 0
        hasR[i] = 0
        sideF[i] = 0.0
        ovfX[i] = 0.0
        efx[i] = 0.0
        efy[i] = 0.0
        supp[i] = 0
        comIn[i] = 0
        supLo[i] = 1e9
        supHi[i] = -1e9
        bSxx[i] = 0.0
        bSyy[i] = 0.0
    for bd in range(MAXBONDS):
        bondIntact[bd] = BOND_DEAD
        bondE[bd] = 0.0
        bondPlast[bd] = 0.0
        bondSlip[bd] = 0.0
        bondFlow[bd] = 0
        bondCreep[bd] = 0.0
        bondOver[bd] = 0.0
        bondSplitAvg[bd] = 0.0
        bondBendAvg[bd] = 0.0
    # Сброс карт связей/занятости: ПАРАЛЛЕЛЬНО по MAXB, BMAPW — внутри (serial).
    # Плоский range(MAXB*BMAPW) на домене 128 даёт ~18.9 млн итераций = больше
    # лимита диспетчера Vulkan (~65535 групп × block_dim ≈ 8.4 млн) -> "Dispatch
    # error: RhiResult(not_supported)". ndrange с тем же произведением тоже не
    # проходит, поэтому делаем параллельных итераций ровно MAXB (24576 -> ~192
    # группы), а BMAPW прогоняем последовательно.
    for i in range(MAXB):
        for j in range(BMAPW):
            bondMap[i, j] = 0
            hasMap[i, j] = 0


@ti.kernel
def update_mass_rho():
    for i in range(bcount[None]):
        bmass[i] = P[None].rho * bsz[i] * bsz[i]


@ti.kernel
def set_ext_force(i: ti.i32, fx: ti.f32, fy: ti.f32):
    """Внешняя сила на блок (Н) — нагружение образца (UCS/бразильский тест).
    Живёт между тиками (forces_reset читает efx/efy каждый подшаг)."""
    efx[i] = fx
    efy[i] = fy


@ti.kernel
def clear_ext_forces():
    for i in range(bcount[None]):
        efx[i] = 0.0
        efy[i] = 0.0


@ti.kernel
def _load_blocks(blocks: ti.types.ndarray()):
    for k in range(blocks.shape[0]):
        if k < MAXB:
            init_block(k, blocks[k, 0], blocks[k, 1], blocks[k, 2], ti.cast(blocks[k, 3], ti.i32))


@ti.kernel
def _load_bedrock(flags: ti.types.ndarray()):
    for k in range(flags.shape[0]):
        if k < MAXB:
            bbedrock[k] = ti.cast(flags[k], ti.i32)


@ti.kernel
def _load_bonds(bonds: ti.types.ndarray()):
    for k in range(bonds.shape[0]):
        # целостность из внешней модели: 1 -> целая связь, 0 -> шов (BOND_JOINT:
        # сохраняет сжатие, как разорванная связь); BOND_DEAD загрузке недоступен
        st = BOND_JOINT
        if ti.cast(bonds[k, 3], ti.i32) == 1:
            st = BOND_INTACT
        add_bond_func(ti.cast(bonds[k, 0], ti.i32), ti.cast(bonds[k, 1], ti.i32), bonds[k, 2], st)


@ti.kernel
def _load_water(water: ti.types.ndarray()):
    for k in range(water.shape[0]):
        water_inject_func(water[k, 0], water[k, 1], water[k, 2])


def build_model(blocks, bonds, water, cavity):
    """Загрузить модель, построенную снаружи (render.py).

    blocks — [(x, y, size, fixed), ...]   позиция центра, размер (сторона), флаг закрепления
    bonds  — [(a, b, rest, intact), ...]  индексы блоков, длина покоя, целостность
    water  — [(x, y, v, ang), ...]        частицы воды
    cavity — (cx, cy, rx, ry) или None    полость (для диагностики заполнения)
    """
    reset_all()
    if cavity:
        cavX[None] = cavity[0]
        cavY[None] = cavity[1]
        cavRX[None] = cavity[2]
        cavRY[None] = cavity[3]
    maxs = 1.0
    if blocks:
        for b in blocks:
            if b[2] > maxs:
                maxs = b[2]
    P[None].maxSize = max(1, int(math.ceil(maxs)))
    if blocks:
        nb = min(len(blocks), MAXB)
        bcount[None] = nb
        _load_blocks(np.asarray(blocks, dtype=np.float32))
    if bonds:
        _load_bonds(np.asarray(bonds, dtype=np.float32))
    water_init_host()
    if water:
        _load_water(np.asarray(water, dtype=np.float32))


@ti.kernel
def rotate_model(angle: ti.f32, px: ti.f32, py: ti.f32):
    """Жёсткий поворот СОБРАННОЙ модели на angle (радиан) вокруг точки (px, py).

    Поворачивается только геометрия модели: блоки (позиции/скорости/углы),
    вода, пыль, источники, центр полости. СИЛА ТЯЖЕСТИ И ПЕРЕГРУЗ ОСТАЮТСЯ
    СТРОГО ВЕРТИКАЛЬНЫМИ, пол и стены домена — мировыми (горизонтальными):
    наклонённая модель «лежит» на горизонтальной границе под вертикальным g.
    Связи не трогаем: поворот — изометрия, brest/энергии корректны. Пружины
    трения (fricS) сбрасываем — тангенциальные упругие сдвиги накоплены в
    старой ориентации и после поворота дали бы ложный импульс."""
    ct = ti.cos(angle)
    st = ti.sin(angle)
    for i in range(bcount[None]):
        dx = bx[i] - px
        dy = by[i] - py
        bx[i] = px + dx * ct - dy * st
        by[i] = py + dx * st + dy * ct
        vx = bvx[i]
        vy = bvy[i]
        bvx[i] = vx * ct - vy * st
        bvy[i] = vx * st + vy * ct
        brot[i] += angle
        dxh = hx[i] - px
        dyh = hy[i] - py
        hx[i] = px + dxh * ct - dyh * st
        hy[i] = py + dxh * st + dyh * ct
    for d in range(dustCount[None]):
        dx = dustX[d] - px
        dy = dustY[d] - py
        dustX[d] = px + dx * ct - dy * st
        dustY[d] = py + dx * st + dy * ct
        vx = dustVX[d]
        vy = dustVY[d]
        dustVX[d] = vx * ct - vy * st
        dustVY[d] = vx * st + vy * ct
    for s in range(srcCount[None]):
        dx = srcX[s] - px
        dy = srcY[s] - py
        srcX[s] = px + dx * ct - dy * st
        srcY[s] = py + dx * st + dy * ct
    dx = cavX[None] - px
    dy = cavY[None] - py
    cavX[None] = px + dx * ct - dy * st
    cavY[None] = py + dx * st + dy * ct
    for f in range(FRIC_SLOTS):
        fricS[f] = 0.0


# ================================================================== источники воды
def nearest_free_cell(x, y, radius=8):
    """Ближайшая пустая (не порода) клетка вокруг (x, y) в координатах КЛЕТОК."""
    cx = int(x)
    cy = int(y)

    def free(ix, iy):
        if ix < 0 or ix >= COLS or iy < 0 or iy >= ROWS:
            return False
        return gOcc[iy * COLS + ix] == 0

    if free(cx, cy):
        return float(x), float(y)
    best = None
    bd = 1.0e18
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if free(cx + dx, cy + dy):
                d = float(dx * dx + dy * dy)
                if d < bd:
                    bd = d
                    best = (cx + dx + 0.5, cy + dy + 0.5)
    if best is None:
        return None
    return best


def place_source(x, y):
    if srcCount[None] >= 4:
        return "Maximum 4 sources"
    spot = nearest_free_cell(x, y)
    if spot is None:
        return "No free cell for the source"
    x, y = spot
    s = srcCount[None]
    srcX[s] = x
    srcY[s] = y
    srcPh[s] = time.time() % 1.0
    srcCount[None] += 1
    rx = float(cavRX[None])
    ry = float(cavRY[None])
    if rx > 1e-6 and ry > 1e-6:
        if ((x - cavX[None]) / rx) ** 2 + ((y - cavY[None]) / ry) ** 2 < 1:
            return "Source: H=%d m * in the working - filling" % P[None].head
    return "Source: H=%d m" % P[None].head


def remove_source(x, y):
    bi = -1
    bd = 9.0
    for i in range(srcCount[None]):
        d = math.hypot(srcX[i] - x, srcY[i] - y)
        if d < bd:
            bd = d
            bi = i
    if bi >= 0:
        last = srcCount[None] - 1
        srcX[bi] = srcX[last]
        srcY[bi] = srcY[last]
        srcPh[bi] = srcPh[last]
        srcCount[None] -= 1
        return True
    return False


def emit_sources():
    if srcCount[None] == 0:
        return
    v0 = math.sqrt(2.0 * G_PHYS * P[None].head)
    water_emit_sources(v0)


# ================================================================== главный цикл
# ЛОКАЛЬНАЯ ФИЗИКА РЕШАТЕЛЯ. Один ТИК = один вызов step_physics(): модель
# продвигается на TICK секунд внутреннего времени, и следующий тик наступает
# только после полного расчёта предыдущего (никакой привязки к кадрам или
# реальному времени — скорость тиков ограничена только CPU). Подшаги интегрирования
# внутри тика — локальная деталь решателя; наружу виден только тик.
_tickN = 0    # номер тика в процессе (для разреженных задач: cleanup/энергия)


def step_physics(mode, dt=None):
    """Продвинуть локальную физику на ОДИН ТИК внутреннего времени (TICK с).

    dt — подшаг интегрирования внутри тика (по умолчанию DT = TICK/SUBSTEPS).
    Тик не зависит от кадров/реального времени: следующий тик стартует только
    после полного расчёта этого и идёт так быстро, как позволяет CPU."""
    global _tickN
    _tickN += 1
    simT[None] += TICK
    frameBrk[None] = 0
    vmaxHit[None] = 0
    decay_bond_damage()
    emit_sources()
    build_hash()
    # гидравлические силы обнуляются ВСЕГДА (см. water_forces_zero): иначе при
    # нуле частиц воды последние wfx/wfy застревали на блоках навечно
    water_step(P[None].lbmTau)
    water_apply_forces(P[None].lbmDrag, P[None].lbmPressure)
    # когда воды нет, флюидные подшаги пропускаются ЦЕЛИКОМ: ~25 пустых лаунчей
    # ядер за тик — заметная доля CPU-накладных расходов (GPU при этом простаивает)

    compute_top_load(mode)
    count_bonds()
    refresh_field()
    compute_side_load(mode)   # боковое давление: прямая сила с боков, растёт с глубиной;
                              # после refresh_field — sigVI уже свежий
    # демпфирование задано на тик (DAMP_*_TICK), а применяется SUBSTEPS раз за тик,
    # SUBSTEPS берётся из P (слайдер «подшагов/тик»): смена на ходу сразу меняет dt
    # и частоту пересборки топологии (TOPO_EVERY ~ 0.25 тика, масштабируется).
    nsub = P[None].substeps
    if nsub <= 0:
        nsub = SUBSTEPS
    topo_every = max(16, nsub // 4)
    if dt is None:
        dt = TICK / nsub
    dfree = DAMP_FREE_TICK ** (dt / TICK)
    dsupp = DAMP_SUPP_TICK ** (dt / TICK)
    drotFree = DAMP_ROT_FREE_TICK ** (dt / TICK)
    drotBond = DAMP_ROT_BOND_TICK ** (dt / TICK)
    # таблицу памяти трения периодически чистим (см. fric_reset): слоты иначе
    # копятся устаревшими ключами и таблица переполняется.
    if _tickN % FRIC_CLEAR_EVERY == 0:
        fric_reset()
    for ss in range(nsub):
        if ss % topo_every == 0:
            # топология (занятость клеток, вертикальное напряжение, контактный хэш,
            # счётчики связей) пересобирается по ТЕКУЩИМ позициям, а не раз в тик;
            # frameBrk сбрасывается вместе с ней, чтобы лимит разрывов не создавал
            # импульсную волну разрушения в первые подшаги тика.
            build_hash()
            count_bonds()
            refresh_field()
            frameBrk[None] = 0
        # весь подшаг — ОДНИМ лаунчем (поле + силы + качание + интегрирование):
        # 384 запуска ядра за тик (по одному на подшаг)
        phys_substep(dt, dfree, dsupp, drotFree, drotBond)
    # cleanup_fallen — СЕРИЙНЫЙ while по всем блокам (снятие с хвоста), на GPU
    # один поток на ~2 мс; вылетевшие обломки вне экрана и 4 тика не мешают
    if _tickN % 4 == 0:
        cleanup_fallen()
    build_hash_now()
    water_step(P[None].lbmTau, update_solid=False)
    water_apply_forces(P[None].lbmDrag, P[None].lbmPressure)
    if simT[None] - compactT[None] >= COMPACT_EVERY_T:
        compact_bonds()
        compactT[None] = simT[None]
    # --- равномерная сводка по всем блокам (каждый 2-й тик) ---
    # Полная энергия каждого блока и интегралы системы + центр тяжести.
    # Считаются для ВСЕХ блоков одинаково — контроль закона сохранения энергии
    # и положения центра тяжести. Разрежено: 4 атомарные редукции по всем
    # блокам/связям на слабом GPU дороже самих подшагов, а сводка - диагностика.
    if _tickN % 2 == 0:
        compute_energy()
        center_of_mass()

# ================================================================== энергия и центр тяжести
@ti.kernel
def compute_energy():
    """Полная энергия КАЖДОГО блока + общие интегралы системы (Дж).

    benergy[i] = ½·m·v² (кинетика) + ½·I·ω² (вращение) + m·g·(H−y) (потенциал)
                 + ½·Σ упругих энергий своих связей.
    y растёт ВНИЗ, поэтому потенциал отсчитывается от пола домена (y=H):
    при падении (y увеличивается) потенциальная энергия УМЕНЬШАЕТСЯ.
    Закреплённые блоки (опора) исключены из интегралов: у них бесконечная
    реакция опоры, их учёт не даёт физически содержательного баланса. Работа
    выполняется ОДИНАКОВО для всех блоков каждый тик — физика равномерна.

    Диссипация (демпфирование, кулоновское трение, пластика) уменьшает полную
    энергию — это честный отток энергии в тепло. Баланс нужен, чтобы доказать,
    что энергия НЕ генерируется из ниоткуда (нет дрейфа/взрывов)."""
    energyKin[None] = 0.0
    energyRot[None] = 0.0
    energyPot[None] = 0.0
    energyEl[None] = 0.0
    g = P[None].g
    for i in range(bcount[None]):
        benergy[i] = 0.0
        if bfixed[i] == 1 or bbedrock[i] != 0:
            continue
        m = bmass[i]
        v2 = bvx[i] * bvx[i] + bvy[i] * bvy[i]
        ke = 0.5 * m * v2
        I = m * bsz[i] * bsz[i] / 6.0
        re = 0.5 * I * bw[i] * bw[i]
        # y направлена ВНИЗ: U = m·g·(H−y), иначе падающий блок «набирает»
        # потенциальную энергию и энергобаланс/HUD показывают обратный знак.
        pe = m * g * (ROWS * LCELL - by[i])
        benergy[i] = ke + pe + re
        energyKin[None] += ke
        energyRot[None] += re
        energyPot[None] += pe
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            e = bondE[bd]
            energyEl[None] += e
            ti.atomic_add(benergy[bondA[bd]], e * 0.5)
            ti.atomic_add(benergy[bondB[bd]], e * 0.5)
    energyTotal[None] = energyKin[None] + energyRot[None] + energyPot[None] + energyEl[None]


@ti.kernel
def center_of_mass():
    """Центр тяжести системы (при однородном g совпадает с центром масс).

    Учитываются только подвижные блоки (bfixed==0); закреплённая «стена» — опора
    и граница модели, а не часть обрушаемого массива."""
    comX[None] = 0.0
    comY[None] = 0.0
    comMass[None] = 0.0
    for i in range(bcount[None]):
        if bfixed[i] == 1 or bbedrock[i] != 0:
            continue
        m = bmass[i]
        comMass[None] += m
        comX[None] += m * bx[i]
        comY[None] += m * by[i]
    if comMass[None] > 0.0:
        comX[None] /= comMass[None]
        comY[None] /= comMass[None]


@ti.kernel
def hud_gather():
    """Снимок всех значений HUD ОДНИМ ядром (слоты hudF — в physstate).

    Всё, что показывает GUI-панель: время, счётчики, заполнение водой,
    энергии, центр тяжести и параметры P. Python читает ОДИН раз
    hudF.to_numpy() на кадр вместо двух десятков read'ов-синков."""
    hudF[HUD_INTACT] = 0.0
    hudF[HUD_FILL] = 0.0
    for i in range(bondCount[None]):
        if bondIntact[i] == 1:
            ti.atomic_add(hudF[HUD_INTACT], 1.0)
    hudF[HUD_FILL] = water_fill_pct_func(cavX[None], cavY[None], cavRX[None], cavRY[None])
    hudF[HUD_PCOUNT] = water_fluid_count_func()
    hudF[HUD_SIMT] = simT[None]
    hudF[HUD_BCOUNT] = bcount[None]
    hudF[HUD_BROKEN] = broken[None]
    hudF[HUD_CRUSH] = breakCrush[None]
    hudF[HUD_FILLFLAG] = filledFlag[None]
    hudF[HUD_ETOT] = energyTotal[None]
    hudF[HUD_EKIN] = energyKin[None]
    hudF[HUD_EPOT] = energyPot[None]
    hudF[HUD_EROT] = energyRot[None]
    hudF[HUD_EEL] = energyEl[None]
    hudF[HUD_COMX] = comX[None]
    hudF[HUD_COMY] = comY[None]
    hudF[HUD_COMASS] = comMass[None]
    hudF[HUD_DEPTH] = P[None].depth
    hudF[HUD_K0] = P[None].k0
    hudF[HUD_FIXEDROWS] = P[None].fixedRows
    hudF[HUD_BRKCAP] = P[None].brkCap
    hudF[HUD_DMGMAX] = P[None].dmgMax
    hudF[HUD_PLASTFLOW] = P[None].plastFlow
    hudF[HUD_WALLS] = P[None].walls
    hudF[HUD_G] = P[None].g
    hudF[HUD_HEAD] = P[None].head
    hudF[HUD_RP] = P[None].rp
    hudF[HUD_MU] = P[None].mu
    hudF[HUD_RCFACTOR] = P[None].rcFactor
    hudF[HUD_MAXSIZE] = P[None].maxSize
    hudF[HUD_E] = P[None].E
    hudF[HUD_RHO] = P[None].rho
    hudF[HUD_PEAKT] = stressPeak[0]
    hudF[HUD_PEAKC] = stressPeak[1]


def hud_snapshot():
    """Снимок HUD для GUI: один лаунч ядра + одно копирование на кадр."""
    hud_gather()
    return hudF.to_numpy()


def energy_report():
    """Сводка энергии системы после последнего тика: (kin, rot, pot, elast, total), Дж."""
    return (energyKin[None], energyRot[None], energyPot[None], energyEl[None], energyTotal[None])


def center_of_mass_report():
    """Центр тяжести системы после последнего тика: (x, y, mass), м/кг."""
    return (comX[None], comY[None], comMass[None])


def frozen_report():
    """Диагностика зависших блоков: не закреплён, почти неподвижен, опоры снизу
    нет. Печатает, какая сила ДЕРЖИТ блок (связь/контакт/трение/фантом).
    Вызывается из GUI примерно раз в 3 с - смотреть КОНСОЛЬ."""
    n = bcount[None]
    if n == 0:
        return 0
    x = bx.to_numpy()[:n]
    y = by.to_numpy()[:n]
    vxb = bvx.to_numpy()[:n]
    vyb = bvy.to_numpy()[:n]
    nb = nbond.to_numpy()[:n]
    fix = bfixed.to_numpy()[:n]
    bed = bbedrock.to_numpy()[:n]
    com = comIn.to_numpy()[:n]
    sup = supp.to_numpy()[:n]
    inc = inContact.to_numpy()[:n]
    cn = cnck.to_numpy()[:n]
    fxb = bfx.to_numpy()[:n]
    fyb = bfy.to_numpy()[:n]
    wfxv = wfx.to_numpy()[:n]
    wfyv = wfy.to_numpy()[:n]
    ovfv = ovf.to_numpy()[:n]
    efxv = efx.to_numpy()[:n]
    efyv = efy.to_numpy()[:n]
    njoint = np.zeros(n, dtype=np.int64)
    if bondCount[None] > 0:
        aa = bondA.to_numpy()[:bondCount[None]]
        bb = bondB.to_numpy()[:bondCount[None]]
        st = bondIntact.to_numpy()[:bondCount[None]]
        for bd in range(bondCount[None]):
            if st[bd] == BOND_JOINT:
                if aa[bd] < n:
                    njoint[aa[bd]] += 1
                if bb[bd] < n:
                    njoint[bb[bd]] += 1
    m = bmass.to_numpy()[:n]
    out = []
    for i in range(n):
        if fix[i] == 1 or bed[i] != 0:
            continue
        if vxb[i] * vxb[i] + vyb[i] * vyb[i] > 2.5e-5:
            continue
        # Блок в верхней точке траектории тоже имеет v≈0, но он не «завис»:
        # у него |a|≈g. Сообщаем только действительно уравновешенные блоки.
        ax_i = fxb[i] / m[i]
        ay_i = fyb[i] / m[i]
        if ax_i * ax_i + ay_i * ay_i > 0.5:
            continue
        if y[i] >= ROWS * LCELL - 0.3:
            continue
        ok = False
        for j in range(n):
            if j == i or abs(x[j] - x[i]) > 0.6:
                continue
            dy = y[j] - y[i]
            if 0.0 < dy < 0.8:
                ok = True
                break
        if ok:
            continue
        out.append(i)
    if out:
        print("[freeze] %d block(s) frozen WITHOUT support below:" % len(out))
        for i in out[:8]:
            zone = ""
            if x[i] < 0.5:
                zone = " LEFT-WALL"
            if x[i] > COLS * LCELL - 0.5:
                zone = " RIGHT-WALL"
            print("   #%d pos=(%.2f,%.2f) nbond=%d njoint=%d comIn=%d supp=%d inC=%d "
                  "cnck=%.0fN mg=%.0fN ax=%+.2f ay=%+.2f wfx=%.0f wfy=%.0f "
                  "ovf=%.0f efx=%.0f efy=%.0f%s"
                  % (i, x[i], y[i], nb[i], int(njoint[i]), com[i], sup[i], inc[i],
                     cn[i], m[i] * 9.81, fxb[i] / m[i], fyb[i] / m[i],
                     wfxv[i], wfyv[i], ovfv[i], efxv[i], efyv[i], zone))
    return len(out)
