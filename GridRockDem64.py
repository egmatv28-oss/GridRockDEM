"""ГЕОМАССИВ/2D · DEM — точка входа, домен 64×64 клетки (GRIDROCKDEM_SIZE=64).

Запуск:  python GridRockDem64.py [--selftest]

Быстрый вариант (меньше блоков, ниже разрешение) — для отладки/слабых машин.
Физика — solver.py, графика — render.py, общий запуск — _bootstrap.py.
"""

from _bootstrap import launch

if __name__ == "__main__":
    launch(64)
