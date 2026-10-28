"""ГЕОМАССИВ/2D · DEM — точка входа, домен 128×128 клеток (GRIDROCKDEM_SIZE=128).

Запуск:  python GridRockDem128.py [--selftest]

Крупный домен (вчетверо больше блоков): дальнее поле вокруг выработки умещается
целиком, видна локализация разрушения в свод, а не артефакт границы.
Физика — solver.py, графика — render.py, общий запуск — _bootstrap.py.
"""

from _bootstrap import launch

if __name__ == "__main__":
    launch(128)
