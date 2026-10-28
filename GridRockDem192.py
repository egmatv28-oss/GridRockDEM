"""ГЕОМАССИВ/2D · DEM — точка входа: домен 192×192 клеток (GRIDROCKDEM_SIZE=192).

Запуск:  python GridRockDem192.py [--selftest] [--frames N]

Домен 192×192 нужен для проверки размерного эффекта: выработка R=10
остаётся той же, а дальнее поле до границы — ~8.6·a (у 128×128 ~5.4·a).
На этой машине Vulkan не выделяет root buffer для 192×192, поэтому
`_bootstrap.launch` по умолчанию выбирает OpenGL; переопределяется
переменной GRIDROCKDEM_ARCH.

Решатель — solver.py, отрисовка — render.py, общий запуск — _bootstrap.py.
"""

from _bootstrap import launch

if __name__ == "__main__":
    launch(192)
