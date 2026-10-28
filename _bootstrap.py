"""Общий запуск ГЕОМАССИВ/2D · DEM для конкретного размера домена.

Решение о размере (GRIDROCKDEM_SIZE) ДОЛЖНО быть принято раньше, чем
импортируется physstate.py — он читает env-переменную на уровне модуля и строит
по ней COLS/ROWS и все Ti-поля (кэш Taichi тоже разведён по размеру). Поэтому
каждый лаунчер (GridRockDem64.py / GridRockDem128.py) лишь задаёт размер и
вызывает launch(size) ЗДЕСЬ, а вся тяжёлая логика (relaunch при кириллическом
пути до site-packages, junction, вызов gui.main) живёт в одном месте.
"""

import os
import subprocess
import sys

JUNCTION = r"C:\py312"
JUNCTION_EXE = JUNCTION + r"\python.exe"


def _is_ascii(s):
    return all(ord(c) < 128 for c in s)


def _taichi_home():
    """Путь до каталога taichi в site-packages — без импорта самого taichi.

    Импорт taichi в родительском процессе заставляет его захватывать lock
    кеша компиляции (ticache.lock), который потом не может взять дочерний
    процесс — отсюда предупреждение "[W] Lock ... failed". Чтобы родитель не
    создавал лишний KernelCompilationManager, ограничиваемся поиском пути.
    """
    for p in sys.path:
        if p and "site-packages" in p.replace("\\", "/").lower():
            return p
    return None


def _needs_relaunch():
    home = _taichi_home()
    try:
        return home is not None and not _is_ascii(home)
    except Exception:
        return False


def _ensure_junction():
    if os.path.exists(JUNCTION_EXE):
        return True
    target = os.path.dirname(sys.executable)
    if _is_ascii(target):
        return False
    try:
        import ctypes
        import ctypes.wintypes as w
        from ctypes import wintypes

        class REPARSE_DATA_BUFFER(ctypes.Structure):
            _fields_ = [("ReparseTag", w.DWORD), ("ReparseDataLength", w.WORD),
                        ("Reserved", w.WORD), ("SubstituteNameOffset", w.WORD),
                        ("SubstituteNameLength", w.WORD), ("PrintNameOffset", w.WORD),
                        ("PrintNameLength", w.WORD), ("Flags", w.DWORD),
                        ("PathBuffer", ctypes.c_char * 512)]

        CreateSymbolicLink = ctypes.windll.kernel32.CreateSymbolicLinkW
        CreateSymbolicLink.restype = w.BOOL
        CreateSymbolicLink.argtypes = [w.LPCWSTR, w.LPCWSTR, w.DWORD]
        ok = CreateSymbolicLink(JUNCTION, target, 1)  # 1 = directory
        return bool(ok) and os.path.exists(JUNCTION_EXE)
    except Exception:
        return False


def launch(size):
    """Запустить GUI с доменом size×size клеток.

    size должен быть задан ДО импорта gui/physstate (см. модульный docstring).
    """
    size = int(size)
    # env наследуется дочерним процессом при relaunch, поэтому достаточно задать
    # его один раз здесь - размер <зашит> через junction.
    os.environ["GRIDROCKDEM_SIZE"] = str(size)
    # 192×192: Vulkan проходит короткую пробу, но падает на materialize
    # ("Failed to allocate root buffer") - на этой машине у него не хватает
    # памяти под корневой буфер. OpenGL тот же объём тянет. Явно заданный
    # GRIDROCKDEM_ARCH имеет приоритет.
    if size >= 192 and not os.environ.get("GRIDROCKDEM_ARCH"):
        os.environ["GRIDROCKDEM_ARCH"] = "opengl"
        print("[GridRockDem] %dx%d: Vulkan root buffer limit -> OpenGL" % (size, size))
    args = sys.argv[1:]
    if "--selftest" in args:
        import gui
        gui.main()
        return
    if _needs_relaunch():
        if _ensure_junction() and os.path.exists(JUNCTION_EXE):
            env = dict(os.environ)
            env["GRIDROCKDEM_SIZE"] = str(size)
            sys.exit(subprocess.call(
                [JUNCTION_EXE, os.path.abspath(sys.argv[0])] + args, env=env))
        print(
            "Taichi не может открыть свои шейдеры: путь до site-packages содержит кириллицу.\n"
            "Скопируйте Python в путь без кириллицы либо создайте junction C:\\py312 -> Python312\n"
            "и запускайте:  C:\\py312\\python.exe " + os.path.basename(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    import gui
    gui.main()
