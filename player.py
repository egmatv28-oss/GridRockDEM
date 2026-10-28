"""Плеер PNG-записи GridRockDem.

Использование:
    python player.py recordings/rec_YYYYmmdd_HHMMSS
    python player.py recordings/rec_... --scale 0.7 --loop

Запись делает recorder.py из GUI (кнопка Record). Плеер не считает физику;
он показывает кадры по их модельным временам, поэтому 1 с симуляции = 1 с
видео. Space — пауза, ←/→ — шаг, Q/Esc — выход.
"""
import argparse
import json
import os
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk


def load_meta(path):
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", help="folder with frames/ and meta.json")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    path = os.path.abspath(args.recording)
    meta = load_meta(path)
    frames_dir = os.path.join(path, "frames")
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".png"))
    if not files:
        raise SystemExit("no PNG frames in recording")
    times = meta.get("sim_times") or []
    fps = float(meta.get("fps") or 60.0)

    root = tk.Tk()
    root.title("GridRockDem playback — %s" % os.path.basename(path))
    label = tk.Label(root, bd=0)
    label.pack()
    state = {"i": 0, "playing": True, "photo": None, "after": None}
    scale = max(0.1, args.scale)

    def show(i):
        img = Image.open(os.path.join(frames_dir, files[i])).convert("RGB")
        if scale != 1.0:
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        state["photo"] = ImageTk.PhotoImage(img)
        label.configure(image=state["photo"])
        root.title("GridRockDem playback - frame %d/%d (%.2f s)"
                   % (i + 1, len(files), times[i] - times[0] if len(times) > i else i / fps))

    def schedule_next(i):
        if not state["playing"]:
            return
        if i + 1 >= len(files):
            if args.loop:
                i = 0
            else:
                state["playing"] = False
                return
        # точное модельное время кадра: 1 с сим = 1 с видео
        if len(times) > i + 1:
            delay = max(1, int((times[i + 1] - times[i]) * 1000.0))
        else:
            delay = max(1, int(1000.0 / fps))
        state["after"] = root.after(delay, lambda: step(i + 1))

    def step(i):
        if not state["playing"]:
            return
        state["i"] = i % len(files)
        show(state["i"])
        schedule_next(state["i"])

    def toggle(_=None):
        state["playing"] = not state["playing"]
        if state["playing"]:
            schedule_next(state["i"])

    def seek(delta):
        state["i"] = (state["i"] + delta) % len(files)
        show(state["i"])

    root.bind("<space>", toggle)
    root.bind("<Left>", lambda e: seek(-1))
    root.bind("<Right>", lambda e: seek(1))
    root.bind("<q>", lambda e: root.destroy())
    root.bind("<Escape>", lambda e: root.destroy())
    show(0)
    schedule_next(0)
    root.mainloop()


if __name__ == "__main__":
    main()
