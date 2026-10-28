"""Простая запись кадров GridRockDem в PNG-последовательность.

1 секунда модельного времени = 1 секунда видео:
    fps = 30, TICK = 1/120 -> кадр каждые 4 тика.
GUI сам останавливает запись на 10 с (300 кадров).


Пишет:
    recordings/<name>/frames/frame_000000.png ...
    recordings/<name>/meta.json   (fps, число кадров, времена, размер)

Воспроизведение: player.py <каталог записи>
"""
import json
import os
import time

import numpy as np


class FrameRecorder:
    def __init__(self, out_dir, fps=30.0, width=None, height=None,
                 tick_hz=120.0, meta=None, gif_max_width=960):
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.width = width
        self.height = height
        self.tick_hz = float(tick_hz)
        self.frames = 0
        self.sim_times = []
        self.sim_start = None
        self.sim_end = None
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        self.meta = dict(meta or {})
        self.gif_max_width = int(gif_max_width or 0)
        self.video_path = None
        self._pil = None

    def _image(self, frame):
        if self._pil is None:
            from PIL import Image
            self._pil = Image
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        # Taichi image field is (X, Y, 3); PNG is (Y, X, 3), and the
        # Taichi canvas/imwrite convention additionally flips Y:
        #   ti.tools.image.cook_image_to_bytes: img.swapaxes(0, 1)[::-1, :]
        # Без [::-1] кадры получаются перевёрнутыми по вертикали.
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 0, 2))[::-1, :, :]
        return self._pil.fromarray(arr, "RGB")

    def capture(self, frame, sim_t):
        path = os.path.join(self.frames_dir, "frame_%06d.png" % self.frames)
        self._image(frame).save(path, optimize=False)
        if self.sim_start is None:
            self.sim_start = float(sim_t)
        self.sim_end = float(sim_t)
        self.sim_times.append(float(sim_t))
        self.frames += 1
        return path

    def make_gif(self):
        """Собрать video.gif из PNG-кадров (Pillow, без ffmpeg)."""
        if self.frames < 2:
            return None
        try:
            from PIL import Image
        except Exception:
            return None
        frames = []
        for i in range(self.frames):
            im = Image.open(os.path.join(self.frames_dir, "frame_%06d.png" % i)).convert("RGB")
            if self.gif_max_width > 0 and im.width > self.gif_max_width:
                h = max(1, int(im.height * self.gif_max_width / im.width))
                im = im.resize((self.gif_max_width, h), Image.Resampling.LANCZOS)
            frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        # Длительности из модельного времени, чтобы 1 с симуляции = 1 с GIF.
        durs = []
        for i in range(len(frames) - 1):
            durs.append(max(10, int(round((self.sim_times[i + 1] - self.sim_times[i]) * 1000.0))))
        if not durs:
            durs = [max(10, int(round(1000.0 / self.fps)))] * len(frames)
        elif len(durs) < len(frames):
            durs.append(durs[-1])
        path = os.path.join(self.out_dir, "video.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=durs, loop=0, optimize=False)
        self.video_path = path
        return path

    def close(self):
        self.make_gif()
        if self.width is None or self.height is None:
            # dimensions are not critical for playback; keep whatever meta has
            pass
        data = dict(self.meta)
        data.update({
            "fps": self.fps,
            "tick_hz": self.tick_hz,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "sim_start": self.sim_start,
            "sim_end": self.sim_end,
            "sim_times": self.sim_times,
            "video": os.path.basename(self.video_path) if self.video_path else None,
            "created": self.created,
        })
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return self.out_dir


