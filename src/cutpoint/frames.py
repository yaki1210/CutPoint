"""Decode a video once into small RGB frames and per-frame curves."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np

from cutpoint.curves import (
    Signals,
    histogram_distance,
    pixel_diff,
    rec709_luma,
    region_slices,
    scaled_size,
)
from cutpoint.detect import DetectConfig
from cutpoint.probe import Probe, ffmpeg_exe


def _read_exact(pipe, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = pipe.read(size - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def read_signals(path: str | Path, probed: Probe, cfg: DetectConfig) -> Signals:
    thumb_w, thumb_h = scaled_size(probed.width, probed.height, cfg.thumb_width)
    cmd = [
        ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={thumb_w}:{thumb_h}",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    err: list[bytes] = []

    def _drain() -> None:
        err.append(proc.stderr.read())

    drain = threading.Thread(target=_drain)
    drain.start()

    frame_bytes = thumb_w * thumb_h * 3
    top, bot = region_slices(thumb_h, cfg.top_crop, cfg.bottom_crop)
    luma: list[float] = []
    pixel: list[float] = []
    hist: list[float] = []
    bottom: list[float] = []
    persist: list[float] = []
    prev2 = prev = None
    prev_bot = None

    while True:
        raw = _read_exact(proc.stdout, frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(thumb_h, thumb_w, 3)
        mid = frame[top:bot]
        low = frame[bot:] if bot < thumb_h else frame[-1:]
        luma.append(rec709_luma(mid))
        if prev is None:
            pixel.append(0.0)
            hist.append(0.0)
            bottom.append(0.0)
            persist.append(-1.0)
        else:
            pixel.append(pixel_diff(prev, mid))
            hist.append(histogram_distance(prev, mid))
            bottom.append(pixel_diff(prev_bot, low))
            persist.append(-1.0)
            if prev2 is not None:
                persist[-2] = pixel_diff(prev2, mid)
        prev2 = prev
        prev = mid.copy()
        prev_bot = low.copy()

    proc.stdout.close()
    return_code = proc.wait()
    drain.join()
    if return_code != 0:
        message = (err[0] if err else b"").decode("utf-8", "replace").strip()
        raise RuntimeError(message or "ffmpeg decode failed")

    count = len(luma)
    if count == 0:
        raise RuntimeError(f"ffmpeg decoded no frames from {path}")

    pts, rate, keyframe = _align_probe(probed, count)
    return Signals(
        pts=pts,
        luma=np.asarray(luma, dtype=np.float64),
        pixel=np.asarray(pixel, dtype=np.float64),
        hist=np.asarray(hist, dtype=np.float64),
        pixel_bottom=np.asarray(bottom, dtype=np.float64),
        rate=rate,
        keyframe=keyframe,
        persist=np.asarray(persist, dtype=np.float64),
        fps=probed.fps if probed.fps > 0 else _fps_from_pts(pts),
        duration=probed.duration if probed.duration > 0 else float(pts[-1]),
        width=probed.width,
        height=probed.height,
        frame_count=count,
    )


def _fps_from_pts(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    deltas = np.diff(pts)
    deltas = deltas[deltas > 1e-6]
    if len(deltas) == 0:
        return 0.0
    return float(1.0 / np.median(deltas))


def _align_probe(probed: Probe, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(probed.pts) == count:
        return probed.pts.copy(), probed.rate.copy(), probed.keyframe.copy()

    fps = probed.fps if probed.fps > 0 else 30.0
    pts = np.arange(count, dtype=np.float64) / fps
    rate = np.zeros(count, dtype=np.float64)
    keyframe = np.zeros(count, dtype=bool)
    shared = min(count, len(probed.pts))
    if shared:
        pts[:shared] = probed.pts[:shared]
        rate[:shared] = probed.rate[:shared]
        keyframe[:shared] = probed.keyframe[:shared]
        if shared < count and fps > 0:
            step = 1.0 / fps
            for i in range(shared, count):
                pts[i] = pts[i - 1] + step
    return pts, rate, keyframe
