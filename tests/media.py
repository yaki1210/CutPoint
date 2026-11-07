"""Encode raw RGB frames for detector fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from cutpoint.probe import ffmpeg_exe


def solid(width: int, height: int, color: tuple[int, int, int], count: int) -> list[np.ndarray]:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return [frame.copy() for _ in range(count)]


def encode_frames(
    path: Path,
    frames: list[np.ndarray],
    fps: int = 30,
    gop: int | None = None,
) -> None:
    height, width = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        "8",
    ]
    if gop is not None:
        cmd += ["-g", str(gop), "-keyint_min", str(gop), "-x264-params", "scenecut=0"]
    cmd.append(str(path))
    raw = b"".join(np.ascontiguousarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    proc = subprocess.run(cmd, input=raw, check=False, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-15:]
        raise RuntimeError("\n".join(tail))
