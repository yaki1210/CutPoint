"""ffprobe wrappers: duration, frame PTS, packet size, keyframes."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WINDOWS_BIN = Path(r"C:\Program Files\ffmpeg\bin")


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    windows = _WINDOWS_BIN / f"{name}.exe"
    if windows.exists():
        return str(windows)
    return name


def ffmpeg_exe() -> str:
    return _tool("ffmpeg")


def ffprobe_exe() -> str:
    return _tool("ffprobe")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def parse_frame_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(value)


@dataclass
class Probe:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    pts: np.ndarray
    rate: np.ndarray
    keyframe: np.ndarray


def probe(path: str | Path) -> Probe:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    meta = _run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    if meta.returncode != 0:
        raise RuntimeError(meta.stderr.strip() or "ffprobe failed")
    payload = json.loads(meta.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    fps = parse_frame_rate(str(video.get("avg_frame_rate") or "0/0"))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    frames = _run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame,pts_time,pkt_size",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    if frames.returncode != 0:
        raise RuntimeError(frames.stderr.strip() or "ffprobe frames failed")

    pts_list: list[float] = []
    rate_list: list[float] = []
    key_list: list[bool] = []
    for line in frames.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            key = int(float(parts[0])) == 1
            pts = float(parts[1])
        except ValueError:
            continue
        try:
            size = float(parts[2])
        except ValueError:
            size = 0.0
        pts_list.append(pts)
        rate_list.append(size)
        key_list.append(key)

    pts = np.asarray(pts_list, dtype=np.float64)
    if fps <= 0 and len(pts) >= 2:
        deltas = np.diff(pts)
        deltas = deltas[deltas > 1e-6]
        if len(deltas):
            fps = float(1.0 / np.median(deltas))
    if duration <= 0 and len(pts):
        frame_dur = 1.0 / fps if fps else 0.0
        duration = float(pts[-1] + frame_dur)

    return Probe(
        path=path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        has_audio=has_audio,
        pts=pts,
        rate=np.asarray(rate_list, dtype=np.float64),
        keyframe=np.asarray(key_list, dtype=bool),
    )
