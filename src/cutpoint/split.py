"""Split a video into one file per detected segment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cutpoint.probe import ffmpeg_exe, probe


def split_video(
    source: str | Path,
    segments: list[dict],
    out_dir: str | Path,
    mode: str = "accurate",
) -> list[Path]:
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode not in {"accurate", "copy"}:
        raise ValueError(f"unknown split mode: {mode}")
    if not segments:
        raise ValueError("no segments to write")

    if len(segments) == 1:
        dest = out_dir / _name(0, segments[0]["start"], segments[0]["end"])
        shutil.copy2(source, dest)
        return [dest]

    has_audio = probe(source).has_audio
    written: list[Path] = []
    for segment in segments:
        dest = out_dir / _name(segment["index"], segment["start"], segment["end"])
        _cut_one(
            source,
            dest,
            float(segment["start"]),
            float(segment["end"]),
            mode=mode,
            has_audio=has_audio,
        )
        written.append(dest)
    return written


def _name(index: int, start: float, end: float) -> str:
    return f"{int(index):03d}_{float(start):.3f}-{float(end):.3f}.mp4"


def _cut_one(
    source: Path,
    dest: Path,
    start: float,
    end: float,
    mode: str,
    has_audio: bool,
) -> None:
    duration = max(end - start, 0.0)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    if mode == "copy":
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "160k"]
        else:
            cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(dest)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError("ffmpeg split failed:\n" + "\n".join(tail))
