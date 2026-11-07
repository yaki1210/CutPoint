"""Contact sheets of the frame before a cut and the frame after it."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cutpoint.probe import ffmpeg_exe


def export_review(source: str | Path, cuts: list[dict], out_dir: str | Path) -> Path:
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
    if not cuts:
        return out_dir

    frames = sorted({max(int(c["frame"]) - 1, 0) for c in cuts} | {int(c["frame"]) for c in cuts})
    raw_dir = out_dir / "_frames"
    raw_dir.mkdir(exist_ok=True)
    for stale in raw_dir.glob("*.png"):
        stale.unlink()
    _extract(source, frames, raw_dir)

    pairs: list[Image.Image] = []
    for index, cut in enumerate(cuts):
        prev = max(int(cut["frame"]) - 1, 0)
        left = Image.open(raw_dir / f"{prev:06d}.png").convert("RGB")
        right = Image.open(raw_dir / f"{int(cut['frame']):06d}.png").convert("RGB")
        pair = _pair(left, right, f"{index:02d}  {cut['time']:.3f}s  {cut['type']}  {cut['score']:.1f}")
        pair.save(out_dir / f"{index:03d}_{cut['time']:.3f}.jpg", quality=85)
        pairs.append(pair)

    montage = _grid(pairs)
    montage_path = out_dir / "montage.jpg"
    montage.save(montage_path, quality=85)
    return montage_path


def _extract(source: Path, frames: list[int], dest: Path) -> None:
    if not frames:
        return
    select = "+".join(f"eq(n\\,{frame})" for frame in frames)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-vf",
        f"select={select},scale=240:-2",
        "-vsync",
        "vfr",
        str(dest / "%06d.png"),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError("ffmpeg review extract failed:\n" + "\n".join(tail))
    written = sorted(dest.glob("*.png"))
    if len(written) != len(frames):
        raise RuntimeError(f"expected {len(frames)} review frames, got {len(written)}")
    temps = []
    for index, image_path in enumerate(written):
        temp = dest / f"_tmp_{index:06d}.png"
        image_path.replace(temp)
        temps.append(temp)
    for temp, frame in zip(temps, frames):
        temp.replace(dest / f"{frame:06d}.png")


def _pair(left: Image.Image, right: Image.Image, caption: str) -> Image.Image:
    gap = 4
    caption_h = 22
    width = left.width + right.width + gap
    height = max(left.height, right.height) + caption_h
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    canvas.paste(left, (0, caption_h))
    canvas.paste(right, (left.width + gap, caption_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), caption, fill=(240, 240, 240))
    return canvas


def _grid(pairs: list[Image.Image], columns: int = 2) -> Image.Image:
    if not pairs:
        return Image.new("RGB", (16, 16))
    cell_w = max(im.width for im in pairs)
    cell_h = max(im.height for im in pairs)
    rows = int(np.ceil(len(pairs) / columns))
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), (0, 0, 0))
    for index, image in enumerate(pairs):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        canvas.paste(image, (x, y))
    return canvas
