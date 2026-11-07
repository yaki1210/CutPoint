"""Per-frame luma, pixel difference, and color-histogram distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Signals:
    pts: np.ndarray
    luma: np.ndarray
    pixel: np.ndarray
    hist: np.ndarray
    pixel_bottom: np.ndarray
    rate: np.ndarray
    keyframe: np.ndarray
    persist: np.ndarray
    fps: float
    duration: float
    width: int
    height: int
    frame_count: int


def scaled_size(width: int, height: int, thumb_width: int = 96) -> tuple[int, int]:
    """Width fixed, height rounded to a multiple of 2, matching ffmpeg scale=W:-2."""
    thumb_width = max(int(thumb_width), 16)
    if width <= 0 or height <= 0:
        return thumb_width, thumb_width
    raw = height * thumb_width / width
    thumb_h = int(round(raw / 2.0) * 2)
    return thumb_width, max(thumb_h, 2)


def region_slices(height: int, top_crop: float, bottom_crop: float) -> tuple[int, int]:
    top = int(round(height * top_crop))
    bot = int(round(height * (1.0 - bottom_crop)))
    if bot - top < 8:
        return 0, height
    return top, bot


def rec709_luma(image: np.ndarray) -> float:
    plane = image.astype(np.float32, copy=False)
    y = 0.2126 * plane[:, :, 0] + 0.7152 * plane[:, :, 1] + 0.0722 * plane[:, :, 2]
    return float(y.mean())


def pixel_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.int16, copy=False) - right.astype(np.int16, copy=False))))


def histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Chi-square distance of 8x8x8 RGB histograms. 0 is identical, 1 is disjoint."""

    def bins(image: np.ndarray) -> np.ndarray:
        q = np.right_shift(image, 5)
        idx = (q[:, :, 0].astype(np.int32) * 64) + (q[:, :, 1].astype(np.int32) * 8) + q[:, :, 2]
        return np.bincount(idx.ravel(), minlength=512).astype(np.float64)

    ha = bins(left)
    hb = bins(right)
    ha /= ha.sum() or 1.0
    hb /= hb.sum() or 1.0
    return float(0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-10)))
