"""Fuse pixel, histogram, luma, and bitrate curves into cut points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cutpoint.curves import Signals


@dataclass
class DetectConfig:
    thumb_width: int = 96
    top_crop: float = 0.08
    bottom_crop: float = 0.28
    z_threshold: float = 3.5
    pixel_floor: float = 18.0
    hist_floor: float = 0.12
    min_gap: float = 0.4
    min_shot: float = 0.4
    window_sec: float = 2.0
    flash_ratio: float = 0.35
    gop_std_max: float = 0.05
    adaptive_k: float = 6.0


@dataclass
class Cut:
    time: float
    frame: int
    type: str
    score: float


def detect_cuts(signals: Signals, cfg: DetectConfig | None = None) -> tuple[list[Cut], dict]:
    cfg = cfg or DetectConfig()
    count = signals.frame_count
    if count == 0:
        return [], {"gop_regular": False, "bitrate_bonus": False}

    fps = signals.fps if signals.fps > 0 else 30.0
    window = max(5, int(round(cfg.window_sec * fps)))
    z_hist = _robust_z(signals.hist, window)
    z_pixel = _robust_z(signals.pixel, window)
    score = 0.55 * z_hist + 0.45 * z_pixel

    gop_regular = _gop_regular(signals.pts, signals.keyframe, cfg.gop_std_max)
    bitrate_bonus = False
    if not gop_regular:
        bonus = _bitrate_bonus(signals.rate, signals.keyframe, window)
        if np.any(bonus):
            bitrate_bonus = True
            score = score + bonus

    hard = _hard_cuts(signals, score, window, cfg)
    fades, fade_spans = _fade_cuts(signals, cfg)
    fades.extend(_black_cuts(signals))
    hard = _drop_inside_fades(hard, fade_spans)
    cuts = _nms(hard + fades, cfg.min_gap)
    cuts = [c for c in cuts if 1.0 / fps < c.time < signals.duration - 1.0 / fps]
    cuts = _apply_min_shot(cuts, signals.duration, cfg.min_shot)
    cuts.sort(key=lambda c: c.time)

    intervals = _key_intervals(signals.pts, signals.keyframe)
    debug = {
        "gop_regular": gop_regular,
        "gop_interval_median": float(np.median(intervals)) if len(intervals) else None,
        "bitrate_bonus": bitrate_bonus,
        "frame_count": count,
        "pixel_floor": cfg.pixel_floor,
        "hist_floor": cfg.hist_floor,
        "z_threshold": cfg.z_threshold,
    }
    return cuts, debug


def result_dict(path, signals: Signals, cuts: list[Cut], debug: dict) -> dict:
    segments = []
    bounds = [0.0] + [c.time for c in cuts] + [float(signals.duration)]
    for index in range(len(bounds) - 1):
        segments.append(
            {
                "index": index,
                "start": _round_time(bounds[index]),
                "end": _round_time(bounds[index + 1]),
            }
        )
    return {
        "source": str(path),
        "duration": _round_time(signals.duration),
        "time_base": "pts",
        "width": signals.width,
        "height": signals.height,
        "fps": signals.fps,
        "cuts": [
            {
                "time": _round_time(c.time),
                "frame": int(c.frame),
                "type": c.type,
                "score": round(float(c.score), 3),
            }
            for c in cuts
        ],
        "segments": segments,
        "debug": debug,
    }


def analyze_video(path, cfg: DetectConfig | None = None) -> dict:
    from cutpoint.frames import read_signals
    from cutpoint.probe import probe

    cfg = cfg or DetectConfig()
    probed = probe(path)
    signals = read_signals(path, probed, cfg)
    if signals.duration <= 0:
        signals.duration = probed.duration
    cuts, debug = detect_cuts(signals, cfg)
    return result_dict(probed.path, signals, cuts, debug)


def _round_time(value: float) -> float:
    return round(float(value) + 0.0, 3)


def _robust_z(values: np.ndarray, window: int) -> np.ndarray:
    count = len(values)
    half = max(window // 2, 1)
    z = np.zeros(count, dtype=np.float64)
    for i in range(count):
        sample = values[max(0, i - half) : min(count, i + half + 1)]
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med)))
        scale = mad / 0.6745
        delta = float(values[i]) - med
        if scale < 1e-6:
            z[i] = 0.0 if abs(delta) < 1e-6 else 20.0
        else:
            z[i] = delta / scale
    return z


def _past_floor(values: np.ndarray, index: int, window: int, base: float, k: float) -> float:
    sample = values[max(0, index - window) : index]
    if len(sample) < 3:
        return base
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med))) / 0.6745
    return max(base, med + k * mad)


def _local_max(values: np.ndarray, index: int) -> bool:
    left = values[index - 1] if index > 0 else -1e9
    right = values[index + 1] if index + 1 < len(values) else -1e9
    return bool(values[index] >= left and values[index] >= right)


def _hard_cuts(signals: Signals, score: np.ndarray, window: int, cfg: DetectConfig) -> list[Cut]:
    cuts: list[Cut] = []
    pixel = signals.pixel
    hist = signals.hist
    for i in range(1, signals.frame_count - 1):
        if not _local_max(score, i):
            continue
        if score[i] < cfg.z_threshold:
            continue
        pixel_line = _past_floor(pixel, i, window, cfg.pixel_floor, cfg.adaptive_k)
        hist_line = _past_floor(hist, i, window, cfg.hist_floor, cfg.adaptive_k)
        if pixel[i] < pixel_line:
            continue
        # Histogram alone flickers when a flat color crosses a bin edge.
        # A large pixel jump can still be a cut when the palette barely changes.
        if hist[i] < hist_line and pixel[i] < max(pixel_line * 2.5, 70.0):
            continue
        if _spread_change(pixel, i):
            continue
        if _is_flash(signals, i, cfg):
            continue
        # A one-frame black separator belongs to the boundary. Start the new
        # shot on the first visible frame after it.
        if (
            signals.luma[i] < 0.08 * 255.0
            and i + 1 < signals.frame_count - 1
            and signals.luma[i + 1] > 40.0
        ):
            i += 1
        cuts.append(Cut(float(signals.pts[i]), i, "hard", float(score[i])))
    return cuts


def _spread_change(pixel: np.ndarray, index: int) -> bool:
    """A modest change spread over several frames is a dissolve or overlay, not a hard cut."""
    if index < 2 or index + 2 >= len(pixel):
        return False
    if pixel[index] >= 45:
        return False
    neighbor = pixel[index - 2] + pixel[index - 1] + pixel[index + 1] + pixel[index + 2]
    return bool(neighbor > 1.2 * pixel[index])


def _is_flash(signals: Signals, index: int, cfg: DetectConfig) -> bool:
    """A one-frame spike whose neighbors match is a flash, including its return frame."""
    pixel = signals.pixel
    persist = signals.persist

    def returned(frame: int, spike: float) -> bool:
        surround = persist[frame]
        return bool(
            surround >= 0
            and surround <= cfg.pixel_floor
            and surround < cfg.flash_ratio * max(spike, 1.0)
        )

    if returned(index, pixel[index]):
        return True
    previous = index - 1
    if previous > 0 and pixel[previous] >= cfg.pixel_floor and returned(previous, max(pixel[index], pixel[previous])):
        return True
    return False


def _fade_cuts(signals: Signals, cfg: DetectConfig) -> tuple[list[Cut], list[tuple[int, int]]]:
    luma = signals.luma
    count = len(luma)
    need = 0.40 * 255.0
    found: list[tuple[float, int, int, float]] = []
    for start in range(count):
        best: tuple[float, int, float] | None = None
        limit = min(count - 1, start + 45)
        for end in range(start + 8, limit + 1):
            delta = float(luma[end] - luma[start])
            if abs(delta) < need:
                continue
            window = luma[start : end + 1]
            span = float(window.max() - window.min())
            if span < 1e-6 or abs(delta) < 0.8 * span:
                continue
            diffs = np.diff(window)
            aligned = np.sum(diffs * np.sign(delta) >= -1.5) / len(diffs)
            if aligned < 0.8:
                continue
            after = luma[end : min(count, end + 5)]
            if len(after) >= 2 and abs(float(after[-1] - after[0])) > 0.25 * abs(delta):
                continue
            slope = abs(delta) / (end - start)
            if best is None or slope > best[0]:
                best = (slope, end, delta)
        if best is not None:
            found.append((best[0], start, best[1], best[2]))

    found.sort(key=lambda item: -item[0])
    chosen: list[tuple[int, int, float]] = []
    for _slope, start, end, delta in found:
        if any(not (end < s or start > e) for s, e, _d in chosen):
            continue
        chosen.append((start, end, delta))

    cuts: list[Cut] = []
    spans: list[tuple[int, int]] = []
    for start, end, delta in chosen:
        if end <= 1 or end >= count - 1:
            continue
        inner = luma[start : end + 1]
        # A single-frame jump with a flat tail is a hard cut, not a fade.
        if float(np.max(np.abs(np.diff(inner)))) > 0.55 * abs(delta):
            continue
        cuts.append(Cut(float(signals.pts[end]), end, "fade", abs(delta) / 255.0))
        spans.append((start, end))
    return cuts, spans


def _black_cuts(signals: Signals) -> list[Cut]:
    luma = signals.luma
    count = len(luma)
    level = 0.08 * 255.0
    cuts: list[Cut] = []
    i = 0
    while i < count:
        if luma[i] >= level:
            i += 1
            continue
        j = i
        while j < count and luma[j] < level:
            j += 1
        if j - i >= 3 and i > 0 and j < count and luma[i - 1] > 40 and luma[j] > 40:
            local = i + int(np.argmin(luma[i:j]))
            score = float(min(luma[i - 1], luma[j]) - luma[local]) / 255.0
            cuts.append(Cut(float(signals.pts[local]), local, "fade", score))
        i = j
    return cuts


def _drop_inside_fades(hard: list[Cut], spans: list[tuple[int, int]]) -> list[Cut]:
    if not spans:
        return hard
    kept = []
    for cut in hard:
        if any(start <= cut.frame <= end for start, end in spans):
            continue
        kept.append(cut)
    return kept


def _nms(cuts: list[Cut], min_gap: float) -> list[Cut]:
    kept: list[Cut] = []
    for cut in sorted(cuts, key=lambda c: -c.score):
        if all(abs(cut.time - other.time) >= min_gap for other in kept):
            kept.append(cut)
    kept.sort(key=lambda c: c.time)
    return kept


def _apply_min_shot(cuts: list[Cut], duration: float, min_shot: float) -> list[Cut]:
    cuts = list(cuts)
    changed = True
    while changed and cuts:
        changed = False
        bounds = [0.0] + [c.time for c in cuts] + [duration]
        lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        for index, length in enumerate(lengths):
            if length >= min_shot - 1e-3:
                continue
            if index == 0:
                cuts = cuts[1:]
            elif index == len(lengths) - 1:
                cuts = cuts[:-1]
            else:
                left = index - 1
                right = index
                drop = right if cuts[left].score >= cuts[right].score else left
                del cuts[drop]
            changed = True
            break
    return cuts


def _key_intervals(pts: np.ndarray, keyframe: np.ndarray) -> np.ndarray:
    times = pts[keyframe] if len(pts) == len(keyframe) else np.asarray([])
    if len(times) < 2:
        return np.asarray([])
    return np.diff(times)


def _gop_regular(pts: np.ndarray, keyframe: np.ndarray, std_max: float) -> bool:
    intervals = _key_intervals(pts, keyframe)
    if len(intervals) < 2:
        return False
    return float(np.std(intervals)) < std_max


def _bitrate_bonus(rate: np.ndarray, keyframe: np.ndarray, window: int) -> np.ndarray:
    bonus = np.zeros(len(rate), dtype=np.float64)
    if len(rate) != len(keyframe):
        return bonus
    for i in range(len(rate)):
        if keyframe[i]:
            continue
        sample = rate[max(0, i - window) : i]
        sample = sample[~keyframe[max(0, i - window) : i]] if len(sample) else sample
        if len(sample) < 3:
            continue
        med = float(np.median(sample))
        if med > 0 and rate[i] > 2.0 * med:
            bonus[i] = 0.3
    return bonus
