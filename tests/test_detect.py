from pathlib import Path

import numpy as np

from cutpoint.curves import Signals, histogram_distance
from cutpoint.detect import detect_cuts
from cutpoint.detect import analyze_video
from tests.media import encode_frames, solid


FPS = 30
WIDTH = 320
HEIGHT = 240


def _signals(luma, pixel=None, hist=None, persist=None, keyframe=None, rate=None) -> Signals:
    count = len(luma)
    return Signals(
        pts=np.arange(count, dtype=np.float64) / FPS,
        luma=np.asarray(luma, dtype=np.float64),
        pixel=np.zeros(count) if pixel is None else np.asarray(pixel, dtype=np.float64),
        hist=np.zeros(count) if hist is None else np.asarray(hist, dtype=np.float64),
        pixel_bottom=np.zeros(count),
        rate=np.zeros(count) if rate is None else np.asarray(rate, dtype=np.float64),
        keyframe=np.zeros(count, dtype=bool) if keyframe is None else np.asarray(keyframe, dtype=bool),
        persist=np.full(count, -1.0) if persist is None else np.asarray(persist, dtype=np.float64),
        fps=FPS,
        duration=count / FPS,
        width=WIDTH,
        height=HEIGHT,
        frame_count=count,
    )


def _near(cuts, expected, frames=1):
    times = [c.time for c in cuts]
    missing = [t for t in expected if not any(abs(found - t) <= frames / FPS for found in times)]
    assert not missing, f"missing {missing} in {times}"


def test_histogram_distance_identical_and_disjoint():
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[:, :] = (255, 0, 0)
    blue = np.zeros_like(red)
    blue[:, :] = (0, 0, 255)
    assert histogram_distance(red, red) < 1e-6
    assert histogram_distance(red, blue) > 0.9


def test_flash_is_rejected_and_real_cut_is_kept():
    count = 90
    pixel = np.zeros(count)
    hist = np.zeros(count)
    pixel[45] = 180
    hist[45] = 0.9
    flash_persist = np.full(count, -1.0)
    flash_persist[45] = 3.0
    flash, _ = detect_cuts(_signals(np.full(count, 40.0), pixel, hist, flash_persist))
    assert flash == []

    cut_persist = np.full(count, -1.0)
    cut_persist[45] = 170.0
    cuts, _ = detect_cuts(_signals(np.full(count, 40.0), pixel, hist, cut_persist))
    assert len(cuts) == 1
    assert cuts[0].type == "hard"
    assert cuts[0].frame == 45


def test_black_frame_between_different_shots_is_kept():
    count = 90
    pixel = np.zeros(count)
    hist = np.zeros(count)
    persist = np.full(count, -1.0)
    luma = np.full(count, 120.0)
    luma[40] = 0.0
    pixel[40] = 200
    hist[40] = 1.0
    persist[40] = 80
    pixel[41] = 150
    hist[41] = 1.0
    persist[41] = 150
    cuts, _ = detect_cuts(_signals(luma, pixel, hist, persist))
    assert any(c.frame == 41 and c.type == "hard" for c in cuts)


def test_same_scene_black_frame_is_dropped():
    count = 90
    pixel = np.zeros(count)
    hist = np.zeros(count)
    persist = np.full(count, -1.0)
    pixel[40] = 200
    hist[40] = 1.0
    persist[40] = 4
    pixel[41] = 180
    hist[41] = 1.0
    persist[41] = 170
    cuts, _ = detect_cuts(_signals(np.full(count, 120.0), pixel, hist, persist))
    assert cuts == []


def test_histogram_flicker_without_pixel_change_is_ignored():
    count = 90
    pixel = np.zeros(count)
    hist = np.zeros(count)
    hist[45] = 0.8
    pixel[45] = 6
    persist = np.full(count, -1.0)
    persist[45] = 6
    cuts, _ = detect_cuts(_signals(np.full(count, 140.0), pixel, hist, persist))
    assert cuts == []


def test_spread_out_overlay_is_not_a_hard_cut():
    count = 90
    pixel = np.zeros(count)
    hist = np.zeros(count)
    persist = np.full(count, -1.0)
    pixel[43:48] = [2, 12, 23, 22, 11]
    hist[43:48] = [0.05, 0.4, 0.7, 0.65, 0.3]
    persist[45] = 40
    cuts, _ = detect_cuts(_signals(np.full(count, 140.0), pixel, hist, persist))
    assert cuts == []


def test_luma_ramp_is_a_fade():
    count = 90
    luma = np.zeros(count)
    luma[30:45] = np.linspace(0, 255, 15)
    luma[45:] = 255
    cuts, _ = detect_cuts(_signals(luma))
    fades = [c for c in cuts if c.type == "fade"]
    assert fades, cuts
    assert any(abs(c.frame - 44) <= 2 for c in fades)
    assert not any(c.type == "hard" for c in cuts)


def test_black_dip_is_a_fade():
    count = 90
    luma = np.full(count, 180.0)
    luma[40:48] = 5
    cuts, _ = detect_cuts(_signals(luma))
    fades = [c for c in cuts if c.type == "fade"]
    assert any(40 <= c.frame <= 47 for c in fades)


def test_encoded_hard_cuts(tmp_path: Path):
    colors = [(220, 20, 20), (20, 180, 40), (20, 40, 220), (230, 210, 20), (180, 20, 180)]
    frames = []
    for color in colors:
        frames.extend(solid(WIDTH, HEIGHT, color, FPS * 2))
    path = tmp_path / "hard.mp4"
    encode_frames(path, frames)
    payload = analyze_video(path)
    times = [c["time"] for c in payload["cuts"]]
    assert len(payload["cuts"]) == 4, payload["cuts"]
    assert all(c["type"] == "hard" for c in payload["cuts"])
    _near_times(times, [2, 4, 6, 8])


def test_encoded_flash_is_ignored(tmp_path: Path):
    frames = solid(WIDTH, HEIGHT, (180, 30, 30), 90)
    frames[45] = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    path = tmp_path / "flash.mp4"
    encode_frames(path, frames)
    payload = analyze_video(path)
    assert payload["cuts"] == [], payload["cuts"]


def test_encoded_bottom_caption_is_ignored(tmp_path: Path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 40, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    base[int(HEIGHT * 0.82) :, :] = (10, 10, 10)
    frames = [base.copy() for _ in range(FPS * 3)]
    for frame in frames[FPS:]:
        frame[int(HEIGHT * 0.88) :, :] = (240, 240, 240)
    path = tmp_path / "caption.mp4"
    encode_frames(path, frames)
    payload = analyze_video(path)
    assert payload["cuts"] == [], payload["cuts"]


def test_regular_gop_on_a_static_shot_is_not_a_cut(tmp_path: Path):
    frames = solid(WIDTH, HEIGHT, (40, 90, 140), FPS * 4)
    path = tmp_path / "static.mp4"
    encode_frames(path, frames, gop=FPS)
    payload = analyze_video(path)
    assert payload["cuts"] == [], payload["cuts"]
    assert payload["debug"]["gop_regular"] is True
    assert payload["debug"]["bitrate_bonus"] is False


def _near_times(found, expected, frames=1):
    missing = [t for t in expected if not any(abs(item - t) <= frames / FPS + 1e-3 for item in found)]
    assert not missing, f"missing {missing} in {found}"
