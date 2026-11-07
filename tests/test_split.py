from pathlib import Path

from cutpoint.detect import analyze_video
from cutpoint.probe import probe
from cutpoint.split import split_video
from tests.media import encode_frames, solid

FPS = 30
WIDTH = 320
HEIGHT = 240


def test_accurate_split_matches_segment_duration(tmp_path: Path):
    colors = [(220, 20, 20), (20, 180, 40), (20, 40, 220), (230, 210, 20), (180, 20, 180)]
    frames = []
    for color in colors:
        frames.extend(solid(WIDTH, HEIGHT, color, FPS * 2))
    source = tmp_path / "hard.mp4"
    encode_frames(source, frames)
    payload = analyze_video(source)
    assert len(payload["segments"]) == 5

    out = tmp_path / "clips"
    written = split_video(source, payload["segments"], out, mode="accurate")
    assert len(written) == 5
    for path, segment in zip(written, payload["segments"]):
        info = probe(path)
        expected = segment["end"] - segment["start"]
        assert abs(info.duration - expected) <= 1.5 / FPS, (path.name, info.duration, expected)
