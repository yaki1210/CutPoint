import os
from pathlib import Path

import pytest

from cutpoint.detect import analyze_video

SAMPLE = Path(
    os.environ.get(
        "CUTPOINT_SAMPLE",
        Path(__file__).resolve().parents[1] / "202611010340字幕版.mp4",
    )
)
# 4.567 is the first frame of the new slide after a one-frame black separator
# (the separator itself is at 4.533). 49.267 is a black flash inside one slide;
# that slide changes at 49.467. 81.167 is a hard cut onto the end card.
KNOWN_HARD_CUTS = (4.567, 49.467, 81.167)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample video is not in the workspace")
def test_sample_finds_obvious_hard_cuts_without_the_gop_grid():
    payload = analyze_video(SAMPLE)
    times = [cut["time"] for cut in payload["cuts"]]
    assert payload["debug"]["frame_count"] == 2891
    assert payload["debug"]["gop_regular"] is True
    assert payload["debug"]["bitrate_bonus"] is False
    assert len(times) < 40, times
    missing = [t for t in KNOWN_HARD_CUTS if not any(abs(found - t) <= 2 / 30 for found in times)]
    assert not missing, f"missing {missing} in {times}"
