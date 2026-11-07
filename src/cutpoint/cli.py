"""Command line: detect cut points and optionally split the video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cutpoint.detect import DetectConfig, analyze_video
from cutpoint.review import export_review
from cutpoint.split import split_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cutpoint", description="用像素差、直方图、亮度和码率曲线找剪辑点")
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect", help="输出剪辑点 JSON")
    _add_detect_args(detect)
    detect.add_argument("--review", type=Path, default=None, help="导出每个剪辑点的前后帧对照")

    split = sub.add_parser("split", help="按已有 JSON 切分视频")
    split.add_argument("input", type=Path)
    split.add_argument("-o", "--output", type=Path, required=True)
    split.add_argument("--cuts", type=Path, required=True)
    split.add_argument("--mode", choices=("accurate", "copy"), default="accurate")

    run = sub.add_parser("run", help="检测并切分")
    _add_detect_args(run)
    run.add_argument("--mode", choices=("accurate", "copy"), default="accurate")
    run.add_argument("--review", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "detect":
            payload = _detect(args)
            if args.review:
                export_review(args.input, payload["cuts"], args.review)
            _print_summary(payload)
        elif args.cmd == "split":
            payload = json.loads(args.cuts.read_text(encoding="utf-8"))
            written = split_video(args.input, payload["segments"], args.output, args.mode)
            print(f"wrote {len(written)} segments to {args.output}")
        else:
            payload = _detect(args)
            if args.review:
                export_review(args.input, payload["cuts"], args.review)
            written = split_video(args.input, payload["segments"], args.output, args.mode)
            _print_summary(payload)
            print(f"wrote {len(written)} segments to {args.output}")
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _add_detect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="剪辑点 JSON，run 时是输出目录")
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--pixel-floor", type=float, default=18.0)
    parser.add_argument("--hist-floor", type=float, default=0.12)
    parser.add_argument("--min-gap", type=float, default=0.4)
    parser.add_argument("--min-shot", type=float, default=0.4)


def _config(args) -> DetectConfig:
    return DetectConfig(
        z_threshold=args.z_threshold,
        pixel_floor=args.pixel_floor,
        hist_floor=args.hist_floor,
        min_gap=args.min_gap,
        min_shot=args.min_shot,
    )


def _detect(args) -> dict:
    payload = analyze_video(args.input, _config(args))
    if args.cmd == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        dest = args.output / "cuts.json"
    else:
        dest = args.output
        dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return payload


def _print_summary(payload: dict) -> None:
    debug = payload.get("debug") or {}
    print(
        f"{len(payload['cuts'])} cuts, {len(payload['segments'])} segments, "
        f"gop_regular={debug.get('gop_regular')}"
    )
    for cut in payload["cuts"]:
        print(f"  {cut['time']:8.3f}  {cut['type']:<4}  {cut['score']:7.2f}  frame {cut['frame']}")
