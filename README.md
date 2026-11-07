# CutPoint

Detect shot boundaries from pixel, color-histogram, brightness, and bitrate curves, then split the video at those cuts.

输入一个视频，输出剪辑点，也可以按这些点切成片段。

## 安装

需要 Python 3.11 或更高版本，以及能在命令行里调用的 `ffmpeg` 和 `ffprobe`。

```bash
pip install -e .
```

只跑检测时的依赖是 NumPy 和 Pillow。`--review` 用 Pillow 拼对照图。

## 用法

```bash
cutpoint detect input.mp4 -o cuts.json --review out/review
cutpoint split  input.mp4 -o clips --cuts cuts.json
cutpoint run    input.mp4 -o clips --review out/review
```

`detect` 写 JSON。`split` 和 `run` 再写出片段。

`split` / `run` 默认 `--mode accurate`：按时间重编码（libx264 CRF 18），切点可以落在任意帧上。`--mode copy` 直接拷流，速度快、画质与源一致，起点会吸附到关键帧。

没有剪辑点时输出整段拷贝，不产生空文件。

常用参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--pixel-floor` | 18 | 中部像素差下限（0–255） |
| `--hist-floor` | 0.12 | 颜色直方图距离下限 |
| `--z-threshold` | 3.5 | 相对本段画面的异常程度 |
| `--min-gap` | 0.4 | 两个剪辑点的最小间隔（秒） |
| `--min-shot` | 0.4 | 短于这个长度的片段会并进相邻段 |

## 检测

每帧四条曲线：

- **像素差**和 **8×8×8 颜色直方图**，只在画面中部计算（去掉顶部 8% 和底部 28%），用来躲开烧录字幕。
- **亮度**，用来找淡变和黑场。
- **码率 / 关键帧**。GOP 间隔几乎固定时不加分，避免把「每秒一个 I 帧」当成剪辑点。

硬切要同时满足：像素差过线，颜色分布也变了或者像素差本身很大，并且前后画面没有马上回到原样。单帧闪光、同镜头里铺开的叠字，不会当成剪辑点。两段画面之间如果只隔了一帧黑场，剪辑点落在黑场后面的第一帧。

`cuts.json` 里每个剪辑点有 `time`、`frame`、`type`（`hard` 或 `fade`）和 `score`。`segments` 由这些点切出，首段从 0 开始，末段到片长。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

合成片测试用 ffmpeg 现做。仓库不附带样片。若要跑真实成片断言，把视频放在仓库根目录并命名为 `202611010340字幕版.mp4`，或设置 `CUTPOINT_SAMPLE` 指向它。没有这个文件时，该项测试会跳过。

## 许可证

[MIT](LICENSE)
