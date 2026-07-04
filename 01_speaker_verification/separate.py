# -*- coding: utf-8 -*-
"""用 Demucs(htdemucs) 剥离背景音乐/伴奏，只保留人声。

为什么自己写：新版 torchaudio 的 save() 需要 torchcodec（未装），demucs 命令行在
保存阶段会报错。这里改用 demucs 的 Python API 做分离，再用 soundfile 写出，规避该问题。

用法：
    python separate.py <src>            # src 是目录：镜像 src/*/*.wav -> audio_vocals/...
    python separate.py <src.wav> <out>  # src 是单文件：输出到 out/<basename>
"""
import os
import sys
import glob
import time

import numpy as np
import soundfile as sf
import subprocess

import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(max(1, os.cpu_count() or 1))

src_arg = sys.argv[1] if len(sys.argv) > 1 else "audio"
out_arg = sys.argv[2] if len(sys.argv) > 2 else "audio_vocals"
SRC = os.path.join(ROOT, src_arg)
OUT = os.path.join(ROOT, out_arg)

print("加载 Demucs htdemucs 模型……", flush=True)
model = get_model("htdemucs")
model.eval()
SR = model.samplerate                    # 44100
voc_idx = model.sources.index("vocals")
print(f"模型就绪。sources={model.sources}, 采样率={SR}\n", flush=True)


MAX_SEP_SEC = 1800.0   # 每个文件最多分离这么长（秒）=30min，避免超长录像吃爆内存
SEP_WINDOWS = 6        # 超长文件：拆成 N 个等长小窗(这里 6×5min)、均匀铺在整段时间轴上再拼接


def _ff_duration(path):
    """ffprobe 取时长（秒）。比 librosa/audioread 快且稳，不会卡在超长文件上。"""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", path],
                             capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _ff_window(path, offset, dur):
    """用 ffmpeg 快速 seek 抽 [offset, offset+dur] -> (2, N) float32 @ SR。
    -ss 是高效定位（不从头解码），对超长文件也快、内存可控；在独立子进程里跑，坏文件不拖垮主程序。"""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{offset:.3f}", "-i", path,
         "-t", f"{dur:.3f}", "-ac", "2", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        capture_output=True).stdout
    a = np.frombuffer(out, dtype=np.float32)
    if a.size < 2:
        return np.zeros((2, 0), dtype=np.float32)
    return a.reshape(-1, 2).T.copy()                           # (2, N)，交错 LR -> 双声道


def load_wav(path):
    """任意音频 -> (2, N) torch 张量 @ 模型采样率、立体声。全程用 ffmpeg，不走 audioread。
    超长文件取 SEP_WINDOWS 个小窗、均匀铺在整段(5%~95%)上拼接，覆盖整场直播。"""
    dur = _ff_duration(path)
    if dur <= 0 or dur <= MAX_SEP_SEC:
        y = _ff_window(path, 0.0, MAX_SEP_SEC if dur <= 0 else dur)
    else:
        win = MAX_SEP_SEC / SEP_WINDOWS
        offsets = np.linspace(dur * 0.05, dur * 0.95 - win, SEP_WINDOWS)
        y = np.concatenate([_ff_window(path, float(o), win) for o in offsets], axis=1)
    return torch.from_numpy(y).float()


def separate_vocals(wav):
    """返回人声轨 (2, N)。按 demucs 标准做均值/方差归一化再还原。"""
    ref = wav.mean(0)
    w = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, w[None], device="cpu", split=True, overlap=0.25)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    return out[voc_idx]


# 收集待处理文件（多格式；输出统一为 .wav 人声）
AUDIO_EXT = ("wav", "m4a", "mp3", "flac", "ogg", "aac", "opus", "webm", "wma")
if SRC.lower().endswith(tuple("." + e for e in AUDIO_EXT)):
    files = [SRC]
    def out_for(f):
        return os.path.join(OUT, os.path.splitext(os.path.basename(f))[0] + ".wav")
else:
    files = []
    for e in AUDIO_EXT:
        files += glob.glob(os.path.join(SRC, "*", "*." + e))
    files = sorted(set(files))
    def out_for(f):
        rel = os.path.relpath(f, SRC)
        return os.path.join(OUT, os.path.splitext(rel)[0] + ".wav")

print(f"待处理 {len(files)} 个文件\n", flush=True)
t_all = time.time()
for f in files:
    outp = out_for(f)
    rel = os.path.relpath(f, ROOT)
    if os.path.exists(outp):                    # 已分离过则跳过，便于断点续跑
        print(f"  - 跳过(已存在) {rel}", flush=True)
        continue
    try:
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        t0 = time.time()
        wav = load_wav(f)
        voc = separate_vocals(wav)
        sf.write(outp, voc.T.cpu().numpy(), SR)
        print(f"  ✓ {rel}  ({wav.shape[1] / SR:.0f}s 音频 -> 用时 {time.time() - t0:.0f}s)", flush=True)
    except Exception as e:                        # 单文件失败不拖垮整批
        print(f"  ! 失败跳过 {rel}: {e}", flush=True)

print(f"\n全部完成，总用时 {time.time() - t_all:.0f}s。人声输出目录：{OUT}", flush=True)
