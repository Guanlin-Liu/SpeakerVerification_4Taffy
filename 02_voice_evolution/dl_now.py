# -*- coding: utf-8 -*-
"""下载 "现在" 锚点：2026-07 两场(07-02 + 07-03)多切片，存入 audio_evo_vocals/NOW/ → 2 session 平均，更稳健。"""
import sys, os
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
import run_pipeline as rp
import add_slices as asl

NOW_VIDEOS = [("BV1w9TB6KE9s", 18600),   # 2026-07-02 哥特新衣回/看世界奇妙物语（已下 e0-e2）
              ("BV1ohTJ6MEZy", 17827)]   # 2026-07-03 原啊躲猫猫/人森（偏早采样避开靠后的 VRC 连麦）
FRACS = [0.08, 0.16, 0.24, 0.32]         # 都在 P1 内
L = 240

VOCDIR = os.path.join(rp.VOC, "NOW")
RAWDIR = os.path.join(rp.RAW, "NOW")
os.makedirs(VOCDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)
rp.get_sep_model()

done = 0
for BV, DUR in NOW_VIDEOS:
    for k, frac in enumerate(FRACS):
        voc_out = os.path.join(VOCDIR, f"{BV}_e{k}_vocals.wav")
        if os.path.exists(voc_out):
            print("skip", BV, k, flush=True)
            continue
        start = max(300, int(frac * DUR))
        raw = os.path.join(RAWDIR, f"{BV}_e{k}.wav")
        got = asl.dl(BV, start, L, raw)
        if not got:
            print(f"!! fail {BV}_e{k}", flush=True)
            continue
        try:
            ok = rp.separate(got, voc_out)
        except Exception as ex:
            print(f"!! sep fail {BV}_e{k} {ex!r}", flush=True)
            ok = False
        if ok:
            try:
                os.remove(got)
            except OSError:
                pass
            done += 1
            print(f"== OK {BV}_e{k}", flush=True)
print(f"=== dl_now DONE {done} new slices ===", flush=True)
