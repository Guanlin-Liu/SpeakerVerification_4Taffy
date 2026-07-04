# -*- coding: utf-8 -*-
"""给 manifest 里每个视频，在已有 mid 切片之外，再抽 2 段(早~0.18 / 晚~0.78)非重叠切片，
降低 within-session 抽样方差。命名 <BVID>_e{k}_vocals.wav，与 mid(<BVID>_vocals.wav) 同属一个 session。
复用 run_pipeline 的下载/分离逻辑。幂等：已存在的切片跳过。用法: python add_slices.py"""
import sys, os, subprocess

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
import run_pipeline as rp   # 复用 YTDLP/FFMPEG_ARGS/COOKIE_ARGS/separate/get_sep_model/read_manifest/etag/RAW/VOC

L = 240                     # 每段 4min
FRACS = [0.18, 0.78]        # 早 / 晚，与已有 mid(~0.35) 错开


def dl(bvid, start, length, out_wav):
    raw_dir = os.path.dirname(out_wav)
    stem = os.path.basename(out_wav)[:-4]
    for e in ("wav", "m4a", "m4a.part", "wav.part"):
        fp = os.path.join(raw_dir, stem + "." + e)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
    tmpl = os.path.join(raw_dir, stem + ".%(ext)s")
    cmd = [rp.YTDLP, "-f", "ba", "--downloader", "ffmpeg",
           "--download-sections", f"*{start}-{start+length}",
           "-x", "--audio-format", "wav", *rp.FFMPEG_ARGS,
           *rp.COOKIE_ARGS, "--no-playlist", "-o", tmpl,
           f"https://www.bilibili.com/video/{bvid}"]
    subprocess.run(cmd)
    if os.path.exists(out_wav) and os.path.getsize(out_wav) >= 4096:
        return out_wav
    if os.path.exists(out_wav):
        os.remove(out_wav)
    return None


def main():
    rows = rp.read_manifest()
    rp.get_sep_model()
    done = skip = fail = 0
    for i, r in enumerate(rows, 1):
        bvid, ep, dur = r["bvid"], rp.etag(r["quarter"]), r["dur"]
        raw_dir = os.path.join(rp.RAW, ep)
        voc_dir = os.path.join(rp.VOC, ep)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(voc_dir, exist_ok=True)
        for k, frac in enumerate(FRACS):
            voc_out = os.path.join(voc_dir, f"{bvid}_e{k}_vocals.wav")
            if os.path.exists(voc_out):
                skip += 1
                continue
            start = max(300, min(int(frac * dur), dur - L - 120))
            raw_wav = os.path.join(raw_dir, f"{bvid}_e{k}.wav")
            got = dl(bvid, start, L, raw_wav)
            if not got and start > 600:                 # 多P兜底：回退到 P1 内，两段错开
                got = dl(bvid, 630 + k * (L + 30), L, raw_wav)
            if not got:
                fail += 1
                print(f"!! fail {bvid}_e{k}", flush=True)
                continue
            try:
                ok = rp.separate(got, voc_out)
            except Exception as ex:
                print(f"!! sep fail {bvid}_e{k} {ex!r}", flush=True)
                ok = False
            if ok:
                try:
                    os.remove(got)
                except OSError:
                    pass
                done += 1
            else:
                fail += 1
        if i % 10 == 0:
            print(f"...{i}/{len(rows)}  done={done} skip={skip} fail={fail}", flush=True)
    print(f"=== add_slices DONE  done={done} skip={skip} fail={fail} ===", flush=True)


if __name__ == "__main__":
    main()
