# -*- coding: utf-8 -*-
"""Taffy 声音演化 · 数据流水线：按 manifest 逐场下载中段切片(wav) -> Demucs 人声分离(soundfile 写盘)。
用 voicerec 的 python 运行（内含 demucs），下载走 yt-dlp.exe 子进程。
分离用 demucs Python API + soundfile 写盘，绕开新版 torchaudio 依赖 torchcodec 的坑；模型只加载一次。
用法:
    python run_pipeline.py all              # 跑全部
    python run_pipeline.py 2021Q3           # 只跑某季度(epoch tag)
    python run_pipeline.py one BV19f4y1V7tM # 只跑单场（验证用）
幂等：已存在 vocals 输出的场次自动跳过。
"""
import os, sys, time, shutil, subprocess
import soundfile as sf
import torch
import torchaudio.functional as AF
from demucs.pretrained import get_model
from demucs.apply import apply_model

SCRATCH = os.path.dirname(os.path.abspath(__file__))          # 02_voice_evolution/
MANIFEST = os.path.join(SCRATCH, "manifest_main.tsv")
ROOT = os.path.dirname(SCRATCH)                                # repo root
RAW = os.path.join(ROOT, "audio_evo")
VOC = os.path.join(ROOT, "audio_evo_vocals")
YTDLP = os.environ.get("YTDLP") or shutil.which("yt-dlp") or "yt-dlp"
_FF = shutil.which("ffmpeg")
FFDIR = os.path.dirname(_FF) if _FF else ""
COOKIES = os.environ.get("BILI_COOKIES", "")   # Netscape cookies.txt 路径；B 站合集/会员清晰度可能需要
FFMPEG_ARGS = ["--ffmpeg-location", FFDIR] if FFDIR else []
COOKIE_ARGS = ["--cookies", COOKIES] if COOKIES else []


def log(*a):
    print(*a, flush=True)


def read_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            rows.append(dict(quarter=p[0], date=p[1], bvid=p[2], dur=int(p[3]),
                             start=int(p[4]), length=int(p[5]), title=p[6]))
    return rows


def etag(quarter):
    return quarter.replace("-", "")  # 2021-Q3 -> 2021Q3


def download_slice(bvid, start, length, raw_dir):
    """只下中段 [start, start+length] 的音频并转 wav。返回 wav 路径或 None。"""
    out_tmpl = os.path.join(raw_dir, bvid + ".%(ext)s")
    # 清掉残留旧文件，否则 yt-dlp 见目标已存在会跳过下载
    for e in ("wav", "m4a", "m4a.part", "wav.part"):
        fp = os.path.join(raw_dir, bvid + "." + e)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
    end = start + length
    cmd = [YTDLP, "-f", "ba", "--downloader", "ffmpeg",
           "--download-sections", f"*{start}-{end}",
           "-x", "--audio-format", "wav",
           *FFMPEG_ARGS, *COOKIE_ARGS,
           "--no-playlist", "-o", out_tmpl,
           f"https://www.bilibili.com/video/{bvid}"]
    log(">> yt-dlp", bvid, f"[{start}-{end}]")
    r = subprocess.run(cmd)
    wav = os.path.join(raw_dir, bvid + ".wav")
    if r.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) >= 4096:
        return wav
    if os.path.exists(wav) and os.path.getsize(wav) < 4096:
        os.remove(wav)                       # 空切片(多P offset 越界等)，删掉
        log("!! empty slice", bvid)
    log("!! download failed", bvid, "rc=", r.returncode)
    return None


_MODEL = None
def get_sep_model():
    global _MODEL
    if _MODEL is None:
        log("loading htdemucs model (once) ...")
        m = get_model("htdemucs")
        m.eval()
        _MODEL = m
    return _MODEL


def separate(raw_wav, out_path):
    """demucs 人声分离，vocals 直接用 soundfile 写到 out_path。返回 True/False。"""
    model = get_sep_model()
    data, sr = sf.read(raw_wav, dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(data.T)                                # [C, T]
    ch = model.audio_channels
    if wav.shape[0] == 1 and ch == 2:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > ch:
        wav = wav[:ch]
    if sr != model.samplerate:
        wav = AF.resample(wav, sr, model.samplerate)
    ref = wav.mean(0)
    wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav_n[None], device="cpu", split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    vocals = sources[model.sources.index("vocals")]               # [C, T]
    sf.write(out_path, vocals.T.numpy(), model.samplerate)
    return os.path.exists(out_path)


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    only_bv = sys.argv[2] if (sel == "one" and len(sys.argv) > 2) else None
    rows = read_manifest()

    if sel == "one":
        rows = [r for r in rows if r["bvid"] == only_bv]
    elif sel != "all":
        rows = [r for r in rows if etag(r["quarter"]) == sel]

    log(f"=== pipeline start: {len(rows)} streams, sel={sel} ===")
    done, skipped, failed = [], [], []
    t_all = time.time()

    for i, r in enumerate(rows, 1):
        bvid, ep = r["bvid"], etag(r["quarter"])
        raw_dir = os.path.join(RAW, ep)
        voc_dir = os.path.join(VOC, ep)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(voc_dir, exist_ok=True)
        voc_out = os.path.join(voc_dir, f"{bvid}_vocals.wav")

        log(f"\n--- [{i}/{len(rows)}] {ep} {r['date']} {bvid} ---")
        if os.path.exists(voc_out):
            log("== skip (vocals exist)")
            skipped.append(bvid)
            continue

        t0 = time.time()
        raw_wav = os.path.join(raw_dir, bvid + ".wav")
        if (not os.path.exists(raw_wav)) or os.path.getsize(raw_wav) < 4096:
            raw_wav = download_slice(bvid, r["start"], r["length"], raw_dir)
            if not raw_wav and r["start"] > 600:          # 多P/offset 越界兜底：回退到中前段
                raw_wav = download_slice(bvid, 600, r["length"], raw_dir)
        if not raw_wav:
            failed.append((bvid, "download"))
            continue

        try:
            ok = separate(raw_wav, voc_out)
        except Exception as e:
            log("!! demucs exception", bvid, repr(e))
            ok = False
        if not ok:
            failed.append((bvid, "demucs"))
            continue

        try:
            os.remove(raw_wav)  # 省磁盘，只留 vocals
        except OSError:
            pass
        dt = time.time() - t0
        log(f"== OK {bvid}  {dt:.0f}s  -> {voc_out}")
        done.append(bvid)

    log(f"\n=== DONE in {(time.time()-t_all)/60:.1f} min | ok={len(done)} skip={len(skipped)} fail={len(failed)} ===")
    if failed:
        log("FAILED:", failed)


if __name__ == "__main__":
    main()
