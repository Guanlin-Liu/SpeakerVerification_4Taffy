# -*- coding: utf-8 -*-
"""Taffy 声音特征随时间演化分析（CAM++ · 人声分离）。

思路（类比 Hubble diagram）：
- 以"现在"(2026-07 单场多切片, NOW 锚点)为基准 anchor，度量各时期声音到 anchor 的 angular distance
- 分 bin：2021 与 2026(1-6月) 用双月(端点细分)，其余用季度
      θ = arccos(cosine)   —— embedding 球面上的真实 metric。
- 校准两条带：
  · same-period floor：每个 epoch 内**跨 session**(不同直播分组)的 θ —— session 噪声底；
  · different-person ceiling：anchor 到 Monaka/Miya 的 θ。
- 地标 landmarks：Echo/Fries(之前身份, kind=prev)、Monaka/Miya(不同人, kind=diff)，仅用于 PCA/MDS 定标。
- intra-epoch outlier rejection：剔除偏离本 epoch 质心过远的段（连麦嘉宾/噪声）。

E×E θ 用 bootstrap(有放回抽段→质心→θ) 求 mean±std；另存各实体全池质心供 PCA/MDS。
输出：results/report_evolution_campp.json + results/evolution_centroids_campp.npz
用法：python analyze_evolution.py [B]
"""
import os
import sys
import json
import glob
import re
import logging
import warnings
import tempfile
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("modelscope").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVO_DIR = os.path.join(ROOT, "audio_evo_vocals")
LM_DIR = os.path.join(ROOT, "audio_vocals")
MANIFEST = os.path.join(ROOT, "audio_evo", "manifest.tsv")
OUT_DIR = os.path.join(ROOT, "results")

SR = 16000
TOP_DB = 30
SEG_SEC = 8.0
MIN_SEC = 3.0
POOL_N = 240            # 每实体最多提向量的段数（多切片后样本更丰富）
DRAW_N = 40            # bootstrap 每轮有放回抽多少段求质心
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED = 42
NOW_DIR = "NOW"          # audio_evo_vocals/NOW/ = "现在"锚点(2026-07 单场多切片)
NOW_DATE = "2026-07-02"  # 锚点日期

# 地标：显示名(内部) -> (相对 LM_DIR 的子目录, kind)
LANDMARKS = [
    ("Echo",     "2_Echo_ref",     "prev"),   # 之前身份
    ("Fries",    "3_Fries_ref",    "prev"),
    ("Monaka",   "5_Monaka_neg",   "diff"),   # 不同人
    ("Miya",     "6_Miya_neg",     "diff"),
]
AUDIO_EXT = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.opus")


def log(m=""):
    print(m, flush=True)


import librosa  # noqa: E402
_w = (np.random.rand(8000).astype(np.float32) - 0.5) * 0.1
librosa.resample(_w, orig_sr=4, target_sr=2)
librosa.effects.split(_w, top_db=30)


# ===================== CAM++ 后端（同 analyze_multi） =====================
log("加载 CAM++ ……")
import soundfile as sf  # noqa: E402
from modelscope.pipelines import pipeline  # noqa: E402
from modelscope.utils.constant import Tasks  # noqa: E402
_SV = pipeline(task=Tasks.speaker_verification,
               model="iic/speech_campplus_sv_zh-cn_16k-common")


def embed_pool(segs):
    if not segs:
        return np.zeros((0, 192), dtype=np.float32)
    tmp = tempfile.mkdtemp(prefix="evo_")
    paths = [os.path.join(tmp, f"{i}.wav") for i in range(len(segs))]
    for p, s in zip(paths, segs):
        sf.write(p, s, SR)
    try:
        embs = np.asarray(_SV(paths, output_emb=True)["embs"], dtype=np.float32)
        if embs.shape[0] != len(segs):
            raise ValueError("batch mismatch")
    except Exception:
        embs = np.zeros((len(segs), 192), dtype=np.float32)
        for i, p in enumerate(paths):
            embs[i] = np.asarray(_SV([p], output_emb=True)["embs"], dtype=np.float32)[0]
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)


log("模型就绪。\n")


# ===================== 音频 → 段 =====================
def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def to_segments(y):
    iv = librosa.effects.split(y, top_db=TOP_DB)
    voiced = np.concatenate([y[s:e] for s, e in iv]) if len(iv) else y
    seg_len, min_len = int(SEG_SEC * SR), int(MIN_SEC * SR)
    segs = [voiced[i:i + seg_len] for i in range(0, len(voiced), seg_len)
            if len(voiced[i:i + seg_len]) >= min_len]
    return segs, len(voiced) / SR


def bv_of(path):
    """从文件名取"视频/session id"：去掉 _vocals 和切片编号后缀。"""
    b = os.path.basename(path)
    b = re.sub(r"_vocals\.wav$", "", b)
    b = re.sub(r"_(e|p|n)\d+$", "", b)
    return b


def bin_label(dt):
    """2021 与 2026(1-6月) 用双月 bin(端点细分)，其余用季度 bin。"""
    y, m = dt.year, dt.month
    if y == 2021 or (y == 2026 and m <= 6):
        b = (m - 1) // 2 * 2 + 1          # 1,3,5,7,9,11
        return f"{y}-{b:02d}"
    return f"{y}Q{(m - 1) // 3 + 1}"


def build_pool(files):
    """files: 一组人声 wav 路径。返回 (embs, src[按视频BV分组], voiced_sec, n_videos)。"""
    files = sorted(set(files))
    if not files:
        return None
    keymap = {}
    all_segs, src, vsec = [], [], 0.0
    for f in files:
        try:
            y = load_audio(f)
        except Exception as e:
            log(f"  ! 读取失败 {os.path.basename(f)} ({e})")
            continue
        sid = keymap.setdefault(bv_of(f), len(keymap))
        s, v = to_segments(y)
        all_segs += s
        src += [sid] * len(s)
        vsec += v
    if not all_segs:
        return None
    src = np.array(src)
    if len(all_segs) > POOL_N:                       # 跨全程随机取 POOL_N 段
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(all_segs), POOL_N, replace=False)
        all_segs = [all_segs[i] for i in idx]
        src = src[idx]
    embs = embed_pool(all_segs)
    return embs, src, vsec, len(keymap)


def outlier_reject(embs, n_std=2.0, hard=0.30, passes=2):
    """迭代剔除偏离质心过远的段（连麦/噪声）。返回保留 index。"""
    keep = np.arange(len(embs))
    for _ in range(passes):
        c = embs[keep].mean(0)
        c /= np.linalg.norm(c) + 1e-9
        cs = embs[keep] @ c
        thr = max(cs.mean() - n_std * cs.std(), hard)
        nk = keep[cs >= thr]
        if len(nk) == len(keep) or len(nk) < DRAW_N:
            keep = nk if len(nk) >= DRAW_N else keep
            break
        keep = nk
    return keep


def centroid(embs):
    c = embs.mean(0)
    return c / (np.linalg.norm(c) + 1e-9)


def stream_reject(embs, src, min_streams=3, thr=0.62):
    """>=min_streams 场时，丢掉质心明显偏离其他场的污染场(median θ-to-others > thr)。至少留 2 场。"""
    streams = sorted(set(src.tolist()))
    if len(streams) < min_streams:
        return np.arange(len(embs)), []
    cent = {s: centroid(embs[src == s]) for s in streams}
    med = {}
    for s in streams:
        ds = [float(np.arccos(np.clip(cent[s] @ cent[t], -1.0, 1.0))) for t in streams if t != s]
        med[s] = float(np.median(ds))
    keep_streams = [s for s in streams if med[s] <= thr]
    if len(keep_streams) < 2:
        keep_streams = sorted(streams, key=lambda s: med[s])[:2]
    dropped = [s for s in streams if s not in keep_streams]
    keep_idx = np.where(np.isin(src, keep_streams))[0]
    return keep_idx, dropped


# ===================== 日期（epoch 代表日期） =====================
def load_dates():
    m = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    m[p[2]] = p[1]          # bvid -> date
    return m


def epoch_mean_date(folder, bvid2date):
    ords = []
    for f in glob.glob(os.path.join(folder, "*_vocals.wav")):
        bv = os.path.basename(f).split("_vocals")[0]
        d = bvid2date.get(bv)
        if d:
            ords.append(datetime.strptime(d, "%Y-%m-%d").toordinal())
    if not ords:
        return None
    return sum(ords) / len(ords)


# ============================== 主流程 ==============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bvid2date = load_dates()
    rng = np.random.default_rng(SEED)

    # ---- 收集时间线文件，按日期分 bin ----
    from collections import defaultdict
    all_files = glob.glob(os.path.join(EVO_DIR, "**", "*_vocals.wav"), recursive=True)
    now_files = [f for f in all_files if os.path.basename(os.path.dirname(f)) == NOW_DIR]
    tl_files = [f for f in all_files if os.path.basename(os.path.dirname(f)) != NOW_DIR]
    binmap = defaultdict(list)              # label -> [(file, ord)]
    for f in tl_files:
        d = bvid2date.get(bv_of(f))
        if not d:
            continue
        o = datetime.strptime(d, "%Y-%m-%d").toordinal()
        binmap[bin_label(datetime.fromordinal(o))].append((f, o))

    ents = []
    log("=" * 70)

    def add_epoch(label, files, ords, kind):
        r = build_pool(files)
        if r is None:
            log(f"  {label:<10} 无音频，跳过")
            return
        embs0, src0, vsec, nstreams = r
        sk, dropped = stream_reject(embs0, src0)
        embs, src = embs0[sk], src0[sk]
        keep = outlier_reject(embs)
        dord = sum(ords) / len(ords) if ords else None
        ents.append(dict(name=label, kind=kind, date_ord=dord,
                         embs=embs[keep], src=src[keep],
                         n_before=len(embs0), n_after=int(len(keep)),
                         voiced=vsec, n_streams=nstreams,
                         n_streams_kept=nstreams - len(dropped), dropped=len(dropped)))
        log(f"  {label:<10} kind={kind:<6} streams={nstreams}"
            f"{'-'+str(len(dropped))+'drop' if dropped else ''} "
            f"voiced={vsec:6.0f}s seg {len(embs0)}->{len(keep)} "
            f"date={datetime.fromordinal(int(dord)).date() if dord else '?'}")

    for label in sorted(binmap, key=lambda L: np.mean([o for _, o in binmap[L]])):
        add_epoch(label, [f for f, _ in binmap[label]], [o for _, o in binmap[label]], "epoch")

    if now_files:                            # "现在"锚点(单场多切片)
        add_epoch("now", now_files, [datetime.strptime(NOW_DATE, "%Y-%m-%d").toordinal()], "anchor")
    else:
        log("!! 未找到 NOW 锚点数据")

    # ---- landmarks ----
    for name, sub, kind in LANDMARKS:
        folder = os.path.join(LM_DIR, sub)
        lm_files = []
        for ext in AUDIO_EXT:
            lm_files += glob.glob(os.path.join(folder, "**", ext), recursive=True)
        r = build_pool(lm_files)
        if r is None:
            log(f"  [LM] {name:<10} 缺失，跳过")
            continue
        embs, src, vsec, nstreams = r
        ents.append(dict(name=name, kind=kind, date_ord=None, embs=embs, src=src,
                         n_before=len(embs), n_after=len(embs), voiced=vsec, n_streams=nstreams,
                         n_streams_kept=nstreams, dropped=0))
        log(f"  [LM] {name:<10} kind={kind:<6} voiced={vsec:6.0f}s seg={len(embs)}")
    log("=" * 70 + "\n")

    names = [e["name"] for e in ents]
    if "now" not in names:
        log(f"!! 未找到 now 锚点，无法定基准。已有：{names}")
        return
    ai = names.index("now")
    E = len(ents)

    # ---- E×E θ bootstrap ----
    log(f"bootstrap θ 矩阵：E={E} B={B} …")
    thetas = np.zeros((B, E, E), dtype=np.float32)
    for b in range(B):
        V = np.zeros((E, 192), dtype=np.float32)
        for k, e in enumerate(ents):
            pool = e["embs"]
            idx = rng.integers(0, len(pool), DRAW_N)
            v = pool[idx].mean(0)
            V[k] = v / (np.linalg.norm(v) + 1e-9)
        cos = np.clip(V @ V.T, -1.0, 1.0)
        thetas[b] = np.arccos(cos)
    th_mean, th_std = thetas.mean(0), thetas.std(0)

    # ---- same-period floor：每 epoch 跨 session θ ----
    floor_per = {}
    for e in ents:
        if e["kind"] != "epoch":            # 锚点是单场，不进 floor
            continue
        streams = sorted(set(e["src"].tolist()))
        if len(streams) >= 2:
            A = streams[0::2]
            B_ = streams[1::2]
            eA = e["embs"][np.isin(e["src"], A)]
            eB = e["embs"][np.isin(e["src"], B_)]
            optimistic = False
        else:                                   # 单 session 兜底：随机对半（偏乐观）
            perm = rng.permutation(len(e["embs"]))
            half = len(perm) // 2
            eA, eB = e["embs"][perm[:half]], e["embs"][perm[half:]]
            optimistic = True
        if len(eA) < 5 or len(eB) < 5:
            continue
        vals = []
        for _ in range(B):
            ca = centroid(eA[rng.integers(0, len(eA), DRAW_N)])
            cb = centroid(eB[rng.integers(0, len(eB), DRAW_N)])
            vals.append(float(np.arccos(np.clip(ca @ cb, -1, 1))))
        floor_per[e["name"]] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                                    optimistic=optimistic)
    fvals = [v["mean"] for v in floor_per.values()]
    floor = dict(mean=float(np.mean(fvals)) if fvals else None,
                 std=float(np.std(fvals)) if fvals else None, per_epoch=floor_per)

    # ---- 组织输出 ----
    anchor_ord = ents[ai]["date_ord"]

    def dt_months(e):
        if e["date_ord"] is None or anchor_ord is None:
            return None
        return round((anchor_ord - e["date_ord"]) / 30.44, 2)

    report = dict(
        model="campp", B=B, anchor="now", seg_sec=SEG_SEC, draw_n=DRAW_N,
        entities=names,
        kinds=[e["kind"] for e in ents],
        dt_months=[dt_months(e) for e in ents],
        dates=[(datetime.fromordinal(int(e["date_ord"])).strftime("%Y-%m-%d")
                if e["date_ord"] else None) for e in ents],
        theta_to_anchor=[[float(th_mean[ai, k]), float(th_std[ai, k])] for k in range(E)],
        theta_mean=th_mean.tolist(), theta_std=th_std.tolist(),
        floor=floor,
        ceiling={e["name"]: [float(th_mean[ai, k]), float(th_std[ai, k])]
                 for k, e in enumerate(ents) if e["kind"] == "diff"},
        pool={e["name"]: dict(kind=e["kind"], n_before=e["n_before"], n_after=e["n_after"],
                              voiced=round(e["voiced"], 1), n_streams=e["n_streams"],
                              n_streams_kept=e.get("n_streams_kept", e["n_streams"]),
                              dropped=e.get("dropped", 0))
              for e in ents},
    )
    with open(os.path.join(OUT_DIR, "report_evolution_campp.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 质心 npz（PCA/MDS 用）----
    cents = np.vstack([centroid(e["embs"]) for e in ents]).astype(np.float32)
    np.savez(os.path.join(OUT_DIR, "evolution_centroids_campp.npz"),
             centroids=cents, names=np.array(names),
             kinds=np.array([e["kind"] for e in ents]),
             dt_months=np.array([dt_months(e) if dt_months(e) is not None else np.nan for e in ents]))

    # ---- 摘要打印 ----
    log("\n【θ 到 anchor(现在)】（rad, mean±std）")
    for k, e in enumerate(ents):
        dm = dt_months(e)
        tag = f"Δt={dm:6.1f}mo" if dm is not None else f"[{e['kind']}]"
        log(f"  {e['name']:<10} {tag}  θ={th_mean[ai,k]:.3f}±{th_std[ai,k]:.3f}")
    if floor["mean"] is not None:
        log(f"\nsame-period floor(跨 session): {floor['mean']:.3f} ± {floor['std']:.3f} rad")
    if report["ceiling"]:
        log("different-person ceiling:")
        for n, (m, s) in report["ceiling"].items():
            log(f"  anchor↔{n}: {m:.3f}±{s:.3f}")
    log("\n已保存 results/report_evolution_campp.json + evolution_centroids_campp.npz")


if __name__ == "__main__":
    main()
