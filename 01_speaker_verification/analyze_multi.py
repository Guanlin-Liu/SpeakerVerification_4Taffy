# -*- coding: utf-8 -*-
"""多次采样(bootstrap)说话人确认：输出带误差棒(mean±std)的相似度。

要点：
- 每个 speaker 先把池内(最多 POOL_N)小段都提声纹向量——**只提一次**；
- bootstrap B 轮：每轮每个实体从自己池里**有放回**抽 DRAW_N 段求平均成声纹，算一次相似度矩阵；
  B 轮 → 逐格 mean ± std。因为只在现成向量上抽样，B 几乎不花算力。
- 额外实体 **Taffy_composite** = 把所有 reference(Echo/Fries/Taffy) 的段合并成一个池再抽（"合并身份"）；
- 同时也把 Echo/Fries/Taffy 当**独立 speaker** 列出（假设不知它们同人），给出别人对各 persona 的相似度。

用法：  python analyze_multi.py <audio_dir> <ecapa|campp> [B]
"""
import os
import sys
import json
import glob
import logging
import warnings
import tempfile
import itertools

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("modelscope").setLevel(logging.ERROR)

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "audio")
MODEL     = (sys.argv[2] if len(sys.argv) > 2 else "ecapa").lower()
B         = int(sys.argv[3]) if len(sys.argv) > 3 else 100
OUT_DIR   = os.path.join(ROOT, "results")
SR        = 16000
TOP_DB    = 30
SEG_SEC   = 8.0
MIN_SEC   = 3.0
POOL_SEC  = 960       # 每个 speaker 最多提多少秒(120 段)的向量作为 bootstrap 池
DRAW_SEC  = 240       # 每轮抽多少秒(30 段)求平均成一个声纹
SEED      = 42
COMPOSITE = "Taffy_composite"
AUDIO_EXT = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.aac", "*.wma", "*.opus", "*.webm")
POOL_N    = int(round(POOL_SEC / SEG_SEC))
DRAW_N    = int(round(DRAW_SEC / SEG_SEC))


def log(m=""):
    print(m, flush=True)


import librosa  # noqa: E402
_w = (np.random.rand(8000).astype(np.float32) - 0.5) * 0.1
librosa.resample(_w, orig_sr=4, target_sr=2)
librosa.effects.split(_w, top_db=30)


# ===================== 声纹模型后端 =====================
log(f"加载模型：{MODEL} ……")
if MODEL == "ecapa":
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _M = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                        savedir=os.path.join(ROOT, "models", "ecapa"),
                                        run_opts={"device": "cpu"},
                                        local_strategy=LocalStrategy.COPY)

    def embed_pool(segs):
        out = np.zeros((len(segs), 192), dtype=np.float32)
        for i, s in enumerate(segs):
            t = torch.from_numpy(s).float().unsqueeze(0)
            with torch.no_grad():
                out[i] = _M.encode_batch(t).squeeze().cpu().numpy()
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

elif MODEL == "campp":
    import soundfile as sf
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    _SV = pipeline(task=Tasks.speaker_verification,
                   model="iic/speech_campplus_sv_zh-cn_16k-common")

    def embed_pool(segs):
        if not segs:
            return np.zeros((0, 192), dtype=np.float32)
        tmp = tempfile.mkdtemp(prefix="campm_")
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
else:
    sys.exit("model 只能是 ecapa 或 campp")
log("模型就绪。\n")


# ===================== 音频 → 段 → 池向量 =====================
def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def to_segments(y):
    iv = librosa.effects.split(y, top_db=TOP_DB)
    voiced = np.concatenate([y[s:e] for s, e in iv]) if len(iv) else y
    seg_len, min_len = int(SEG_SEC * SR), int(MIN_SEC * SR)
    segs = [voiced[i:i + seg_len] for i in range(0, len(voiced), seg_len)
            if len(voiced[i:i + seg_len]) >= min_len]
    if not segs and len(voiced) >= SR:
        segs = [voiced]
    return segs, len(voiced) / SR


def role_of(n):
    n = n.lower()
    return "test" if ("test" in n or "unknown" in n) else "negative" if "neg" in n else "reference"


def display_name(folder):
    parts = folder.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    if parts and parts[-1].lower() in ("test", "ref", "reference", "neg", "negative", "unknown"):
        parts = parts[:-1]
    return "_".join(parts) if parts else folder


def build_pool(folder):
    base = os.path.basename(folder.rstrip("/\\"))
    files = []
    for ext in AUDIO_EXT:
        files += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    files = sorted(set(files))
    if not files:
        return None
    segs, vsec = [], 0.0
    for f in files:
        try:
            y = load_audio(f)
        except Exception as e:
            log(f"  ! 读取失败 {os.path.basename(f)} ({e})")
            continue
        s, v = to_segments(y)
        segs += s
        vsec += v
    if not segs:
        return None
    if len(segs) > POOL_N:                       # 限制提向量的段数(控制耗时)，跨全程随机取
        rng = np.random.default_rng(SEED)
        segs = [segs[i] for i in rng.choice(len(segs), POOL_N, replace=False)]
    return {"name": display_name(base), "role": role_of(base),
            "voiced": vsec, "embs": embed_pool(segs)}


# ============================== 主流程 ==============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = os.path.basename(AUDIO_DIR.rstrip("/\\")) or "audio"
    log(f"分析目录：{tag}　模型：{MODEL}　B={B}\n" + "=" * 64)

    sources = {}
    for fo in sorted(d for d in glob.glob(os.path.join(AUDIO_DIR, "*")) if os.path.isdir(d)):
        s = build_pool(fo)
        if s is None:
            continue
        sources[s["name"]] = s
        log(f"  {s['name']:<16} 角色={s['role']:<9} 有效语音={s['voiced']:7.1f}s "
            f"池内段={len(s['embs'])}")
    log("=" * 64 + "\n")
    if not sources:
        log("无可用音频。")
        return

    tests = [n for n, s in sources.items() if s["role"] == "test"]
    refs  = [n for n, s in sources.items() if s["role"] == "reference"]
    negs  = [n for n, s in sources.items() if s["role"] == "negative"]
    entities = tests + refs + ([COMPOSITE] if refs else []) + negs
    comp_pool = np.vstack([sources[n]["embs"] for n in refs]) if refs else None

    # ---- bootstrap：B 轮，每轮抽样→声纹→矩阵 ----
    rng = np.random.default_rng(SEED)
    E = len(entities)
    mats = np.zeros((B, E, E), dtype=np.float32)
    for b in range(B):
        V = np.zeros((E, sources[entities[0] if entities[0] != COMPOSITE else refs[0]]["embs"].shape[1]),
                     dtype=np.float32)
        for k, name in enumerate(entities):
            pool = comp_pool if name == COMPOSITE else sources[name]["embs"]
            idx = rng.integers(0, len(pool), DRAW_N)            # 有放回
            v = pool[idx].mean(0)
            V[k] = v / (np.linalg.norm(v) + 1e-9)
        mats[b] = V @ V.T
    mean, std = mats.mean(0), mats.std(0)

    def cell(a, b):
        i, j = entities.index(a), entities.index(b)
        return mean[i, j], std[i, j]

    # ---- 打印矩阵（均值；括号内 std）----
    short = {n: (n[:9] if n != COMPOSITE else "T_compos") for n in entities}
    log("【相似度矩阵 均值】（越接近 1 越像；下一节给关键项的 ±标准差）")
    log("            " + "".join(f"{short[n]:>10}" for n in entities))
    for a in entities:
        log(f"{short[a]:>11} " + "".join(f"{mean[entities.index(a), entities.index(b)]:>10.3f}"
                                         for b in entities))
    log("")

    # ---- 校准（带误差棒）----
    same_pairs = list(itertools.combinations(refs, 2))
    same_vals = [cell(a, b) for a, b in same_pairs]
    diff_vals = [cell(ng, r) for ng in negs for r in (refs + [COMPOSITE])]
    log("【校准】已知同一人(X 内部各 persona 两两) 相似度：")
    for (a, b), (m, sd) in zip(same_pairs, same_vals):
        log(f"    {a}↔{b}: {m:.3f} ± {sd:.3f}")
    if same_vals:
        sm = [m for m, _ in same_vals]
        log(f"    → 同一人区间 ~ {min(sm):.3f} … {max(sm):.3f}")
    log("")

    report = {"model": MODEL, "audio_dir": tag, "B": B, "entities": entities,
              "mean": mean.tolist(), "std": std.tolist(),
              "pool": {n: {"role": s["role"], "voiced": round(s["voiced"], 1),
                           "n": int(len(s["embs"]))} for n, s in sources.items()}}

    # ---- 判决：待测 & 负样本 对 Taffy_composite / 各 persona ----
    if tests and refs:
        log("【判决】各对象对 X 的相似度（mean ± std）：")
        comp_ref = [cell(r, COMPOSITE) for r in refs]      # 已知同人 vs 合并身份(上参考)
        log(f"  ·〔同人基准〕Echo/Fries/Taffy vs {COMPOSITE}: "
            + ", ".join(f"{r}:{m:.3f}±{sd:.3f}" for r, (m, sd) in zip(refs, comp_ref)))
        log("")
        probes = tests + negs
        comp_scores = {p: cell(p, COMPOSITE) for p in probes}
        for p in probes:
            m, sd = comp_scores[p]
            per = {r: cell(p, r) for r in refs}
            role = sources[p]["role"]
            log(f"  ◆ {p}（{role}）")
            log(f"      vs {COMPOSITE}: {m:.3f} ± {sd:.3f}")
            log("      vs 各 persona  : " + ", ".join(f"{r}:{mm:.3f}±{ss:.3f}"
                                                    for r, (mm, ss) in per.items()))
        log("")
        # 关键显著性：待测 vs 最强负样本（对 composite）
        for t in tests:
            tm, ts = comp_scores[t]
            log(f"  ▶ 关键比较：{t} 对 {COMPOSITE} = {tm:.3f}±{ts:.3f}")
            for ng in negs:
                nm, ns = comp_scores[ng]
                gap = tm - nm
                csd = (ts ** 2 + ns ** 2) ** 0.5
                sig = "显著高于" if gap > 2 * csd else ("略高于" if gap > 0 else "未高于")
                log(f"        − {ng}({nm:.3f}±{ns:.3f}): 差 {gap:+.3f}，合并σ {csd:.3f} → {t} {sig} 该负样本"
                    + ("（差>2σ）" if gap > 2 * csd else "（差≤2σ，重叠）"))
            report.setdefault("verdict", {})[t] = {
                "vs_composite": [float(tm), float(ts)],
                "vs_negatives": {ng: [float(comp_scores[ng][0]), float(comp_scores[ng][1])] for ng in negs}}
        log("")

    with open(os.path.join(OUT_DIR, f"report_multi_{MODEL}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"已保存：results/report_multi_{MODEL}_{tag}.json")


if __name__ == "__main__":
    main()
