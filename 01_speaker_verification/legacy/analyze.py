# -*- coding: utf-8 -*-
"""
说话人确认（Speaker Verification）分析脚本
============================================

目标：判断 audio/1_unknown 里的人，是否与已确认的说话人 X（audio/2_X, 3_X, 4_X）为同一人。
方法：把每段音频转成"声纹向量"(ECAPA-TDNN, 192维) → 比较余弦相似度 → 用你自己的数据校准判决边界。

文件夹约定（放在 audio/ 下，每个子文件夹是一个"来源"）：
    1_unknown   待测（不确定是不是 X）
    2_X 3_X 4_X 已确认 = X（参考）
    5_negative  已确认 ≠ X（负样本，可放多个别人，每人可单独子文件夹或混放）

判决逻辑（关键）：不依赖固定阈值，而是：
    - 用 X 的几份材料两两比，得到"同一人"的分数范围（same-speaker 区间）；
    - 用负样本 vs X，得到"不同人"的分数范围（different-speaker 区间）；
    - 看待测者落在哪个区间，给出结论 + 置信度。

运行：  conda run -n voicerec python analyze.py
依赖：  torch(cpu) / speechbrain / librosa / soundfile / numpy / scikit-learn（已装在 voicerec 环境）
"""

import os
import sys
import json
import glob
import random
import itertools
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # 屏蔽一些无关紧要的库警告，让输出干净
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows 下关掉 symlink 警告

# ----------------------------- 可调参数 -----------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_DIR = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "audio")  # 可传 audio_vocals 做对比
MODEL_DIR = os.path.join(ROOT, "models", "ecapa")
OUT_DIR   = os.path.join(ROOT, "results")

SR        = 16000     # 目标采样率（ECAPA 要求 16kHz）
TOP_DB    = 30        # 静音切除阈值：低于峰值 TOP_DB(dB) 的段视为静音并剔除。越小切得越狠
SEG_SEC   = 8.0       # 每个分析小段的长度（秒）
MIN_SEC   = 3.0       # 小段最短长度；短于此则丢弃（太短的段声纹不稳）
SAMPLE_SEC = 240      # 长音频随机截取上限：有效语音超过此值就随机抽样到约此时长（0=不限）
SEED      = 42        # 随机采样种子（可复现）
AUDIO_EXT = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.aac", "*.wma", "*.opus", "*.webm")
# -------------------------------------------------------------------


def log(msg=""):
    print(msg, flush=True)


# ===================== 1. 先预热 librosa，再加载声纹模型 =====================
# 重要：必须在导入 speechbrain 之前先"预热"librosa 的惰性子模块。
# 否则 librosa 首次调用时内部的 inspect.stack() 会遍历 sys.modules、对每个模块访问
# __file__，从而探到 speechbrain 注册的 k2_fsa 惰性模块并触发其导入；k2 没装 → librosa 调用崩溃。
# 先用一小段假信号把 librosa.core.audio 和 librosa.effects 都加载好，问题即消失。
import librosa  # noqa: E402
_warm = (np.random.rand(8000).astype(np.float32) - 0.5) * 0.1
librosa.resample(_warm, orig_sr=4, target_sr=2)   # 预热 librosa.core.audio（load/resample 所在）
librosa.effects.split(_warm, top_db=30)           # 预热 librosa.effects（静音切除所在）

log("加载 ECAPA-TDNN 声纹模型（CPU）……")
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
_MODEL = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODEL_DIR,
    run_opts={"device": "cpu"},
    local_strategy=LocalStrategy.COPY,   # Windows 非管理员：用复制代替 symlink，避免 WinError 1314
)
log("模型就绪。\n")


# ===================== 2. 音频 → 声纹向量 的工具函数 =====================
def load_audio(path):
    """读任意格式音频 → 16k 单声道 float32（依赖已安装的 ffmpeg 解码 mp3/m4a 等）。"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def to_segments(y):
    """静音切除 + 切成若干等长小段。返回 [小段波形...] 和 有效语音秒数。"""
    intervals = librosa.effects.split(y, top_db=TOP_DB)        # 找出非静音区间
    voiced = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) else y
    seg_len, min_len = int(SEG_SEC * SR), int(MIN_SEC * SR)
    segs = []
    for start in range(0, len(voiced), seg_len):
        chunk = voiced[start:start + seg_len]
        if len(chunk) >= min_len:
            segs.append(chunk)
    if not segs and len(voiced) >= SR:        # 兜底：材料很短时，至少用 >=1s 的整段
        segs = [voiced]
    return segs, len(voiced) / SR


def embed(sig):
    """单段波形 → L2 归一化的 192 维声纹向量。"""
    t = torch.from_numpy(sig).float().unsqueeze(0)            # (1, T)
    with torch.no_grad():
        e = _MODEL.encode_batch(t).squeeze().cpu().numpy()   # (192,)
    return e / (np.linalg.norm(e) + 1e-9)


def cos(a, b):
    """余弦相似度（向量已归一化，等价于点积）。范围约 -1~1，越大越像。"""
    return float(np.dot(a, b))


# ===================== 3. 把一个"来源"文件夹 → 声纹 =====================
class Source:
    def __init__(self, name, role):
        self.name = name          # 文件夹名，如 '2_X'
        self.role = role          # 'reference' / 'test' / 'negative'
        self.seg_embs = []        # 该来源所有小段的声纹向量
        self.voiced_sec = 0.0     # 有效语音总时长
        self.n_files = 0

    @property
    def voiceprint(self):
        """该来源的整体声纹 = 所有小段向量取平均再归一化（材料越多越稳）。"""
        v = np.mean(self.seg_embs, axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def self_consistency(self):
        """自一致性：把小段分两半各求声纹再比。代表'同一人同录音'的相似度上限参考。"""
        if len(self.seg_embs) < 2:
            return None
        mid = len(self.seg_embs) // 2
        h1 = np.mean(self.seg_embs[:mid], axis=0)
        h2 = np.mean(self.seg_embs[mid:], axis=0)
        h1 /= np.linalg.norm(h1) + 1e-9
        h2 /= np.linalg.norm(h2) + 1e-9
        return cos(h1, h2)


def role_of(folder_name):
    n = folder_name.lower()
    if "test" in n or "unknown" in n:
        return "test"
    if "neg" in n:
        return "negative"
    return "reference"          # _ref 后缀或默认都按参考


def display_name(folder_name):
    """从 '1_Cui_test' 提取展示名 'Cui'（去掉序号前缀与角色后缀），用于矩阵标签。"""
    parts = folder_name.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    if parts and parts[-1].lower() in ("test", "ref", "reference", "neg", "negative", "unknown"):
        parts = parts[:-1]
    return "_".join(parts) if parts else folder_name


def build_source(folder):
    base = os.path.basename(folder.rstrip("/\\"))
    src = Source(display_name(base), role_of(base))
    files = []
    for ext in AUDIO_EXT:
        files += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    files = sorted(set(files))
    if not files:
        return None
    all_segs = []
    for f in files:
        try:
            y = load_audio(f)
        except Exception as e:
            log(f"  ! 读取失败，跳过：{os.path.basename(f)}  ({e})")
            continue
        segs, vsec = to_segments(y)
        all_segs += segs
        src.voiced_sec += vsec
        src.n_files += 1
    if not all_segs:
        return None
    cap = int(round(SAMPLE_SEC / SEG_SEC)) if SAMPLE_SEC else 0
    if cap and len(all_segs) > cap:                       # 长音频：随机抽样到约 SAMPLE_SEC
        all_segs = random.Random(SEED).sample(all_segs, cap)
    src.seg_embs = [embed(s) for s in all_segs]
    return src


# ============================== 4. 主流程 ==============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(AUDIO_DIR):
        log(f"找不到音频目录：{AUDIO_DIR}")
        return

    # 4.1 逐文件夹建声纹
    folders = sorted(d for d in glob.glob(os.path.join(AUDIO_DIR, "*")) if os.path.isdir(d))
    sources = {}
    log("=" * 60)
    log("逐来源提取声纹：")
    for fo in folders:
        src = build_source(fo)
        nm = os.path.basename(fo)
        if src is None:
            log(f"  - {nm:<12} （空文件夹或无有效音频，跳过）")
            continue
        sources[src.name] = src
        sc = src.self_consistency()
        sc_s = f"，自一致性={sc:.3f}" if sc is not None else ""
        log(f"  - {nm:<12} 角色={src.role:<9} 文件={src.n_files} "
            f"有效语音={src.voiced_sec:5.1f}s 小段={len(src.seg_embs)}{sc_s}")
    log("=" * 60 + "\n")

    if not sources:
        log("还没有任何可用音频。把素材放进 audio/ 各子文件夹后再运行即可。")
        return

    refs = [s for s in sources.values() if s.role == "reference"]
    tests = [s for s in sources.values() if s.role == "test"]
    negs = [s for s in sources.values() if s.role == "negative"]

    # 4.2 来源级相似度矩阵
    names = list(sources.keys())
    log("【相似度矩阵】（来源整体声纹两两余弦相似度，越接近 1 越像同一人）")
    header = "            " + "".join(f"{n:>12}" for n in names)
    log(header)
    matrix = {}
    for a in names:
        row = []
        for b in names:
            v = cos(sources[a].voiceprint, sources[b].voiceprint)
            row.append(v)
            matrix[f"{a}|{b}"] = v
        log(f"{a:>12}" + "".join(f"{v:>12.3f}" for v in row))
    log("")

    # 4.3 校准：同一人区间 & 不同人区间
    same_ref = [cos(a.voiceprint, b.voiceprint) for a, b in itertools.combinations(refs, 2)]
    diff_ref = []
    for ng in negs:
        for r in refs:
            diff_ref.append(cos(ng.voiceprint, r.voiceprint))

    report = {"matrix": matrix, "sources": {
        n: {"role": s.role, "n_files": s.n_files,
            "voiced_sec": round(s.voiced_sec, 1), "n_segments": len(s.seg_embs),
            "self_consistency": s.self_consistency()} for n, s in sources.items()}}

    log("【校准】用你自己的数据确定判决边界：")
    if same_ref:
        log(f"  · 已知同一人(X 内部) 相似度区间：{min(same_ref):.3f} ~ {max(same_ref):.3f}"
            f"  (均值 {np.mean(same_ref):.3f})   ← 这就是'同一人'长什么样")
        report["same_speaker_range"] = [min(same_ref), max(same_ref), float(np.mean(same_ref))]
    else:
        log("  · 还没有 ≥2 份 X 参考，无法标定'同一人'区间（先把 2_X/3_X/4_X 放好）")
    if diff_ref:
        log(f"  · 已知不同人(负样本 vs X) 相似度区间：{min(diff_ref):.3f} ~ {max(diff_ref):.3f}"
            f"  (均值 {np.mean(diff_ref):.3f})   ← 这就是'不同人'长什么样")
        report["diff_speaker_range"] = [min(diff_ref), max(diff_ref), float(np.mean(diff_ref))]
    else:
        log("  · 还没有负样本，无法标定'不同人'区间（建议放 5_negative，结论更可靠）")
    log("")

    # 4.4 判决：待测者 vs X
    if tests and refs:
        # 把所有 X 的小段合并成一个"注册声纹"（identification 风格，更稳）
        x_pool = np.mean([e for r in refs for e in r.seg_embs], axis=0)
        x_pool /= np.linalg.norm(x_pool) + 1e-9

        log("【判决】待测来源 vs 说话人 X：")
        verdicts = {}
        for t in tests:
            s_pool = cos(t.voiceprint, x_pool)
            per_ref = {r.name: cos(t.voiceprint, r.voiceprint) for r in refs}
            log(f"  ◆ {t.name}")
            log(f"      vs X(合并声纹) 相似度 = {s_pool:.3f}")
            log("      vs 各 X 来源       = " +
                ", ".join(f"{k}:{v:.3f}" for k, v in per_ref.items()))

            verdict, conf = decide(s_pool, same_ref, diff_ref)
            log(f"      → 结论：{verdict}{conf}\n")
            verdicts[t.name] = {"score_vs_X": s_pool, "per_ref": per_ref,
                                "verdict": verdict}
        report["verdicts"] = verdicts
    elif tests and not refs:
        log("【判决】检测到待测来源，但还没有 X 参考（2_X/3_X/4_X）。先放参考再判决。\n")
    else:
        log("【判决】暂无待测来源（1_unknown 为空）。可先只放 2/3/4 验证方法是否有效。\n")

    # 4.5 落盘
    tag = os.path.basename(AUDIO_DIR.rstrip("/\\")) or "audio"
    with open(os.path.join(OUT_DIR, f"report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"详细结果已保存：{os.path.join('results', f'report_{tag}.json')}")


def decide(score, same_ref, diff_ref):
    """根据校准区间给出结论 + 置信度描述。没有负样本时退化为模型经验阈值并提示。"""
    if same_ref and diff_ref:
        same_m, diff_m = np.mean(same_ref), np.mean(diff_ref)
        boundary = (same_m + diff_m) / 2.0
        if score >= max(same_ref) or score >= same_m:
            return "同一人（= X）", f"（落入'同一人'区间，边界≈{boundary:.3f}，余量 +{score - boundary:.3f}）"
        if score <= min(diff_ref) or score <= diff_m:
            return "不同人（≠ X）", f"（落入'不同人'区间，边界≈{boundary:.3f}，余量 {score - boundary:+.3f}）"
        side = "偏向同一人" if score >= boundary else "偏向不同人"
        return "边界模糊，需谨慎", f"（在两区间之间，{side}，边界≈{boundary:.3f}）"
    # 没有完整校准：用 ECAPA 在 VoxCeleb 上的经验阈值（仅供参考）
    thr = 0.25
    if score >= thr:
        return "倾向同一人", f"（无负样本校准，按经验阈值 {thr}，可靠性打折）"
    return "倾向不同人", f"（无负样本校准，按经验阈值 {thr}，可靠性打折）"


if __name__ == "__main__":
    main()
