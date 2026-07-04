# -*- coding: utf-8 -*-
"""CAM++（中文 CN-Celeb 训练）版说话人确认分析。

与 analyze.py 同样的切片/平均/校准/判决逻辑，只把声纹模型换成 3D-Speaker CAM++，
用作独立的第二意见（交叉验证）。CAM++ 经 modelscope 的 SV pipeline 取 embedding。

用法：  python analyze_campp.py [audio | audio_vocals]
"""
import os
import sys
import json
import glob
import random
import logging
import warnings
import tempfile
import itertools

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("modelscope").setLevel(logging.ERROR)

# ----------------------------- 参数（与 analyze.py 保持一致）-----------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_DIR = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "audio")
OUT_DIR   = os.path.join(ROOT, "results")
SR        = 16000
TOP_DB    = 30
SEG_SEC   = 8.0
MIN_SEC   = 3.0
SAMPLE_SEC = 240      # 长音频随机截取上限（0=不限）
SEED      = 42        # 随机采样种子（可复现）
AUDIO_EXT = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.aac", "*.wma", "*.opus", "*.webm")
# -------------------------------------------------------------------------------------


def log(msg=""):
    print(msg, flush=True)


# 先预热 librosa 再导入 modelscope（与 analyze.py 同样的防雷措施）
import librosa  # noqa: E402
_warm = (np.random.rand(8000).astype(np.float32) - 0.5) * 0.1
librosa.resample(_warm, orig_sr=4, target_sr=2)
librosa.effects.split(_warm, top_db=30)

log("加载 CAM++（中文）声纹模型……")
from modelscope.pipelines import pipeline          # noqa: E402
from modelscope.utils.constant import Tasks        # noqa: E402

_SV = pipeline(task=Tasks.speaker_verification,
               model="iic/speech_campplus_sv_zh-cn_16k-common")
log("模型就绪。\n")


# ============================ 音频 → 声纹 ============================
def cos(a, b):
    return float(np.dot(a, b))


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def to_segments(y):
    intervals = librosa.effects.split(y, top_db=TOP_DB)
    voiced = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) else y
    seg_len, min_len = int(SEG_SEC * SR), int(MIN_SEC * SR)
    segs = []
    for start in range(0, len(voiced), seg_len):
        chunk = voiced[start:start + seg_len]
        if len(chunk) >= min_len:
            segs.append(chunk)
    if not segs and len(voiced) >= SR:
        segs = [voiced]
    return segs, len(voiced) / SR


def embed_segments(segs):
    """一批小段波形 -> (n,192) L2 归一化向量。写临时 wav 后交给 CAM++ pipeline。"""
    if not segs:
        return np.zeros((0, 192), dtype=np.float32)
    tmp = tempfile.mkdtemp(prefix="camp_")
    paths = [os.path.join(tmp, f"{i}.wav") for i in range(len(segs))]
    for p, s in zip(paths, segs):
        sf.write(p, s, SR)
    try:
        r = _SV(paths, output_emb=True)
        embs = np.asarray(r["embs"], dtype=np.float32)
        if embs.shape[0] != len(segs):           # 批量行为异常则逐段兜底
            raise ValueError("batch size mismatch")
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


# ============================ 来源建模 ============================
class Source:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.seg_embs = []
        self.voiced_sec = 0.0
        self.n_files = 0

    @property
    def voiceprint(self):
        v = np.mean(self.seg_embs, axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def self_consistency(self):
        if len(self.seg_embs) < 2:
            return None
        mid = len(self.seg_embs) // 2
        h1 = np.mean(self.seg_embs[:mid], axis=0); h1 /= np.linalg.norm(h1) + 1e-9
        h2 = np.mean(self.seg_embs[mid:], axis=0); h2 /= np.linalg.norm(h2) + 1e-9
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
    src.seg_embs = list(embed_segments(all_segs))
    return src


# ============================ 主流程（与 analyze.py 一致）============================
def decide(score, same_ref, diff_ref):
    if same_ref and diff_ref:
        same_m, diff_m = np.mean(same_ref), np.mean(diff_ref)
        boundary = (same_m + diff_m) / 2.0
        if score >= max(same_ref) or score >= same_m:
            return "同一人（= X）", f"（落入'同一人'区间，边界≈{boundary:.3f}，余量 +{score - boundary:.3f}）"
        if score <= min(diff_ref) or score <= diff_m:
            return "不同人（≠ X）", f"（落入'不同人'区间，边界≈{boundary:.3f}，余量 {score - boundary:+.3f}）"
        side = "偏向同一人" if score >= boundary else "偏向不同人"
        return "边界模糊，需谨慎", f"（在两区间之间，{side}，边界≈{boundary:.3f}）"
    thr = 0.30
    return ("倾向同一人" if score >= thr else "倾向不同人"), f"（无完整校准，按经验阈值 {thr}）"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(AUDIO_DIR):
        log(f"找不到音频目录：{AUDIO_DIR}")
        return
    log(f"分析目录：{os.path.basename(AUDIO_DIR)}（模型：CAM++ 中文）")

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
        log("无可用音频。")
        return

    refs = [s for s in sources.values() if s.role == "reference"]
    tests = [s for s in sources.values() if s.role == "test"]
    negs = [s for s in sources.values() if s.role == "negative"]

    names = list(sources.keys())
    log("【相似度矩阵】（CAM++ 余弦相似度）")
    log("            " + "".join(f"{n:>12}" for n in names))
    matrix = {}
    for a in names:
        row = []
        for b in names:
            v = cos(sources[a].voiceprint, sources[b].voiceprint)
            row.append(v); matrix[f"{a}|{b}"] = v
        log(f"{a:>12}" + "".join(f"{v:>12.3f}" for v in row))
    log("")

    same_ref = [cos(a.voiceprint, b.voiceprint) for a, b in itertools.combinations(refs, 2)]
    diff_ref = [cos(ng.voiceprint, r.voiceprint) for ng in negs for r in refs]
    report = {"model": "campp", "audio_dir": os.path.basename(AUDIO_DIR), "matrix": matrix,
              "sources": {n: {"role": s.role, "voiced_sec": round(s.voiced_sec, 1),
                              "n_segments": len(s.seg_embs),
                              "self_consistency": s.self_consistency()}
                          for n, s in sources.items()}}

    log("【校准】")
    if same_ref:
        log(f"  · 已知同一人区间：{min(same_ref):.3f} ~ {max(same_ref):.3f}  (均值 {np.mean(same_ref):.3f})")
        report["same_speaker_range"] = [min(same_ref), max(same_ref), float(np.mean(same_ref))]
    if diff_ref:
        log(f"  · 已知不同人区间：{min(diff_ref):.3f} ~ {max(diff_ref):.3f}  (均值 {np.mean(diff_ref):.3f})")
        report["diff_speaker_range"] = [min(diff_ref), max(diff_ref), float(np.mean(diff_ref))]
    log("")

    if tests and refs:
        x_pool = np.mean([e for r in refs for e in r.seg_embs], axis=0)
        x_pool /= np.linalg.norm(x_pool) + 1e-9
        log("【判决】待测来源 vs 说话人 X：")
        verdicts = {}
        for t in tests:
            s_pool = cos(t.voiceprint, x_pool)
            per_ref = {r.name: cos(t.voiceprint, r.voiceprint) for r in refs}
            log(f"  ◆ {t.name}")
            log(f"      vs X(合并声纹) 相似度 = {s_pool:.3f}")
            log("      vs 各 X 来源       = " + ", ".join(f"{k}:{v:.3f}" for k, v in per_ref.items()))
            verdict, conf = decide(s_pool, same_ref, diff_ref)
            log(f"      → 结论：{verdict}{conf}\n")
            verdicts[t.name] = {"score_vs_X": s_pool, "per_ref": per_ref, "verdict": verdict}
        report["verdicts"] = verdicts

    tag = os.path.basename(AUDIO_DIR.rstrip("/\\")) or "audio"
    with open(os.path.join(OUT_DIR, f"report_campp_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"详细结果已保存：{os.path.join('results', f'report_campp_{tag}.json')}")


if __name__ == "__main__":
    main()
