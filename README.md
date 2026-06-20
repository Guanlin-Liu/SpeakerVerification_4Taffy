# Speaker Verification

Turn the subjective question *"do these two voices sound alike?"* into a quantitative, objective decision: use pretrained speaker‑embedding models to judge whether different audio clips come from the **same person**. Supports vocal‑separation preprocessing, two‑model cross‑checking, and bootstrap **error bars**.

> This is a **general‑purpose tool** — drop in your **own audio** and plug in **other models**. The PNGs in `figures/` are from one case study, included only as a reference example.

## How it works

Each clip → split into short segments (silence removed) → a pretrained model extracts a speaker embedding (e.g. 192‑d) per segment → average the segment vectors into that source's "voiceprint" → compute **cosine similarity** between sources. The decision does **not** rely on a fixed threshold; it **self‑calibrates** the boundary from known same‑person (within‑reference) and different‑person (negative) samples.

- **Models** (different architectures / training data → agreement means higher confidence)
  - `analyze.py` — **ECAPA‑TDNN** (SpeechBrain, VoxCeleb / English)
  - `analyze_campp.py` — **CAM++** (3D‑Speaker / ModelScope, CN‑Celeb / Chinese)
  - `analyze_multi.py` — either model + **bootstrap error bars (mean ± std)** + a composite reference
- **Optional preprocessing**: `separate.py` uses Demucs (htdemucs) to strip background music / game audio, keeping only the voice
- **CPU‑only**: inference on small data, no GPU needed

## Project layout

```
.
├── analyze.py          # ECAPA single run
├── analyze_campp.py    # CAM++ single run (same logic, different model)
├── analyze_multi.py    # bootstrap error bars + composite (python analyze_multi.py <dir> <ecapa|campp> [B])
├── separate.py         # Demucs vocal separation: audio/ -> audio_vocals/
├── download.py         # yt-dlp helper to fetch audio into audio/<folder>/
├── make_figures.py     # render results/report_multi_*.json into tables/figures
├── requirements.txt
├── figures/            # example result images (one case study's 4 model×data outputs; committed for reference)
├── audio/              # YOUR raw audio (gitignored; follow the naming convention below)
├── audio_vocals/       # separated output (gitignored, mirrors audio/)
├── models/  results/  tables/   # model cache / reports / matrix CSVs (gitignored)
```

## Data naming convention

Under `audio/`, each subfolder is one "source", named `<index>_<name>_<role>`. The role is auto‑detected from the suffix:

| suffix | meaning |
|---|---|
| `_test` | the source under test (is it the target person X?) |
| `_ref`  | reference (confirmed to be X; multiple recordings / periods allowed) |
| `_neg`  | negative (a confirmed different person; **one folder per person**, never mix people) |

Example: `1_alice_test/`, `2_bob_ref/`, `3_bob_ref/`, `4_carol_neg/`. Put one or more audio files (wav/mp3/m4a/flac…) in each folder; the scripts segment and average automatically.

## Setup

```bash
conda create -n voicerec python=3.11 -y
conda activate voicerec
pip install torch==2.12.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
You also need **ffmpeg** on PATH (to decode mp3/m4a). Embedding / separation models download to cache on first run.

## Usage

```bash
# 1) (optional) download audio (bring your own yt-dlp)
python download.py 2_bob_ref "https://..."

# 2) (optional) vocal separation: audio/ -> audio_vocals/
python separate.py audio audio_vocals

# 3) single runs (raw and separated; two models to cross-check)
python analyze.py        audio
python analyze.py        audio_vocals
python analyze_campp.py  audio
python analyze_campp.py  audio_vocals

# 4) analysis with error bars + composite (B = resampling rounds)
python analyze_multi.py  audio_vocals  campp  100

# 5) render the multi reports into tables/figures
python make_figures.py
```

> On a Chinese (GBK) Windows console, `conda run` can crash on UTF‑8 output. Call the env python directly with UTF‑8 forced:
> `PYTHONUTF8=1 .../envs/voicerec/python.exe analyze.py audio`

## Using other models / other data

- **Other data**: just place your audio under `audio/` following the naming convention — no code changes.
- **Other model**: the decision / calibration logic is **model‑agnostic**; only replace the embedding step.
  - In `analyze.py` / `analyze_campp.py`, find `embed(...)` (takes a 16 kHz waveform, returns one **L2‑normalized** vector) and swap in your model.
  - In `analyze_multi.py`, the backend lives in `embed_pool(segs)` (takes a batch of segments, returns an `(n, D)` array of normalized vectors); add a branch next to `ecapa` / `campp`.
  - Segment length, sample rate, etc. are constants at the top of each file (`SEG_SEC`, `SR`, …).

## Reading the results

- **Similarity matrix**: each cell is the cosine similarity between two sources' voiceprints (closer to 1 = more alike).
- **Same‑person baseline**: pairwise similarity among references (same person, different recordings) — the realistic "same person" level and the yardstick for the test source (not 1.0).
- **Composite** (`analyze_multi.py`): pools all `_ref` segments into one merged reference voiceprint.
- **Error bars**: bootstrap over B resampling rounds → mean ± std; less data → wider bars, honestly reflecting uncertainty.
- **Verdict**: whether the test source is significantly closer to the target than the negatives, and whether it reaches the same‑person baseline.
- Reports are saved to `results/report_*.json`.

## Known issues (Windows)

- SpeechBrain's default symlink fetch fails on Windows without admin (WinError 1314) → the code uses `LocalStrategy.COPY`.
- librosa must be "warmed up" before `import speechbrain`, otherwise its lazy submodule import triggers speechbrain's `k2_fsa` and crashes (see comments in `analyze.py`).
- Demucs 4 + torchaudio 2.11 `save` needs torchcodec → `separate.py` uses ffmpeg / soundfile instead, with windowing for very long files and subprocess isolation.
- ModelScope (CAM++) needs a few extra runtime deps on first use (see comments in `requirements.txt`).

## Roadmap

- [ ] **Voice evolution over time**: for a single speaker, track both speaker‑embedding similarity and acoustic / prosodic features (F0, formants, …) to test whether the voice changes across periods.

## Acknowledgements

[SpeechBrain](https://github.com/speechbrain/speechbrain) · [3D‑Speaker / CAM++](https://github.com/modelscope/3D-Speaker) · [Demucs](https://github.com/facebookresearch/demucs) · [ModelScope](https://github.com/modelscope/modelscope)

<!-- ============================================================
中文版 README（保留在文件中，GitHub 页面不渲染显示）
============================================================

# Speaker Verification 声纹比对工作流

把"两段声音听起来像不像"这个主观问题，变成可量化的客观判定：用预训练的说话人嵌入（speaker embedding）模型，判断不同音频是否来自同一个人。支持人声分离预处理、双模型交叉验证、以及 bootstrap 误差棒。

本仓库是通用工具——你可以放入自己的音频、换用其他模型来测试。figures/ 里是一个案例研究的结果图，仅作示例参考。

## 方法概述
每段音频 → 切成若干小段（去静音）→ 预训练模型提取声纹向量（如 192 维）→ 段向量平均得到该来源的"声纹" → 计算来源之间的余弦相似度。判决不依赖固定阈值，而是用已知"同一人"（参考内部）与"不同人"（负样本）的样本自校准出决策边界。

- 模型（架构/训练语料不同，结论一致则更可信）
  - analyze.py        — ECAPA-TDNN（SpeechBrain，VoxCeleb 英文训练）
  - analyze_campp.py  — CAM++（3D-Speaker / ModelScope，CN-Celeb 中文训练）
  - analyze_multi.py  — 任一模型 + bootstrap 误差棒（均值 ± 标准差）+ 合成参考（composite）
- 可选预处理：separate.py 用 Demucs(htdemucs) 剥离背景音乐/伴奏，只留人声
- 全程 CPU：仅推理、数据量小，无需 GPU

## 数据目录命名约定
在 audio/ 下，每个子文件夹是一个"来源"，命名为 <序号>_<名字>_<角色>，角色由后缀自动识别：
  _test  待测者（要判断是否 = 目标人 X）
  _ref   参考（已确认是目标人 X 的录音；可多份不同时期）
  _neg   负样本（已确认不是 X 的其他人；每人单独一个文件夹，切勿混入多人）
例如：1_alice_test/、2_bob_ref/、3_bob_ref/、4_carol_neg/。文件夹里放一个或多个音频，脚本自动切片、求平均。

## 环境搭建
  conda create -n voicerec python=3.11 -y
  conda activate voicerec
  pip install torch==2.12.0 torchaudio==2.11.0 (CPU 版，见上方英文)
  pip install -r requirements.txt
另需 ffmpeg 在 PATH 上（解码 mp3/m4a 等）。模型首次运行自动下载缓存。

## 使用
  python separate.py audio audio_vocals          # 人声分离
  python analyze.py        audio / audio_vocals   # ECAPA
  python analyze_campp.py  audio / audio_vocals   # CAM++
  python analyze_multi.py  audio_vocals campp 100 # 误差棒 + composite
  python make_figures.py                          # 出表/图
Windows 中文控制台用 conda run 可能因 GBK 崩；直接调用环境 python 并强制 PYTHONUTF8=1 更稳。

## 如何换用其他模型 / 其他数据
- 换数据：按命名约定把音频放进 audio/，无需改代码。
- 换模型：判决/校准逻辑与模型无关，只替换"提声纹向量"这一步——
  analyze.py / analyze_campp.py 里的 embed(...)（输入 16k 波形，返回一个 L2 归一化向量）；
  analyze_multi.py 里的 embed_pool(segs)（输入一批小段，返回 (n, D) 归一化数组），按 ecapa/campp 写法加分支。
  段长、采样率等在文件顶部常量（SEG_SEC、SR…）可调。

## 结果解读
- 相似度矩阵：行列交点 = 两来源声纹余弦相似度（越接近 1 越像）。
- 同人基准：参考来源（同一人不同录音）两两相似度——同人真实水平，判断尺子（非 1.0）。
- composite：把所有 _ref 段合并成一个"合成声纹"作统一基准。
- 误差棒：bootstrap B 次重采样得均值 ± 标准差；数据越少棒越宽，如实反映不确定度。
- 判决：待测者是否显著高于负样本、是否达到同人基准。报告存于 results/report_*.json。

## 已知坑（Windows）
- SpeechBrain 默认 symlink 取模型在无管理员的 Windows 上失败（WinError 1314）→ 已用 LocalStrategy.COPY。
- 必须在 import speechbrain 前预热 librosa，否则惰性子模块导入触发 k2_fsa 而崩（见 analyze.py 注释）。
- Demucs 4 + torchaudio 2.11 的 save 需 torchcodec → separate.py 改用 ffmpeg/soundfile，超长文件分窗、子进程隔离。
- ModelScope(CAM++) 首次需补装若干运行期依赖（见 requirements.txt 注释）。

============================================================ -->
