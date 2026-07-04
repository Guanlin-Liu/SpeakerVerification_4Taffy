# Speaker Verification & Voice Evolution

Two related, **CPU‑only** tools built on pretrained speaker embeddings:

1. **Speaker verification** (`01_speaker_verification/`) — turn the subjective *"do these two voices sound alike?"* into a quantitative **same‑person / different‑person** decision, with vocal‑separation preprocessing, two‑model cross‑checking, and bootstrap **error bars**.
2. **Voice evolution over time** (`02_voice_evolution/`) — for a **single** speaker, measure how the voice **drifts across years**, calibrated against session noise and reference landmarks — a *"voice Hubble diagram"*.

> Both are **general‑purpose** — drop in your **own audio** and plug in **other models**. The PNGs in `figures/` are from one case study (a VTuber), included only as reference examples.

## Repository layout

```
.
├── 01_speaker_verification/        # Part 1 — same person?
│   ├── analyze_multi.py            #   bootstrap error bars + composite reference
│   ├── separate.py  download.py    #   Demucs vocal separation / yt-dlp helper
│   ├── make_figures.py             #   render results/report_multi_*.json → tables/figures
│   └── legacy/ analyze.py analyze_campp.py    # earlier single-model versions (superseded)
├── 02_voice_evolution/             # Part 2 — does the voice change over time?
│   ├── analyze_evolution.py        #   θ-to-"now" + session-noise floor + PCA/MDS  (general)
│   ├── make_figures_evolution.py   #   Hubble diagram + 2D trajectory
│   ├── enum_series.py  select_streams.py  select_streams_aug.py   # dataset sampling (Bilibili)
│   ├── run_pipeline.py  add_slices.py  dl_now.py                  # download slices + separation
│   └── extract_summary.py  manifest_main.tsv
├── figures/                        # example result images (committed for reference)
├── requirements.txt
└── (audio*/  audio_evo*/  models/  results/  tables/  video/  → gitignored)
```

## Part 1 — Speaker verification (`01_speaker_verification/`)

Each clip → split into short segments (silence removed) → a pretrained model extracts a speaker embedding (e.g. 192‑d) per segment → average into that source's "voiceprint" → **cosine similarity** between sources. The decision does **not** use a fixed threshold; it **self‑calibrates** from known same‑person (references) and different‑person (negatives) samples.

- **Models** (different architectures / data → agreement means higher confidence)
  - `legacy/analyze.py` — ECAPA‑TDNN (SpeechBrain, VoxCeleb / English)
  - `legacy/analyze_campp.py` — CAM++ (3D‑Speaker / ModelScope, CN‑Celeb / Chinese)
  - `analyze_multi.py` — either model + **bootstrap error bars (mean ± std)** + a composite reference
- **Optional preprocessing**: `separate.py` (Demucs htdemucs) strips background music / game audio, keeping only the voice.

### Data naming convention
Under `audio/`, each subfolder is one "source", named `<index>_<name>_<role>`; role auto‑detected from the suffix:

| suffix | meaning |
|---|---|
| `_test` | the source under test (is it target person X?) |
| `_ref`  | reference (confirmed X; multiple recordings / periods allowed) |
| `_neg`  | negative (a confirmed different person; **one folder per person**, never mix) |

Example: `1_alice_test/`, `2_bob_ref/`, `3_bob_ref/`, `4_carol_neg/`.

### Usage
```bash
python 01_speaker_verification/download.py 2_bob_ref "https://..."     # (optional) fetch audio
python 01_speaker_verification/separate.py audio audio_vocals          # (optional) vocal separation
python 01_speaker_verification/analyze_multi.py audio_vocals campp 100  # error bars + composite (B rounds)
python 01_speaker_verification/make_figures.py                         # render tables/figures
```

### Reading the results
- **Similarity matrix** — cosine similarity between two voiceprints (→1 = more alike).
- **Same‑person baseline** — pairwise similarity among references (same person, different recordings): the realistic "same person" level and the yardstick for the test source (not 1.0).
- **Composite** — pools all `_ref` segments into one merged reference voiceprint.
- **Error bars** — bootstrap over B rounds → mean ± std; less data → wider bars.
- **Verdict** — whether the test source is significantly closer to X than the negatives, and whether it reaches the same‑person baseline. Reports → `results/report_*.json`.

## Part 2 — Voice evolution over time (`02_voice_evolution/`)

**Question**: does one person's voice actually change over the years, or is it just perception?

**Metric**: **θ = arccos(cosine)** — the angular distance between two voiceprints on the embedding sphere. Pick the **current** voice as the anchor ("now") and measure θ(now, period) for every past period → a **"voice Hubble diagram"** (angular "recession" vs lookback time).

**Calibration (the crux)** — θ is meaningless without a scale:
- **Same‑period floor** — θ between two *independent streams of the SAME period* = the irreducible **session noise** (mic, mood, room, content). A period only counts as "changed" if it rises **above** this band (like peculiar‑velocity dispersion in a Hubble diagram).
- **Landmarks** — a genuinely different person / the speaker's previous identities, placed on the 2D map as scale references.
- **PCA on per‑period centroids + classical MDS on the θ‑matrix** → a 2D trajectory that tells a **smooth drift** (one connected cloud) from a **discrete jump**.

**Run the analysis (general — bring your own data):**
```bash
# expects  audio_evo_vocals/<period>/*.wav  (one folder per time bin) + a  NOW/  anchor folder,
# and  audio_evo/manifest.tsv  mapping each clip id → date.
python 02_voice_evolution/analyze_evolution.py        # → results/report_evolution_campp.json (+ .npz)
python 02_voice_evolution/make_figures_evolution.py   # → figures/evolution_fig1_hubble.png, _fig2_pca_mds.png
```

**Rebuild the case‑study dataset (Bilibili‑specific, for reproducibility):**
```bash
python 02_voice_evolution/enum_series.py         # a Bilibili series → taffy_series.tsv
python 02_voice_evolution/select_streams.py      # quarterly sampling → manifest_main.tsv
python 02_voice_evolution/select_streams_aug.py  # top up to N streams per quarter
python 02_voice_evolution/run_pipeline.py all    # 7-min slices + Demucs → audio_evo_vocals/
python 02_voice_evolution/add_slices.py          # extra early/late slices per video (lower variance)
python 02_voice_evolution/dl_now.py              # the "now" anchor (latest streams)
```
> Needs `yt-dlp` + `ffmpeg` on PATH (or set env `YTDLP`); Bilibili cookies optional via `BILI_COOKIES=/path/to/cookies.txt`. The channel id and sampling rules in `enum_series.py` / `select_streams.py` are specific to this case study — edit them for your own speaker.

**Result (case study, ~5 years):** a mild, gradual, **continuous** drift — the last ~2 years sit **within** session noise of "now" (stable), while earlier periods rise above it; **no discrete jump** (one connected cloud in PCA/MDS). See `figures/evolution_fig1_hubble.png` and `figures/evolution_fig2_pca_mds.png`.

## Setup

```bash
conda create -n voicerec python=3.11 -y
conda activate voicerec
pip install torch==2.12.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Also needs **ffmpeg** on PATH (to decode mp3/m4a and for slice extraction). Embedding / separation models download to cache on first run.

> On a Chinese (GBK) Windows console, `conda run` can crash on UTF‑8 output — call the env python directly with UTF‑8 forced: `PYTHONUTF8=1 .../envs/voicerec/python.exe ...`.

## Using other models / other data

- **Other data**: place audio under the expected folders — no code changes (Part 1: the `audio/` naming convention; Part 2: `audio_evo_vocals/<period>/`).
- **Other model**: the decision / calibration logic is **model‑agnostic**; only replace the embedding step — `embed_pool(segs)` (returns an `(n, D)` array of **L2‑normalized** vectors) in `analyze_multi.py` and `analyze_evolution.py`. Segment length / sample rate are top‑of‑file constants (`SEG_SEC`, `SR`, …).

## Known issues (Windows)

- SpeechBrain's default symlink fetch fails without admin (WinError 1314) → the code uses `LocalStrategy.COPY`.
- librosa must be "warmed up" before `import speechbrain` (see comments in `legacy/analyze.py`).
- Demucs 4 + torchaudio 2.11 `save` needs torchcodec → separation uses the demucs Python API + soundfile instead.
- ModelScope (CAM++) needs a few extra runtime deps on first use (see `requirements.txt`).
- `yt-dlp` slice download (`--download-sections`) can truncate or mis‑offset on multi‑part (分P) videos; the pipeline retries / falls back and skips empty slices.

## Roadmap

- [x] **Voice evolution over time** — done, see `02_voice_evolution/` (θ‑to‑"now" Hubble diagram + session‑noise floor + PCA/MDS trajectory).

## Acknowledgements

[SpeechBrain](https://github.com/speechbrain/speechbrain) · [3D‑Speaker / CAM++](https://github.com/modelscope/3D-Speaker) · [Demucs](https://github.com/facebookresearch/demucs) · [ModelScope](https://github.com/modelscope/modelscope) · [yt-dlp](https://github.com/yt-dlp/yt-dlp)

<!-- ============================================================
中文版 README（保留在文件中，GitHub 页面不渲染显示）
============================================================

# 声纹比对 & 声音演化

基于预训练说话人嵌入（speaker embedding）的两个 CPU 工具：

1. 声纹比对（01_speaker_verification/）——把"两段声音像不像"变成可量化的"同一人/不同人"判定；支持人声分离、双模型交叉验证、bootstrap 误差棒。
2. 声音演化（02_voice_evolution/）——针对单个人，度量其声音随年份的漂移，用 session 噪声与地标校准——一张"声音 Hubble 图"。

figures/ 是一个案例（某 VTuber）的结果图，仅作示例；两者都是通用工具，可放自己的音频、换别的模型。

## Part 1 声纹比对
每段音频 → 切小段(去静音) → 模型提 192 维声纹 → 段向量平均 → 余弦相似度。判决不靠固定阈值，用已知"同人(参考)/不同人(负样本)"自校准。
- 模型：legacy/analyze.py(ECAPA)、legacy/analyze_campp.py(CAM++)、analyze_multi.py(任一模型 + 误差棒 + composite)
- 可选：separate.py 用 Demucs 剥背景音，只留人声
- 数据命名：audio/ 下每个子夹是一个来源，<序号>_<名字>_<角色>，角色后缀 _test/_ref/_neg（负样本每人单独一夹）
- 用法：见上方英文（脚本都在 01_speaker_verification/ 下）

## Part 2 声音演化
问题：一个人的声音这些年到底变没变？
度量：θ = arccos(余弦)，嵌入球面上两声纹的夹角。以"现在"为锚点，量每个时期到现在的 θ——即"声音 Hubble 图"。
校准（关键）：
- same-period floor：同一时期两场不同直播之间的 θ = 不可约的 session 噪声；某期要"高出这条带"才算真的变了（类比 Hubble 图里的本动速度弥散）。
- 地标：真·不同人 / 之前身份，放到 PCA/MDS 二维图上当尺子。
- PCA(各期质心) + classical MDS(θ 矩阵) → 二维轨迹，区分"平滑渐变(连成一团)"与"断崖跳变"。
分析（通用）：analyze_evolution.py（读 audio_evo_vocals/<期>/*.wav + NOW/ 锚点 + audio_evo/manifest.tsv 的日期）→ report + npz；make_figures_evolution.py → 两张图。
复现塔菲数据集（B 站专用）：enum_series → select_streams(_aug) → run_pipeline/add_slices/dl_now。需 yt-dlp/ffmpeg 在 PATH（或设 YTDLP），B 站 cookies 用 BILI_COOKIES 环境变量；enum_series/select_streams 里的频道 id 与选材规则是本案例专用，换人需自行调整。
结论（本案例·五年）：温和、渐进、连续的漂移；近两年在 session 噪声以内(稳定)，早期高出噪声；全程无突变(PCA/MDS 一整团连续云)。

## 环境
conda create -n voicerec python=3.11 -y；装 CPU 版 torch/torchaudio；pip install -r requirements.txt；ffmpeg 在 PATH。首次运行自动下载模型缓存。Windows 中文控制台直接调 env python 并 PYTHONUTF8=1 更稳。

============================================================ -->
