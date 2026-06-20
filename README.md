# Speaker Verification 声纹比对工作流

把"两段声音听起来像不像"这个**主观问题**，变成可量化的**客观判定**：用预训练的说话人嵌入（speaker embedding）模型，判断不同音频是否来自**同一个人**。支持人声分离预处理、双模型交叉验证、以及 bootstrap **误差棒**。

> 本仓库是**通用工具**——你可以放入**自己的音频**、换用**其他模型**来测试。`figures/` 里是一个案例研究的结果图，仅作示例参考。

## 方法概述

每段音频 → 切成若干小段（去静音）→ 预训练模型提取声纹向量（如 192 维）→ 段向量平均得到该来源的"声纹" → 计算来源之间的**余弦相似度**。判决**不依赖固定阈值**，而是用已知"同一人"（参考内部）与"不同人"（负样本）的样本**自校准**出决策边界。

- **模型**（架构/训练语料不同，结论一致则更可信）
  - `analyze.py` — **ECAPA-TDNN**（SpeechBrain，VoxCeleb 英文训练）
  - `analyze_campp.py` — **CAM++**（3D-Speaker / ModelScope，CN-Celeb 中文训练）
  - `analyze_multi.py` — 任一模型 + **bootstrap 误差棒（均值 ± 标准差）** + 合成参考（composite）
- **可选预处理**：`separate.py` 用 Demucs(htdemucs) 剥离背景音乐/伴奏，只留人声
- **全程 CPU**：仅推理、数据量小，无需 GPU

## 项目结构

```
.
├── analyze.py          # ECAPA 单次分析
├── analyze_campp.py    # CAM++ 单次分析（同逻辑、换模型）
├── analyze_multi.py    # bootstrap 误差棒 + composite（python analyze_multi.py <dir> <ecapa|campp> [B]）
├── separate.py         # Demucs 人声分离：audio/ -> audio_vocals/
├── download.py         # yt-dlp 下载音频到 audio/<folder>/
├── make_figures.py     # 把 results/report_multi_*.json 渲染成表/图
├── requirements.txt
├── figures/            # 示例结果图（一个案例研究的 4 组 model×data 输出，已提交作参考）
├── audio/              # 你的原始音频（gitignore；按下面约定放置）
├── audio_vocals/       # 人声分离结果（gitignore，结构镜像 audio/）
├── models/  results/  tables/   # 模型缓存 / 报告 / 矩阵 CSV（gitignore）
```

### 数据目录命名约定

在 `audio/` 下，每个子文件夹是一个"来源"，命名为 `<序号>_<名字>_<角色>`，角色由后缀自动识别：

| 角色后缀 | 含义 |
|---|---|
| `_test` | 待测者（要判断是否 = 目标人 X） |
| `_ref`  | 参考（已确认是目标人 X 的录音；可多份不同时期的录音） |
| `_neg`  | 负样本（已确认**不是** X 的其他人；**每人单独一个文件夹**，切勿把多人混进一个） |

例如：`1_alice_test/`、`2_bob_ref/`、`3_bob_ref/`、`4_carol_neg/`。文件夹里放一个或多个音频（wav/mp3/m4a/flac…），脚本自动切片、求平均。矩阵里只显示中间的名字。

## 环境搭建

```bash
conda create -n voicerec python=3.11 -y
conda activate voicerec
pip install torch==2.12.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
另需 **ffmpeg** 在 PATH 上（解码 mp3/m4a 等）。声纹/分离模型首次运行自动下载缓存。

## 使用

```bash
# 1)（可选）下载音频（需自备 yt-dlp）
python download.py 2_bob_ref "https://..."

# 2)（可选）人声分离：audio/ -> audio_vocals/
python separate.py audio audio_vocals

# 3) 单次分析（原始 与 分离后各跑；两个模型交叉验证）
python analyze.py        audio
python analyze.py        audio_vocals
python analyze_campp.py  audio
python analyze_campp.py  audio_vocals

# 4) 带误差棒 + composite 的分析（B=重采样次数）
python analyze_multi.py  audio_vocals  campp  100

# 5) 把 multi 报告渲染成表/图
python make_figures.py
```

> Windows 中文控制台用 `conda run` 可能因 GBK 编码崩；直接调用环境 python 并强制 UTF-8 更稳：
> `PYTHONUTF8=1 .../envs/voicerec/python.exe analyze.py audio`

## 如何换用其他模型 / 其他数据

- **换数据**：只需按上面的命名约定把你的音频放进 `audio/`，无需改代码。
- **换模型**：判决/校准逻辑与模型**无关**，只需替换"提声纹向量"这一步——
  - 在 `analyze.py` / `analyze_campp.py` 里，找到 `embed(...)`（输入一段 16k 波形，返回一个 **L2 归一化**向量）；换成你的模型即可。
  - 在 `analyze_multi.py` 里，模型后端集中在 `embed_pool(segs)`（输入一批小段、返回 `(n, D)` 归一化向量数组）；按 `ecapa`/`campp` 的写法新增一个分支即可。
  - 段长、采样率等在文件顶部常量（`SEG_SEC`、`SR`…）可调。

## 结果解读

- **相似度矩阵**：行列交点 = 两来源声纹的余弦相似度（越接近 1 越像）。
- **同人基准**：参考来源（同一人不同录音）两两相似度——代表"同一人在不同录音下"的真实水平，是判断待测者的尺子（而非 1.0）。
- **composite**（`analyze_multi.py`）：把所有 `_ref` 的段合并成一个"合成声纹"作为统一基准。
- **误差棒**：bootstrap B 次重采样得到均值 ± 标准差；数据越少误差棒越宽，如实反映不确定度。
- **判决**：待测者对目标人的相似度，是否显著高于负样本、是否达到"同人基准"水平。
- 报告存于 `results/report_*.json`。

## 已知坑（Windows）

- SpeechBrain 默认 symlink 取模型在无管理员的 Windows 上失败（WinError 1314）→ 代码已用 `LocalStrategy.COPY`。
- 必须在 `import speechbrain` 前"预热"librosa，否则其惰性子模块导入会触发 speechbrain 的 `k2_fsa` 而崩（见 `analyze.py` 注释）。
- Demucs 4 + torchaudio 2.11 的 `save` 需 torchcodec → `separate.py` 改用 ffmpeg/soundfile，并对超长文件分窗、子进程隔离。
- ModelScope(CAM++) 首次需补装若干运行期依赖（见 `requirements.txt` 注释）。

## Roadmap

- [ ] **声线随时间演化研究**：对单个 speaker，用"声纹相似度 + 声学/韵律特征（基频/共振峰等）"双指标，分析声音是否随时间变化。

## 致谢 / 依赖

[SpeechBrain](https://github.com/speechbrain/speechbrain) · [3D-Speaker / CAM++](https://github.com/modelscope/3D-Speaker) · [Demucs](https://github.com/facebookresearch/demucs) · [ModelScope](https://github.com/modelscope/modelscope)
