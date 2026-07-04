# -*- coding: utf-8 -*-
"""增补采样：把每季度补到 N_TARGET 场。选材打分：talk 优先、剧情/配音游戏降权、偏长。
新选中的行 append 到 manifest_main.tsv，并同步到 audio_evo/manifest.tsv。仅打印 ASCII。"""
import re, os

SCRATCH = os.path.dirname(os.path.abspath(__file__))
SERIES = os.path.join(SCRATCH, "taffy_series.tsv")
MANIFEST = os.path.join(SCRATCH, "manifest_main.tsv")
PROJ_MANIFEST = os.path.join(os.path.dirname(SCRATCH), "audio_evo", "manifest.tsv")

N_TARGET = 4
MIN_DUR = 1800
SLICE_LEN = 420
BLOCK = re.compile(r"联动|连麦|连线|公演|人气赛|水友赛|唱|歌回|K歌|合唱|PK|测试|抽奖|生日会|见面会")
TALK = re.compile(r"杂谈|闲聊|聊天|聊会|唠|读空气|睡前|二创|鉴赏|随便聊|吃饭")
# 剧情/配音重(人声易污染)——软降权
STORY = re.compile(r"恋爱|视觉小说|gal|AVG|逆转|弹丸|我推|完蛋|美女|侦探|文字|奇异人生|底特律|"
                   r"极乐|港诡|烟火|纸嫁衣|尼尔|巫师|赛博|生化|天国|明末|卧龙|女神异闻|P5|如龙|大表哥")


def quarter(d):
    y, m, _ = d.split("-")
    return f"{y}-Q{(int(m)-1)//3+1}"


def slice_plan(dur):
    L = SLICE_LEN if dur >= 2 * SLICE_LEN else int(dur * 0.5)
    start = max(int(0.35 * dur), 300)
    if start + L > dur - 120:
        start = max(60, dur - 120 - L)
    return start, L


def score(dur, title):
    s = 0.0
    if TALK.search(title):
        s += 3.0
    if STORY.search(title):
        s -= 1.5
    s += min(dur / 3600.0, 4.0) * 0.4      # 偏长（封顶4h）
    return s


# 已有选择
existing_bv, byq_have = set(), {}
with open(MANIFEST, encoding="utf-8") as f:
    hdr = next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        existing_bv.add(p[2])
        byq_have[p[0]] = byq_have.get(p[0], 0) + 1

# 全部候选按季度
byq_cand = {}
with open(SERIES, encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        date, bvid, dur, title = p[0], p[1], int(p[2]), p[3]
        byq_cand.setdefault(quarter(date), []).append((date, bvid, dur, title))

new_rows = []
for q in sorted(k for k in byq_cand if not k.startswith("2026-Q3")):
    need = N_TARGET - byq_have.get(q, 0)
    if need <= 0:
        continue
    cand = [c for c in byq_cand[q] if c[2] >= MIN_DUR and c[1] not in existing_bv and not BLOCK.search(c[3])]
    cand.sort(key=lambda c: c[0])                       # 按日期
    n = len(cand)
    picks = []
    for i in range(min(need, n)):                       # 日期分段，每段取分最高
        seg = cand[i * n // need:(i + 1) * n // need] or [cand[min(i, n - 1)]]
        picks.append(max(seg, key=lambda c: score(c[2], c[3])))
    for date, bvid, dur, title in picks:
        s, L = slice_plan(dur)
        new_rows.append((q, date, bvid, dur, s, L, title))

# append
with open(MANIFEST, "a", encoding="utf-8") as f:
    for r in new_rows:
        f.write("\t".join(str(x) for x in r) + "\n")
# 同步到项目（供 analyze 读日期）
import shutil
shutil.copyfile(MANIFEST, PROJ_MANIFEST)

print("added rows:", len(new_rows))
from collections import Counter
c = Counter(r[0] for r in new_rows)
for k in sorted(c):
    print(f"  {k}  +{c[k]}  (had {byq_have.get(k,0)} -> {byq_have.get(k,0)+c[k]})")
