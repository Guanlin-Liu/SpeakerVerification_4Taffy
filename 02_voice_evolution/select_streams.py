# -*- coding: utf-8 -*-
"""从 taffy_series.tsv 里，按季度自动选采样场次，生成 manifest（主选 + 备选）。仅打印 ASCII 统计。"""
import re, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taffy_series.tsv")
DST = os.path.dirname(os.path.abspath(__file__))

MIN_DUR = 1800          # >=30 min
SLICE_LEN = 420         # 7 min 中段切片
PICKS = 2               # 每季度主选
END_PICKS = 3           # 首尾季度多选一场（anchor）
BLOCK = re.compile(r"联动|连麦|连线|公演|人气赛|水友赛|唱|歌回|K歌|合唱|PK|测试|抽奖|生日会|见面会")

def quarter(d):
    y, m, _ = d.split("-")
    return f"{y}-Q{(int(m)-1)//3+1}"

def slice_plan(dur):
    L = SLICE_LEN if dur >= 2*SLICE_LEN else int(dur*0.5)
    start = max(int(0.35*dur), 300)
    if start + L > dur - 120:
        start = max(60, dur - 120 - L)
    return start, L

rows = []
with open(SRC, encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        date, bvid, dur, title = p[0], p[1], int(p[2]), p[3]
        rows.append((date, bvid, dur, title))

# 分季度
byq = {}
for r in rows:
    byq.setdefault(quarter(r[0]), []).append(r)

qs = sorted(k for k in byq if not k.startswith("2026-Q3"))  # 丢掉只有 2 场的尾巴 stub

def pick_spread(cands, k):
    """按日期排序后均分 k 段，各取中位一场，保证季度内时间均匀。"""
    cands = sorted(cands, key=lambda r: r[0])
    n = len(cands)
    out = []
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        seg = cands[lo:hi] if hi > lo else cands[lo:lo+1]
        out.append(seg[len(seg)//2])
    return out

main, alt = [], []
for qi, q in enumerate(qs):
    cands = [r for r in byq[q] if r[2] >= MIN_DUR and not BLOCK.search(r[3])]
    k = END_PICKS if (qi == 0 or qi == len(qs)-1) else PICKS
    chosen = pick_spread(cands, min(k, len(cands)))
    chosen_bv = {c[1] for c in chosen}
    for c in chosen:
        s, L = slice_plan(c[2])
        main.append((q, c[0], c[1], c[2], s, L, c[3]))
    # 备选：每段中位之外，各段再取一个，供替换
    extra = [c for c in pick_spread(cands, min(2*k, len(cands))) if c[1] not in chosen_bv]
    for c in extra[:k]:
        alt.append((q, c[0], c[1], c[2], c[3]))

# 用户手动指定必须包含的场次（纯人声等）
FORCE = [("2021-Q3", "2021-08-13", "BV19f4y1V7tM", 7670, "【直播回放】看流星雨！ 2021年8月12日23点场")]
have = {r[2] for r in main}
for q, date, bv, dur, title in FORCE:
    if bv not in have:
        s, L = slice_plan(dur)
        main.append((q, date, bv, dur, s, L, title))
main.sort(key=lambda r: (r[0], r[1]))

with open(os.path.join(DST, "manifest_main.tsv"), "w", encoding="utf-8") as f:
    f.write("quarter\tdate\tbvid\tdur_s\tslice_start\tslice_len\ttitle\n")
    for r in main:
        f.write("\t".join(str(x) for x in r) + "\n")

with open(os.path.join(DST, "manifest_alt.tsv"), "w", encoding="utf-8") as f:
    f.write("quarter\tdate\tbvid\tdur_s\ttitle\n")
    for r in alt:
        f.write("\t".join(str(x) for x in r) + "\n")

print("quarters:", len(qs))
print("main picks:", len(main), " alt picks:", len(alt))
tot = sum(r[5] for r in main)
print(f"total slice audio: {tot/60:.0f} min ({tot/3600:.1f} h) to download+separate")
