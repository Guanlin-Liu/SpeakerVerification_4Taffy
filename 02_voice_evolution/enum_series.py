# -*- coding: utf-8 -*-
"""枚举 B 站合集(series)所有分集：bvid / 日期 / 时长 / 标题 -> TSV。只打印 ASCII 统计，标题写文件避免控制台乱码。"""
import os, json, sys, time, urllib.request
from http.cookiejar import MozillaCookieJar
from collections import Counter

MID = 1265680561
SID = 210676
COOKIES = os.environ.get("BILI_COOKIES", "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taffy_series.tsv")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

cj = MozillaCookieJar()
try:
    cj.load(COOKIES, ignore_discard=True, ignore_expires=True)
    print("cookies loaded:", len(cj))
except Exception as e:
    print("cookie load failed (continuing without):", e)

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [("User-Agent", UA), ("Referer", f"https://space.bilibili.com/{MID}/")]


def get(pn, ps=100):
    url = (f"https://api.bilibili.com/x/series/archives?mid={MID}&series_id={SID}"
           f"&only_normal=true&sort=asc&pn={pn}&ps={ps}")
    for attempt in range(3):
        try:
            with opener.open(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            print(f"  page {pn} attempt {attempt} error: {e}")
            time.sleep(2)
    raise SystemExit(f"give up page {pn}")


first = get(1)
print("code:", first.get("code"), "msg:", first.get("message"))
if first.get("code") != 0:
    print("RAW:", json.dumps(first)[:600])
    raise SystemExit("api returned non-zero code")

total = first["data"]["page"]["total"]
print("total archives:", total)
arch = list(first["data"]["archives"])
pages = (total + 99) // 100
for pn in range(2, pages + 1):
    time.sleep(0.6)
    arch += get(pn)["data"]["archives"]
print("collected:", len(arch))

rows = []
for a in arch:
    d = time.strftime("%Y-%m-%d", time.localtime(a["pubdate"]))
    rows.append((d, a["bvid"], a.get("duration", 0), a["title"].replace("\t", " ").replace("\n", " ")))
rows.sort()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("date\tbvid\tdur_s\ttitle\n")
    for d, b, du, t in rows:
        f.write(f"{d}\t{b}\t{du}\t{t}\n")

print("span:", rows[0][0], "->", rows[-1][0])
print("wrote:", OUT)

# 每季度候选场次数（ASCII 直方，用于定 bin）
def q(d):
    y, m, _ = d.split("-")
    return f"{y}-Q{(int(m)-1)//3+1}"

c = Counter(q(r[0]) for r in rows)
print("--- streams per quarter ---")
for k in sorted(c):
    print(f"{k}  {c[k]:3d}  " + "#" * min(c[k], 60))
