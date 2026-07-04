# -*- coding: utf-8 -*-
"""用 yt-dlp 下载音频到 audio/<folder>/。

yt-dlp 不在 voicerec 环境里（在你的 GPTSoVits 环境）。本脚本按顺序查找 yt-dlp：
  1) 环境变量 YT_DLP 指定的可执行文件
  2) GPTSoVits 环境的 Scripts/yt-dlp.exe
  3) PATH 上的 yt-dlp

用法：
  python download.py <folder> <url> [more_urls...]
  python download.py <folder> --section 00:10:00-00:14:00 <url>

示例：
  python download.py 4_Taffy_ref "https://www.youtube.com/watch?v=xxxx"
  python download.py 1_Cui_test --section 00:05:00-00:09:00 "https://..."

输出为压缩 m4a（长音频省空间；分析时再随机截取）。需要 ffmpeg 在 PATH。
"""
import os
import sys
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CANDIDATES = [
    os.environ.get("YT_DLP"),
    r"D:\Codes\Anaconda\envs\GPTSoVits\Scripts\yt-dlp.exe",
    shutil.which("yt-dlp"),
]


def find_ytdlp():
    for c in _CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit("找不到 yt-dlp。请用环境变量 YT_DLP 指定其完整路径，或把它加入 PATH。")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)

    folder = args[0]
    rest = args[1:]
    section = None
    if rest and rest[0] == "--section":
        section = rest[1]
        rest = rest[2:]
    urls = rest
    if not urls:
        sys.exit("请至少提供一个 URL。")

    outdir = os.path.join(ROOT, "audio", folder)
    os.makedirs(outdir, exist_ok=True)
    ytdlp = find_ytdlp()
    print(f"使用 yt-dlp: {ytdlp}\n输出目录: {outdir}\n", flush=True)

    for url in urls:
        cmd = [ytdlp, "-x", "--audio-format", "m4a",
               "-o", os.path.join(outdir, "%(title)s.%(ext)s")]
        if section:
            cmd += ["--download-sections", f"*{section}", "--force-keyframes-at-cuts"]
        cmd.append(url)
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print("\n完成 ->", outdir)


if __name__ == "__main__":
    main()
