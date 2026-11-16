"""完成檢查：python tools/check.py

依 AGENTS.md 第 8 節，自動檢查講義檔、目錄檔與隱私。
只用 Python 標準函式庫；有任何 ❌ 時結束代碼為 1。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"
REQUIRED_HEADINGS = ["## 本講重點總覽", "## 名詞對照表", "## 自我檢核"]
REQUIRED_FILES = ["AGENTS.md", "CLAUDE.md", "index.html", "lesson.html",
                  "glossary.html", "about.html", "assets/style.css",
                  "assets/common.js", ".nojekyll", "templates/lecture-template.md"]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+")
PHONE_RE = re.compile(r"09\d{2}-?\d{3}-?\d{3}")
KEY_RE = re.compile(r"(sk-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,})")

errors, warnings, oks = [], [], []


def ok(msg):
    oks.append(msg)


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


# 1. 必要檔案
missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
if missing:
    err("缺少必要檔案：" + "、".join(missing))
else:
    ok("必要檔案都在（含 .nojekyll）")

# 2. 讀目錄
try:
    data = json.loads((NOTES / "lessons.json").read_text(encoding="utf-8"))
    lessons = data["lessons"]
    ok(f"lessons.json 格式正確，共 {len(lessons)} 講")
except Exception as e:  # noqa: BLE001
    print(f"❌ lessons.json 讀取失敗：{e}")
    sys.exit(1)

ids = [l.get("id", "") for l in lessons]
if ids != sorted(ids):
    err("lessons.json 沒有依講次排序：" + ", ".join(ids))
if len(set(ids)) != len(ids):
    err("lessons.json 有重複的 id")

files = sorted(p.name for p in NOTES.glob("*.md"))
bad_names = [f for f in files if not re.fullmatch(r"lecture-\d{2}\.md", f)]
if bad_names:
    err("檔名不符合 lecture-NN.md：" + "、".join(bad_names))

file_ids = {re.findall(r"\d{2}", f)[0] for f in files if f not in bad_names}
not_listed = sorted(file_ids - set(ids))
no_file = sorted(set(ids) - file_ids)
if not_listed:
    err("有講義檔但沒登記在 lessons.json：" + "、".join(not_listed))
if no_file:
    err("lessons.json 有登記但找不到講義檔：" + "、".join(no_file))
if not not_listed and not no_file:
    ok("講義檔與 lessons.json 一一對應")

# 3. 逐講檢查
for l in lessons:
    lid = l.get("id", "??")
    path = NOTES / f"lecture-{lid}.md"
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0].strip() if text else ""
    problems = []

    for key in ("id", "title", "subtitle", "summary", "tags"):
        if not l.get(key):
            problems.append(f"lessons.json 缺 {key}")

    m = re.fullmatch(r"# 第 (\d{2}) 講｜(.+)", first)
    if not m:
        problems.append(f"第一行格式不對：{first[:40]}")
    else:
        if m.group(1) != lid:
            problems.append(f"第一行講次 {m.group(1)} 和 id {lid} 不一致")
        heading = m.group(2)
        t, s = l.get("title", ""), l.get("subtitle", "")
        if heading not in (f"{t}：{s}", f"{t}{s}"):
            problems.append(f"第一行標題和 lessons.json 不一致：「{heading}」")

    for h in REQUIRED_HEADINGS:
        if not re.search(rf"^{re.escape(h)}\s*$", text, re.M):
            problems.append(f"缺少固定標題「{h}」")

    if "**最重要的觀念**" not in text:
        problems.append("缺少「**最重要的觀念**」段落")

    quiz = text.split("## 自我檢核", 1)[-1]
    n_q = len(re.findall(r"^\d+\. ", quiz, re.M))
    if not 5 <= n_q <= 25:
        problems.append(f"自我檢核有 {n_q} 題（應為 5～25 題）")

    video = l.get("video", "")
    if not video:
        warn(f"第 {lid} 講沒有 video（上課錄影 ID）")
    elif not re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
        problems.append(f"video 應該是 11 字元的 YouTube ID，不是網址：{video}")
    else:
        head = "\n".join(text.splitlines()[:8])
        linked = re.findall(r"(?:live/|youtu\.be/|[?&]v=)([A-Za-z0-9_-]{11})", head)
        if linked and video not in linked:
            problems.append(f"資料來源的錄影連結（{linked[0]}）和 video（{video}）不一致")

    summary = l.get("summary", "")
    if summary and not 30 <= len(summary) <= 80:
        warn(f"第 {lid} 講 summary 有 {len(summary)} 字（建議 40～70 字）")
    for tag in l.get("tags", []):
        if tag.lower() not in text.lower():
            warn(f"第 {lid} 講的 tag「{tag}」沒有出現在講義內文")

    # 「學號@nccu.edu.tw」這類說明寫法不會被抓到（@ 前面不是英數字）
    for em in EMAIL_RE.finditer(text):
        problems.append(f"疑似個人 Email：{em.group(0)}")
    if PHONE_RE.search(text):
        problems.append("疑似手機號碼")
    if KEY_RE.search(text):
        problems.append("疑似 API key")

    if problems:
        for p in problems:
            err(f"第 {lid} 講：{p}")
    else:
        ok(f"第 {lid} 講：格式、標題、隱私檢查通過（自我檢核 {n_q} 題）")

# 4. 輸出
for m in oks:
    print("✅", m)
for m in warnings:
    print("⚠️ ", m)
for m in errors:
    print("❌", m)
print()
if errors:
    print(f"共 {len(errors)} 個錯誤，請修正後再跑一次。")
    sys.exit(1)
print("全部通過 🎉" + (f"（{len(warnings)} 個提醒可以順手看一下）" if warnings else ""))
