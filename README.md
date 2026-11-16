# 生成式 AI 課程講義

「生成式 AI 文字與圖像」（115 學年度上學期）的個人講義網站：每一講一頁，依投影片架構整理，穿插 💬 老師補充，附名詞對照表與自我檢核。

- 網站：https://donna12301230.github.io/genai-lecture-notes/
- 給 AI 的規則：[AGENTS.md](AGENTS.md)（`CLAUDE.md` 只是指向它）

## 本機預覽

```bash
python -m http.server 8000
```

打開 http://localhost:8000 。直接雙擊 html 檔會讀不到 Markdown。

## 新增一講

對 AI 說：「請先讀 AGENTS.md，把第 NN 講加進網站。」並附上講義原稿。詳細步驟見 AGENTS.md 第 5 節。

## 檢查

```bash
python tools/check.py
```
