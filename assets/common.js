/* 共用程式：目錄、頁首頁尾、Markdown 轉換。
 * 新增講義不需要改這個檔案（見 AGENTS.md 第 5 節）。 */

const SITE = {
  name: "生成式 AI 課程講義",
  repo: "https://github.com/Donna12301230/genai-lecture-notes",
};

/* ---------- 小工具 ---------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch (e) { /* 無痕模式或封鎖儲存時就不記錄 */ }
  return null;
}

async function loadLessons() {
  const res = await fetch("notes/lessons.json", { cache: "no-cache" });
  if (!res.ok) throw new Error("讀不到 notes/lessons.json");
  return res.json();
}
async function loadNote(id) {
  const res = await fetch(`notes/lecture-${id}.md`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`讀不到 notes/lecture-${id}.md`);
  return res.text();
}

/* ---------- 頁首頁尾 ---------- */
function renderChrome(active) {
  const nav = [
    ["index.html", "講義目錄", "home"],
    ["glossary.html", "名詞總表", "glossary"],
    ["about.html", "關於本站", "about"],
  ];
  const header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML = `<div class="wrap">
    <a class="brand" href="index.html">生成式 AI <span>課程講義</span></a>
    <nav class="site-nav" aria-label="主選單">
      ${nav.map(([href, label, key]) => `<a href="${href}"${key === active ? ' aria-current="page"' : ""}>${label}</a>`).join("")}
    </nav></div>`;
  document.body.prepend(header);

  const footer = document.createElement("footer");
  footer.className = "site-footer";
  footer.innerHTML = `個人學習筆記，依課堂投影片與上課內容整理；內容若有錯誤以老師課堂為準。<br>
    <a href="${SITE.repo}">GitHub 原始碼</a> · 這個網站依 <a href="${SITE.repo}/blob/main/AGENTS.md">AGENTS.md</a> 的規則由 AI 協作維護`;
  document.body.append(footer);
}

/* ---------- Markdown → HTML（保護數學式） ---------- */
function renderMarkdown(md) {
  const math = [];
  const keep = (tex, display) => {
    math.push({ tex, display });
    return `KATEXPH${math.length - 1}END`;
  };
  // 只在程式碼以外的地方找數學式
  const parts = md.split(/(^```[\s\S]*?^```)/m);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i]
      .split(/(`+[^`\n]*?`+)/)
      .map((seg, j) => {
        if (j % 2 === 1) return seg; // 行內程式碼
        return seg
          .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => keep(t, true))
          .replace(/\$(?=\S)([^$\n]+?)(?<=\S)\$/g, (_, t) => keep(t, false));
      })
      .join("");
  }
  let html = marked.parse(parts.join(""), { gfm: true, breaks: true });
  html = html.replace(/KATEXPH(\d+)END/g, (_, n) => {
    const { tex, display } = math[+n];
    try {
      return katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false });
    } catch (e) {
      return esc(tex);
    }
  });
  return html;
}

function renderInline(md) {
  const html = renderMarkdown(md);
  return html.replace(/^<p>|<\/p>\s*$/g, "");
}

/* ---------- 講義頁的後處理：依 AGENTS.md 第 4 節的格式加效果 ---------- */
function enhanceArticle(root, id) {
  // 表格：可橫向捲動
  root.querySelectorAll("table").forEach((t) => {
    const w = document.createElement("div");
    w.className = "table-wrap";
    t.replaceWith(w);
    w.append(t);
  });

  // 💬 老師補充 → 便利貼框（含下方緊接的條列）
  [...root.querySelectorAll(":scope > p")].forEach((p) => {
    if (!p.textContent.trim().startsWith("💬 老師補充")) return;
    const box = document.createElement("aside");
    box.className = "teacher-note";
    p.before(box);
    box.append(p);
    let next = box.nextElementSibling;
    while (next && (next.tagName === "UL" || next.tagName === "OL")) {
      const n = next.nextElementSibling;
      box.append(next);
      next = n;
    }
  });

  // 最重要的觀念 → 綠色重點框
  root.querySelectorAll(":scope > p").forEach((p) => {
    const first = p.firstElementChild;
    if (first && first.tagName === "STRONG" && first.textContent.startsWith("最重要的觀念")) p.classList.add("key-idea");
  });

  // 標題 id、課堂口述標記
  let n = 0;
  root.querySelectorAll("h2, h3").forEach((h) => {
    const text = h.textContent.trim();
    if (text === "名詞對照表") h.id = "glossary";
    else if (text === "自我檢核") h.id = "quiz";
    else h.id = `s${++n}`;
    if (text.startsWith("💬")) {
      h.classList.add("spoken");
      h.innerHTML = h.innerHTML.replace("💬", "").trim();
    }
  });

  // 自我檢核 → 「我會了」勾選
  const quizHead = root.querySelector("#quiz");
  let list = quizHead && quizHead.nextElementSibling;
  while (list && list.tagName !== "OL" && list.tagName !== "H2") list = list.nextElementSibling;
  if (list && list.tagName === "OL") {
    list.classList.add("quiz");
    const items = [...list.children];
    const meter = document.createElement("p");
    meter.className = "quiz-meter";
    quizHead.after(meter);
    const update = () => {
      const done = items.filter((li) => li.classList.contains("done")).length;
      meter.textContent = `已掌握 ${done} / ${items.length} 題（只記在你自己的瀏覽器）`;
      store(`quiz-${id}-done`, String(done));
    };
    store(`quiz-${id}-total`, String(items.length));
    items.forEach((li, i) => {
      const key = `quiz-${id}-${i + 1}`;
      const q = document.createElement("span");
      q.className = "q";
      while (li.firstChild) q.append(li.firstChild);
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox"> 我會了`;
      const box = label.querySelector("input");
      box.checked = store(key) === "1";
      li.classList.toggle("done", box.checked);
      box.addEventListener("change", () => {
        li.classList.toggle("done", box.checked);
        store(key, box.checked ? "1" : null);
        update();
      });
      li.append(q, label);
    });
    update();
  }

  // mermaid 流程圖
  const blocks = root.querySelectorAll("pre > code.language-mermaid");
  if (blocks.length && window.mermaid) {
    blocks.forEach((code) => {
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      code.parentElement.replaceWith(div);
    });
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral", securityLevel: "strict" });
    mermaid.run({ nodes: root.querySelectorAll(".mermaid") }).catch(() => {});
  }
}

/* ---------- 上課錄影 ---------- */
function videoBlock(videoId) {
  const wrap = document.createElement("section");
  wrap.className = "video";
  wrap.setAttribute("aria-label", "上課錄影");
  wrap.innerHTML = `<div class="video-frame">
      <iframe src="https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}" title="上課錄影"
        loading="lazy" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>
    </div>
    <p class="video-note">🎬 上課錄影　·　講義裡的「影片約 H:MM」可以點，播放器會跳到那個時間　·
      <a href="https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}" target="_blank" rel="noopener">在 YouTube 開啟</a></p>`;
  const iframe = wrap.querySelector("iframe");
  wrap.seek = (sec) => {
    iframe.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}?start=${sec}&autoplay=1`;
    wrap.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  return wrap;
}

// 把含「影片」字樣段落裡的 H:MM 或 H:MM:SS 變成可點的時間連結
function linkTimestamps(root, onSeek) {
  const re = /(\d{1,2}):(\d{2})(?::(\d{2}))?/g;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const el = node.parentElement;
      if (!el || el.closest("code, pre, a, .katex, .video")) return NodeFilter.FILTER_REJECT;
      const block = el.closest("p, li, h2, h3, h4, td, blockquote");
      if (!block || !block.textContent.includes("影片")) return NodeFilter.FILTER_REJECT;
      re.lastIndex = 0;
      return re.test(node.data) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const frag = document.createDocumentFragment();
    let last = 0;
    node.data.replace(re, (m, h, mm, ss, offset) => {
      frag.append(node.data.slice(last, offset));
      const sec = +h * 3600 + +mm * 60 + (ss ? +ss : 0);
      const a = document.createElement("a");
      a.href = "#";
      a.className = "timestamp";
      a.textContent = m;
      a.title = `跳到影片 ${m}`;
      a.addEventListener("click", (e) => { e.preventDefault(); onSeek(sec); });
      frag.append(a);
      last = offset + m.length;
      return m;
    });
    frag.append(node.data.slice(last));
    node.replaceWith(frag);
  });
}

/* ---------- 名詞對照表解析（名詞總表用） ---------- */
function parseGlossary(md) {
  const m = md.split(/^## 名詞對照表\s*$/m)[1];
  if (!m) return [];
  const section = m.split(/^## /m)[0];
  const rows = [];
  section.split("\n").forEach((line) => {
    const t = line.trim();
    if (!t.startsWith("|") || /^\|[\s:|-]+\|$/.test(t)) return;
    const cells = t.replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((c) => c.trim());
    if (cells.length < 2 || cells[0] === "名詞") return;
    rows.push({ term: cells[0], desc: cells.slice(1).join("｜") });
  });
  return rows;
}
