"""Gradio web app for the PDF English-fiction annotator.

Users upload an English PDF, choose a CEFR level, and download the annotated
result — no install required. Runs on Hugging Face Spaces, Render, Railway or
any Docker host.

Launch locally::

    pip install -r webapp/requirements.txt
    python webapp/app.py

The app auto-downloads the ECDICT dictionary (~65 MB) and required NLTK data
on first use, caching them under a writable data directory.
"""

import os
import random
import sys
import tempfile
import threading
import urllib.parse

import gradio as gr

# Make the parent package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotator.config import AnnotationConfig  # noqa: E402
from annotator.nltk_setup import ensure_nltk_data  # noqa: E402
from annotator.pipeline import annotate_pdf  # noqa: E402
from prepare_assets import DB_FILENAME, ensure_ecdict_database  # noqa: E402

# A writable directory for the cached dictionary. HF Spaces / most PaaS allow
# writing under the app dir or /tmp.
DATA_DIR = os.environ.get("ANNOTATOR_DATA_DIR") or os.path.join(
    tempfile.gettempdir(), "pdf-annotator-data"
)
ECDICT_PATH = os.path.join(DATA_DIR, DB_FILENAME)
# Small hand-compiled Ming/Qing official-title glossary shipped in the repo;
# resolved relative to this file so it works regardless of the process cwd.
HISTORICAL_GLOSSARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ming_qing_titles.csv",
)

# --- Dictionary presets --------------------------------------------------
# The UI now asks the reader to "choose a dictionary" rather than a raw CEFR
# level. Each named dictionary maps to an internal CEFR threshold that controls
# how many / how rare the annotated words are. The built-in Ming/Qing novel
# glossary is always consulted first regardless of the choice.
DICTIONARY_TO_LEVEL = {
    "通俗词典 · 注释最多（入门）": "A2",
    "常用词典 · 注释较多": "B1",
    "文学词典 · 注释适中（推荐）": "B2",
    "典雅词典 · 仅注释生僻词（进阶）": "C1",
}
DICTIONARY_CHOICES = list(DICTIONARY_TO_LEVEL)
DEFAULT_DICTIONARY = "文学词典 · 注释适中（推荐）"


# --- Literary background artwork -----------------------------------------
# A hand-built SVG evoking a rainy Jiangnan (Jinling) evening: misty ancient
# rooftops and a pagoda, two ladies in qipao under oil-paper umbrellas, and a
# foreground shelf of English book spines. Rendered once at import time and
# embedded as a data-URI so no external image hosting is required.


def _roof(cx, ridge_y, width, height, color, opacity=0.9, body=True):
    """A single sweeping Chinese roof silhouette with upturned eaves."""
    w = float(width)
    h = float(height)
    left = cx - w / 2.0
    right = cx + w / 2.0
    eave_y = ridge_y + h
    flare = w * 0.12
    p0 = (left - flare, eave_y - h * 0.24)
    c1 = (left + w * 0.04, eave_y - h * 0.02)
    p1 = (cx, eave_y + h * 0.06)
    c2 = (right - w * 0.04, eave_y - h * 0.02)
    p2 = (right + flare, eave_y - h * 0.24)
    c3 = (cx, ridge_y - h * 0.12)
    d = ("M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f "
         "Q%.1f,%.1f %.1f,%.1f Q%.1f,%.1f %.1f,%.1f Z") % (
        p0[0], p0[1], c1[0], c1[1], p1[0], p1[1],
        c2[0], c2[1], p2[0], p2[1], c3[0], c3[1], p0[0], p0[1])
    parts = ["<path d='%s' fill='%s' opacity='%.2f'/>" % (d, color, opacity)]
    if body:
        bx = left + w * 0.16
        bw = w * 0.68
        bh = h * 3.6
        parts.append(
            "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' "
            "fill='%s' opacity='%.2f'/>" % (bx, eave_y - 2, bw, bh, color, opacity))
    return "".join(parts)


def _pagoda(cx, base_y, scale, color):
    """A tapering multi-tier pagoda silhouette."""
    parts = []
    th = 26 * scale
    tw = 130 * scale
    y = base_y
    for i in range(5):
        w = tw * (1 - i * 0.14)
        parts.append(_roof(cx, y - th, w, th, color, 0.85, body=False))
        wall_w = w * 0.55
        parts.append(
            "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' "
            "fill='%s' opacity='0.85'/>" % (cx - wall_w / 2, y, wall_w, th, color))
        y -= th * 1.85
    parts.append(
        "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' "
        "stroke-width='%.1f' opacity='0.85'/>"
        % (cx, y + th, cx, y - 26 * scale, color, 3 * scale))
    parts.append(
        "<circle cx='%.1f' cy='%.1f' r='%.1f' fill='%s' opacity='0.85'/>"
        % (cx, y - 26 * scale, 5 * scale, color))
    return "<g>%s</g>" % "".join(parts)


def _umbrella_figure(x, ground, r, canopy, dress):
    """A lady in a qipao holding a round oil-paper umbrella."""
    cy = ground - 250.0
    dome = "M%.1f,%.1f A%.1f,%.1f 0 0 1 %.1f,%.1f Z" % (x - r, cy, r, r, x + r, cy)
    parts = ["<path d='%s' fill='%s' opacity='0.92'/>" % (dome, canopy)]
    parts.append(
        "<g stroke='#3a2f26' stroke-width='1.4' opacity='0.35'>"
        "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f'/>"
        "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f'/>"
        "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f'/>"
        "</g>" % (
            x, cy, x - r * 0.55, cy - r * 0.70,
            x, cy, x, cy - r,
            x, cy, x + r * 0.55, cy - r * 0.70))
    parts.append(
        "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' "
        "stroke-width='2.4'/>" % (x, cy - r, x, cy - r - 14, canopy))
    parts.append(
        "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#4a3b2c' "
        "stroke-width='2.2' opacity='0.85'/>" % (x, cy, x, ground))
    dress_path = ("M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f L%.1f,%.1f "
                  "Q%.1f,%.1f %.1f,%.1f Z") % (
        x - 15, cy + 58,
        x - 24, cy + 150, x - 19, ground,
        x + 19, ground,
        x + 24, cy + 150, x + 15, cy + 58)
    parts.append("<path d='%s' fill='%s' opacity='0.95'/>" % (dress_path, dress))
    parts.append("<circle cx='%.1f' cy='%.1f' r='16' fill='#2c2320'/>" % (x, cy + 42))
    parts.append("<circle cx='%.1f' cy='%.1f' r='11' fill='#e7d3ba'/>" % (x, cy + 46))
    return "<g>%s</g>" % "".join(parts)


def _bookshelf(rng, width, height):
    """A foreground row of English book spines forming a shelf."""
    palette = [
        "#8a5a44", "#6b7a52", "#4a6675", "#9a7b4f", "#7a5566", "#5f6b6e",
        "#a8704a", "#556b5a", "#8b6d3f", "#6a5a7a", "#3f5a66", "#7d4f4a",
    ]
    parts = ["<rect x='0' y='%d' width='%d' height='%d' fill='#4b3b2c' "
             "opacity='0.92'/>" % (height - 156, width, 156)]
    x = 6
    while x < width - 6:
        w = rng.randint(24, 46)
        if x + w > width - 6:
            w = (width - 6) - x
        if w < 12:
            break
        h = rng.randint(112, 150)
        y = height - h
        color = rng.choice(palette)
        parts.append("<rect x='%d' y='%d' width='%d' height='%d' rx='2' "
                     "fill='%s'/>" % (x, y, w, h, color))
        parts.append("<rect x='%d' y='%d' width='%d' height='4' "
                     "fill='#000000' opacity='0.12'/>" % (x, y, w))
        ly = y + rng.randint(22, 42)
        parts.append("<rect x='%d' y='%d' width='%d' height='3' "
                     "fill='#f3ecd8' opacity='0.5'/>" % (x + 5, ly, max(6, w - 10)))
        parts.append("<rect x='%d' y='%d' width='%d' height='3' "
                     "fill='#f3ecd8' opacity='0.35'/>" % (x + 5, ly + 9, max(4, w - 14)))
        x += w + rng.randint(0, 2)
    return "<g>%s</g>" % "".join(parts)


def _lerp_hex(c1, c2, t):
    a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(
        int(round(a[j] + (b[j] - a[j]) * t)) for j in range(3))


def _background_data_uri():
    rng = random.Random(20240723)
    width, height = 1600, 1000
    p = ["<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' "
         "preserveAspectRatio='xMidYMid slice'>" % (width, height)]
    # Warm dusk sky built from solid colour bands (renders in every engine,
    # unlike SVG gradients which some renderers ignore).
    bands = 48
    for i in range(bands):
        t = i / float(bands - 1)
        col = _lerp_hex("#f2e9d5", "#dcc9a3", t)
        y0 = height * i / float(bands)
        p.append("<rect x='0' y='%.1f' width='%d' height='%.1f' fill='%s'/>"
                 % (y0, width, height / float(bands) + 1.0, col))
    p.append("<circle cx='1200' cy='250' r='130' fill='#faf3dc' opacity='0.6'/>")
    p.append("<circle cx='1200' cy='250' r='92' fill='#fdf8e8' opacity='0.75'/>")
    p.append("<path d='M0,470 C220,405 380,455 560,432 C760,404 900,455 1120,422 "
             "C1320,392 1470,442 1600,418 L1600,1000 L0,1000 Z' "
             "fill='#c8b78f' opacity='0.5'/>")
    p.append(_pagoda(1360, 470, 0.85, "#b09a72"))
    p.append(_roof(240, 520, 360, 66, "#7b6950", 0.72))
    p.append(_roof(560, 560, 300, 58, "#6c5b43", 0.80))
    p.append(_roof(150, 610, 280, 54, "#5d4d39", 0.86))
    p.append(_roof(430, 640, 250, 50, "#574837", 0.90))
    p.append(_roof(900, 590, 320, 62, "#5a4a38", 0.85))
    p.append(_roof(1120, 630, 260, 52, "#4f4030", 0.90))
    # Soft mist drifting over the rooftops.
    p.append("<rect x='0' y='430' width='%d' height='210' fill='#f6f0e2' "
             "opacity='0.32'/>" % width)
    p.append("<rect x='0' y='560' width='%d' height='240' fill='#f4ecda' "
             "opacity='0.34'/>" % width)
    rain = []
    for _ in range(140):
        rx = rng.uniform(0, width)
        ry = rng.uniform(0, 800)
        rl = rng.uniform(12, 26)
        rain.append("<line x1='%.0f' y1='%.0f' x2='%.0f' y2='%.0f'/>"
                    % (rx, ry, rx - 7, ry + rl))
    p.append("<g stroke='#b9ab8b' stroke-width='1' opacity='0.38'>%s</g>"
             % "".join(rain))
    p.append("<ellipse cx='730' cy='845' rx='560' ry='58' fill='#cbbc99' "
             "opacity='0.5'/>")
    p.append(_umbrella_figure(645, 815, 80, "#9c3b39", "#3b4a67"))
    p.append(_umbrella_figure(805, 830, 72, "#33445f", "#82486a"))
    p.append(_bookshelf(rng, width, height))
    p.append("</svg>")
    return "data:image/svg+xml;utf8," + urllib.parse.quote("".join(p))


_BACKGROUND_DATA_URI = _background_data_uri()

_CSS_TEMPLATE = """
gradio-app {
  background:
    linear-gradient(rgba(244,238,222,0.34), rgba(226,214,187,0.52)),
    url("__BG__") center bottom / cover no-repeat fixed !important;
}
.gradio-container {
  background: transparent !important;
  max-width: 1060px !important;
  margin: 0 auto !important;
}
#app-header { text-align: center; padding: 30px 20px 6px; color: #3a2f26; }
#app-header .ah-kicker {
  display: inline-block; letter-spacing: 0.35em; font-size: 12px;
  color: #2e6b45; border: 1px solid rgba(46,107,69,0.45);
  border-radius: 999px; padding: 4px 16px; margin-bottom: 14px;
  background: rgba(253,250,243,0.7);
}
#app-header h1 {
  font-family: "Noto Serif SC", Georgia, "Songti SC", serif;
  font-weight: 700; font-size: 34px; margin: 6px 0; color: #33281f;
  text-shadow: 0 1px 0 rgba(255,255,255,0.55);
}
#app-header .ah-sub {
  max-width: 690px; margin: 12px auto 4px; line-height: 1.95;
  font-size: 15px; color: #5a4a3a;
}
#app-header .ah-note { font-size: 12.5px; color: #8a7a63; margin-top: 8px; }
.paper-card {
  background: rgba(253,250,243,0.90) !important;
  border: 1px solid rgba(120,95,60,0.22) !important;
  border-radius: 14px !important;
  box-shadow: 0 10px 34px rgba(60,45,30,0.20) !important;
  padding: 20px !important;
  backdrop-filter: blur(2px);
}
.paper-card label span, .paper-card .gr-check-radio { color: #3a2f26 !important; }
.paper-card input, .paper-card textarea, .paper-card .wrap-inner,
.paper-card .secondary-wrap, .paper-card .container .wrap {
  background: rgba(255,253,248,0.96) !important;
}
button.primary, .primary {
  background: #2e8b57 !important; border: none !important; color: #fff !important;
  font-weight: 600 !important; letter-spacing: 0.06em;
}
button.primary:hover, .primary:hover { background: #276f47 !important; }
#app-footer {
  text-align: center; padding: 16px 12px 26px; color: #6a5a45; font-size: 13px;
  line-height: 1.9;
}
#app-footer a {
  color: #2e6b45; text-decoration: none;
  border-bottom: 1px solid rgba(46,107,69,0.4);
}
footer { display: none !important; }
"""

CUSTOM_CSS = _CSS_TEMPLATE.replace("__BG__", _BACKGROUND_DATA_URI)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Noto Serif SC"), "Georgia", "serif"],
)

HEADER_HTML = """
<div id="app-header">
  <div class="ah-kicker">江 南 夜 雨 · 书 斋</div>
  <h1>📖 英文小说 · 中文注释书斋</h1>
  <p class="ah-sub">上传一本英文原版小说的 PDF，为其中的<b>生词、短语与习语</b>
  添上简洁的中文释义。注释安放在页面右侧扩展出的留白处，<b>不遮挡原文</b>，
  原书排版与目录链接保持如初。</p>
  <div class="ah-note">词典已在服务端预先建立索引 · 免费服务器首次唤醒可能需要片刻</div>
</div>
"""

FOOTER_HTML = """
<div id="app-footer">
  免费开源 · MIT 许可证 · 词典来自
  <a href="https://github.com/skywind3000/ECDICT" target="_blank">ECDICT</a>
  与内置明清小说词典 ·
  <a href="https://github.com/Tao-Yang/pdf-english-fiction-annotator" target="_blank">源码</a><br/>
  请仅对你拥有合法权利的 PDF 使用本工具
</div>
"""

_READY = False
_ASSET_LOCK = threading.Lock()


def _ensure_assets(progress=None) -> None:
    """Prepare NLTK data and the disk-backed ECDICT database once."""
    global _READY
    if _READY:
        return
    with _ASSET_LOCK:
        if _READY:
            return
        if progress:
            progress(0.05, desc="准备语言数据 (NLTK)…")
        ensure_nltk_data()

        def update_status(message):
            if progress:
                progress(0.2, desc=message)

        ensure_ecdict_database(DATA_DIR, update_status)
        _READY = True


def annotate(pdf_file, dictionary, start_page, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("请先上传一个英文 PDF 文件。")

    level = DICTIONARY_TO_LEVEL.get(dictionary, "B2")
    _ensure_assets(progress)

    src_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    out_dir = tempfile.mkdtemp(prefix="annotated-")
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, "%s-annotated-%s.pdf" % (stem, level))

    config = AnnotationConfig(
        cefr_level=level,
        ecdict_path=ECDICT_PATH,
        historical_glossary_path=HISTORICAL_GLOSSARY_PATH,
    )
    if start_page is not None and str(start_page).strip() != "":
        try:
            # UI value is a 1-based page number; config.start_page is 0-based.
            config.start_page = max(0, int(start_page) - 1)
        except (TypeError, ValueError):
            config.start_page = 0
    else:
        # Web uploads are usually short excerpts, so annotate from the start by
        # default instead of skipping front matter of a specific book.
        config.start_page = 0

    progress(0.4, desc="正在读取词汇并生成注释…")
    try:
        written = annotate_pdf(
            input_path=src_path,
            output_path=out_path,
            config=config,
            progress=False,
        )
    except Exception as exc:
        raise gr.Error("注释失败：%s" % exc) from exc
    progress(1.0, desc="完成")
    return written


with gr.Blocks(title="英文小说中文注释书斋", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER_HTML)
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["paper-card"]):
            pdf_in = gr.File(label="英文 PDF", file_types=[".pdf"], type="filepath")
            dictionary = gr.Dropdown(
                choices=DICTIONARY_CHOICES,
                value=DEFAULT_DICTIONARY,
                label="选择字典",
                info="字典越“典雅”，注释的词越少越精；内置明清小说词典始终优先生效",
            )
            start_page = gr.Number(
                label="正文起始页（可选）",
                info="从第几页开始注释（1 = 第一页）。留空则从第一页开始，可填大一点以跳过前置页",
                value=None,
                precision=0,
            )
            run = gr.Button("开始注释", variant="primary")
        with gr.Column(scale=1, elem_classes=["paper-card"]):
            pdf_out = gr.File(label="下载带注释的 PDF")

    run.click(annotate, inputs=[pdf_in, dictionary, start_page], outputs=pdf_out)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
