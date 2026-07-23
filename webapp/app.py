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
import base64
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
# The reader picks one or more "dictionaries" (rendered as calligraphy tiles,
# multi-select). Each tile corresponds to a vocabulary tier / CEFR threshold.
# When several are selected the annotations cover every selected tier, i.e. the
# most inclusive (easiest) tier wins. The built-in Ming/Qing novel glossary is
# always applied on top regardless of the choice.
DICTIONARY_TILES = [
    ("通俗词典", "注释最多 · 入门", "A2"),
    ("常用词典", "注释较多", "B1"),
    ("文学词典", "注释适中 · 推荐", "B2"),
    ("典雅词典", "仅注释生僻词 · 进阶", "C1"),
]
DICTIONARY_CHOICES = [name for name, _desc, _lv in DICTIONARY_TILES]
NAME_TO_LEVEL = {name: lv for name, _desc, lv in DICTIONARY_TILES}
DEFAULT_DICTIONARIES = ["文学词典"]
# Lower rank == more words annotated (easier / more inclusive tier).
_LEVEL_RANK = {"A2": 0, "B1": 1, "B2": 2, "C1": 3}


# --- Literary background artwork -----------------------------------------
# A soft, blurred wash derived from the app icon (a misty Jiangnan water town
# with a moon-gate arch, a stone bridge and a stack of English classics),
# embedded as a data-URI so no external image hosting is required.
_BG_ASSET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "background.jpg")
with open(_BG_ASSET, "rb") as _bg_f:
    _BACKGROUND_DATA_URI = (
        "data:image/jpeg;base64,"
        + base64.b64encode(_bg_f.read()).decode("ascii"))


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


def _house(cx, ground, w, h, wall, roof, op=0.85):
    """A Jiangnan waterside house: white-washed wall + dark tiled roof."""
    parts = ["<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='%s' "
             "opacity='%.2f'/>" % (cx - w / 2, ground - h, w, h, wall, op)]
    roof_h = h * 0.30
    parts.append("<g opacity='%.2f'>%s</g>"
                 % (op, _roof(cx, ground - h - roof_h, w * 1.12, roof_h, roof,
                              1.0, body=False)))
    win = min(1.0, op * 0.75)
    parts.append("<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='#565a4c' "
                 "opacity='%.2f'/>" % (cx - w * 0.28, ground - h * 0.60, w * 0.20,
                                       h * 0.24, win))
    parts.append("<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='#565a4c' "
                 "opacity='%.2f'/>" % (cx + w * 0.08, ground - h * 0.60, w * 0.20,
                                       h * 0.24, win))
    return "".join(parts)


def _bridge(cx, crown_y, span, color):
    """An arched Jiangnan stone bridge over the canal."""
    left = cx - span / 2.0
    right = cx + span / 2.0
    base = crown_y + span * 0.30
    th = span * 0.06 + 12
    top = "M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f" % (left, base, cx, crown_y, right, base)
    bot = " L%.1f,%.1f Q%.1f,%.1f %.1f,%.1f Z" % (
        right, base + th, cx, crown_y + th, left, base + th)
    parts = ["<path d='%s%s' fill='%s' opacity='0.92'/>" % (top, bot, color)]
    ar = span * 0.16
    parts.append("<path d='M%.1f,%.1f A%.1f,%.1f 0 0 1 %.1f,%.1f' fill='none' "
                 "stroke='%s' stroke-width='4' opacity='0.55'/>"
                 % (cx - ar, base + th + 4, ar, ar, cx + ar, base + th + 4, color))
    posts = ["<g stroke='%s' stroke-width='3' opacity='0.85'>" % color]
    for k in range(-2, 3):
        px = cx + k * span * 0.16
        tt = (px - left) / (right - left)
        yy = (1 - tt) ** 2 * base + 2 * (1 - tt) * tt * crown_y + tt * tt * base
        posts.append("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f'/>"
                     % (px, yy, px, yy - 16))
    posts.append("</g>")
    parts.append("".join(posts))
    return "".join(parts)


def _willow(ax, ay, direction, scale, color):
    """A fuller cluster of drooping willow branches framing a top corner."""
    branch = ["<g stroke='%s' stroke-width='%.1f' fill='none' opacity='0.48'>"
              % (color, 2.0 * scale)]
    leaf = ["<g fill='%s' opacity='0.5'>" % color]
    for j in range(11):
        anx = ax + direction * j * 24 * scale
        any_ = ay + j * 5
        length = (150 + (j % 5) * 42 + (j * 13) % 70) * scale
        sway = direction * (28 + (j * 19) % 78) * scale
        ex = anx + sway
        ey = any_ + length
        cx1 = anx + direction * 10 * scale
        cy1 = any_ + length * 0.55
        branch.append("<path d='M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f'/>"
                      % (anx, any_, cx1, cy1, ex, ey))
        for tt in (0.48, 0.66, 0.80, 0.92):
            mt = 1 - tt
            bx = mt * mt * anx + 2 * mt * tt * cx1 + tt * tt * ex
            by = mt * mt * any_ + 2 * mt * tt * cy1 + tt * tt * ey
            leaf.append("<ellipse cx='%.1f' cy='%.1f' rx='%.1f' ry='%.1f' "
                        "transform='rotate(%.0f %.1f %.1f)'/>"
                        % (bx, by, 3.0 * scale, 8.0 * scale, direction * 40, bx, by))
    branch.append("</g>")
    leaf.append("</g>")
    return "".join(branch) + "".join(leaf)


def _background_data_uri():
    rng = random.Random(20240811)
    width, height = 1600, 1000
    p = ["<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' "
         "preserveAspectRatio='xMidYMid slice'>" % (width, height)]
    # Soft misty sepia-green sky built from solid colour bands (renders
    # everywhere), warmed to echo the app icon's nostalgic tone.
    bands = 48
    for i in range(bands):
        t = i / float(bands - 1)
        col = _lerp_hex("#e9e4ce", "#b3b48a", t)
        y0 = height * i / float(bands)
        p.append("<rect x='0' y='%.1f' width='%d' height='%.1f' fill='%s'/>"
                 % (y0, width, height / float(bands) + 1.0, col))
    # Pale veiled sun softening the sky.
    p.append("<circle cx='1180' cy='250' r='160' fill='#f2ecd6' opacity='0.5'/>")
    p.append("<circle cx='1180' cy='250' r='100' fill='#f8f3df' opacity='0.55'/>")
    # Distant treeline.
    p.append("<path d='M0,500 C240,440 420,492 620,468 C820,444 980,494 1200,462 "
             "C1360,438 1500,486 1600,462 L1600,1000 L0,1000 Z' "
             "fill='#93976b' opacity='0.38'/>")
    # Far pagoda + misty far houses of the water town.
    p.append(_pagoda(1410, 470, 0.70, "#63614c"))
    for hx, hg, hw, hh in [(120, 590, 120, 90), (250, 585, 150, 108),
                           (400, 596, 118, 84), (1180, 600, 150, 106),
                           (1320, 590, 128, 94)]:
        p.append(_house(hx, hg, hw, hh, "#ebe7d3", "#4c4a3d", 0.66))
    # Heavy veils of mist over the midground for a painterly look.
    p.append("<rect x='0' y='452' width='%d' height='210' fill='#ece7d2' "
             "opacity='0.50'/>" % width)
    p.append("<rect x='0' y='500' width='%d' height='150' fill='#f0ecd8' "
             "opacity='0.34'/>" % width)
    # Calm canal water.
    p.append("<rect x='0' y='740' width='%d' height='120' fill='#9aa47a' "
             "opacity='0.48'/>" % width)
    p.append("<rect x='0' y='740' width='%d' height='120' fill='#c4c6a2' "
             "opacity='0.22'/>" % width)
    # Waterside houses (left cluster) softened by mist.
    p.append(_house(210, 760, 152, 150, "#efebd9", "#48463a", 0.82))
    p.append(_house(360, 770, 128, 128, "#e9e5d2", "#434136", 0.82))
    p.append(_house(72, 774, 128, 140, "#e5e1cf", "#403e33", 0.82))
    # Empty arched stone bridge over the canal (no figures).
    p.append(_bridge(720, 662, 300, "#a3a683"))
    # A translucent moon-gate arch framing the left, echoing the app icon.
    p.append("<circle cx='150' cy='430' r='372' fill='none' stroke='#6c6a51' "
             "stroke-width='42' opacity='0.22'/>")
    # Willow foliage framing the two top corners.
    p.append("<ellipse cx='80' cy='40' rx='180' ry='120' fill='#66703f' "
             "opacity='0.46'/>")
    p.append("<ellipse cx='1530' cy='44' rx='190' ry='120' fill='#5d6739' "
             "opacity='0.46'/>")
    p.append(_willow(150, 60, 1, 1.0, "#66703f"))
    p.append(_willow(1470, 64, -1, 1.05, "#5c6636"))
    # Gentle drifting drizzle (subtle).
    rain = []
    for _ in range(90):
        rx = rng.uniform(0, width)
        ry = rng.uniform(0, 820)
        rl = rng.uniform(10, 22)
        rain.append("<line x1='%.0f' y1='%.0f' x2='%.0f' y2='%.0f'/>"
                    % (rx, ry, rx - 6, ry + rl))
    p.append("<g stroke='#b7bd97' stroke-width='1' opacity='0.28'>%s</g>"
             % "".join(rain))
    # Foreground shelf of English classics.
    p.append(_bookshelf(rng, width, height))
    # Cohesive warm sepia-green tint over everything.
    p.append("<rect x='0' y='0' width='%d' height='%d' fill='#5c5a3d' "
             "opacity='0.08'/>" % (width, height))
    p.append("</svg>")
    return "data:image/svg+xml;utf8," + urllib.parse.quote("".join(p))


# The scene helpers above are retained for reference; the active background is
# the blurred icon wash loaded into ``_BACKGROUND_DATA_URI`` near the top.

_CSS_TEMPLATE = """
gradio-app {
  background:
    linear-gradient(rgba(238,232,214,0.46), rgba(223,215,195,0.56)),
    url("__BG__") center center / cover no-repeat fixed !important;
}
.gradio-container {
  background: transparent !important;
  width: 68vw !important;
  max-width: 1500px !important;
  min-width: 720px !important;
  margin: 0 auto !important;
  padding-bottom: 20px !important;
}

/* ---- Header (brush calligraphy) ---- */
#app-header { text-align: center; padding: 34px 20px 2px; color: #33402c; }
#app-header .ah-seal {
  display: inline-block; background: #a5352f; color: #fbe7cf;
  font-family: "Ma Shan Zheng", "Noto Serif SC", serif;
  font-size: 22px; letter-spacing: 5px; padding: 7px 15px 4px;
  border-radius: 9px; transform: rotate(-3deg);
  box-shadow: 0 4px 12px rgba(120,30,25,0.35); margin-bottom: 4px;
}
#app-header h1 {
  font-family: "Ma Shan Zheng", "Noto Serif SC", cursive;
  font-size: 60px; line-height: 1.15; letter-spacing: 14px;
  margin: 8px 0 2px; color: #24311d; font-weight: 400;
  text-shadow: 0 2px 0 rgba(255,255,255,0.5), 0 8px 22px rgba(40,60,30,0.28);
}
#app-header .ah-sub {
  font-family: "Ma Shan Zheng", "Noto Serif SC", serif;
  font-size: 27px; color: #3f5233; margin: 2px 0 10px; letter-spacing: 4px;
}
#app-header .ah-desc {
  max-width: 760px; margin: 8px auto 0; line-height: 1.9;
  font-size: 22px; letter-spacing: 2px; color: #3f5233; font-weight: 400;
  font-family: "Ma Shan Zheng", "Noto Serif SC", cursive;
}

/* ---- Dictionary picker: calligraphy tiles, multi-select ---- */
#pick-title {
  text-align: center; margin: 20px 0 10px;
  font-family: "Ma Shan Zheng", "Noto Serif SC", serif;
  font-size: 30px; letter-spacing: 8px; color: #2e6b45;
}
#dict-picker { margin: 0 auto 4px; border: none !important; background: transparent !important; }
#dict-picker [data-testid="checkbox-group"],
#dict-picker fieldset, #dict-picker .wrap {
  display: flex !important; flex-wrap: wrap !important;
  gap: 18px !important; justify-content: center !important;
  border: none !important; background: transparent !important;
}
#dict-picker label {
  flex: 1 1 180px; min-width: 168px; max-width: 260px;
  display: flex !important; align-items: center; justify-content: center;
  padding: 22px 12px; border-radius: 18px;
  border: 2px solid rgba(70,90,55,0.30);
  background: rgba(250,250,242,0.86) !important;
  cursor: pointer; transition: all .16s ease;
  font-family: "Ma Shan Zheng", "Noto Serif SC", cursive;
  font-size: 33px; color: #37472d; letter-spacing: 4px;
  box-shadow: 0 6px 16px rgba(50,60,40,0.12);
}
#dict-picker label:hover { border-color: #3f7a4e; transform: translateY(-3px); }
#dict-picker input[type="checkbox"] {
  position: absolute !important; opacity: 0 !important; width: 0 !important;
  height: 0 !important; margin: 0 !important;
}
#dict-picker label:has(input:checked) {
  background: #3f7a4e !important; color: #fdfbf0 !important;
  border-color: #2e6b45; box-shadow: 0 10px 24px rgba(46,107,69,0.38);
}
#dict-legend {
  text-align: center; color: #46583a; font-size: 13px; line-height: 2;
  margin: 4px auto 16px; max-width: 780px; font-family: "Noto Serif SC", serif;
}
#dict-legend b { color: #2e6b45; font-weight: 600; }
#dict-legend span { margin: 0 9px; white-space: nowrap; }

/* ---- Operation cards (fill 2/3 of the screen) ---- */
.paper-card {
  background: rgba(252,251,244,0.92) !important;
  border: 1px solid rgba(70,90,55,0.22) !important;
  border-radius: 18px !important;
  box-shadow: 0 14px 40px rgba(40,55,30,0.22) !important;
  padding: 26px !important; min-height: 440px;
  backdrop-filter: blur(2px);
}
.card-title {
  font-family: "Ma Shan Zheng", "Noto Serif SC", serif;
  font-size: 27px; color: #2e6b45; letter-spacing: 4px;
  margin-bottom: 14px; text-align: center;
  border-bottom: 1px dashed rgba(70,90,55,0.3); padding-bottom: 8px;
}
.paper-card label span { color: #3a4a30 !important; }
.paper-card input, .paper-card textarea {
  background: rgba(255,254,248,0.96) !important;
}
#pdf-in .wrap, #pdf-in .file-preview, #pdf-out .wrap, #pdf-out .file-preview {
  min-height: 250px !important;
}

/* ---- Run button (brush style) ---- */
#run-btn button, button.primary, .primary {
  background: #3f7a4e !important; border: none !important; color: #fdfbf0 !important;
  font-family: "Ma Shan Zheng", "Noto Serif SC", serif !important;
  font-size: 23px !important; letter-spacing: 8px; padding: 14px 10px !important;
  border-radius: 12px !important; font-weight: 400 !important;
}
#run-btn button:hover, button.primary:hover, .primary:hover {
  background: #316040 !important;
}

/* ---- Footer ---- */
#app-footer {
  text-align: center; padding: 18px 12px 30px; color: #46583a;
  font-size: 13px; line-height: 1.95; font-family: "Noto Serif SC", serif;
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
    font=[gr.themes.GoogleFont("Noto Serif SC"),
          gr.themes.GoogleFont("Ma Shan Zheng"),
          "Georgia", "serif"],
)

HEADER_HTML = """
<div id="app-header">
  <h1>英 文 原 著 伴 读</h1>
  <p class="ah-sub">你的第一本英文原著，我陪你读完</p>
  <p class="ah-desc">上传一本英文原版小说的 PDF，为其中的生词、短语与习语添上简洁的中文释义。注释安放在页面右侧扩展出的留白处，不遮挡原文，原书排版与目录链接保持如初。</p>
</div>
"""

LEGEND_HTML = "<div id='dict-legend'>" + "".join(
    "<span><b>%s</b> · %s</span>" % (name, desc)
    for name, desc, _lv in DICTIONARY_TILES
) + "<br/>可多选，注释将覆盖所选各档词汇；内置<b>明清小说词典</b>始终生效。</div>"

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


def annotate(pdf_file, dictionaries, start_page, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("请先上传一个英文 PDF 文件。")

    names = [d for d in (dictionaries or []) if d in NAME_TO_LEVEL]
    if not names:
        names = DEFAULT_DICTIONARIES
    # The most inclusive (easiest) selected tier wins, so every chosen
    # dictionary's vocabulary is covered.
    level = min((NAME_TO_LEVEL[n] for n in names), key=lambda lv: _LEVEL_RANK[lv])
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


with gr.Blocks(title="伴读 · 英文原著中文注释", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML("<div id='pick-title'>选 择 字 典 · 可 多 选</div>")
    dictionaries = gr.CheckboxGroup(
        choices=DICTIONARY_CHOICES,
        value=DEFAULT_DICTIONARIES,
        show_label=False,
        container=False,
        elem_id="dict-picker",
    )
    gr.HTML(LEGEND_HTML)
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes=["paper-card"]):
            gr.HTML("<div class='card-title'>上 传 原 著</div>")
            pdf_in = gr.File(
                label="英文 PDF", file_types=[".pdf"], type="filepath",
                elem_id="pdf-in",
            )
            start_page = gr.Number(
                label="正文起始页（可选）",
                info="从第几页开始注释（1 = 第一页）。留空则从第一页开始，可填大一点以跳过前置页",
                value=None,
                precision=0,
            )
            run = gr.Button("开 始 注 释", variant="primary", elem_id="run-btn")
        with gr.Column(scale=1, elem_classes=["paper-card"]):
            gr.HTML("<div class='card-title'>取 回 译 注</div>")
            pdf_out = gr.File(label="下载带注释的 PDF", elem_id="pdf-out")

    run.click(annotate, inputs=[pdf_in, dictionaries, start_page], outputs=pdf_out)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
