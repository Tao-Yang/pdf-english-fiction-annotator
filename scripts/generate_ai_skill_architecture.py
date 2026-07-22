"""Generate a presentation-grade architecture diagram for the AI Skill."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2400, 1350
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets"
FONT_DIR = Path(r"C:\Users\uidv3390\.copilot\skills\canvas-design\canvas-fonts")
CJK = Path(r"C:\Windows\Fonts\msyh.ttc")

BG = "#071019"
PANEL = "#0B1824"
PANEL_2 = "#0D1E2C"
WHITE = "#ECF8F3"
MUTED = "#7896A3"
CYAN = "#21D4C2"
GREEN = "#8BE36A"
GOLD = "#F2C14E"
BLUE = "#5FA8FF"
GRID = "#163040"


def font(name, size):
    path = FONT_DIR / name
    return ImageFont.truetype(str(path), size)


def cjk(size):
    return ImageFont.truetype(str(CJK), size)


F_TITLE = font("BigShoulders-Bold.ttf", 78)
F_SUB = font("GeistMono-Regular.ttf", 23)
F_STAGE = font("BigShoulders-Bold.ttf", 34)
F_CODE = font("GeistMono-Regular.ttf", 18)
F_CODE_B = font("GeistMono-Bold.ttf", 18)
F_CN_32 = cjk(32)
F_CN_26 = cjk(26)
F_CN_22 = cjk(22)
F_CN_18 = cjk(18)
F_CN_16 = cjk(16)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def glow_box(im, box, color, radius=24, blur=22, alpha=100):
    glow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    rgba = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) + (alpha,)
    gd.rounded_rectangle(box, radius=radius, fill=rgba)
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(blur)))


def arrow(draw, points, color=CYAN, width=4, dashed=False, head=12):
    if dashed:
        for a, b in zip(points[:-1], points[1:]):
            x1, y1 = a
            x2, y2 = b
            length = max(abs(x2 - x1), abs(y2 - y1))
            steps = max(1, int(length / 15))
            for i in range(0, steps, 2):
                t1, t2 = i / steps, min(1, (i + 1) / steps)
                draw.line((x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                           x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2), fill=color, width=width)
    else:
        draw.line(points, fill=color, width=width, joint="curve")
    x2, y2 = points[-1]
    x1, y1 = points[-2]
    if abs(x2 - x1) >= abs(y2 - y1):
        s = 1 if x2 > x1 else -1
        tri = [(x2, y2), (x2 - s * head, y2 - head * .65), (x2 - s * head, y2 + head * .65)]
    else:
        s = 1 if y2 > y1 else -1
        tri = [(x2, y2), (x2 - head * .65, y2 - s * head), (x2 + head * .65, y2 - s * head)]
    draw.polygon(tri, fill=color)


def text_center(draw, box, text, fnt, fill=WHITE):
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    x = (box[0] + box[2] - (r - l)) / 2
    y = (box[1] + box[3] - (b - t)) / 2 - t
    draw.text((x, y), text, font=fnt, fill=fill)


def stage_header(draw, x, y, number, title, subtitle, color):
    draw.text((x, y), number, font=F_CODE_B, fill=color)
    draw.text((x + 52, y - 9), title, font=F_STAGE, fill=WHITE)
    draw.text((x + 52, y + 34), subtitle, font=F_CN_16, fill=MUTED)


def node(im, draw, box, title, subtitle, color=CYAN, icon=None, strong=False):
    glow_box(im, box, color, blur=18, alpha=50 if strong else 28)
    rounded(draw, box, 22, PANEL_2 if strong else PANEL, color, 2 if strong else 1)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1, y1, x1 + 8, y2), radius=4, fill=color)
    if icon:
        rounded(draw, (x1 + 24, y1 + 21, x1 + 68, y1 + 65), 10, "#102B36", color, 1)
        text_center(draw, (x1 + 24, y1 + 21, x1 + 68, y1 + 65), icon, F_CODE_B, color)
        tx = x1 + 84
    else:
        tx = x1 + 28
    draw.text((tx, y1 + 18), title, font=F_CN_26 if strong else F_CN_22, fill=WHITE)
    draw.text((tx, y1 + 58), subtitle, font=F_CN_16, fill=MUTED)


def chip(draw, box, label, color):
    rounded(draw, box, 16, "#0A1620", color, 1)
    text_center(draw, box, label, F_CODE, color)


def make_diagram():
    im = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(im)

    # Deep-space gradient and disciplined technical grid.
    grad = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gp = grad.load()
    for y in range(H):
        for x in range(W):
            d1 = ((x - 1500) ** 2 + (y - 480) ** 2) ** .5
            d2 = ((x - 330) ** 2 + (y - 960) ** 2) ** .5
            a1 = max(0, 1 - d1 / 1300)
            a2 = max(0, 1 - d2 / 900)
            gp[x, y] = (8, int(42 * a1), int(48 * a1 + 24 * a2), int(150 * max(a1, a2)))
    im.alpha_composite(grad)
    draw = ImageDraw.Draw(im)
    for x in range(70, W, 70):
        draw.line((x, 235, x, 1190), fill=GRID, width=1)
    for y in range(245, 1191, 70):
        draw.line((70, y, W - 70, y), fill=GRID, width=1)

    # Header.
    draw.text((90, 62), "AI SKILL", font=F_TITLE, fill=WHITE)
    draw.text((395, 62), "//", font=F_TITLE, fill=CYAN)
    draw.text((490, 62), "CONTEXT-AWARE PDF ANNOTATION SYSTEM", font=F_TITLE, fill=WHITE)
    draw.text((94, 156), "FROM USER INTENT TO A NON-OBSTRUCTIVE, DOWNLOADABLE READING EXPERIENCE", font=F_SUB, fill=MUTED)
    chip(draw, (2060, 72, 2295, 116), "SYSTEM MAP  /  01", CYAN)
    draw.line((90, 208, 2310, 208), fill="#214354", width=2)

    # Stage labels.
    stage_header(draw, 100, 266, "01", "ACCESS", "意图与 Skill 编排", CYAN)
    stage_header(draw, 475, 266, "02", "REASONING CORE", "语言理解与注释决策", GREEN)
    stage_header(draw, 1640, 266, "03", "DOCUMENT ENGINE", "布局、重建与质量闭环", GOLD)

    # Access column.
    access_boxes = [
        (100, 360, 400, 465, "用户意图", "PDF · 水平 · 页码 · 密度", "IN"),
        (100, 510, 400, 615, "Skill 触发器", "语义匹配 · 默认值补全", "SK"),
        (100, 660, 400, 765, "任务编排", "Web / CLI · 参数校验", "OR"),
    ]
    for i, (x1, y1, x2, y2, title, sub, ic) in enumerate(access_boxes):
        node(im, draw, (x1, y1, x2, y2), title, sub, CYAN, ic, strong=(i == 1))
        if i < 2:
            arrow(draw, [(250, y2 + 7), (250, access_boxes[i + 1][1] - 8)], CYAN, 3)

    # Reasoning enclosure.
    rounded(draw, (465, 345, 1570, 940), 34, "#081721", "#235B58", 2)
    draw.text((500, 372), "INTELLIGENCE PIPELINE", font=F_CODE_B, fill=GREEN)
    draw.text((1328, 374), "LIVE", font=F_CODE_B, fill=GREEN)
    draw.ellipse((1301, 379, 1313, 391), fill=GREEN)

    row1 = [
        (505, 430, 810, 535, "PDF 结构分析", "文本层 · 坐标 · 页面几何", "01"),
        (865, 430, 1170, 535, "正文识别", "章节 · 阅读顺序 · 去噪", "02"),
        (1225, 430, 1530, 535, "候选项抽取", "单词 · 短语 · 习语", "03"),
    ]
    for box in row1:
        node(im, draw, box[:4], box[4], box[5], GREEN, box[6])
    arrow(draw, [(812, 482), (855, 482)], GREEN, 3)
    arrow(draw, [(1172, 482), (1215, 482)], GREEN, 3)

    decision = (680, 600, 1360, 735)
    node(im, draw, decision, "语境感知难度决策", "CEFR × 词频 × 词性 × 词形 × 固定搭配", GREEN, "AI", strong=True)
    arrow(draw, [(1375, 535), (1375, 568), (1020, 568), (1020, 590)], GREEN, 3)

    # Knowledge plane.
    draw.text((510, 790), "KNOWLEDGE PLANE", font=F_CODE_B, fill=MUTED)
    chips = [
        (510, 835, 735, 885, "WORDFREQ", BLUE),
        (755, 835, 980, 885, "NLTK / POS", CYAN),
        (1000, 835, 1225, 885, "WORDNET", GREEN),
        (1245, 835, 1530, 885, "ECDICT / SQLITE", GOLD),
    ]
    for x1, y1, x2, y2, label, color in chips:
        chip(draw, (x1, y1, x2, y2), label, color)
        arrow(draw, [((x1 + x2) // 2, y1 - 8), ((x1 + x2) // 2, 770), (1020, 770), (1020, 745)], color, 2, dashed=True, head=8)

    # Main hand-off.
    arrow(draw, [(405, 712), (445, 712), (445, 482), (495, 482)], CYAN, 5, head=15)
    arrow(draw, [(1575, 670), (1628, 670)], GOLD, 5, head=15)

    # Document engine.
    doc_boxes = [
        (1640, 360, 2260, 465, "注释空间规划", "原词定位 · 右栏扩展 · 防碰撞", "LY"),
        (1640, 510, 2260, 615, "视觉标注渲染", "下划线 · 引出线 · 中文栅格标签", "UI"),
        (1640, 660, 2260, 765, "PDF 无损重建", "原文排版 · 链接 · 书签 · 可搜索文本", "PDF"),
        (1640, 810, 2260, 915, "质量闭环", "边界检查 · 视觉抽查 · 输出验证", "QA"),
    ]
    for i, box in enumerate(doc_boxes):
        node(im, draw, box[:4], box[4], box[5], GOLD if i < 3 else GREEN, box[6], strong=(i == 2))
        if i < len(doc_boxes) - 1:
            arrow(draw, [(1950, box[3] + 7), (1950, doc_boxes[i + 1][1] - 8)], GOLD, 3)

    # QA feedback loop.
    arrow(draw, [(2268, 862), (2310, 862), (2310, 412), (2270, 412)], GREEN, 3, dashed=True, head=10)
    draw.text((2180, 640), "ITERATE", font=F_CODE, fill=GREEN)

    # Delivery bar.
    glow_box(im, (100, 990, 2260, 1160), CYAN, blur=28, alpha=24)
    rounded(draw, (100, 990, 2260, 1160), 28, "#091823", "#244B5B", 2)
    draw.text((132, 1018), "DELIVERY & OPERATIONS", font=F_CODE_B, fill=CYAN)
    infra = [
        (132, 1070, 430, 1128, "GRADIO WEB APP", CYAN),
        (460, 1070, 750, 1128, "LOCAL CLI", BLUE),
        (780, 1070, 1105, 1128, "DOCKER IMAGE", GREEN),
        (1135, 1070, 1460, 1128, "RENDER CLOUD", GOLD),
        (1490, 1070, 1845, 1128, "GITHUB ACTIONS", CYAN),
        (1875, 1070, 2225, 1128, "DOWNLOAD PDF", WHITE),
    ]
    for x1, y1, x2, y2, label, color in infra:
        chip(draw, (x1, y1, x2, y2), label, color)
    for i in range(len(infra) - 1):
        arrow(draw, [(infra[i][2] + 5, 1099), (infra[i + 1][0] - 6, 1099)], MUTED, 2, head=7)

    # Footer notes and metadata.
    draw.text((100, 1215), "DESIGN PRINCIPLE", font=F_CODE_B, fill=CYAN)
    draw.text((100, 1250), "不替代阅读，只消除查词中断。", font=F_CN_22, fill=WHITE)
    draw.text((865, 1215), "NON-OBSTRUCTIVE", font=F_CODE_B, fill=GREEN)
    draw.text((865, 1250), "原文零遮挡 · 注释一一对应 · 结果可下载", font=F_CN_22, fill=WHITE)
    draw.text((1690, 1215), "OPEN SYSTEM", font=F_CODE_B, fill=GOLD)
    draw.text((1690, 1250), "MIT · PYTHON · PYMUPDF · SQLITE", font=F_CODE, fill=WHITE)
    draw.text((2295, 1280), "v1.0", font=F_CODE, fill=MUTED, anchor="ra")

    # Precision markers.
    for x, y in [(70, 235), (2330, 235), (70, 1190), (2330, 1190)]:
        draw.line((x - 12, y, x + 12, y), fill=CYAN, width=2)
        draw.line((x, y - 12, x, y + 12), fill=CYAN, width=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "ai-skill-architecture.png"
    pdf = OUT_DIR / "ai-skill-architecture.pdf"
    im.convert("RGB").save(png, quality=96, optimize=True)
    im.convert("RGB").save(pdf, "PDF", resolution=180.0)
    print(png)
    print(pdf)


if __name__ == "__main__":
    make_diagram()
