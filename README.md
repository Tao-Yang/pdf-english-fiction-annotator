# pdf-english-fiction-annotator

为英文小说 PDF 添加**简明中文词汇注释**，注释显示在词语右侧扩展出的页边空白处，**不遮挡原文**。
Add concise, non-obstructive **Chinese vocabulary annotations** to English-fiction
PDFs. Glosses are placed in a widened right margin, aligned to each word, and the
original page content, links and bookmarks are preserved untouched.
<img width="720" height="584" alt="image" src="https://github.com/user-attachments/assets/f9c98a51-9042-4183-ba60-d141d523b356" />



---

## 🌐 在线体验 / Web App

不想安装任何东西？直接用网页版：

- **在线使用**：打开 https://pdf-english-fiction-annotator.onrender.com/，上传英文 PDF、选择难度，推荐B2，点击“开始注释”即可下载结果。

- **本地运行网页版**：

  ```bash
  pip install -r webapp/requirements.txt
  python webapp/app.py          # 打开 http://localhost:7860
  ```

- **Docker 部署**（Hugging Face Spaces / Render / Railway 等）：

  ```bash
  docker build -t pdf-annotator .
  docker run -p 7860:7860 pdf-annotator
  ```

首次注释会自动下载词典（约 65 MB）与语言数据，请保持联网。

---

## Features / 功能

- **词语难度自适应** — 用 `wordfreq` 的 Zipf 频率结合 CEFR 等级，只标注对目标读者偏难的实词。
- **词组 / 习语识别** — 用 NLTK 词性标注与词形还原，优先匹配短语动词、固定搭配和习语。
- **中文释义** — 基于 [ECDICT](https://github.com/skywind3000/ECDICT) 词典给出语境化的简明中文义。
- **明清历史文化词库（四类可选）** — 内置四套独立词库，覆盖 ECDICT 缺失的历史文化专名，并优先于通用词典命中、不受频率/词性过滤限制：
  - **明史职官** (`official_titles.csv`) — 明清官职、机构、科举等名词的中文对照。
  - **人物生平** (`figures.csv`) — 明清小说与史书人物的简要生平注释（含威妥玛拼音等旧式罗马字形，如 Hsi-men Ch'ing、Hsiang Yü）。
  - **地名** (`places.csv`) — 古今地名、府县、名胜的中文对照与背景。
  - **俚语 / 习语** (`idioms.csv`) — 小说中的俚语、俗谚与固定说法。

  四套词库以 `term,chinese` CSV 形式分开维护，可按需增删或单独选用；也可整体传入 `--historical-glossary data/glossaries`。
- **图片化标签，兼容所有阅读器** — 中文标签渲染为 PNG 图片嵌入，避免部分阅读器不显示 Type0/Identity-H 中文字体的问题。
- **页面右侧扩展 + 防重叠布局** — 原页无缩放地放在左侧，注释放入新增右边栏，自上而下避让。
- **绿色下划线 + 淡黄标签** — 词语加绿色下划线，标签为带绿色左边缘标记的淡黄圆角框。
- **保留导航** — 目录内部链接和书签在重建页面后被重新写回，翻页与跳转保持可用。

## How it works / 工作原理

对每个正文页：

1. 提取纯文本并挑选值得注释的词/词组（`annotator/selector.py`）：先按短语（词组/习语）从长到短匹配，再处理单词；历史文化词库命中的专名优先保留，不受 CEFR 频率与词性过滤限制。
2. 用 `page.search_for` 定位每个词的矩形。
3. 新建更宽的页面，用 `show_pdf_page` 原样贴上原页内容（保留可复制文本与排版）。
4. 绿色下划线标出词语，右边栏放置图片化的中文标签，自上而下防碰撞对齐。
5. 用 `insert_link` 重新写回内部链接，`set_toc` 复制书签。

输出 PDF 页数与源文件一致，导航保持可用。

## Project layout / 目录结构

```
pdf-english-fiction-annotator/
├── annotator/                # 核心库
│   ├── config.py             # AnnotationConfig：所有可调参数
│   ├── dictionary.py         # ECDICT + 历史文化词库懒加载查询
│   ├── selector.py           # 选词：wordfreq + NLTK + 词库优先命中
│   ├── renderer.py           # PIL 中文标签栅格化
│   ├── pipeline.py           # 主流程：扩页、绘制、复制链接/书签
│   ├── nltk_setup.py         # 按需下载 NLTK 语料
│   └── cli.py                # 命令行入口
├── data/
│   ├── ecdict.csv            # ECDICT 词典（不入库，按需下载）
│   ├── glossaries/           # 明清历史文化词库（四套，已入库）
│   │   ├── official_titles.csv  # 明史职官：官职、机构、科举
│   │   ├── figures.csv          # 人物生平：明清小说 / 史书人物
│   │   ├── places.csv           # 地名：古今地名、府县、名胜
│   │   └── idioms.csv           # 俚语 / 习语：俗谚、固定说法
│   └── README.md             # 词典与词库说明
├── webapp/
│   ├── app.py                # 网页版（Gradio）
│   └── requirements.txt      # 网页版依赖
├── scripts/
│   ├── download_ecdict.py    # 下载 ECDICT 词典
│   └── fix_links.py          # 独立工具：把源 PDF 链接补回重建后的 PDF
├── docs/                     # GitHub Pages 介绍页
├── .github/workflows/        # 部署 Pages 的自动化流程
├── Dockerfile                # 网页版容器镜像（一键部署）
├── SKILL.md                  # 方法论说明（VS Code Agent Skill）
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Install / 安装

需要 Python 3.7+。

```bash
git clone https://github.com/Tao-Yang/pdf-english-fiction-annotator.git
cd pdf-english-fiction-annotator

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt      # 或 pip install -e .
python -m scripts.download_ecdict    # 下载词典到 data/ecdict.csv
```

首次运行会自动下载所需的 NLTK 语料（POS 标注器、WordNet、分词器）。

明清历史文化词库（`data/glossaries/` 下的四套 CSV）已随仓库一起提供，无需额外下载即可生效。

> **字体**：默认使用 `C:\Windows\Fonts\msyh.ttc`（微软雅黑）。
> 其它系统请用 `--font` 指定一个支持中文的字体文件，例如 Noto Sans CJK SC。

## Usage / 用法

命令行：

```bash
# 默认 CEFR B2、简体中文
python -m annotator.cli novel.pdf

# 指定等级、输出路径、报告文件与词典
python -m annotator.cli novel.pdf -o novel-annotated.pdf --level C1 \
    --report report.json --ecdict data/ecdict.csv

# 指定正文起始页（跳过前置内容）与自定义字体
python -m annotator.cli novel.pdf --start-page 143 --font "/path/NotoSansCJKsc.otf"

# 自定义历史文化词库：可传目录（合并其中所有 *.csv）或单个 CSV
python -m annotator.cli novel.pdf --historical-glossary data/glossaries
python -m annotator.cli novel.pdf --historical-glossary data/glossaries/figures.csv
```

安装后也可用入口命令：

```bash
annotate-fiction novel.pdf --level B2
```

作为库调用：

```python
from annotator import AnnotationConfig, annotate_pdf

config = AnnotationConfig(cefr_level="B2", ecdict_path="data/ecdict.csv")
annotate_pdf("novel.pdf", "novel-annotated.pdf", config=config)
```

补回丢失的目录链接（针对已用其它工具生成、丢了链接的 PDF）：

```bash
python -m scripts.fix_links source.pdf annotated.pdf annotated-fixed.pdf
```

## Options / 主要参数

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--level` | CEFR 等级 A1–C2，越高标注越少越难 | `B2` |
| `--ecdict` | ECDICT csv 路径 | `data/ecdict.csv` |
| `--historical-glossary` | 明清历史文化词库；可为目录（合并其中所有 `*.csv`）或单个 CSV，优先于 ECDICT 命中 | `data/glossaries` |
| `--start-page` | 正文起始页（0 基），跳过前置内容 | `143` |
| `--min-notes` / `--max-notes` | 每页注释数量范围 | `5` / `12` |
| `--margin-width` | 右侧扩展宽度（点） | `205` |
| `--font` | 支持中文的字体文件 | 微软雅黑 |
| `--report` | 输出 JSON 运行报告 | 无 |
| `--quiet` | 静默模式，不打印逐页进度 | 关 |

更多可调项见 [`annotator/config.py`](annotator/config.py)。

## Notes / 说明

- 不会覆盖源文件；输出默认为 `<原名>-annotated-<LEVEL>.pdf`。
- ECDICT 词典与小说 PDF **不入库**（见 [.gitignore](.gitignore)），请勿提交受版权保护的原著；`data/glossaries/` 下的四套历史文化词库则随仓库提供。
- 历史文化词库为 `term,chinese` 两列 CSV，优先于 ECDICT 命中且不受频率/词性过滤限制；可自行增删条目或新增 CSV 以扩充覆盖面。
- 词库对威妥玛拼音等旧式罗马字形（如 `Hsi-men Ch'ing`、`Hsiang Yü`）做了兼容：查词时会归一化弯引号/直引号并容忍变音符号。
- 扫描版 PDF 需先 OCR；本工具面向已含可提取文本的 PDF。

## License / 许可

- 本项目代码：MIT（见 [LICENSE](LICENSE)）。
- ECDICT 词典：MIT，版权归其作者所有。
- 请自行确保对所注释小说文本拥有合法使用权。
