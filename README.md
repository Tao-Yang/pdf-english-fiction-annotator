# pdf-english-fiction-annotator

为英文小说 PDF 添加**简明中文词汇注释**，注释显示在词语右侧扩展出的页边空白处，**不遮挡原文**。
Add concise, non-obstructive **Chinese vocabulary annotations** to English-fiction
PDFs. Glosses are placed in a widened right margin, aligned to each word, and the
original page content, links and bookmarks are preserved untouched.

---

## 🌐 在线体验 / Web App

不想安装任何东西？直接用网页版：

- **在线使用**：打开 [Web App](https://huggingface.co/spaces/Tao-Yang/pdf-english-fiction-annotator)
  或 [下载页](https://tao-yang.github.io/pdf-english-fiction-annotator/)，上传英文 PDF、选择难度，点击“开始注释”即可下载结果。
- **一键部署（免令牌，推荐）**：点击
  [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Tao-Yang/pdf-english-fiction-annotator)
  ，用 GitHub 登录并确认，Render 会读取仓库里的 `render.yaml` + `Dockerfile` 自动构建上线，随后分享它给出的 `https://…onrender.com` 网址即可，用户打开即用、无需下载。
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
- **图片化标签，兼容所有阅读器** — 中文标签渲染为 PNG 图片嵌入，避免部分阅读器不显示 Type0/Identity-H 中文字体的问题。
- **页面右侧扩展 + 防重叠布局** — 原页无缩放地放在左侧，注释放入新增右边栏，自上而下避让。
- **绿色下划线 + 淡黄标签** — 词语加绿色下划线，标签为带绿色左边缘标记的淡黄圆角框。
- **保留导航** — 目录内部链接和书签在重建页面后被重新写回，翻页与跳转保持可用。

## How it works / 工作原理

对每个正文页：

1. 提取纯文本并挑选值得注释的词/词组（`annotator/selector.py`）。
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
│   ├── dictionary.py         # ECDICT 懒加载词典
│   ├── selector.py           # 选词：wordfreq + NLTK
│   ├── renderer.py           # PIL 中文标签栅格化
│   ├── pipeline.py           # 主流程：扩页、绘制、复制链接/书签
│   ├── nltk_setup.py         # 按需下载 NLTK 语料
│   └── cli.py                # 命令行入口
├── webapp/
│   ├── app.py                # 网页版（Gradio）
│   └── requirements.txt      # 网页版依赖
├── scripts/
│   ├── download_ecdict.py    # 下载 ECDICT 词典
│   └── fix_links.py          # 独立工具：把源 PDF 链接补回重建后的 PDF
├── docs/                     # GitHub Pages 介绍页
├── .github/workflows/        # 部署 Pages 的自动化流程
├── Dockerfile                # 网页版容器镜像（一键部署）
├── data/                     # ECDICT csv 放这里（不入库）
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
| `--start-page` | 正文起始页（0 基），跳过前置内容 | `143` |
| `--min-notes` / `--max-notes` | 每页注释数量范围 | `5` / `12` |
| `--margin-width` | 右侧扩展宽度（点） | `205` |
| `--font` | 支持中文的字体文件 | 微软雅黑 |
| `--report` | 输出 JSON 运行报告 | 无 |

更多可调项见 [`annotator/config.py`](annotator/config.py)。

## Notes / 说明

- 不会覆盖源文件；输出默认为 `<原名>-annotated-<LEVEL>.pdf`。
- ECDICT 词典与小说 PDF **不入库**（见 [.gitignore](.gitignore)），请勿提交受版权保护的原著。
- 扫描版 PDF 需先 OCR；本工具面向已含可提取文本的 PDF。

## License / 许可

- 本项目代码：MIT（见 [LICENSE](LICENSE)）。
- ECDICT 词典：MIT，版权归其作者所有。
- 请自行确保对所注释小说文本拥有合法使用权。
