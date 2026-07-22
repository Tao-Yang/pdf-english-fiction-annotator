# pdf-english-fiction-annotator
Every word has a story. Let's discover them together.
pdf-english-fiction-annotator/
├── annotator/            # 核心库（已从之前删除的临时脚本重建）
│   ├── config.py         # AnnotationConfig：CEFR、颜色、页边、字体等全部参数
│   ├── dictionary.py     # ECDICT 懒加载词典
│   ├── selector.py       # 选词：wordfreq(Zipf) + NLTK 词性/词形还原 + 词组识别
│   ├── renderer.py       # PIL 中文标签栅格化（绿色左边缘 + 淡黄框，兼容所有阅读器）
│   ├── pipeline.py       # 主流程：扩页、绿色下划线、防碰撞布局、复制链接/书签
│   ├── nltk_setup.py     # 按需下载 NLTK 语料
│   └── cli.py            # 命令行入口 annotate-fiction
├── scripts/
│   ├── download_ecdict.py  # 下载 65MB 词典
│   └── fix_links.py        # 独立工具：补回目录内部链接
├── data/README.md        # 词典放置说明（csv 不入库）
├── SKILL.md              # 方法论说明（原 skill 文件副本）
├── README.md, LICENSE(MIT), requirements.txt, pyproject.toml, .gitignore
