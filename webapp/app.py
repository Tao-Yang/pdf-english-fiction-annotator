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
import sys
import tempfile
import urllib.request

import gradio as gr

# Make the parent package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotator.config import CEFR_ZIPF_THRESHOLD, AnnotationConfig  # noqa: E402
from annotator.nltk_setup import ensure_nltk_data  # noqa: E402
from annotator.pipeline import annotate_pdf  # noqa: E402

ECDICT_URL = "https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv"

# A writable directory for the cached dictionary. HF Spaces / most PaaS allow
# writing under the app dir or /tmp.
DATA_DIR = os.environ.get("ANNOTATOR_DATA_DIR") or os.path.join(
    tempfile.gettempdir(), "pdf-annotator-data"
)
ECDICT_PATH = os.path.join(DATA_DIR, "ecdict.csv")

_READY = False


def _ensure_assets(progress=None) -> None:
    """Download NLTK data and the ECDICT dictionary once."""
    global _READY
    if _READY:
        return
    if progress:
        progress(0.05, desc="准备语言数据 (NLTK)…")
    ensure_nltk_data()

    if not (os.path.isfile(ECDICT_PATH) and os.path.getsize(ECDICT_PATH) > 1_000_000):
        os.makedirs(DATA_DIR, exist_ok=True)
        if progress:
            progress(0.15, desc="下载词典 (约 65 MB)…")
        urllib.request.urlretrieve(ECDICT_URL, ECDICT_PATH)
    _READY = True


def annotate(pdf_file, level, start_page, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("请先上传一个英文 PDF 文件。")

    _ensure_assets(progress)

    src_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    out_dir = tempfile.mkdtemp(prefix="annotated-")
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, "%s-annotated-%s.pdf" % (stem, level))

    config = AnnotationConfig(cefr_level=level, ecdict_path=ECDICT_PATH)
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

    progress(0.4, desc="正在注释，请稍候…")
    written = annotate_pdf(
        input_path=src_path,
        output_path=out_path,
        config=config,
        progress=False,
    )
    progress(1.0, desc="完成")
    return written


with gr.Blocks(title="PDF 英文小说中文注释工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📖 PDF 英文小说中文注释工具

        上传一本英文原版小说的 PDF，为其中的**生词、短语和习语**自动加上简洁的
        中文释义。注释放在页面右侧扩展出的空白处，**不遮挡原文**，原书排版与目录
        链接保持不变。

        > 首次注释会自动下载词典（约 65 MB），可能需要等待一会儿。
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            pdf_in = gr.File(label="英文 PDF", file_types=[".pdf"], type="filepath")
            level = gr.Dropdown(
                choices=sorted(CEFR_ZIPF_THRESHOLD),
                value="B2",
                label="难度 (CEFR)",
                info="级别越高，注释的词越少越难",
            )
            start_page = gr.Number(
                label="正文起始页（可选）",
                info="从第几页开始注释（1 = 第一页）。留空则从第一页开始，可填大一点以跳过前置页",
                value=None,
                precision=0,
            )
            run = gr.Button("开始注释", variant="primary")
        with gr.Column(scale=1):
            pdf_out = gr.File(label="下载带注释的 PDF")

    run.click(annotate, inputs=[pdf_in, level, start_page], outputs=pdf_out)

    gr.Markdown(
        "免费开源 · MIT 许可证 · 词典来自 "
        "[ECDICT](https://github.com/skywind3000/ECDICT) · "
        "[源码](https://github.com/Tao-Yang/pdf-english-fiction-annotator)。"
        "请仅对你拥有合法权利的 PDF 使用本工具。"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
