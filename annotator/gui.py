"""One-click desktop GUI for the PDF English-fiction annotator.

Run it directly::

    python -m annotator.gui

or launch the bundled Windows ``.exe`` / ``Start-Annotator.bat``.

The window lets a user pick an English PDF, choose a CEFR level and press
*Annotate*. On the very first run it automatically downloads the required
NLTK corpora and the ECDICT dictionary (~65 MB), showing progress in the log
box, so no command-line steps are needed.
"""

import os
import sys
import threading
import traceback
import urllib.request

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - headless environments
    raise SystemExit("Tkinter is required to run the GUI: %s" % exc)

from .config import CEFR_ZIPF_THRESHOLD, AnnotationConfig
from .nltk_setup import ensure_nltk_data
from .pipeline import annotate_pdf

ECDICT_URL = "https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv"


def app_base_dir() -> str:
    """Directory used to store the downloaded dictionary and outputs.

    When frozen by PyInstaller this is the folder that contains the ``.exe``;
    otherwise it is the project root (the parent of this package).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_ecdict_path() -> str:
    return os.path.join(app_base_dir(), "data", "ecdict.csv")


class AnnotatorApp:
    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        root.title("PDF 英文小说中文注释工具")
        root.geometry("680x520")
        root.minsize(620, 460)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.level_var = tk.StringVar(value="B2")
        self.start_page_var = tk.StringVar(value="")
        self._worker = None

        pad = {"padx": 10, "pady": 6}

        # --- Input file -----------------------------------------------------
        frm = ttk.Frame(root)
        frm.pack(fill="x", **pad)
        ttk.Label(frm, text="英文 PDF：").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(frm, text="选择…", command=self.pick_input).grid(row=0, column=2)

        ttk.Label(frm, text="输出 PDF：").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frm, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=(6, 0)
        )
        ttk.Button(frm, text="选择…", command=self.pick_output).grid(
            row=1, column=2, pady=(6, 0)
        )
        frm.columnconfigure(1, weight=1)

        # --- Options --------------------------------------------------------
        opts = ttk.LabelFrame(root, text="选项")
        opts.pack(fill="x", **pad)
        ttk.Label(opts, text="难度 (CEFR)：").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            opts,
            textvariable=self.level_var,
            values=sorted(CEFR_ZIPF_THRESHOLD),
            width=6,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", pady=6)
        ttk.Label(
            opts, text="（级别越高，注释的词越少越难）"
        ).grid(row=0, column=2, sticky="w", padx=8)

        ttk.Label(opts, text="正文起始页（可选）：").grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 8)
        )
        ttk.Entry(opts, textvariable=self.start_page_var, width=8).grid(
            row=1, column=1, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            opts, text="留空则自动跳过前置页（默认从第 143 页）"
        ).grid(row=1, column=2, sticky="w", padx=8, pady=(0, 8))

        # --- Run button + progress -----------------------------------------
        run_row = ttk.Frame(root)
        run_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_row, text="开始注释", command=self.start)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(run_row, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        # --- Log ------------------------------------------------------------
        logf = ttk.LabelFrame(root, text="运行日志")
        logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=12, wrap="word", state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

        self._log("欢迎使用。请选择一个英文 PDF，然后点击“开始注释”。")
        self._log("首次运行会自动下载词典（约 65 MB）与语言数据，请保持联网。")

    # -- UI helpers ---------------------------------------------------------
    def pick_input(self) -> None:
        path = filedialog.askopenfilename(
            title="选择英文 PDF", filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")]
        )
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                stem, _ = os.path.splitext(path)
                self.output_var.set("%s-annotated-%s.pdf" % (stem, self.level_var.get()))

    def pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存为", defaultextension=".pdf", filetypes=[("PDF 文件", "*.pdf")]
        )
        if path:
            self.output_var.set(path)

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def log(self, msg: str) -> None:  # thread-safe wrapper
        self.root.after(0, self._log, msg)

    def set_progress(self, value: float) -> None:
        self.root.after(0, self.progress.configure, {"value": value})

    # -- Run ----------------------------------------------------------------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        input_path = self.input_var.get().strip()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("缺少输入", "请先选择一个有效的英文 PDF 文件。")
            return
        self.run_btn.configure(state="disabled")
        self.progress.configure(value=0)
        self._worker = threading.Thread(target=self._run_safe, daemon=True)
        self._worker.start()

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception:  # noqa: BLE001 - surface any failure to the user
            self.log("发生错误：\n" + traceback.format_exc())
            self.root.after(
                0, messagebox.showerror, "注释失败", "运行出错，详情见日志。"
            )
        finally:
            self.root.after(0, self.run_btn.configure, {"state": "normal"})

    def _ensure_ecdict(self) -> str:
        path = default_ecdict_path()
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.log("正在下载 ECDICT 词典（约 65 MB）…")

        def hook(block_num, block_size, total_size):
            if total_size > 0:
                done = min(block_num * block_size, total_size)
                self.set_progress(done * 100 / total_size)

        urllib.request.urlretrieve(ECDICT_URL, path, hook)
        self.log("词典下载完成。")
        self.set_progress(0)
        return path

    def _run(self) -> None:
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip() or None
        level = self.level_var.get()

        self.log("准备语言数据（NLTK）…")
        ensure_nltk_data()

        ecdict_path = self._ensure_ecdict()

        config = AnnotationConfig(cefr_level=level, ecdict_path=ecdict_path)
        start_page = self.start_page_var.get().strip()
        if start_page:
            try:
                config.start_page = int(start_page)
            except ValueError:
                self.log("起始页无效，忽略。")

        self.log("开始注释：%s" % input_path)
        self.progress.configure(mode="indeterminate")
        self.root.after(0, self.progress.start, 15)

        written = annotate_pdf(
            input_path=input_path,
            output_path=output_path,
            config=config,
            progress=False,
        )

        self.root.after(0, self.progress.stop)
        self.progress.configure(mode="determinate", value=100)
        self.log("完成！已生成：\n%s" % written)
        self.root.after(
            0,
            messagebox.showinfo,
            "完成",
            "注释完成，已保存到：\n%s" % written,
        )


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    AnnotatorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
