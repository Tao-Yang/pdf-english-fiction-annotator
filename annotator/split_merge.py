"""Split a large PDF into page-range parts, and merge annotated parts back.

Used by the webapp to work around Render free-tier's RAM ceiling and
CPU-throttling-triggered restarts on very long books: instead of annotating
an 800+ page book in one long-lived request, the user splits it into a few
parts (each comfortably within the size that has been proven to annotate
reliably in one request), annotates each part separately, then merges the
annotated parts back into a single final PDF.
"""

import os

import fitz  # PyMuPDF


def split_pdf(input_path: str, max_pages: int, out_dir: str) -> list:
    """Split ``input_path`` into consecutive parts of at most ``max_pages`` pages.

    Returns the list of output file paths, in page order (part 1, part 2, ...).
    """
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_paths = []
    with fitz.open(input_path) as doc:
        total = len(doc)
        n_parts = (total + max_pages - 1) // max_pages
        for i in range(n_parts):
            start = i * max_pages
            end = min(start + max_pages, total) - 1
            part = fitz.open()
            part.insert_pdf(doc, from_page=start, to_page=end)
            out_path = os.path.join(out_dir, "%s_part%02d.pdf" % (stem, i + 1))
            part.save(out_path, deflate=True)
            part.close()
            out_paths.append(out_path)
    return out_paths


def merge_pdfs(paths: list, output_path: str) -> None:
    """Concatenate PDFs in the given order into a single ``output_path`` file."""
    if not paths:
        raise ValueError("no input files to merge")
    out = fitz.open()
    try:
        for p in paths:
            with fitz.open(p) as part:
                out.insert_pdf(part)
        out.save(output_path, garbage=4, deflate=True)
    finally:
        out.close()
