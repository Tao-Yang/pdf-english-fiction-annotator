"""Standalone utility: copy internal links from a source PDF into a rebuilt PDF.

``show_pdf_page`` copies page pixels but not link annotations, so a rebuilt
document can lose its table-of-contents navigation. If you produced an
annotated PDF with an older tool that dropped links, run this to graft the
original links back on (matched page-by-page):

    python -m scripts.fix_links source.pdf annotated.pdf annotated-fixed.pdf

Pages must correspond 1:1 (same count) between source and annotated PDFs.
"""

import sys

import fitz


def fix_links(source_path: str, target_path: str, output_path: str) -> None:
    src = fitz.open(source_path)
    dst = fitz.open(target_path)
    if len(src) != len(dst):
        raise SystemExit(
            "Page count mismatch: source=%d target=%d" % (len(src), len(dst))
        )

    copied = 0
    for pno in range(len(src)):
        dst_page = dst[pno]
        existing = {
            (round(l["from"].x0, 1), round(l["from"].y0, 1)) for l in dst_page.get_links()
        }
        for link in src[pno].get_links():
            key = (round(link["from"].x0, 1), round(link["from"].y0, 1))
            if key in existing:
                continue
            payload = {"kind": link["kind"], "from": link["from"]}
            for k in ("page", "to", "zoom", "uri", "file", "nameddest"):
                if k in link:
                    payload[k] = link[k]
            try:
                dst_page.insert_link(payload)
                copied += 1
            except Exception:
                continue

    toc = src.get_toc(simple=False)
    if toc and not dst.get_toc():
        dst.set_toc(toc)

    dst.save(output_path, garbage=4, deflate=True)
    dst.close()
    src.close()
    print("Copied %d links -> %s" % (copied, output_path))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python -m scripts.fix_links source.pdf annotated.pdf output.pdf",
            file=sys.stderr,
        )
        raise SystemExit(1)
    fix_links(sys.argv[1], sys.argv[2], sys.argv[3])
