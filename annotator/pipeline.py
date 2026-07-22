"""End-to-end annotation pipeline.

For each body page:

1. Extract plain text and pick words/phrases to annotate (``WordSelector``).
2. Locate each surface form's rectangle(s) with ``page.search_for``.
3. Create a new, wider page (original width + margin) and stamp the original
   page content onto it via ``show_pdf_page`` — this preserves the visual
   layout untouched while giving us blank space on the right.
4. Draw a green underline under each annotated word and place a rasterised
   Chinese label in the right margin, aligned to the word's baseline, using a
   simple top-down collision-avoidance layout.
5. Re-attach internal link annotations (``show_pdf_page`` copies pixels, not
   links) so the table of contents keeps working, and carry over the outline /
   bookmarks.

The output PDF has the same page count and navigation as the source.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from .config import AnnotationConfig
from .dictionary import Dictionary
from .renderer import LabelRenderer
from .selector import Selected, WordSelector


class _Placement:
    __slots__ = ("selected", "word_rect", "png")

    def __init__(self, selected: Selected, word_rect: fitz.Rect, png: bytes) -> None:
        self.selected = selected
        self.word_rect = word_rect
        self.png = png


def annotate_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[AnnotationConfig] = None,
    report_path: Optional[str] = None,
    progress: bool = True,
) -> str:
    """Annotate ``input_path`` and write the result to ``output_path``.

    Returns the output path actually written.
    """
    config = config or AnnotationConfig()
    if output_path is None:
        stem, _ext = os.path.splitext(input_path)
        output_path = "%s-annotated-%s.pdf" % (stem, config.cefr_level.upper())

    dictionary = Dictionary(config.ecdict_path)
    selector = WordSelector(config, dictionary)
    renderer = LabelRenderer(config)

    src = fitz.open(input_path)
    out = fitz.open()

    stats = {
        "input": input_path,
        "output": output_path,
        "pages": len(src),
        "pages_annotated": 0,
        "annotations": 0,
        "cefr_level": config.cefr_level.upper(),
    }

    for pno in range(len(src)):
        src_page = src[pno]
        page_doc, annotated = _build_page(
            src_page, selector, renderer, config, pno
        )
        # Import the finished page into the output document.
        out.insert_pdf(page_doc, from_page=0, to_page=0)
        page_doc.close()

        if annotated:
            stats["pages_annotated"] += 1
            stats["annotations"] += annotated
        if progress and (pno % 25 == 0 or pno == len(src) - 1):
            print(
                "  page %d/%d  (annotated pages: %d, notes: %d)"
                % (pno + 1, len(src), stats["pages_annotated"], stats["annotations"])
            )

    # Preserve bookmarks / outline.
    toc = src.get_toc(simple=False)
    if toc:
        out.set_toc(toc)

    out.save(output_path, garbage=4, deflate=True)
    out.close()
    src.close()

    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(
        "Done: %d/%d pages annotated, %d notes -> %s"
        % (stats["pages_annotated"], stats["pages"], stats["annotations"], output_path)
    )
    return output_path


def _build_page(
    src_page: fitz.Page,
    selector: WordSelector,
    renderer: LabelRenderer,
    config: AnnotationConfig,
    pno: int,
) -> Tuple[fitz.Document, int]:
    """Return a single-page document containing the widened, annotated page.

    The *document* (not the page) is returned so the caller holds a strong
    reference; ``page.parent`` is only a weak proxy and the page would be
    orphaned if the document were garbage-collected.
    """
    src_rect = src_page.rect
    new_w = src_rect.width + config.margin_width
    new_h = src_rect.height

    tmp = fitz.open()
    page = tmp.new_page(width=new_w, height=new_h)

    # Stamp the original page pixels onto the left region.
    page.show_pdf_page(
        fitz.Rect(0, 0, src_rect.width, src_rect.height),
        src_page.parent,
        src_page.number,
    )

    # Re-attach internal links so navigation keeps working.
    _copy_links(src_page, page)

    if pno < config.start_page:
        return tmp, 0

    text = src_page.get_text("text")
    selections = selector.select_from_text(text)
    if not selections:
        return tmp, 0

    placements = _locate(src_page, selections, renderer, config)
    if not placements:
        return tmp, 0

    _draw(page, placements, config, src_rect.width)
    return tmp, len(placements)


def _locate(
    src_page: fitz.Page,
    selections: List[Selected],
    renderer: LabelRenderer,
    config: AnnotationConfig,
) -> List[_Placement]:
    placements: List[_Placement] = []
    for sel in selections:
        rect = _search(src_page, sel.surface)
        if rect is None:
            continue
        label_w = config.margin_width - 8
        png = renderer.render(sel.gloss, label_w, config.label_height)
        placements.append(_Placement(sel, rect, png))
    # Top-down order for stable collision avoidance.
    placements.sort(key=lambda p: p.word_rect.y0)
    return placements


def _search(page: fitz.Page, surface: str) -> Optional[fitz.Rect]:
    for variant in _quote_variants(surface):
        hits = page.search_for(variant)
        if hits:
            return hits[0]
    return None


def _quote_variants(surface: str) -> List[str]:
    variants = [surface]
    if "'" in surface:
        variants.append(surface.replace("'", "\u2019"))
    if "\u2019" in surface:
        variants.append(surface.replace("\u2019", "'"))
    return variants


def _draw(
    page: fitz.Page,
    placements: List[_Placement],
    config: AnnotationConfig,
    content_width: float,
) -> None:
    green = config.green_rgb_pdf
    margin_x = content_width + 4
    label_w = config.margin_width - 8
    last_bottom = -1e9

    for p in placements:
        rect = p.word_rect
        # Green underline under the source word.
        y = rect.y1 + 0.5
        page.draw_line(
            fitz.Point(rect.x0, y),
            fitz.Point(rect.x1, y),
            color=green,
            width=config.underline_width,
        )

        # Vertical position for the label: aligned to the word, pushed down to
        # avoid overlapping the previous label.
        top = max(rect.y0, last_bottom + config.label_gap)
        bottom = top + config.label_height
        if bottom > page.rect.height - 4:
            bottom = page.rect.height - 4
            top = bottom - config.label_height
        last_bottom = bottom

        box = fitz.Rect(margin_x, top, margin_x + label_w, bottom)
        page.insert_image(box, stream=p.png, keep_proportion=False)


def _copy_links(src_page: fitz.Page, dst_page: fitz.Page) -> None:
    for link in src_page.get_links():
        payload: Dict = {"kind": link["kind"], "from": link["from"]}
        for key in ("page", "to", "zoom", "uri", "file", "nameddest"):
            if key in link:
                payload[key] = link[key]
        try:
            dst_page.insert_link(payload)
        except Exception:
            # Skip malformed links rather than aborting the whole run.
            continue
