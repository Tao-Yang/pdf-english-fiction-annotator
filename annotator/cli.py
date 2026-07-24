"""Command-line interface for the annotator.

Examples
--------
Annotate a novel with defaults (CEFR B2, Simplified Chinese)::

    annotate-fiction novel.pdf

Choose a level, custom output and a report file::

    annotate-fiction novel.pdf -o novel-annotated.pdf --level C1 \
        --report report.json --ecdict data/ecdict.csv
"""

import argparse
import sys

from .config import CEFR_ZIPF_THRESHOLD, AnnotationConfig
from .nltk_setup import ensure_nltk_data
from .pipeline import annotate_pdf, annotate_pdf_parallel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotate-fiction",
        description="Add non-obstructive Chinese margin annotations to an "
        "English-fiction PDF.",
    )
    parser.add_argument("input", help="Path to the source English PDF.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PDF path (default: <input>-annotated-<LEVEL>.pdf).",
    )
    parser.add_argument(
        "--level",
        default="B2",
        choices=sorted(CEFR_ZIPF_THRESHOLD),
        help="Target CEFR reading level; higher = fewer, harder words (default: B2).",
    )
    parser.add_argument(
        "--ecdict",
        default="data/ecdict.csv",
        help="Path to the ECDICT csv dictionary (default: data/ecdict.csv).",
    )
    parser.add_argument(
        "--historical-glossary",
        default="data/glossaries",
        help="Path to the historical/cultural glossary csv or directory of "
        "csvs covering official titles, place names, figures and idioms "
        "(default: data/glossaries). Checked before ECDICT.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="0-based page index where body text begins (skips front matter).",
    )
    parser.add_argument(
        "--min-notes", type=int, default=None, help="Minimum notes per page."
    )
    parser.add_argument(
        "--max-notes", type=int, default=None, help="Maximum notes per page."
    )
    parser.add_argument(
        "--margin-width",
        type=float,
        default=None,
        help="Points of right margin added for annotations (default: 205).",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Path to a CJK-capable font file for rendering labels.",
    )
    parser.add_argument("--report", help="Optional path to write a JSON run report.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-page progress output."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Split the book into chunks and annotate them concurrently across "
        "this many worker processes (default: 1, i.e. the plain sequential "
        "pipeline). Each worker pays its own one-time NLTK/wordfreq/dictionary "
        "load (~150-200MB), so keep this conservative on memory-constrained "
        "hosts.",
    )
    parser.add_argument(
        "--chunk-pages",
        type=int,
        default=10,
        help="Pages per chunk when --workers > 1 (default: 10).",
    )
    parser.add_argument(
        "--literary-translation",
        action="store_true",
        help="Enable the optional 'master translator' mode: long sentences "
        "and poem/verse quotations additionally get a full literary "
        "translation (Fu Donghua / Zhu Shenghao / Xu Yuanchong / Yu "
        "Guangzhong style, auto-selected) via the Claude API. Requires "
        "network access and an ANTHROPIC_API_KEY (or --anthropic-api-key).",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic API key for --literary-translation (default: the "
        "ANTHROPIC_API_KEY environment variable).",
    )
    parser.add_argument(
        "--literary-long-sentence-words",
        type=int,
        default=None,
        help="Minimum word count for a prose sentence to qualify for full "
        "literary translation (default: 35).",
    )
    parser.add_argument(
        "--literary-max-per-page",
        type=int,
        default=None,
        help="Max literary translations requested per page (default: 1).",
    )
    parser.add_argument(
        "--literary-max-total",
        type=int,
        default=None,
        help="Max literary translations requested across the whole run "
        "(default: 40).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config = AnnotationConfig(
        cefr_level=args.level,
        ecdict_path=args.ecdict,
        historical_glossary_path=args.historical_glossary,
    )
    if args.start_page is not None:
        config.start_page = args.start_page
    if args.min_notes is not None:
        config.min_notes_per_page = args.min_notes
    if args.max_notes is not None:
        config.max_notes_per_page = args.max_notes
    if args.margin_width is not None:
        config.margin_width = args.margin_width
    if args.font is not None:
        config.font_path = args.font
    if args.literary_translation:
        config.enable_literary_translation = True
    if args.anthropic_api_key is not None:
        config.literary_translator_api_key = args.anthropic_api_key
    if args.literary_long_sentence_words is not None:
        config.literary_long_sentence_words = args.literary_long_sentence_words
    if args.literary_max_per_page is not None:
        config.literary_max_per_page = args.literary_max_per_page
    if args.literary_max_total is not None:
        config.literary_max_total = args.literary_max_total

    print("Preparing NLTK data ...")
    ensure_nltk_data()

    try:
        if args.workers > 1:
            annotate_pdf_parallel(
                input_path=args.input,
                output_path=args.output,
                config=config,
                report_path=args.report,
                chunk_pages=args.chunk_pages,
                max_workers=args.workers,
            )
        else:
            annotate_pdf(
                input_path=args.input,
                output_path=args.output,
                config=config,
                report_path=args.report,
                progress=not args.quiet,
            )
    except FileNotFoundError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
