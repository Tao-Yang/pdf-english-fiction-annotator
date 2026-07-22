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
from .pipeline import annotate_pdf


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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config = AnnotationConfig(cefr_level=args.level, ecdict_path=args.ecdict)
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

    print("Preparing NLTK data ...")
    ensure_nltk_data()

    try:
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
