"""Configuration for the annotation pipeline.

All tunable parameters live here so the CLI and library share a single source
of truth. Default values reproduce the settings validated on an 875-page
English novel (CEFR B2, Simplified Chinese, green raster labels).
"""

import glob
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


def _default_font_path() -> str:
    """Return a CJK-capable font path that exists on the current OS.

    Works on Windows (dev), Linux containers (Render / HF Spaces, where
    ``fonts-noto-cjk`` is installed) and macOS. An explicit override may be
    given via the ``ANNOTATOR_FONT_PATH`` environment variable.
    """
    override = os.environ.get("ANNOTATOR_FONT_PATH")
    candidates = []
    if override:
        candidates.append(override)
    candidates += [
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        # Linux (Debian/Ubuntu fonts-noto-cjk)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    # Last resort: scan common font dirs for any Noto CJK face.
    for pattern in (
        "/usr/share/fonts/**/NotoSansCJK*.*",
        "/usr/share/fonts/**/NotoSerifCJK*.*",
        "/usr/share/fonts/**/*CJK*.*",
    ):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    # Fall back to the Windows path so dev machines keep working; the renderer
    # will raise a clear error if it is genuinely missing.
    return r"C:\Windows\Fonts\msyh.ttc"


# CEFR reading levels mapped to a wordfreq Zipf threshold. A word is treated as
# "worth annotating" when its Zipf frequency is at or below the threshold, i.e.
# rarer words get annotated. Higher target level -> lower threshold -> fewer,
# harder words selected.
CEFR_ZIPF_THRESHOLD = {
    "A1": 5.0,
    "A2": 4.6,
    "B1": 4.2,
    "B2": 3.7,
    "C1": 3.2,
    "C2": 2.8,
}

# Per-level cap on notes-per-page. Easier levels (looser threshold, larger
# candidate pool) get a higher cap than stricter levels so the four tiers
# differ in *quantity* as well as content: 通俗 (A2) always shows the most
# notes and 典雅 (C1) the fewest, matching the "由多到少" nesting
# requirement. B2 keeps the original validated default of 12.
CEFR_MAX_NOTES = {
    "A1": 20,
    "A2": 16,
    "B1": 14,
    "B2": 12,
    "C1": 9,
    "C2": 7,
}


@dataclass
class AnnotationConfig:
    """Tunable options for a single annotation run."""

    # --- Reader profile -----------------------------------------------------
    cefr_level: str = "B2"
    # "simplified" or "traditional" (only affects which dictionary column /
    # post-processing is used; ECDICT ships Simplified Chinese).
    chinese_variant: str = "simplified"

    # --- Selection density --------------------------------------------------
    min_notes_per_page: int = 5
    # ``None`` (the default) derives the cap from ``CEFR_MAX_NOTES`` based on
    # ``cefr_level`` -- see ``notes_cap()``. Set explicitly (e.g. via the
    # ``--max-notes`` CLI flag) to override for every level.
    max_notes_per_page: Optional[int] = None
    # Skip proper nouns (capitalised mid-sentence tokens tagged NNP/NNPS).
    skip_proper_nouns: bool = True
    # Longest phrase (in words) to look up as an idiom / collocation.
    max_phrase_len: int = 4

    # --- Page geometry (points) --------------------------------------------
    # Width added to the right of each page to hold annotations. The reference
    # novel was 612pt wide and expanded to 817pt (612 + 205).
    margin_width: float = 205.0
    # First 0-based page index that contains body text worth annotating.
    # Front matter (title, contents, dedication ...) is skipped.
    start_page: int = 143
    # Vertical height reserved per annotation label, in points.
    label_height: float = 15.0
    # Minimum vertical gap between stacked labels (collision avoidance).
    label_gap: float = 2.0

    # --- Styling ------------------------------------------------------------
    # Rasterisation DPI-equivalent scale for label PNGs. Higher = crisper text.
    raster_scale: float = 3.0
    # Font used to render Chinese glyphs. Must be a CJK-capable font file.
    # Resolved per-OS (Windows / Linux container / macOS); override with the
    # ANNOTATOR_FONT_PATH environment variable.
    font_path: str = field(default_factory=_default_font_path)
    # Green used for the underline, left-edge marker and label text.
    green_rgb_pdf: Tuple[float, float, float] = (0.086, 0.510, 0.231)
    green_hex: str = "#16823b"
    # Pale-yellow label background.
    box_rgb_pdf: Tuple[float, float, float] = (1.0, 0.992, 0.94)
    box_hex: str = "#fff3a8"
    # Underline thickness in points.
    underline_width: float = 0.8
    # Thin leader line from the underlined word to its margin annotation.
    leader_line_width: float = 0.45

    # --- Dictionary ---------------------------------------------------------
    # Path to ECDICT csv (word,phonetic,definition,translation,pos,...).
    ecdict_path: str = field(default="data/ecdict.csv")
    # Directory of hand-compiled ``term,chinese`` glossary CSVs (Ming/Qing
    # official titles, place names, historical figures, idioms/slang). All
    # ``*.csv`` files in the directory are merged. Checked before ECDICT and
    # supports multi-word phrases (e.g. "grand secretary", "West Lake").
    historical_glossary_path: str = field(default="data/glossaries")

    # --- Literary "master translator" mode (optional) -----------------------
    # When enabled, long/complex prose sentences and poem/verse quotations
    # embedded in the novel are additionally sent to the Claude API for a
    # full literary translation into Chinese, rendered in the style of one
    # of four celebrated 20th-century translators (see
    # ``annotator.literary_translation.TRANSLATOR_STYLES``), shown as a
    # block annotation alongside the ordinary per-word glosses. Off by
    # default: it requires network access, an Anthropic API key, and
    # materially slows down annotation, so it must be an explicit opt-in.
    enable_literary_translation: bool = False
    # A sentence (outside a detected poem block) must have at least this
    # many words to be considered "long" enough to warrant a full-sentence
    # translation instead of only per-word glosses.
    literary_long_sentence_words: int = 35
    # Safety caps on API usage / added latency: at most this many long-
    # sentence/poem translations are requested per page ...
    literary_max_per_page: int = 1
    # ... and at most this many across a single run (a full-length novel can
    # contain thousands of long sentences; without a hard ceiling that would
    # mean thousands of blocking network calls).
    literary_max_total: int = 40
    # Anthropic model used for literary translation calls.
    literary_translator_model: str = "claude-opus-4-6"
    # Explicit API key; if ``None``, falls back to the ``ANTHROPIC_API_KEY``
    # environment variable. Never hard-code a real key here.
    literary_translator_api_key: Optional[str] = None
    # Colour used for the underline/leader/left-edge marker of a literary-
    # translation block, distinct from the green word-gloss colour so the
    # two annotation kinds are visually distinguishable at a glance.
    literary_accent_rgb_pdf: Tuple[float, float, float] = (0.541, 0.200, 0.141)
    literary_accent_hex: str = "#8a3324"
    # Pale parchment background for literary-translation blocks (word-gloss
    # labels use ``box_hex`` instead).
    literary_box_hex: str = "#fdf1e6"

    def zipf_threshold(self) -> float:
        """Return the Zipf frequency cutoff for the configured CEFR level."""
        level = self.cefr_level.upper()
        if level not in CEFR_ZIPF_THRESHOLD:
            raise ValueError(
                "Unknown CEFR level %r; choose from %s"
                % (self.cefr_level, ", ".join(sorted(CEFR_ZIPF_THRESHOLD)))
            )
        return CEFR_ZIPF_THRESHOLD[level]

    def notes_cap(self) -> int:
        """Return the max-notes-per-page cap for this run.

        Uses the explicit ``max_notes_per_page`` override if one was set,
        otherwise derives a level-scaled default from ``CEFR_MAX_NOTES`` so
        easier CEFR levels always allow at least as many notes as stricter
        ones.
        """
        if self.max_notes_per_page is not None:
            return self.max_notes_per_page
        level = self.cefr_level.upper()
        if level not in CEFR_MAX_NOTES:
            raise ValueError(
                "Unknown CEFR level %r; choose from %s"
                % (self.cefr_level, ", ".join(sorted(CEFR_MAX_NOTES)))
            )
        return CEFR_MAX_NOTES[level]
