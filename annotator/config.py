"""Configuration for the annotation pipeline.

All tunable parameters live here so the CLI and library share a single source
of truth. Default values reproduce the settings validated on an 875-page
English novel (CEFR B2, Simplified Chinese, green raster labels).
"""

import glob
import os
from dataclasses import dataclass, field
from typing import Tuple


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
    max_notes_per_page: int = 12
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

    def zipf_threshold(self) -> float:
        """Return the Zipf frequency cutoff for the configured CEFR level."""
        level = self.cefr_level.upper()
        if level not in CEFR_ZIPF_THRESHOLD:
            raise ValueError(
                "Unknown CEFR level %r; choose from %s"
                % (self.cefr_level, ", ".join(sorted(CEFR_ZIPF_THRESHOLD)))
            )
        return CEFR_ZIPF_THRESHOLD[level]
