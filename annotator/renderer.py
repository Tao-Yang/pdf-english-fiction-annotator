"""Rasterise Chinese annotation labels to PNG images.

Why rasterise instead of drawing text directly? Some PDF viewers do not render
embedded Type0 / Identity-H CJK fonts, leaving annotations invisible even when
the font is correctly embedded. Rendering each label to a small PNG makes the
result viewer-independent: the annotation is a plain image every reader can
show.

Each label is a pale-yellow rounded box with a solid green left edge marker and
green Chinese text, matching the underline colour drawn on the source word.
"""

import io
from functools import lru_cache
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from .config import AnnotationConfig


class LabelRenderer:
    """Render annotation glosses to RGBA PNG bytes."""

    def __init__(self, config: AnnotationConfig) -> None:
        self.config = config
        self._scale = config.raster_scale

    @lru_cache(maxsize=None)
    def _font(self, px: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.config.font_path, px)

    def render(self, text: str, box_width_pt: float, box_height_pt: float) -> bytes:
        """Return PNG bytes for a single label sized to fit the given box.

        ``box_width_pt`` / ``box_height_pt`` are in PDF points; the raster is
        produced at ``raster_scale`` times that size for crisp text.
        """
        w = max(8, int(round(box_width_pt * self._scale)))
        h = max(8, int(round(box_height_pt * self._scale)))

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        box_rgb = _hex_to_rgb(self.config.box_hex)
        green_rgb = _hex_to_rgb(self.config.green_hex)

        radius = max(2, int(round(2 * self._scale)))
        # Rounded pale-yellow background.
        draw.rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=radius, fill=box_rgb + (255,)
        )
        # Solid green left-edge marker.
        marker_w = max(2, int(round(2 * self._scale)))
        draw.rectangle([0, 0, marker_w, h - 1], fill=green_rgb + (255,))

        # Auto-fit the font so the text fills the box without clipping.
        pad_x = marker_w + int(round(2 * self._scale))
        pad_y = int(round(1 * self._scale))
        avail_w = w - pad_x - int(round(2 * self._scale))
        avail_h = h - 2 * pad_y
        font = self._fit_font(draw, text, avail_w, avail_h)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        ty = pad_y + (avail_h - text_h) // 2 - bbox[1]
        draw.text((pad_x, ty), text, font=font, fill=green_rgb + (255,))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _fit_font(
        self, draw: "ImageDraw.ImageDraw", text: str, avail_w: int, avail_h: int
    ) -> ImageFont.FreeTypeFont:
        px = max(8, avail_h)
        while px >= 8:
            font = self._font(px)
            bbox = draw.textbbox((0, 0), text, font=font)
            if (bbox[2] - bbox[0]) <= avail_w and (bbox[3] - bbox[1]) <= avail_h:
                return font
            px -= 1
        return self._font(8)

    # --- Multi-line "literary translation" blocks --------------------------
    # Used for long-sentence/poem full translations (see
    # ``annotator.literary_translation``), which are taller, wrapped,
    # multi-line labels rather than a single short word gloss. A header line
    # (translator credit, e.g. "【许渊冲译】") is drawn in the accent colour
    # above the wrapped Chinese translation body.

    def measure_block(self, header: str, body: str, box_width_pt: float) -> float:
        """Return the height (in points) needed to render a block label."""
        img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        w = max(8, int(round(box_width_pt * self._scale)))
        pad_x = int(round(4 * self._scale))
        pad_y = int(round(3 * self._scale))
        avail_w = max(1, w - 2 * pad_x)
        header_font = self._font(max(8, int(round(8 * self._scale))))
        body_font = self._font(max(8, int(round(9 * self._scale))))
        header_lines = self._wrap(draw, header, header_font, avail_w)
        body_lines = self._wrap(draw, body, body_font, avail_w)
        gap = int(round(2 * self._scale))
        total_px = (
            2 * pad_y
            + self._line_height(draw, header_font) * len(header_lines)
            + gap
            + self._line_height(draw, body_font) * len(body_lines)
        )
        return total_px / self._scale

    def render_block(
        self,
        header: str,
        body: str,
        box_width_pt: float,
        box_height_pt: float,
        accent_hex: str,
        box_hex: str,
    ) -> bytes:
        """Return PNG bytes for a wrapped, multi-line translation block."""
        w = max(8, int(round(box_width_pt * self._scale)))
        h = max(8, int(round(box_height_pt * self._scale)))

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        box_rgb = _hex_to_rgb(box_hex)
        accent_rgb = _hex_to_rgb(accent_hex)

        radius = max(2, int(round(2 * self._scale)))
        draw.rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=radius, fill=box_rgb + (255,)
        )
        marker_w = max(2, int(round(2 * self._scale)))
        draw.rectangle([0, 0, marker_w, h - 1], fill=accent_rgb + (255,))

        pad_x = marker_w + int(round(3 * self._scale))
        pad_y = int(round(3 * self._scale))
        avail_w = max(1, w - pad_x - int(round(3 * self._scale)))
        header_font = self._font(max(8, int(round(8 * self._scale))))
        body_font = self._font(max(8, int(round(9 * self._scale))))

        header_lines = self._wrap(draw, header, header_font, avail_w)
        body_lines = self._wrap(draw, body, body_font, avail_w)

        y = pad_y
        for ln in header_lines:
            draw.text((pad_x, y), ln, font=header_font, fill=accent_rgb + (255,))
            y += self._line_height(draw, header_font)
        y += int(round(2 * self._scale))
        text_rgb = (58, 42, 32)  # dark warm brown, distinct from word glosses
        for ln in body_lines:
            draw.text((pad_x, y), ln, font=body_font, fill=text_rgb + (255,))
            y += self._line_height(draw, body_font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _line_height(draw: "ImageDraw.ImageDraw", font: ImageFont.FreeTypeFont) -> int:
        bbox = draw.textbbox((0, 0), "汉字Ag", font=font)
        return int((bbox[3] - bbox[1]) * 1.35)

    @staticmethod
    def _wrap(
        draw: "ImageDraw.ImageDraw",
        text: str,
        font: ImageFont.FreeTypeFont,
        avail_w: int,
    ) -> List[str]:
        """Greedy character-wrap (CJK-safe: wraps by character, not word).

        Existing newlines (used for poem line breaks) are kept as hard
        breaks rather than being merged into a single wrapped paragraph.
        """
        lines: List[str] = []
        for raw_line in text.split("\n"):
            current = ""
            for ch in raw_line:
                trial = current + ch
                bbox = draw.textbbox((0, 0), trial, font=font)
                if (bbox[2] - bbox[0]) > avail_w and current:
                    lines.append(current)
                    current = ch
                else:
                    current = trial
            lines.append(current)
        return lines or [""]


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
