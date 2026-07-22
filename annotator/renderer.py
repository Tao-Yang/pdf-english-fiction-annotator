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
from typing import Tuple

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


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
