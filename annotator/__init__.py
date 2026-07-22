"""PDF English-fiction annotator.

Add concise, non-obstructive Chinese margin annotations to English-fiction PDFs.
"""

from .config import AnnotationConfig
from .pipeline import annotate_pdf

__all__ = ["AnnotationConfig", "annotate_pdf"]
__version__ = "1.0.0"
