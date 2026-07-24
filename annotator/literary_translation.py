"""Optional "master translator" mode for long sentences and verse quotations.

When ``AnnotationConfig.enable_literary_translation`` is on, long/complex
prose sentences and poem/verse excerpts embedded in the novel are sent to the
Claude API for a full literary translation into Chinese, rendered in the
style of one of four celebrated 20th-century translators, as an *additional*
block annotation alongside the ordinary per-word glosses:

* **Fu Donghua (傅东华)** -- his "Gone with the Wind" / 《飘》 translation is
  the archetype of naturalised, fluent narrative prose. Used for long,
  plain narrative/descriptive sentences.
* **Zhu Shenghao (朱生豪)** -- his Shakespeare translations read as natural,
  rhythmic stage dialogue. Used for long dialogue / exclamatory rhetorical
  sentences.
* **Xu Yuanchong (许渊冲)** -- "three beauties" (意美/音美/形美) verse
  translation that preserves rhyme, meter and line breaks. Used for detected
  poem/verse quotations.
* **Yu Guangzhong (余光中)** -- refined, lyrical literary-essay prose. Used
  for long, clause-heavy reflective/descriptive sentences that are not
  dialogue.

This module is entirely optional and additive: if the feature is off, a
required dependency/API key is missing, or a particular call fails, the rest
of the pipeline (word-level glosses) is completely unaffected.
"""

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import AnnotationConfig

try:
    from nltk.tokenize import sent_tokenize
except Exception:  # pragma: no cover - nltk is a hard dependency elsewhere
    sent_tokenize = None


# --- Translator personas ---------------------------------------------------

TRANSLATOR_STYLES: Dict[str, Dict[str, str]] = {
    "fu_donghua": {
        "label": "傅东华",
        "system": (
            "你是翻译家傅东华（《飘》中译本译者）。请把给出的英文长句译成"
            "地道、流畅的现代汉语叙事散文，采用归化译法：可适度调整语序、"
            "拆分或合并分句，使中文读起来自然顺畅，像中文小说本身写就的一样，"
            "但不得增删原文没有的情节信息。只输出译文本身，不要解释、不要"
            "加引号、注释或署名。"
        ),
    },
    "zhu_shenghao": {
        "label": "朱生豪",
        "system": (
            "你是翻译家朱生豪（莎士比亚戏剧中译本译者）。请把给出的英文长句"
            "或独白译成有舞台朗诵感、抑扬顿挫的中文，保留原文的修辞气势与"
            "情感强度，用词典雅但不生僻。只输出译文本身，不要解释、不要"
            "加引号、注释或署名。"
        ),
    },
    "xu_yuanchong": {
        "label": "许渊冲",
        "system": (
            "你是翻译家许渊冲，主张诗歌翻译要兼顾“意美、音美、形美”。请把"
            "给出的英文诗句译成同样分行的中文诗句：每一行原文对应一行译文，"
            "行与行之间用换行符分隔，不要合并成一段；尽量押韵、保持节奏感和"
            "对仗，同时忠实传达原意。只输出译文本身，不要解释、不要加引号、"
            "注释或署名。"
        ),
    },
    "yu_guangzhong": {
        "label": "余光中",
        "system": (
            "你是诗人、翻译家余光中。请把给出的英文长句译成典雅凝练、富有"
            "文学意境的中文散文，讲究音节与节奏之美，避免翻译腔。只输出"
            "译文本身，不要解释、不要加引号、注释或署名。"
        ),
    },
}

# Guard against pathological over-long "sentences" (e.g. mis-tokenized runs)
# bloating the API request.
_MAX_INPUT_CHARS = 800


@dataclass
class LiteraryPassage:
    """A detected long sentence or poem quotation worth a full translation."""

    kind: str  # "poem" | "sentence"
    translator_key: str
    text: str  # normalised original English (newline-joined for poems)
    anchor_head: str  # first few words, used to locate the passage via
    #                    ``page.search_for``


def _normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def _head_words(s: str, n: int = 6) -> str:
    return " ".join(s.split()[:n])


def _is_poem_run(lines: List[str], col_width: float) -> bool:
    if len(lines) < 2:
        return False
    return all(0 < len(ln) <= 0.75 * col_width for ln in lines)


def detect_passages(
    page_text: str, config: AnnotationConfig
) -> List[LiteraryPassage]:
    """Find long-sentence and poem/verse passages in a page's plain text.

    Poems are detected with a simple layout heuristic that works directly
    off ``page.get_text("text")`` (no bbox extraction needed): a run of two
    or more consecutive non-blank lines that are all noticeably shorter than
    the page's typical wrapped-prose line width is treated as a verse
    quotation, since prose paragraphs wrap close to the column width while
    typeset verse breaks each line at the poet's own line breaks. Everything
    else is treated as ordinary prose and split into sentences (NLTK); a
    sentence qualifies as "long" once it has at least
    ``config.literary_long_sentence_words`` words.
    """
    raw_lines = page_text.split("\n")
    non_blank = [ln.strip() for ln in raw_lines if ln.strip()]
    if not non_blank:
        return []
    lengths = sorted(len(ln) for ln in non_blank)
    # Typical wrapped-prose width ~= the page's own 75th-percentile line
    # length (long lines dominate a prose page; the last line of each
    # paragraph is short, same as poem lines, but is the minority).
    col_width = lengths[max(0, int(len(lengths) * 0.75) - 1)]
    if col_width < 20:
        # Page has essentially no long lines (e.g. a title page, or a page
        # that is itself mostly verse); fall back to the max observed
        # length so the poem heuristic still has a meaningful baseline.
        col_width = max(lengths)

    passages: List[LiteraryPassage] = []
    prose_lines: List[str] = []
    seen_poem_texts = set()
    i = 0
    n = len(raw_lines)
    while i < n:
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue
        run = [line]
        j = i + 1
        while (
            j < n
            and raw_lines[j].strip()
            and len(raw_lines[j].strip()) <= 0.75 * col_width
        ):
            run.append(raw_lines[j].strip())
            j += 1
        if len(run) >= 2 and _is_poem_run(run, col_width):
            poem_text = "\n".join(run)
            if poem_text not in seen_poem_texts:
                seen_poem_texts.add(poem_text)
                passages.append(
                    LiteraryPassage(
                        kind="poem",
                        translator_key="xu_yuanchong",
                        text=poem_text[:_MAX_INPUT_CHARS],
                        anchor_head=_head_words(run[0]),
                    )
                )
            i = j
            continue
        prose_lines.append(line)
        i += 1

    prose_text = _normalize_ws(" ".join(prose_lines))
    if prose_text and sent_tokenize is not None:
        try:
            sentences = sent_tokenize(prose_text)
        except Exception:
            sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < config.literary_long_sentence_words:
                continue
            is_dialogue = (
                '"' in sent
                or "\u201c" in sent
                or sent.endswith(("!", "\u2019\u2019", "?"))
            )
            if is_dialogue:
                translator_key = "zhu_shenghao"
            elif sent.count(",") >= 3:
                # Long, comma-heavy, non-dialogue sentences read as
                # clause-rich, reflective/descriptive prose -- Yu
                # Guangzhong's more literary-essay register.
                translator_key = "yu_guangzhong"
            else:
                translator_key = "fu_donghua"
            passages.append(
                LiteraryPassage(
                    kind="sentence",
                    translator_key=translator_key,
                    text=sent[:_MAX_INPUT_CHARS],
                    anchor_head=_head_words(sent),
                )
            )
    return passages


class LiteraryTranslator:
    """Fail-safe wrapper around the Claude API for literary passages.

    Any failure (missing dependency, missing API key, network error, rate
    limit ...) is caught and turns :meth:`translate` into a no-op returning
    ``None`` -- literary translation is a best-effort enhancement layered on
    top of the ordinary word-gloss pipeline, never a hard requirement.
    """

    def __init__(self, config: AnnotationConfig, budget: Optional[int] = None) -> None:
        self.config = config
        self.remaining = config.literary_max_total if budget is None else budget
        self._cache: Dict[str, Optional[str]] = {}
        self._client = None
        self._warned = False
        self.available = False
        api_key = config.literary_translator_api_key or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
            self.available = True
        except Exception:
            self.available = False

    def translate(self, passage: LiteraryPassage) -> Optional[str]:
        if not self.available or self.remaining <= 0:
            return None
        cache_key = hashlib.sha256(
            (passage.translator_key + "\x00" + passage.text).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        style = TRANSLATOR_STYLES[passage.translator_key]
        text: Optional[str] = None
        try:
            response = self._client.messages.create(
                model=self.config.literary_translator_model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                system=style["system"],
                messages=[{"role": "user", "content": passage.text}],
            )
            text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
        except Exception as exc:
            if not self._warned:
                print(
                    "Literary translation unavailable (%s); continuing "
                    "without it." % exc,
                    file=sys.stderr,
                )
                self._warned = True
            text = None
        self._cache[cache_key] = text or None
        if text:
            self.remaining -= 1
        return text or None
