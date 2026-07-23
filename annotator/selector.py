"""Select which words / phrases on a page deserve a Chinese annotation.

Selection combines three signals:

* **Rarity** — ``wordfreq.zipf_frequency`` gives a 0-8 Zipf score; rarer words
  (score at or below the CEFR threshold) are candidates.
* **Part of speech** — NLTK POS tags let us skip proper nouns and keep content
  words (nouns, verbs, adjectives, adverbs). Lemmatisation lets us look up the
  dictionary head form while still highlighting the inflected surface form.
* **Dictionary coverage** — a candidate is only kept if the dictionary can
  supply a Chinese gloss.

Multi-word idioms / collocations are detected greedily (longest first) before
single words so phrases like "give up" are preferred over the bare verb.
"""

import re
from typing import Dict, List, Optional, Tuple

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from wordfreq import zipf_frequency

from .config import AnnotationConfig
from .dictionary import Dictionary


# A "word" token: starts with a Unicode letter, followed by any mix of
# Unicode letters, apostrophes (straight ' or the curly \u2019/\u2018 used by
# scanned/typeset PDFs), or hyphens. This must stay Unicode-aware (not
# ASCII-only [A-Za-z]) so that romanized Chinese names using diacritics such
# as u-umlaut (Y\u00fc, L\u00fc) or apostrophe-marked aspirates (Wade-Giles
# forms like Ch\u2019i, P\u2019an) are still recognised as annotation
# candidates instead of being silently skipped before any dictionary lookup.
_ALPHA_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|['\u2018\u2019-])*$")
_lemmatizer = WordNetLemmatizer()
_tokenizer = TreebankWordTokenizer()

# Scholarly annotated translations (e.g. David Tod Roy's) typeset footnote
# markers as a bare digit glued directly onto the end of the previous word or
# its trailing punctuation, with no separating space: "Pang:3", "Y\u00fc-chi4",
# "naught.6", "...beauty?\"8". Once the PDF is flattened to plain text this
# merges the digit onto the token ("Pang:3", "naught.6"), which then fails
# every regex/dictionary check and silently blocks annotation of that
# word/phrase. Strip such digits before tokenizing. This never touches
# genuine numbers, which are always preceded by whitespace in prose (e.g.
# "Chapter 1", "ten thousand").
_FOOTNOTE_MARKER_RE = re.compile(
    r"(?<=[A-Za-z\u00C0-\u024F:,.;'\u2018\u2019\"\u201c\u201d)])"
    r"\d{1,2}(?=\s|$|[A-Z\"\u201c\u2019'])"
)


def _strip_footnote_markers(text: str) -> str:
    return _FOOTNOTE_MARKER_RE.sub("", text)


def _wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


class Selected:
    """A chosen word/phrase and its gloss."""

    __slots__ = ("surface", "gloss", "priority")

    def __init__(self, surface: str, gloss: str, priority: bool = False) -> None:
        self.surface = surface
        self.gloss = gloss
        # True when the gloss came from the curated historical/cultural
        # glossary (place names, figures, idioms) rather than the
        # general-purpose ECDICT lookup. Used to keep these entries from
        # being crowded out by the per-page annotation density cap.
        self.priority = priority


class WordSelector:
    """Decide which tokens on a page get annotated."""

    def __init__(self, config: AnnotationConfig, dictionary: Dictionary) -> None:
        self.config = config
        self.dictionary = dictionary
        self._threshold = config.zipf_threshold()

    def select_from_text(self, text: str) -> List[Selected]:
        """Return de-duplicated selections for a page's plain text."""
        text = _strip_footnote_markers(text)
        spans = _tokenizer.span_tokenize(text)
        tokens: List[Tuple[str, int, int]] = [
            (text[s:e], s, e) for s, e in spans
        ]
        surfaces = [t[0] for t in tokens]
        tags = pos_tag(surfaces)

        chosen: Dict[str, Selected] = {}
        used = [False] * len(tokens)

        # --- phrases first (longest first) ----------------------------------
        for length in range(self.config.max_phrase_len, 1, -1):
            for i in range(len(tokens) - length + 1):
                if any(used[i : i + length]):
                    continue
                words = surfaces[i : i + length]
                if not all(_ALPHA_RE.match(w) for w in words):
                    continue
                phrase = " ".join(w.lower() for w in words)
                extra = self.dictionary.extra_gloss(phrase)
                gloss = extra or self.dictionary.gloss(phrase)
                if gloss:
                    key = phrase
                    if key not in chosen:
                        chosen[key] = Selected(
                            " ".join(words), gloss, priority=bool(extra)
                        )
                    for j in range(i, i + length):
                        used[j] = True

        # --- single words ---------------------------------------------------
        for i, (surface, _s, _e) in enumerate(tokens):
            if used[i]:
                continue
            if not _ALPHA_RE.match(surface):
                continue

            # Curated historical/cultural terms (place names, figures,
            # idioms) are always annotated, bypassing the proper-noun and
            # corpus-frequency filters below: they are rare/unknown to a
            # general reader regardless of what NLTK's POS tagger or
            # everyday-English frequency statistics say about them.
            extra = self.dictionary.extra_gloss(surface.lower())
            if extra:
                key = surface.lower()
                if key not in chosen:
                    chosen[key] = Selected(surface, extra, priority=True)
                continue

            tag = tags[i][1]
            if self.config.skip_proper_nouns and tag in ("NNP", "NNPS"):
                continue
            # Only content words.
            if not (tag.startswith(("N", "V", "J", "R"))):
                continue
            lemma = _lemmatizer.lemmatize(surface.lower(), _wordnet_pos(tag))
            if zipf_frequency(lemma, "en") > self._threshold:
                continue
            gloss = self._lookup(surface.lower(), lemma)
            if not gloss:
                continue
            key = lemma
            if key not in chosen:
                chosen[key] = Selected(surface, gloss)

        selections = list(chosen.values())
        return self._limit_density(selections)

    def _lookup(self, surface: str, lemma: str) -> Optional[str]:
        return self.dictionary.gloss(surface) or self.dictionary.gloss(lemma)

    def _limit_density(self, selections: List[Selected]) -> List[Selected]:
        cap = self.config.max_notes_per_page
        if len(selections) <= cap:
            return selections
        # Curated historical/cultural glossary hits (place names, figures,
        # idioms) take priority over ordinary vocabulary picks so a dense
        # page doesn't crowd out the terms a reader most needs context for.
        priority = [s for s in selections if s.priority]
        rest = [s for s in selections if not s.priority]
        if len(priority) >= cap:
            return priority[:cap]
        return priority + rest[: cap - len(priority)]
