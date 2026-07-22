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


_ALPHA_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
_lemmatizer = WordNetLemmatizer()
_tokenizer = TreebankWordTokenizer()


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

    __slots__ = ("surface", "gloss")

    def __init__(self, surface: str, gloss: str) -> None:
        self.surface = surface
        self.gloss = gloss


class WordSelector:
    """Decide which tokens on a page get annotated."""

    def __init__(self, config: AnnotationConfig, dictionary: Dictionary) -> None:
        self.config = config
        self.dictionary = dictionary
        self._threshold = config.zipf_threshold()

    def select_from_text(self, text: str) -> List[Selected]:
        """Return de-duplicated selections for a page's plain text."""
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
                gloss = self.dictionary.gloss(phrase)
                if gloss:
                    key = phrase
                    if key not in chosen:
                        chosen[key] = Selected(" ".join(words), gloss)
                    for j in range(i, i + length):
                        used[j] = True

        # --- single words ---------------------------------------------------
        for i, (surface, _s, _e) in enumerate(tokens):
            if used[i]:
                continue
            if not _ALPHA_RE.match(surface):
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
        return selections[:cap]
