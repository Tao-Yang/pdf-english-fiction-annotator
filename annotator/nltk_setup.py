"""Ensure required NLTK corpora are available.

The selector needs the POS tagger, tokenizer and WordNet. This helper downloads
them on demand so first-time users don't hit cryptic ``LookupError`` messages.
"""

import nltk

_REQUIRED = [
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt", "punkt"),
]


def ensure_nltk_data(quiet: bool = True) -> None:
    for resource, package in _REQUIRED:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=quiet)
