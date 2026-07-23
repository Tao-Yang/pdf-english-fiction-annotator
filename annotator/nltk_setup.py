"""Ensure required NLTK corpora are available.

The selector needs the POS tagger, tokenizer and WordNet. This helper downloads
them on demand so first-time users don't hit cryptic ``LookupError`` messages.
"""

import nltk

# NLTK 3.8.2+ renamed several resources (e.g. the Perceptron tagger gained an
# ``_eng`` suffix and ``punkt`` became ``punkt_tab``). We try both the new and
# legacy names and simply skip any package the installed NLTK doesn't know
# about, so the code works across NLTK versions.
_REQUIRED = [
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
]


def ensure_nltk_data(quiet: bool = True) -> None:
    for resource, package in _REQUIRED:
        try:
            nltk.data.find(resource)
        except (LookupError, OSError):
            # Some NLTK 3.8.x releases mis-resolve the newer "punkt_tab"
            # resource path (raising a raw OSError instead of LookupError
            # when a sibling "punkt" PY3 zip-safe path exists locally). Treat
            # any lookup failure the same way: attempt a download and move
            # on -- a sibling old/new-name entry in _REQUIRED covers the same
            # capability if this particular package name isn't available.
            try:
                nltk.download(package, quiet=quiet)
            except Exception:
                # Package name not available in this NLTK version; a sibling
                # entry (old/new name) covers the same capability.
                pass
