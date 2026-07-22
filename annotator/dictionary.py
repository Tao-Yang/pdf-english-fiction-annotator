"""ECDICT-backed English -> Chinese lookup.

ECDICT (https://github.com/skywind3000/ECDICT) is a large CSV dictionary with
columns: word, phonetic, definition, translation, pos, ...

We only need a compact ``word -> Chinese translation`` map. The full CSV is
~65 MB, so it is loaded lazily and cached in memory the first time a lookup is
requested. Only the ``word`` and ``translation`` columns are retained.
"""

import csv
import os
import re
from typing import Dict, Optional


# ECDICT translations pack several senses onto one line separated by "\n" and
# each sense is prefixed with a part-of-speech tag such as "n. ". We keep the
# first one or two short senses so the margin note stays readable.
_POS_PREFIX_RE = re.compile(r"^[a-z]{1,5}\.\s*")


class Dictionary:
    """Lazy in-memory ECDICT lookup keyed by lower-cased headword."""

    def __init__(self, ecdict_path: str) -> None:
        self._path = ecdict_path
        self._table: Optional[Dict[str, str]] = None

    def _ensure_loaded(self) -> Dict[str, str]:
        if self._table is not None:
            return self._table

        if not os.path.isfile(self._path):
            raise FileNotFoundError(
                "ECDICT csv not found at %r. Download it first, e.g.:\n"
                "  curl -L -o %s "
                "https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv"
                % (self._path, self._path)
            )

        table: Dict[str, str] = {}
        # utf-8-sig strips a possible BOM; newline='' is required by csv.
        with open(self._path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                word = (row.get("word") or "").strip().lower()
                translation = (row.get("translation") or "").strip()
                if word and translation:
                    table[word] = translation
        self._table = table
        return table

    def gloss(self, word: str) -> Optional[str]:
        """Return a short Chinese gloss for ``word`` or ``None``.

        The raw ECDICT translation may contain several newline-separated senses
        with POS prefixes; this trims it to the first one or two concise senses.
        """
        table = self._ensure_loaded()
        raw = table.get(word.strip().lower())
        if not raw:
            return None
        return self._condense(raw)

    @staticmethod
    def _condense(raw: str) -> str:
        senses = [s.strip() for s in raw.replace("\\n", "\n").split("\n") if s.strip()]
        cleaned = []
        for sense in senses:
            sense = _POS_PREFIX_RE.sub("", sense).strip()
            # Drop trailing English gloss noise, keep the Chinese-bearing part.
            if sense:
                cleaned.append(sense)
            if len(cleaned) >= 2:
                break
        if not cleaned:
            return raw.strip()
        note = "；".join(cleaned)
        # Keep the margin note tight.
        if len(note) > 18:
            note = note[:18] + "…"
        return note
