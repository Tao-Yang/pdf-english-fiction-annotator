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
import sqlite3
from typing import Dict, Optional


# ECDICT translations pack several senses onto one line separated by "\n" and
# each sense is prefixed with a part-of-speech tag such as "n. ". We keep the
# first one or two short senses so the margin note stays readable.
_POS_PREFIX_RE = re.compile(r"^[a-z]{1,5}\.\s*")


class Dictionary:
    """Lazy in-memory ECDICT lookup keyed by lower-cased headword.

    An optional ``extra_path`` points at a small CSV of hand-compiled
    ``term,chinese`` pairs (e.g. Ming/Qing official titles and historical
    terminology commonly found in scholarly novel translations but absent
    from a general-purpose dictionary like ECDICT). Entries in that file take
    priority over ECDICT and support multi-word phrases.
    """

    def __init__(self, ecdict_path: str, extra_path: Optional[str] = None) -> None:
        self._path = ecdict_path
        self._table: Optional[Dict[str, str]] = None
        self._db: Optional[sqlite3.Connection] = None
        self._cache: Dict[str, Optional[str]] = {}
        self._extra_path = extra_path
        self._extra_table: Optional[Dict[str, str]] = None

    def _load_extra(self) -> Dict[str, str]:
        if self._extra_table is not None:
            return self._extra_table
        table: Dict[str, str] = {}
        if self._extra_path and os.path.isfile(self._extra_path):
            with open(self._extra_path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    term = (row.get("term") or "").strip().lower()
                    gloss = (row.get("chinese") or "").strip()
                    if term and gloss:
                        table[term] = gloss
        self._extra_table = table
        return table

    def _raw_gloss(self, word: str) -> Optional[str]:
        key = word.strip().lower()
        if key in self._cache:
            return self._cache[key]

        extra = self._load_extra().get(key)
        if extra:
            if len(self._cache) >= 8192:
                self._cache.clear()
            self._cache[key] = extra
            return extra

        if self._path.lower().endswith((".sqlite", ".sqlite3", ".db")):
            if self._db is None:
                if not os.path.isfile(self._path):
                    raise FileNotFoundError("ECDICT database not found at %r" % self._path)
                self._db = sqlite3.connect(
                    "file:%s?mode=ro" % os.path.abspath(self._path).replace("\\", "/"),
                    uri=True,
                )
            row = self._db.execute(
                "SELECT translation FROM entries WHERE word = ?", (key,)
            ).fetchone()
            raw = row[0] if row else None
        else:
            raw = self._ensure_loaded().get(key)

        # Keep repeated phrase / lemma probes off disk while bounding memory.
        if len(self._cache) >= 8192:
            self._cache.clear()
        self._cache[key] = raw
        return raw

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
        raw = self._raw_gloss(word)
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
