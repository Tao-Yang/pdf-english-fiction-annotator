"""ECDICT-backed English -> Chinese lookup.

ECDICT (https://github.com/skywind3000/ECDICT) is a large CSV dictionary with
columns: word, phonetic, definition, translation, pos, ...

We only need a compact ``word -> Chinese translation`` map. The full CSV is
~65 MB, so it is loaded lazily and cached in memory the first time a lookup is
requested. Only the ``word`` and ``translation`` columns are retained.
"""

import csv
import glob
import os
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Union


# ECDICT translations pack several senses onto one line separated by "\n" and
# each sense is prefixed with a part-of-speech tag such as "n. ". We keep the
# first one or two short senses so the margin note stays readable.
_POS_PREFIX_RE = re.compile(r"^[a-z]{1,5}\.\s*")

# Wade-Giles/scholarly romanizations mark aspirated consonants with an
# apostrophe (Ch'i, P'an) which PDFs frequently typeset as a curly quote
# (\u2019 or \u2018) rather than a straight ASCII apostrophe. Normalizing both
# forms to "'" lets curated glossary entries (written with a plain ASCII
# apostrophe) match tokens extracted from the PDF regardless of which quote
# character the source uses.
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u02bc\u00b4`]")


def _normalize_key(word: str) -> str:
    return _APOSTROPHE_RE.sub("'", word.strip().lower())


class Dictionary:
    """Lazy in-memory ECDICT lookup keyed by lower-cased headword.

    An optional ``extra_path`` points at hand-compiled ``term,chinese`` CSVs
    of historical/cultural glossary content (Ming/Qing official titles,
    place names, historical figures, idioms and slang commonly found in
    scholarly novel translations but absent from a general-purpose
    dictionary like ECDICT). Entries in these files take priority over
    ECDICT and support multi-word phrases. ``extra_path`` may be a single
    CSV file, a directory (all ``*.csv`` files inside it are merged), or a
    list/tuple of either.
    """

    def __init__(
        self,
        ecdict_path: str,
        extra_path: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        self._path = ecdict_path
        self._table: Optional[Dict[str, str]] = None
        self._db: Optional[sqlite3.Connection] = None
        self._cache: Dict[str, Optional[str]] = {}
        self._extra_path = extra_path
        self._extra_table: Optional[Dict[str, str]] = None

    def _extra_csv_paths(self) -> List[str]:
        raw = self._extra_path
        if not raw:
            return []
        candidates: Sequence[str] = [raw] if isinstance(raw, str) else raw
        paths: List[str] = []
        for candidate in candidates:
            if os.path.isdir(candidate):
                paths.extend(sorted(glob.glob(os.path.join(candidate, "*.csv"))))
            elif os.path.isfile(candidate):
                paths.append(candidate)
        return paths

    def _load_extra(self) -> Dict[str, str]:
        if self._extra_table is not None:
            return self._extra_table
        table: Dict[str, str] = {}
        for path in self._extra_csv_paths():
            with open(path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    term = _normalize_key(row.get("term") or "")
                    gloss = (row.get("chinese") or "").strip()
                    if term and gloss:
                        table[term] = gloss
        self._extra_table = table
        return table

    def _raw_gloss(self, word: str) -> Optional[str]:
        key = _normalize_key(word)
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

    def extra_gloss(self, word: str) -> Optional[str]:
        """Look up ``word`` in the curated historical/cultural glossary only.

        Unlike :meth:`gloss`, this never falls back to ECDICT. It is used to
        force-annotate proper nouns (place names, historical figures) and
        idioms that are deliberately curated for this project even when the
        general word-selection heuristics (POS tag, corpus frequency) would
        otherwise skip them. The result is condensed the same way as
        :meth:`gloss` so it still fits the single-line margin label.
        """
        raw = self._load_extra().get(_normalize_key(word))
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
