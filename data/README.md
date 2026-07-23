# ECDICT data

The dictionary CSV is **not** committed to the repository because it is ~65 MB.

Download it once with:

```bash
python -m scripts.download_ecdict
```

This writes `ecdict.csv` into this folder. The file is MIT-licensed and comes
from [skywind3000/ECDICT](https://github.com/skywind3000/ECDICT).

You can also point the CLI at any ECDICT-format CSV via `--ecdict /path/to.csv`.
The only columns used are `word` and `translation`.

# Historical / cultural glossary

`glossaries/` holds several small, hand-compiled `term,chinese` CSVs covering
vocabulary commonly found in scholarly English translations of classical
Chinese novels (e.g. David Tod Roy's *The Plum in the Golden Vase*) but
missing from a general-purpose dictionary like ECDICT:

- `official_titles.csv` — Ming/Qing central- and local-government official
  titles, examination-system terms, and related institutions/customs, e.g.
  "grand secretary", "censor-in-chief", "grand coordinator".
- `places.csv` — historical place names, e.g. "Nanjing", "Yingtian
  prefecture", "West Lake", "the Grand Canal".
- `figures.csv` — historical figures who commonly appear in Ming-era fiction
  and its introductions/annotations, e.g. "Zhu Yuanzhang", "the Wanli
  Emperor", "Zheng He", "Wang Yangming".
- `idioms.csv` — period slang, honorifics and cultural idioms, e.g. "Son of
  Heaven", "to kowtow", "feng shui".

These have no everyday English meaning ECDICT would know. They are committed
directly to the repo (a few KB each) since, unlike ECDICT, they do not need
to be downloaded or built into SQLite. Terms and their standard English
renderings follow long-established sinological convention (notably Charles
O. Hucker's official-title nomenclature) and standard historical facts (era
names, reign years), cross-checked against Wikipedia (CC BY-SA 4.0) for
accuracy; these files are original work, not a copy of any single source's
dataset or prose.

`Dictionary` merges every `*.csv` file in this directory (or a single file,
or a list of paths) and checks it before ECDICT, supporting multi-word
phrases (up to `AnnotationConfig.max_phrase_len` words). Single-word entries
found here (e.g. proper nouns like place names or people) are always
annotated, bypassing the usual proper-noun and word-frequency filters.
Override the path via `--historical-glossary /path/to/dir-or-file.csv` on
the CLI or `AnnotationConfig.historical_glossary_path`.


