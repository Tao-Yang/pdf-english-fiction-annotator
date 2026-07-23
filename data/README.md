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

# Ming/Qing official-title glossary

`ming_qing_titles.csv` is a small, hand-compiled `term,chinese` list covering
Ming/Qing central- and local-government official titles, examination-system
terms, and related historical/cultural vocabulary commonly found in scholarly
English translations of classical Chinese novels (e.g. David Tod Roy's
*The Plum in the Golden Vase*) but missing from a general-purpose dictionary
like ECDICT — phrases such as "grand secretary", "censor-in-chief" or
"grand coordinator" have no everyday English meaning ECDICT would know.

It is committed directly to the repo (a few KB) since, unlike ECDICT, it does
not need to be downloaded or built into SQLite. Terms and their standard
English renderings follow long-established sinological convention (notably
Charles O. Hucker's official-title nomenclature), cross-checked against
Wikipedia (CC BY-SA 4.0) for accuracy; this file itself is original work, not
a copy of any single source's dataset or prose.

`Dictionary` checks this glossary before ECDICT and supports multi-word
phrases (up to `AnnotationConfig.max_phrase_len` words). Override the path via
`--historical-glossary /path/to.csv` on the CLI or
`AnnotationConfig.historical_glossary_path`.

