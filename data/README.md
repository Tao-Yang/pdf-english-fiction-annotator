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
