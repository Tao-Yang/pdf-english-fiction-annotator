"""Download the ECDICT dictionary CSV used for English -> Chinese glosses.

    python -m scripts.download_ecdict            # -> data/ecdict.csv
    python -m scripts.download_ecdict path.csv   # custom destination

ECDICT is MIT-licensed: https://github.com/skywind3000/ECDICT
"""

import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv"


def download(dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print("Downloading ECDICT (~65 MB) from %s ..." % URL)

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = done * 100 / total_size
            sys.stdout.write("\r  %6.2f%% (%d/%d bytes)" % (pct, done, total_size))
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, dest, _hook)
    print("\nSaved to %s" % dest)


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "ecdict.csv")
    download(destination)
