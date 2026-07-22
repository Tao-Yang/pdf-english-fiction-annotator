"""Build a compact, disk-backed ECDICT database for the web app.

The source CSV is about 65 MB but expands to several hundred MB when loaded
into a Python dict. That can OOM a free Render instance. SQLite keeps lookups
fast while using almost no Python heap. This script runs during Docker build;
the app also calls it as a cold-start fallback.
"""

import argparse
import csv
import os
import sqlite3
import sys
import urllib.request
from typing import Callable, Optional

ECDICT_URL = "https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv"
DB_FILENAME = "ecdict.sqlite3"


def _database_is_ready(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 1_000_000:
        return False
    try:
        db = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(path).replace("\\", "/"), uri=True)
        row = db.execute("SELECT COUNT(*) FROM entries").fetchone()
        db.close()
        return bool(row and row[0] > 100_000)
    except (sqlite3.Error, OSError):
        return False


def ensure_ecdict_database(
    data_dir: str,
    status: Optional[Callable[[str], None]] = None,
) -> str:
    """Return a ready SQLite dictionary, downloading/building when necessary."""
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, DB_FILENAME)
    if _database_is_ready(db_path):
        return db_path

    csv_path = os.path.join(data_dir, "ecdict.csv")
    download_path = csv_path + ".download"
    if not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 1_000_000):
        if status:
            status("正在下载词典（约 65 MB）…")
        urllib.request.urlretrieve(ECDICT_URL, download_path)
        os.replace(download_path, csv_path)

    if status:
        status("正在创建低内存词典索引（仅首次需要）…")

    temp_db = db_path + ".building"
    if os.path.exists(temp_db):
        os.remove(temp_db)
    db = sqlite3.connect(temp_db)
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute(
            "CREATE TABLE entries (word TEXT PRIMARY KEY, translation TEXT NOT NULL) WITHOUT ROWID"
        )
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            batch = []
            for row in reader:
                word = (row.get("word") or "").strip().lower()
                translation = (row.get("translation") or "").strip()
                if word and translation:
                    batch.append((word, translation))
                if len(batch) >= 5000:
                    db.executemany(
                        "INSERT OR REPLACE INTO entries VALUES (?, ?)", batch
                    )
                    batch = []
            if batch:
                db.executemany("INSERT OR REPLACE INTO entries VALUES (?, ?)", batch)
        db.commit()
    finally:
        db.close()

    os.replace(temp_db, db_path)
    try:
        os.remove(csv_path)
    except OSError:
        pass
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    def safe_print(message: str) -> None:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(message.encode(encoding, errors="replace").decode(encoding))

    path = ensure_ecdict_database(args.data_dir, safe_print)
    print("ECDICT database ready:", path, os.path.getsize(path))


if __name__ == "__main__":
    main()
