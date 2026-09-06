"""
Get a training corpus onto disk.

Tries to download TinyShakespeare (~1.1 MB, public domain). If there is no
network, falls back to the 120 KB sample bundled in this repo so you can train
immediately either way.

    python scratch/prepare_data.py                  # download, or use bundled sample
    python scratch/prepare_data.py --input mine.txt # use your own corpus instead

On using your own text: anything works -- your notes, your journal, exported
chat logs, a book you have the rights to. Aim for at least ~1 MB. Below that the
model memorizes instead of generalizing, which is interesting to watch once and
then not useful.
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TARGET = DATA_DIR / "input.txt"
SAMPLE = DATA_DIR / "sample.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None, help="path to your own .txt corpus")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        src = Path(args.input)
        if not src.exists():
            raise SystemExit(f"No such file: {src}")
        shutil.copy(src, TARGET)
        print(f"Copied your corpus -> {TARGET}")
    else:
        try:
            print(f"Downloading TinyShakespeare from {URL} ...")
            urllib.request.urlopen(URL, timeout=30)  # fail fast if unreachable
            urllib.request.urlretrieve(URL, TARGET)
            print(f"Downloaded -> {TARGET}")
        except Exception as e:
            print(f"Download failed ({e.__class__.__name__}: {e}).")
            print(f"Falling back to the bundled sample at {SAMPLE}")
            shutil.copy(SAMPLE, TARGET)

    text = TARGET.read_text(encoding="utf-8", errors="ignore")
    print()
    print(f"  characters:        {len(text):,}")
    print(f"  distinct symbols:  {len(set(text))}  (this becomes your vocab_size)")
    print(f"  approx. words:     {len(text.split()):,}")
    print()
    print("Preview:")
    print("-" * 60)
    print(text[:400])
    print("-" * 60)
    print()
    print("Next:  python scratch/train.py")


if __name__ == "__main__":
    main()
