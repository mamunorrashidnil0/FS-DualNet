"""
Extract the FIRST N images from a (huge, possibly truncated) .tar.gz without needing the whole file.

Why this works: a .tar.gz is one sequential stream, so everything before the cut-off point is readable even if the
download stopped at 18 GB of 29 GB. You only need ~10-20k images, so a partial file is enough.

1) from a partial / full file on disk:
     python extract_first_n.py sdv2.tar.gz --n 15000 --out "D:/DiffusionForensics/Images/train/celebahq/sdv2"
2) STREAM straight from a link (stops downloading after N images; link must be a plain direct URL):
     curl -L "<DIRECT_LINK>" | python extract_first_n.py - --n 15000 --out /content/sdv2

Images are written flat (file name only; no sub-folders, no path tricks). Broken/cut-off images are skipped.
Note: you get the first N in archive order, not a random sample -- say so in the paper if the order is not random.
"""
import argparse
import io
import os
import sys
import tarfile
import zlib

from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="path to .tar.gz, or - to read the stream from stdin")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fobj = sys.stdin.buffer if args.src == "-" else open(args.src, "rb")
    count = skipped = 0
    try:
        tf = tarfile.open(fileobj=fobj, mode="r|gz")  # streaming mode: forward-only, never needs the end of the file
        for m in tf:
            if not m.isfile() or os.path.splitext(m.name)[1].lower() not in EXT:
                continue
            data = tf.extractfile(m).read()
            try:
                Image.open(io.BytesIO(data)).verify()
            except Exception:
                skipped += 1
                continue
            name = os.path.basename(m.name)
            dst = os.path.join(args.out, name)
            k = 0
            while os.path.exists(dst):  # same file name in different sub-folders
                k += 1
                dst = os.path.join(args.out, f"{k}_{name}")
            with open(dst, "wb") as f:
                f.write(data)
            count += 1
            if count % 1000 == 0:
                print(f"  {count} images extracted...", flush=True)
            if count >= args.n:
                break
    except (EOFError, tarfile.ReadError, zlib.error, OSError) as e:
        print(f"archive ended early (expected for a partial download): {e}")
    print(f"DONE: {count} images -> {args.out} (skipped {skipped} unreadable)")
    if count < args.n:
        print(f"note: only {count} < {args.n} available in what was readable")


if __name__ == "__main__":
    main()
