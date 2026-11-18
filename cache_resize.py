"""
One-time speed-up: resize every prepared image to SIZE x SIZE (PIL bicubic, the same resize train_eval.py applies) and
store it as a lossless PNG. Big source PNGs (e.g. 1 MB Stable-Diffusion images) are slow to decode on Colab's 2 CPU cores;
256x256 PNGs decode several times faster, so every epoch and every robustness test gets faster.
The pipeline's first step is Resize((256,256)) anyway, so what the network sees stays the same (PNG is lossless).
Run check_bias.py on the ORIGINAL data first: resizing hides size/format differences.

  python cache_resize.py --src /content/data --dst /content/data256 --size 256
Layout (<split>/<generator>/{real,fake}) is preserved; symlinked folders are followed.
"""
import argparse
import os
from multiprocessing import Pool

from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def work(job):
    src, dst, size = job
    if os.path.exists(dst):
        return 0
    try:
        im = Image.open(src).convert("RGB").resize((size, size), Image.BICUBIC)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst, format="PNG", compress_level=1)
        return 1
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    jobs = []
    for dp, _, files in os.walk(args.src, followlinks=True):
        rel = os.path.relpath(dp, args.src)
        for f in files:
            if os.path.splitext(f)[1].lower() in EXT:
                jobs.append((os.path.join(dp, f), os.path.join(args.dst, rel, f + ".png"), args.size))
    print(f"{len(jobs)} images -> {args.dst} ({args.size}x{args.size}) with {args.procs} processes", flush=True)
    done = bad = 0
    with Pool(args.procs) as p:
        for i, r in enumerate(p.imap_unordered(work, jobs, chunksize=32), 1):
            done += r == 1
            bad += r == -1
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"DONE: wrote {done}, unreadable {bad}")


if __name__ == "__main__":
    main()
