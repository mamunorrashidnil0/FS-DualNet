"""
Shrink DiffusionForensics (celebahq) on YOUR PC before zipping/uploading to Drive.
Randomly keeps N images per folder, keeping the same folder layout, and copies only the celebahq face domain.

  python make_subset.py --src "D:/DiffusionForensics" --dst "D:/DF_small" --train_n 10000 --test_n 2000

Then zip the folder  D:/DF_small/DiffusionForensics  ->  DiffusionForensics.zip  and upload that.
Needs only Python 3 (no extra packages).
"""
import argparse
import os
import random
import shutil

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def child_ci(parent, name):
    if not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(parent, d)):
            return d
    return None


def find_images(root, depth=3):
    queue = [(root, 0)]
    while queue:
        d, k = queue.pop(0)
        hit = child_ci(d, "images")
        if hit:
            return os.path.join(d, hit)
        if k < depth:
            queue += [(os.path.join(d, s), k + 1) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--domain", default="celebahq")
    ap.add_argument("--train_n", type=int, default=10000, help="images kept per folder in train/ (real and each generator)")
    ap.add_argument("--test_n", type=int, default=2000, help="images kept per folder in test/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = find_images(args.src)
    if images is None:
        raise SystemExit(f"no 'Images' folder under {args.src}")
    rng = random.Random(args.seed)
    out_root = os.path.join(args.dst, "DiffusionForensics", "Images")
    total = 0
    for split in sorted(os.listdir(images)):
        dom = child_ci(os.path.join(images, split), args.domain)
        if dom is None:
            continue
        dom_path = os.path.join(images, split, dom)
        n_keep = args.train_n if split.lower() == "train" else args.test_n
        for sub in sorted(os.listdir(dom_path)):
            sub_path = os.path.join(dom_path, sub)
            if not os.path.isdir(sub_path):
                continue
            files = []
            for dp, _, fs in os.walk(sub_path):
                files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in EXT]
            files.sort()
            keep = rng.sample(files, n_keep) if len(files) > n_keep else files
            size = 0
            for f in keep:
                dst = os.path.join(out_root, split, dom, sub, os.path.relpath(f, sub_path))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst)
                size += os.path.getsize(dst)
            total += size
            print(f"{split}/{dom}/{sub}: kept {len(keep)} of {len(files)} images ({size / 1e9:.2f} GB)")
    print(f"\nDONE. subset size: {total / 1e9:.2f} GB  ->  zip the folder {os.path.join(args.dst, 'DiffusionForensics')}")


if __name__ == "__main__":
    main()
