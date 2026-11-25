"""
Build per-generator real/fake folders from DeepFakeFace (DFF).

Expected input (each zip extracted into its own folder):
    <root>/wiki/        real images (IMDB-WIKI)
    <root>/inpainting/  Stable Diffusion Inpainting fakes   (diffusion)
    <root>/text2img/    Stable Diffusion v1.5 fakes         (diffusion)
    <root>/insight/     InsightFace face-swap fakes         (NOT diffusion -> good unseen-generator test)
DFF keeps the same relative paths as IMDB-WIKI, so a real image and its fakes share a key.

The 80/10/10 train/val/test split is decided by a hash of that key, so a real image and every fake made
from it always land in the same split (no leakage between train and test).

Output (symlinks):  <out>/<split>/<generator>/real/...   and   <out>/<split>/<generator>/fake/...

  python prep_dff.py --root /content/dff --out /content/data
"""
import argparse
import hashlib
import os
import shutil

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_base(root, name):
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        raise SystemExit(f"missing folder: {d}")
    while True:  # some zips contain a top-level folder with the same name
        entries = os.listdir(d)
        if len(entries) == 1 and entries[0] == name and os.path.isdir(os.path.join(d, name)):
            d = os.path.join(d, name)
        else:
            return d


def scan(base):
    out = {}
    for dp, _, files in os.walk(base):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXT:
                p = os.path.join(dp, f)
                out[os.path.relpath(p, base).replace(os.sep, "/")] = p
    return out


def split_of(key, train=80, val=10):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < train else ("val" if h < train + val else "test")


def link_all(items, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for key, src in items:
        dst = os.path.join(out_dir, key.replace("/", "__"))
        if not os.path.lexists(dst):
            src = os.path.abspath(src)
            try:
                os.symlink(src, dst)
            except OSError:  # Windows without admin: hard link (same drive) or copy
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", nargs="+", default=["inpainting", "text2img", "insight"])
    args = ap.parse_args()

    real = scan(find_base(args.root, "wiki"))
    print(f"real (wiki): {len(real)} images")
    for g in args.gens:
        fake = scan(find_base(args.root, g))
        common = sorted(set(real) & set(fake))
        if len(common) >= 0.5 * min(len(real), len(fake)):
            rk, fk = common, common  # paired keys: keeps real/fake balanced and leak-free
            mode = f"paired by path ({len(common)} pairs)"
        else:
            rk, fk = sorted(real), sorted(fake)
            mode = "WARNING: file names do not match wiki, split done independently"
        print(f"{g}: {len(fake)} fakes | {mode}")
        for split in ["train", "val", "test"]:
            r_items = [(k, real[k]) for k in rk if split_of(k) == split]
            f_items = [(k, fake[k]) for k in fk if split_of(k) == split]
            link_all(r_items, os.path.join(args.out, split, g, "real"))
            link_all(f_items, os.path.join(args.out, split, g, "fake"))
            print(f"   {split:5s}: real {len(r_items):6d} | fake {len(f_items):6d}")


if __name__ == "__main__":
    main()
