"""
Convert DiffusionForensics (face domain = celebahq) into the layout train_eval.py expects.

Handles the real RecDrive layout:
    <root>/Images/train/celebahq/{real, sdv2}
    <root>/Images/test/celebahq/{real, sdv2, dalle2, if, midjourney}
    (celebahq has NO val split -> a validation set is carved out of train)
Folder names are matched case-insensitively ("Images" == "images").

Output, one folder per generator (symlinks):
    <out>/<split>/<generator>/real
    <out>/<split>/<generator>/fake

  python prep_data.py --root /content/DiffusionForensics --out /content/data_df
  # train: --train /content/data_df/train/sdv2    val: --val /content/data_df/val/sdv2
  # test : sdv2=.../test/sdv2  dalle2=.../test/dalle2  if=.../test/if  midjourney=.../test/midjourney
"""
import argparse
import hashlib
import os
import shutil
import subprocess

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def child_ci(parent, name):
    """Case-insensitive lookup of a sub-directory."""
    if not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return None


def show_tree(root, depth=3, limit=40):
    """Print the folder tree (dirs only) so a wrong layout is obvious from the error message."""
    n = 0
    for dp, dirs, _ in os.walk(root):
        dirs.sort()
        k = os.path.relpath(dp, root).count(os.sep) if dp != root else 0
        print("   " * k + os.path.basename(dp) + "/")
        n += 1
        if k >= depth - 1:
            dirs[:] = []
        if n >= limit:
            print("   ...")
            return


def find_images(root, depth=3):
    queue = [(root, 0)]
    while queue:
        d, k = queue.pop(0)
        hit = child_ci(d, "images")
        if hit:
            return hit
        if k < depth:
            queue += [(os.path.join(d, s), k + 1) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
    return None


def list_images(folder):
    out = {}
    for dp, _, files in os.walk(folder, followlinks=True):  # nested class folders (ImageNet style) are fine
        for f in files:
            if os.path.splitext(f)[1].lower() in EXT:
                p = os.path.join(dp, f)
                out[os.path.relpath(p, folder).replace(os.sep, "/")] = p
    return out


def pct(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


def safe_link_file(src, dst):
    """symlink -> hard link (works on Windows without admin, same drive) -> copy."""
    src = os.path.abspath(src)
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def link_files(items, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for key, src in items:
        dst = os.path.join(out_dir, key.replace("/", "__"))
        if not os.path.lexists(dst):
            safe_link_file(src, dst)


def link_dir(src, dst):
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
        return
    except OSError:
        pass
    if os.name == "nt":  # Windows junction: no admin rights needed
        r = subprocess.run(["cmd", "/c", "mklink", "/J", dst, os.path.abspath(src)], capture_output=True)
        if r.returncode == 0:
            return
    link_files(sorted(list_images(src).items()), dst)  # last resort: link/copy every file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing Images/ (or a parent of it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="celebahq")
    ap.add_argument("--val_frac", type=float, default=0.1, help="carved from train when the domain has no val split")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"ERROR: folder {args.root} does not exist. Did the unzip step fail? Check the zip path/name in Drive.")
    images = find_images(args.root)
    if images is None:
        print(f"ERROR: no 'Images' folder found under {args.root}. What I see there:")
        show_tree(args.root)
        raise SystemExit(1)
    print("using", images)

    doms = {}
    for split in sorted(os.listdir(images)):
        dom = child_ci(os.path.join(images, split), args.domain)
        real = child_ci(dom, "real") if dom else None
        if real is None:
            continue
        gens = {g: os.path.join(dom, g) for g in sorted(os.listdir(dom))
                if g.lower() != "real" and os.path.isdir(os.path.join(dom, g))}
        doms[split.lower()] = (real, gens)
    if not doms:
        print(f"ERROR: domain '{args.domain}' (with a 'real' folder) not found in any split. What I see in {images}:")
        show_tree(images)
        raise SystemExit(1)

    carve = "train" in doms and "val" not in doms
    if carve:
        print(f"note: '{args.domain}' has no val split -> {int(args.val_frac * 100)}% of train is held out as val")
    cut = int(args.val_frac * 100)

    for split, (real, gens) in doms.items():
        if split == "train" and carve:
            real_files = list_images(real)
            for g, gpath in gens.items():
                fake_files = list_images(gpath)
                for part in ("train", "val"):
                    sel = lambda d: [(k, p) for k, p in sorted(d.items()) if (pct(k) < cut) == (part == "val")]
                    r, f = sel(real_files), sel(fake_files)
                    link_files(r, os.path.join(args.out, part, g, "real"))
                    link_files(f, os.path.join(args.out, part, g, "fake"))
                    print(f"  {part:5s}/{g}: real {len(r)} | fake {len(f)}")
        else:
            n_real = len(list_images(real))
            for g, gpath in gens.items():
                link_dir(real, os.path.join(args.out, split, g, "real"))
                link_dir(gpath, os.path.join(args.out, split, g, "fake"))
                print(f"  {split:5s}/{g}: real {n_real} | fake {len(list_images(gpath))}")


if __name__ == "__main__":
    main()
