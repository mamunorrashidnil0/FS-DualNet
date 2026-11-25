"""
Pick the images for the format-fair v2-vs-DIRE comparison and write the DIRE job list.

celebahq test generators (default dalle2, midjourney): n real + n fake per set, only images whose OFFICIAL DIRE map
exists too (same file stem), reals distinct across sets. DeepFakeFace (optional): n_dff real (wiki) + n_dff fake.
Creates (all links):
  <out>/orig/<set>/{real,fake}/      original images  (DIRE is computed from these; v2 reads a 256 cache of them)
  <out>/official/<set>/{real,fake}/  official DIRE maps of the same celebahq images (real = JPEG, fake = PNG)
and <jobs>: one line per map to compute: <image> TAB <perturbation> TAB <maps>/<perturbation>/<set>/<class>/<stem>.png

  python make_dire_subsets.py --raw /data/DiffusionForensics --out /tmp/sub --maps /kaggle/working/dire_maps \
      --jobs /kaggle/working/dire_jobs.txt --dff_real /tmp/dff/wiki --dff_fake /tmp/dff/text2img
"""
import argparse
import os
import random

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def child(d, name):
    """case-insensitive child folder"""
    for c in os.listdir(d):
        if c.lower() == name.lower() and os.path.isdir(os.path.join(d, c)):
            return os.path.join(d, c)
    raise FileNotFoundError(f"no folder '{name}' in {d}: {sorted(os.listdir(d))}")


def find_root(raw, kind):
    """<raw>/<Images|dire>/test/celebahq (the DiffusionForensics layout)"""
    return child(child(child(raw, kind), "test"), "celebahq")


def images_by_stem(d):
    out = {}
    for dp, _, fs in os.walk(d, followlinks=True):
        for f in sorted(fs):
            if f.lower().endswith(EXT):
                out.setdefault(os.path.splitext(f)[0], os.path.join(dp, f))
    return out


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder that contains Images/ and dire/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps", required=True, help="where the computed DIRE maps go")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--sets", nargs="+", default=["dalle2", "midjourney"])
    ap.add_argument("--n", type=int, default=100, help="real and fake images per celebahq set")
    ap.add_argument("--conds", nargs="+", default=["clean", "jpeg_q65", "jpeg_q30", "blur_s3"])
    ap.add_argument("--dff_real", default="")
    ap.add_argument("--dff_fake", default="")
    ap.add_argument("--n_dff", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_root, dire_root = find_root(args.raw, "Images"), find_root(args.raw, "dire")
    real_img, real_dire = images_by_stem(child(img_root, "real")), images_by_stem(child(dire_root, "real"))
    real_pool = sorted(set(real_img) & set(real_dire))
    need = args.n * len(args.sets)
    if len(real_pool) < need:
        raise SystemExit(f"only {len(real_pool)} real images have an official DIRE map, need {need}: lower --n")
    reals = rng.sample(real_pool, need)
    jobs, counts = [], {}
    for k, s in enumerate(args.sets):
        fake_img, fake_dire = images_by_stem(child(img_root, s)), images_by_stem(child(dire_root, s))
        pool = sorted(set(fake_img) & set(fake_dire))
        fakes = rng.sample(pool, min(args.n, len(pool)))
        for cls, stems, imgs, dires in [("real", reals[k * args.n:(k + 1) * args.n], real_img, real_dire),
                                        ("fake", fakes, fake_img, fake_dire)]:
            for st in stems:
                link(imgs[st], os.path.join(args.out, "orig", s, cls, os.path.basename(imgs[st])))
                link(dires[st], os.path.join(args.out, "official", s, cls, os.path.basename(dires[st])))
                for c in args.conds:
                    jobs.append((imgs[st], c, os.path.join(args.maps, c, s, cls, st + ".png")))
            counts[f"{s}/{cls}"] = len(stems)
    if args.dff_real and args.dff_fake:
        for cls, d in [("real", args.dff_real), ("fake", args.dff_fake)]:
            files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(d, followlinks=True) for f in fs if f.lower().endswith(EXT))
            pick = rng.sample(files, min(args.n_dff, len(files)))
            for i, src in enumerate(pick):
                name = f"{i:04d}_{os.path.basename(src)}"  # unique: DFF uses the same names in every folder
                link(src, os.path.join(args.out, "orig", "dff", cls, name))
                jobs.append((src, "clean", os.path.join(args.maps, "clean", "dff", cls, os.path.splitext(name)[0] + ".png")))
            counts[f"dff/{cls}"] = len(pick)
    os.makedirs(os.path.dirname(os.path.abspath(args.jobs)), exist_ok=True)
    with open(args.jobs, "w") as f:
        f.writelines(f"{a}\t{b}\t{c}\n" for a, b, c in jobs)
    print("images:", counts)
    print(f"{len(jobs)} DIRE maps to compute ({', '.join(args.conds)} for celebahq; clean for dff) -> {args.jobs}")


if __name__ == "__main__":
    main()
