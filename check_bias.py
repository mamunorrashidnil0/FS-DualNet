"""
Shortcut / dataset-bias check: do REAL and FAKE images differ in file format, JPEG quantisation, resolution or bytes per pixel?
If they do, a detector (especially a FFT/DCT frequency branch, which sees JPEG block artefacts directly) can score high by
spotting the FORMAT instead of the diffusion artefacts, and then collapse on another dataset.

  python check_bias.py --roots /content/data/train/inpainting /content/data/test/dalle2 --n 300
  python check_bias.py --real D:/.../celebahq/real --fake D:/.../celebahq/sdv2 --n 300
"""
import argparse
import os
import random
import statistics as st
from collections import Counter

from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample(folder, n, rng):
    files = []
    for dp, _, fs in os.walk(folder, followlinks=True):
        files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in EXT]
    return rng.sample(files, min(n, len(files)))


def stats(files, where=""):
    if not files:
        raise SystemExit(f"ERROR: no images found in {where!r}. Is the folder empty / not extracted yet (.tar)? Check the path.")
    fmts, w, h, bpp, qsum = Counter(), [], [], [], []
    for f in files:
        with Image.open(f) as im:
            fmts[im.format] += 1
            w.append(im.width)
            h.append(im.height)
            q = getattr(im, "quantization", None)
            if q:
                qsum.append(sum(q[0]))
        bpp.append(os.path.getsize(f) * 8 / (w[-1] * h[-1]))
    n = len(files)
    return {"n": n, "fmt": {k: f"{100 * v / n:.0f}%" for k, v in fmts.items()}, "jpeg_share": fmts.get("JPEG", 0) / n,
            "w": st.median(w), "h": st.median(h), "pixels": st.median([a * b for a, b in zip(w, h)]),
            "bpp": st.median(bpp), "qsum": st.median(qsum) if qsum else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[], help="folders that contain real/ and fake/")
    ap.add_argument("--real", help="alternative: a real-images folder ...")
    ap.add_argument("--fake", help="... and a fake-images folder to compare")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n_warn = 0
    pairs = [(root, os.path.join(root, "real"), os.path.join(root, "fake")) for root in args.roots]
    if args.real and args.fake:
        pairs.append((f"{args.real}  vs  {args.fake}", args.real, args.fake))
    if not pairs:
        raise SystemExit("give --roots <dir with real/ and fake/> or --real <dir> --fake <dir>")
    for root, real_dir, fake_dir in pairs:
        r = stats(sample(real_dir, args.n, rng), real_dir)
        f = stats(sample(fake_dir, args.n, rng), fake_dir)
        print(f"\n== {root}")
        for name, s in [("real", r), ("fake", f)]:
            print(f"  {name}: n={s['n']} | formats {s['fmt']} | median size {int(s['w'])}x{int(s['h'])} | "
                  f"bits/pixel {s['bpp']:.2f} | JPEG q-table sum {s['qsum']}")
        warn = []
        if abs(r["jpeg_share"] - f["jpeg_share"]) > 0.2:
            warn.append(f"JPEG share differs: real {100 * r['jpeg_share']:.0f}% vs fake {100 * f['jpeg_share']:.0f}%")
        ratio = r["pixels"] / f["pixels"]
        if ratio > 1.5 or ratio < 1 / 1.5:
            warn.append(f"resolution differs: real {int(r['w'])}x{int(r['h'])} vs fake {int(f['w'])}x{int(f['h'])}")
        if r["bpp"] / f["bpp"] > 2 or r["bpp"] / f["bpp"] < 0.5:
            warn.append(f"bits/pixel differ x{r['bpp'] / f['bpp']:.1f} (different compression/content statistics)")
        for w_ in warn:
            print("  WARNING:", w_)
        if not warn:
            print("  OK: no obvious format/resolution shortcut between real and fake")
        n_warn += len(warn)
    if n_warn:
        print("\n=> Possible shortcut. Run the control experiment: train_eval.py --jpeg_all 90 (re-encodes EVERY image, real and fake, "
              "identically) and report both results. If accuracy drops a lot, the first result was format, not diffusion artefacts.")


if __name__ == "__main__":
    main()
