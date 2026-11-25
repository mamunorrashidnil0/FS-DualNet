"""
Evaluate a saved checkpoint (FS-DualNet v2 or the DIRE ResNet-50) on folders that contain real/ and fake/ sub-folders.

  python eval_ckpt.py --ckpt runs/v2.pt --sets dalle2=/data/dalle2 dff=/data/dff --perturb jpeg_q30 --out preds
  several evaluations with one model load:
  python eval_ckpt.py --ckpt runs/v2.pt --run "out/clean:none:dalle2=/d/a,dff=/d/b" --run "out/q30:jpeg_q30:dalle2=/d/a"

--perturb is applied after the 256 resize and before the center crop (same as train_eval.py --robust).
--png_to_jpeg defaults to what the checkpoint was trained with (format-fair DIRE models: 95).
Writes <out>.npz (per set: <name>__y, <name>__p) and <out>.json (AUC / AP / ACC@0.5 per set).
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from fs_dualnet import build_model, load_model_state
from train_eval import Blur, FaceDataset, Jpeg, build_tf, load_ckpt, metrics_from, parse_named, predict

PERTS = {"none": None, "blur_s1": Blur(1), "blur_s2": Blur(2), "blur_s3": Blur(3), "jpeg_q65": Jpeg(65), "jpeg_q30": Jpeg(30)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", default=[], help="name=folder (folder has real/ and fake/)")
    ap.add_argument("--perturb", choices=list(PERTS), default="none")
    ap.add_argument("--run", action="append", default=[], help="OUT:PERTURB:name=folder,name=folder (repeatable)")
    ap.add_argument("--png_to_jpeg", type=int, default=None, help="override the checkpoint's setting (0 = off)")
    ap.add_argument("--out", default="", help="output path without extension")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_ckpt(args.ckpt, device)
    model = build_model(ck).to(device)
    load_model_state(model, ck["model"])
    model.eval()
    p2j = ck.get("png_to_jpeg") if args.png_to_jpeg is None else args.png_to_jpeg
    runs = [(args.out, args.perturb, parse_named(args.sets))] if args.sets else []
    for r in args.run:
        out, pert, sets = r.split(":", 2)
        runs.append((out, pert, parse_named(sets.split(","))))
    if not runs or not all(o for o, _, _ in runs):
        raise SystemExit("give --sets with --out, or one or more --run OUT:PERTURB:name=folder,...")
    for out, pert, sets in runs:
        tf = build_tf(False, ck.get("resize", 256), ck.get("img_size", 224), perturb=PERTS[pert])
        arrays, summary = {}, {"ckpt": args.ckpt, "perturb": pert, "png_to_jpeg": p2j or 0, "sets": {}}
        for name, path in sets:
            ds = FaceDataset(path, tf, png_to_jpeg=p2j or None)
            dl = DataLoader(ds, args.bs, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            y, p = predict(model, dl, device, device.type == "cuda")
            arrays[f"{name}__y"], arrays[f"{name}__p"] = y.astype(np.int8), p.astype(np.float32)
            m = metrics_from(y, p)
            summary["sets"][name] = {k: m[k] for k in ("acc", "ap", "auc")} | {"n": int(len(y))}
            print(f"{os.path.basename(out):24s} {pert:8s} {name:12s} n={len(y):5d} | "
                  f"AUC {100 * m['auc']:6.2f} | AP {100 * m['ap']:6.2f} | ACC@0.5 {100 * m['acc']:6.2f}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(out + ".npz", **arrays)
        json.dump(summary, open(out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
