"""
Train / evaluate FS-DualNet following the DIRE (Wang et al., ICCV 2023) experimental protocol.

Protocol taken from the base paper
  * images resized to 256x256 (bicubic); train: random crop 224 + h-flip (p=0.5); test: center crop 224
  * loss: binary cross-entropy (their Eq. 10)
  * metrics: ACC (threshold 0.5) and AP  -- we also report AUC (used in the mid-defence slides)
  * robustness: Gaussian blur sigma = 1,2,3 and JPEG quality = 65,30 applied to test images (their Sec. 4.4)

Each dataset root must look like:   <root>/real/**.png   and   <root>/fake/**.png

Example:
  python train_eval.py \
      --train DiFF=/data/DiFF DeepFakeFace=/data/DFF \
      --val   /data/val_split \
      --test  FFpp=/data/FFpp DiffusionForensics=/data/DiffForensics \
      --epochs 10 --bs 32 --robust
"""
import argparse
import io
import json
import os
import random
import time

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms as T

from fs_dualnet import FSDualNet

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------- perturbations (DIRE Sec. 4.4) ----------------------------- #
class Blur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))


class Jpeg:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlurJpeg:
    """Optional training augmentation (off by default): random blur and/or JPEG with prob p each."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = Blur(random.uniform(0.0, 3.0))(img)
        if random.random() < self.p:
            img = Jpeg(random.randint(30, 95))(img)
        return img


def build_tf(train, resize=256, crop=224, perturb=None, aug=False, jpeg_all=None):
    ops = [T.Resize((resize, resize), interpolation=T.InterpolationMode.BICUBIC)]
    if jpeg_all:  # control experiment: identical JPEG re-encoding of EVERY image (real and fake) removes format shortcuts
        ops.append(Jpeg(jpeg_all))
    if train:
        if aug:
            ops.append(RandomBlurJpeg())
        ops += [T.RandomCrop(crop), T.RandomHorizontalFlip(0.5)]
    else:
        if perturb is not None:
            ops.append(perturb)
        ops.append(T.CenterCrop(crop))
    ops.append(T.ToTensor())  # model normalises internally
    return T.Compose(ops)


# ----------------------------- data ----------------------------- #
class FaceDataset(Dataset):
    """real -> 0, fake -> 1"""

    def __init__(self, root, transform, max_per_class=None, seed=0):
        self.transform = transform
        self.items = []
        rng = random.Random(seed)
        for label, name in enumerate(["real", "fake"]):
            paths = []
            for dp, _, files in os.walk(os.path.join(root, name), followlinks=True):
                for f in files:
                    if os.path.splitext(f)[1].lower() in EXT:
                        paths.append(os.path.join(dp, f))
            paths.sort()
            if max_per_class and len(paths) > max_per_class:
                paths = rng.sample(paths, max_per_class)
            self.items += [(p, label) for p in paths]
        if not self.items:
            raise RuntimeError(f"no images found under {root}/(real|fake)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), torch.tensor(label, dtype=torch.float32)


def parse_named(items):
    out = []
    for it in items or []:
        name, _, path = it.partition("=")
        out.append((name, path) if path else (os.path.basename(it.rstrip("/")), it))
    return out


# ----------------------------- eval ----------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logit = model(x.to(device, non_blocking=True))
        ps.append(torch.sigmoid(logit.float()).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {
        "acc": accuracy_score(y, p >= 0.5),  # threshold 0.5, as in DIRE
        "ap": average_precision_score(y, p),
        "auc": roc_auc_score(y, p),
    }


def fmt(m):
    return f"ACC {100 * m['acc']:.2f} | AP {100 * m['ap']:.2f} | AUC {100 * m['auc']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--branch", choices=["both", "freq", "spatial"], default="both")
    ap.add_argument("--fusion", choices=["attn", "concat"], default="attn")
    ap.add_argument("--tag", default=None, help="run name; default <branch>_<fusion>")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--max_per_class", type=int, default=None, help="subset size per class (quick debug runs)")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=224, help="crop size fed to the network")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone_lr_mult", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--aug", action="store_true", help="random blur+JPEG training augmentation (extra, not in DIRE)")
    ap.add_argument("--robust", action="store_true", help="blur sigma 1,2,3 / JPEG q 65,30 test (DIRE Sec. 4.4)")
    ap.add_argument("--jpeg_all", type=int, default=0, help="CONTROL: re-encode every image as JPEG at this quality (e.g. 90)")
    ap.add_argument("--no_resume", action="store_true", help="ignore last_<tag>.pt and start from scratch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tag = args.tag or (f"{args.branch}_{args.fusion}" if args.branch == "both" else args.branch)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pt")
    res_path = os.path.join(args.out_dir, f"results_{tag}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    S, R, K = args.img_size, args.resize, args.max_per_class

    mk = lambda ds, shuffle: DataLoader(ds, args.bs, shuffle=shuffle, num_workers=args.workers,
                                        pin_memory=device.type == "cuda", drop_last=shuffle)
    train_tf = build_tf(True, R, S, aug=args.aug, jpeg_all=args.jpeg_all)
    train_dl = mk(ConcatDataset([FaceDataset(p, train_tf, K) for _, p in parse_named(args.train)]), True)
    val_dl = mk(FaceDataset(args.val, build_tf(False, R, S, jpeg_all=args.jpeg_all), K), False)
    tests = parse_named(args.test)
    test_ds = lambda path, **kw: mk(FaceDataset(path, build_tf(False, R, S, jpeg_all=args.jpeg_all, **kw), K), False)

    model = FSDualNet(img_size=S, pretrained=True, branch=args.branch, fusion=args.fusion).to(device)
    bb = list(model.spatial.parameters()) if model.spatial is not None else []
    bb_ids = {id(p) for p in bb}
    groups = [{"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": args.lr}]
    if bb:
        groups.append({"params": bb, "lr": args.lr * args.backbone_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    crit = nn.BCEWithLogitsLoss()  # BCE, DIRE Eq. (10)
    print(f"run '{tag}': branch={args.branch} fusion={args.fusion} | train imgs={len(train_dl.dataset)} | device={device}")

    # ---- resume: a disconnect (Colab / laptop sleep) costs at most one epoch ----
    last_path = os.path.join(args.out_dir, f"last_{tag}.pt")
    start_ep, best, elapsed_s = 1, 0.0, 0.0
    if os.path.exists(last_path) and not args.no_resume:
        ck = torch.load(last_path, map_location=device)
        if ck.get("epochs") == args.epochs:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            start_ep, best, elapsed_s = ck["epoch"] + 1, ck["best"], ck["elapsed_s"]
            print(f"RESUMING '{tag}' after epoch {ck['epoch']} of {args.epochs} (best val AUC so far {best:.4f})")
        else:
            print(f"found {last_path} but it was made with epochs={ck.get('epochs')} != {args.epochs}: starting from scratch")
    t0 = time.time() - elapsed_s
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        run = 0.0
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{len(train_dl)} loss {run / i:.4f}")
        m = evaluate(model, val_dl, device, amp)
        print(f"== epoch {ep} ({(time.time() - t0) / 60:.1f} min): train loss {run / len(train_dl):.4f} | val {fmt(m)}")
        if m["auc"] > best:
            best = m["auc"]
            torch.save({"model": model.state_dict(), **model.config(), "resize": R, "val_auc": best}, ckpt_path)
        tmp = last_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "epochs": args.epochs,
                    "elapsed_s": time.time() - t0}, tmp)
        os.replace(tmp, last_path)  # atomic: a crash while saving never corrupts the previous checkpoint

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    in_dom = evaluate(model, val_dl, device, amp)
    print(f"\n[in-domain val] {fmt(in_dom)}")
    result = {"tag": tag, "branch": args.branch, "fusion": args.fusion, "epochs": args.epochs,
              "train_sets": [n for n, _ in parse_named(args.train)], "jpeg_all": args.jpeg_all,
              "train_minutes": round((time.time() - t0) / 60, 1), "val": in_dom, "tests": {}, "robust": {}}

    for name, path in tests:
        m = evaluate(model, test_ds(path), device, amp)
        m["auc_drop"] = in_dom["auc"] - m["auc"]
        result["tests"][name] = m
        print(f"[cross-domain] {name}: {fmt(m)} | AUC drop {100 * m['auc_drop']:+.2f}")

    if args.robust:
        perts = [("blur_s1", Blur(1)), ("blur_s2", Blur(2)), ("blur_s3", Blur(3)),
                 ("jpeg_q65", Jpeg(65)), ("jpeg_q30", Jpeg(30))]
        print("\n[robustness] AP (%) -- same metric as DIRE Fig. 4")
        for name, path in tests:
            result["robust"][name] = {}
            for pname, pert in perts:
                m = evaluate(model, test_ds(path, perturb=pert), device, amp)
                result["robust"][name][pname] = m["ap"]
            print(f"  {name}: " + " | ".join(f"{k}: {100 * v:.1f}" for k, v in result["robust"][name].items()))

    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {ckpt_path} and {res_path}")


if __name__ == "__main__":
    main()
