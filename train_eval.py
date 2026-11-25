"""
Train / evaluate FS-DualNet following the DIRE (Wang et al., ICCV 2023) experimental protocol.

Protocol taken from the base paper
  * images resized to 256x256 (bicubic); train: random crop 224 + h-flip (p=0.5); test: center crop 224
  * loss: binary cross-entropy (their Eq. 10)
  * metrics: ACC (threshold 0.5) and AP  -- we also report AUC
  * robustness: Gaussian blur sigma = 1,2,3 and JPEG quality = 65,30 applied to test images (their Sec. 4.4)

--arch v2 (default): FS-DualNet (NPR-residual patch DCT + frozen CLIP ViT + patch cross-attention + MIL);
           use it with --degrade (label-independent rescale/blur/JPEG training augmentation)
--branch resnet50: the DIRE baseline classifier (ResNet-50), trained on DIRE maps

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
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms as T

from fs_dualnet import DIREClassifier, FSDualNetV2, load_model_state, model_state
from metrics_utils import acc_at, best_threshold

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_ckpt(path, device):
    """Our own checkpoints: PyTorch >= 2.6 defaults to weights_only=True, which rejects them."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # very old PyTorch without the argument
        return torch.load(path, map_location=device)


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


_RS = getattr(Image, "Resampling", Image)  # Pillow >= 9.1 moved the resampling constants


class RandomDegrade:
    """v2 degradation-invariant training: the SAME random post-processing for real AND fake images (label-independent),
    so file format, JPEG quality and resize history (shortcuts that can separate real from fake) stop separating the classes.
    Order follows a typical sharing pipeline: rescale -> blur -> JPEG."""

    def __init__(self, p_rescale=0.3, p_blur=0.3, p_jpeg=0.7, scale=(0.5, 1.0), sigma=(0.1, 3.0), quality=(30, 100)):
        self.p_rescale, self.p_blur, self.p_jpeg = p_rescale, p_blur, p_jpeg
        self.scale, self.sigma, self.quality = scale, sigma, quality

    def __call__(self, img):
        if random.random() < self.p_rescale:
            w, h = img.size
            s = random.uniform(*self.scale)
            down = random.choice([_RS.BILINEAR, _RS.BICUBIC, _RS.LANCZOS])
            img = img.resize((max(8, round(w * s)), max(8, round(h * s))), down)
            img = img.resize((w, h), random.choice([_RS.BILINEAR, _RS.BICUBIC]))
        if random.random() < self.p_blur:
            img = Blur(random.uniform(*self.sigma))(img)
        if random.random() < self.p_jpeg:
            img = Jpeg(random.randint(*self.quality))(img)
        return img


def build_tf(train, resize=256, crop=224, perturb=None, aug=False, jpeg_all=None, degrade=False):
    ops = [T.Resize((resize, resize), interpolation=T.InterpolationMode.BICUBIC)]
    if jpeg_all:  # control experiment: identical JPEG re-encoding of EVERY image (real and fake) removes format shortcuts
        ops.append(Jpeg(jpeg_all))
    if train:
        if aug:
            ops.append(RandomBlurJpeg())
        if degrade:
            ops.append(RandomDegrade())
        ops += [T.RandomCrop(crop), T.RandomHorizontalFlip(0.5)]
    else:
        if perturb is not None:
            ops.append(perturb)
        ops.append(T.CenterCrop(crop))
    ops.append(T.ToTensor())  # model normalises internally
    return T.Compose(ops)


# ----------------------------- data ----------------------------- #
class FaceDataset(Dataset):
    """real -> 0, fake -> 1.
    png_to_jpeg=Q: every PNG file is re-saved as JPEG quality Q when loaded (rule depends on the file format only,
    not on the label). Used for DIRE maps: the official real maps are JPEG q95 and the fake maps PNG."""

    def __init__(self, root, transform, max_per_class=None, seed=0, png_to_jpeg=None):
        self.transform, self.png_to_jpeg = transform, png_to_jpeg
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
        img = Image.open(path).convert("RGB")
        if self.png_to_jpeg and path.lower().endswith(".png"):
            img = Jpeg(self.png_to_jpeg)(img)
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def parse_named(items):
    out = []
    for it in items or []:
        name, _, path = it.partition("=")
        out.append((name, path) if path else (os.path.basename(it.rstrip("/")), it))
    return out


# ----------------------------- eval ----------------------------- #
GATE = {}  # mean attention-gate weight on the frequency branch for the last predict() call (None if no attention fusion)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ys, ps, gw = [], [], []
    n_fp32 = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logit, aux = model(x, return_aux=True)
        if not torch.isfinite(logit).all():  # fp16 overflow -> redo this batch in full precision
            n_fp32 += 1
            logit, aux = model(x.float(), return_aux=True)
        ps.append(torch.sigmoid(logit.float()).cpu())
        ys.append(y)
        if aux.get("branch_weights") is not None:
            gw.append(aux["branch_weights"][:, 0].float().cpu())
    GATE["freq"] = float(torch.cat(gw).mean()) if gw else None
    if n_fp32:
        print(f"  note: {n_fp32} batch(es) overflowed in fp16 and were recomputed in fp32")
    p = torch.cat(ps).numpy()
    if not np.isfinite(p).all():
        raise SystemExit(f"model output is NaN/inf for {int((~np.isfinite(p)).sum())} images even in fp32 (training diverged)")
    return torch.cat(ys).numpy(), p


def metrics_from(y, p):
    return {
        "acc": float(accuracy_score(y, p >= 0.5)),  # threshold 0.5, as in DIRE (plain floats: safe to pickle / json)
        "ap": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def evaluate(model, loader, device, amp):
    return metrics_from(*predict(model, loader, device, amp))


def bce_from(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fmt(m):
    return f"ACC {100 * m['acc']:.2f} | AP {100 * m['ap']:.2f} | AUC {100 * m['auc']:.2f}"


TIME_UP = 3  # exit code: time budget (--deadline) reached; checkpoints are saved, run the same command again to resume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--dev", nargs="*", default=[], help="DEV sets (name=path) from a generator NOT used for training: used only to pick the decision threshold, never for training/early stopping")
    ap.add_argument("--arch", choices=["v2"], default="v2", help="FS-DualNet v2 (--branch resnet50 trains the DIRE baseline instead)")
    ap.add_argument("--branch", choices=["both", "freq", "spatial", "resnet50"], default="both")
    ap.add_argument("--spatial_backbone", choices=["clip", "effnet"], default="clip", help="v2: frozen CLIP ViT or fine-tuned EfficientNet-B4")
    ap.add_argument("--clip_name", default="openai/clip-vit-base-patch16", help="v2: Hugging Face CLIP model ('tiny' = offline test model)")
    ap.add_argument("--freq_input", choices=["residual", "raw"], default="residual", help="v2: DCT of the NPR residual or of the raw image")
    ap.add_argument("--moddrop", type=float, default=0.3, help="v2: modality-dropout probability (0 = off)")
    ap.add_argument("--aux_weight", type=float, default=0.3, help="v2: weight of the frequency-only / spatial-only auxiliary losses (0 = off)")
    ap.add_argument("--degrade", action="store_true", help="label-independent rescale/blur/JPEG training augmentation (part of v2)")
    ap.add_argument("--warmup", type=float, default=0.03, help="v2: linear warm-up, fraction of all training steps")
    ap.add_argument("--clip_grad", type=float, default=None, help="gradient-norm clipping (default: 1.0 for v2, off for the DIRE baseline)")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: stop cleanly (exit code 3, resumable) before this time")
    ap.add_argument("--tag", default=None, help="run name; default v2 / v2_<branch> / resnet50")
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
    ap.add_argument("--png_to_jpeg", type=int, default=0, help="re-save every PNG input as JPEG at this quality (format-fair DIRE maps: 95)")
    ap.add_argument("--no_resume", action="store_true", help="ignore last_<tag>.pt and start from scratch")
    ap.add_argument("--select", choices=["loss", "auc", "last"], default="loss",
                    help="checkpoint selection on the VALIDATION set: lowest val BCE loss (default; does not saturate), "
                         "highest val AUC (saturates at 100%% on in-domain data -> picks an arbitrary early epoch), or last epoch")
    ap.add_argument("--no_pretrained", action="store_true", help="random init of EfficientNet (only for offline code tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.deadline and time.time() > args.deadline - 300:
        print("time budget reached before this run started: run the same command again in a new session")
        sys.exit(TIME_UP)
    v2 = args.branch != "resnet50"  # resnet50 = the DIRE baseline classifier
    clip_grad = args.clip_grad if args.clip_grad is not None else (1.0 if v2 else 0.0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if v2:
        tag = args.tag or ("v2" if args.branch == "both" else f"v2_{args.branch}")
    else:
        tag = args.tag or args.branch
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pt")
    res_path = os.path.join(args.out_dir, f"results_{tag}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    S, R, K = args.img_size, args.resize, args.max_per_class

    mk = lambda ds, shuffle: DataLoader(ds, args.bs, shuffle=shuffle, num_workers=args.workers,
                                        pin_memory=device.type == "cuda", drop_last=shuffle)
    train_tf = build_tf(True, R, S, aug=args.aug, jpeg_all=args.jpeg_all, degrade=args.degrade)
    P2J = args.png_to_jpeg or None
    train_dl = mk(ConcatDataset([FaceDataset(p, train_tf, K, png_to_jpeg=P2J) for _, p in parse_named(args.train)]), True)
    val_dl = mk(FaceDataset(args.val, build_tf(False, R, S, jpeg_all=args.jpeg_all), K, png_to_jpeg=P2J), False)
    tests = parse_named(args.test)
    test_ds = lambda path, **kw: mk(FaceDataset(path, build_tf(False, R, S, jpeg_all=args.jpeg_all, **kw), K, png_to_jpeg=P2J), False)

    if v2:
        model = FSDualNetV2(img_size=S, pretrained=not args.no_pretrained, branch=args.branch,
                            spatial_backbone=args.spatial_backbone, freq_input=args.freq_input, moddrop=args.moddrop,
                            aux=args.aux_weight > 0, clip_name=args.clip_name).to(device)
        bb = model.backbone_parameters()  # fine-tuned EfficientNet (ablation) only; CLIP is frozen
    else:
        model = DIREClassifier(img_size=S, pretrained=not args.no_pretrained).to(device)
        bb = list(model.spatial.parameters())
    bb_ids = {id(p) for p in bb}
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in bb_ids], "lr": args.lr}]
    if bb:
        groups.append({"params": bb, "lr": args.lr * args.backbone_lr_mult})
    trainable = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    total_steps = args.epochs * len(train_dl)
    if v2:  # linear warm-up, then cosine (transformer layers trained from scratch)
        warm = max(1, int(args.warmup * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else
                                                  0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total_steps - warm))))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    crit = nn.BCEWithLogitsLoss()  # BCE, DIRE Eq. (10)
    desc = (f"arch=v2 branch={args.branch} spatial={args.spatial_backbone} freq_input={args.freq_input} moddrop={model.cfg['moddrop']} "
            f"aux={args.aux_weight if model.cfg['aux'] else 0} degrade={args.degrade}") if v2 else "DIRE classifier (ResNet-50)"
    print(f"run '{tag}': {desc} | trainable params {sum(p.numel() for p in trainable) / 1e6:.2f}M "
          f"| train imgs={len(train_dl.dataset)} | device={device}")

    # ---- resume: a disconnect (Colab / laptop sleep) costs at most one epoch ----
    last_path = os.path.join(args.out_dir, f"last_{tag}.pt")
    better = {"loss": lambda new, old: new < old, "auc": lambda new, old: new > old, "last": lambda new, old: True}[args.select]
    start_ep, best, best_ep, elapsed_s = 1, (float("inf") if args.select == "loss" else -1.0), 0, 0.0
    if os.path.exists(last_path) and not args.no_resume:
        ck = load_ckpt(last_path, device)
        if ck.get("epochs") == args.epochs and ck.get("select", "auc") == args.select and ck.get("cfg", model.config()) == model.config():
            load_model_state(model, ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            start_ep, best, best_ep, elapsed_s = ck["epoch"] + 1, ck["best"], ck.get("best_ep", 0), ck["elapsed_s"]
            print(f"RESUMING '{tag}' after epoch {ck['epoch']} of {args.epochs} (best epoch so far {best_ep}, select={args.select})")
        else:
            print(f"found {last_path} made with epochs={ck.get('epochs')}, select={ck.get('select', 'auc')}, model {ck.get('cfg')} "
                  f"(now {args.epochs}, {args.select}, {model.config()}): starting from scratch")
    t0 = time.time() - elapsed_s
    t_session = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        run = 0.0
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if v2:
                    logit, aux = model(x, return_aux=True)
                    main_loss = crit(logit.float(), y)
                    # auxiliary single-domain heads: each branch must be discriminative on its own (training only)
                    loss = main_loss + args.aux_weight * sum(crit(a.float(), y) for a in aux.get("aux_logits", []))
                else:
                    loss = main_loss = crit(model(x).float(), y)
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, clip_grad)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += main_loss.item()
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{len(train_dl)} loss {run / i:.4f}")
        vy, vp = predict(model, val_dl, device, amp)
        m = metrics_from(vy, vp)
        m["loss"] = bce_from(vy, vp)
        score = m["loss"] if args.select == "loss" else m["auc"]
        if better(score, best):
            best, best_ep = score, ep
            torch.save({"model": model_state(model), **model.config(), "resize": R, "epoch": ep, "select": args.select,
                        "val_score": float(score), "png_to_jpeg": args.png_to_jpeg}, ckpt_path)
        tmp = last_path + ".tmp"
        torch.save({"model": model_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "best_ep": best_ep, "select": args.select,
                    "epochs": args.epochs, "elapsed_s": time.time() - t0, "cfg": model.config()}, tmp)
        os.replace(tmp, last_path)  # atomic: a crash while saving never corrupts the previous checkpoint
        # printed only AFTER both checkpoints are on disk: seeing this line means epoch {ep} survives a disconnect
        print(f"== epoch {ep} ({(time.time() - t0) / 60:.1f} min): train loss {run / len(train_dl):.4f} | val {fmt(m)} "
              f"| val loss {m['loss']:.4f}{' | saved as best' if best_ep == ep else ''}", flush=True)
        per_ep = (time.time() - t_session) / (ep - start_ep + 1)
        if args.deadline and ep < args.epochs and time.time() + 1.5 * per_ep > args.deadline:
            print(f"time budget: epoch {ep} saved, the next epoch would not finish in time -> stopping. "
                  "Run the same command again (new session) to resume.", flush=True)
            sys.exit(TIME_UP)

    if args.deadline and time.time() > args.deadline:
        print("time budget reached after training: run the same command again to do the evaluation", flush=True)
        sys.exit(TIME_UP)
    load_model_state(model, load_ckpt(ckpt_path, device)["model"])
    preds = {}  # per-image (label, P(fake)) of every evaluated set -> preds_<tag>.npz, for later analysis without re-running
    vy, vp = predict(model, val_dl, device, amp)
    preds["val"] = (vy, vp)
    in_dom = metrics_from(vy, vp)
    print(f"\n[in-domain val] {fmt(in_dom)} | selected epoch {best_ep} of {args.epochs} (select={args.select})")
    result = {"tag": tag, "arch": "v2" if v2 else "dire", "branch": args.branch, "fusion": "patch" if v2 else None,
              "epochs": args.epochs, "train_sets": [n for n, _ in parse_named(args.train)], "jpeg_all": args.jpeg_all, "png_to_jpeg": args.png_to_jpeg,
              "select": args.select, "selected_epoch": best_ep, "seed": args.seed, "aug": bool(args.aug),
              "degrade": bool(args.degrade), "model_cfg": model.config(),
              "train_minutes": round((time.time() - t0) / 60, 1), "val": in_dom, "tests": {}, "robust": {}}

    # ---- DEV sets (different generator than training): pick ONE decision threshold, no test labels involved ----
    thr_dev = None
    if args.dev:
        dev_y, dev_p = [], []
        result["dev"] = {}
        for name, path in parse_named(args.dev):
            y, p = predict(model, test_ds(path), device, amp)
            preds[f"dev__{name}"] = (y, p)
            result["dev"][name] = metrics_from(y, p)
            print(f"[dev] {name}: {fmt(result['dev'][name])}")
            dev_y.append(y)
            dev_p.append(p)
        thr_dev, bal = best_threshold(np.concatenate(dev_y), np.concatenate(dev_p), balanced=True)
        result["thr_dev"] = float(thr_dev)
        print(f"[dev] decision threshold chosen on the dev sets: {thr_dev:.3f} (balanced acc there {100 * bal:.2f})")

    for name, path in tests:
        y, p = predict(model, test_ds(path), device, amp)
        preds[f"test__{name}"] = (y, p)
        m = metrics_from(y, p)
        m["auc_drop"] = in_dom["auc"] - m["auc"]
        if GATE.get("freq") is not None:
            m["gate_freq"] = GATE["freq"]  # how much the attention fusion relies on the frequency branch (0..1)
        # UPPER BOUND only: uses the test labels to pick the threshold. Never report it as the accuracy.
        m["thr_oracle"], m["acc_oracle"] = best_threshold(y, p, balanced=False)
        extra = f" | ACC@oracle(upper bound) {100 * m['acc_oracle']:.2f}"
        if thr_dev is not None:
            m["acc_devthr"] = acc_at(y, p, thr_dev)
            extra = f" | ACC@dev-thr {100 * m['acc_devthr']:.2f}" + extra
        if "gate_freq" in m:
            extra += f" | gate on freq {m['gate_freq']:.2f}"
        result["tests"][name] = m
        print(f"[cross-domain] {name}: {fmt(m)}{extra} | AUC drop {100 * m['auc_drop']:+.2f}")

    if args.robust:
        perts = [("blur_s1", Blur(1)), ("blur_s2", Blur(2)), ("blur_s3", Blur(3)),
                 ("jpeg_q65", Jpeg(65)), ("jpeg_q30", Jpeg(30))]
        print("\n[robustness] AP (%) -- same metric as DIRE Fig. 4")
        for name, path in tests:
            result["robust"][name] = {}
            for pname, pert in perts:
                y, p = predict(model, test_ds(path, perturb=pert), device, amp)
                preds[f"robust__{name}__{pname}"] = (y, p)
                result["robust"][name][pname] = metrics_from(y, p)["ap"]
            print(f"  {name}: " + " | ".join(f"{k}: {100 * v:.1f}" for k, v in result["robust"][name].items()))

    np.savez_compressed(os.path.join(args.out_dir, f"preds_{tag}.npz"),
                        **{f"{k}__y": v[0].astype(np.int8) for k, v in preds.items()},
                        **{f"{k}__p": v[1].astype(np.float32) for k, v in preds.items()})
    with open(res_path, "w") as f:  # written LAST: its existence marks the run as finished (run_all.py skips it)
        json.dump(result, f, indent=2)
    print(f"\nsaved {ckpt_path}, {res_path} and preds_{tag}.npz")


if __name__ == "__main__":
    main()
