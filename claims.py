"""
Data-driven claim checker: reads runs/results_*.json (+ latency.json) and tells you which claims
your OWN numbers support. Use it to decide what to write in the abstract/conclusion.

  python claims.py --dir runs_v2 --main v2 --baseline v2_spatial --unseen dalle2 midjourney

Rule of thumb used: difference >= 1 point = SUPPORTED, within +-1 point = INCONCLUSIVE (single seed), else NOT SUPPORTED.
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs")
ap.add_argument("--margin", type=float, default=1.0, help="points needed to call a difference real")
ap.add_argument("--main", default="v2", help="the proposed run")
ap.add_argument("--others", nargs="*", default=None, help="runs to compare against (default: every other run)")
ap.add_argument("--baseline", default=None, help="baseline for the generalization-gap / robustness claims (default: spatial, or v2_spatial for v2)")
ap.add_argument("--unseen", nargs="*", default=None, help="test sets that count as unseen (default: every test set that is not a training generator, dev_ or x_)")
args = ap.parse_args()

runs = {}
for p in glob.glob(os.path.join(args.dir, "results_*.json")):
    r = json.load(open(p))
    if r["tag"] != "debug":
        runs[r["tag"]] = r
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def unseen(r):
    """test sets that count for the main claims: not a training generator, not a dev_ set, not an x_ extra (e.g. non-diffusion).
    With --unseen: exactly those sets (a run that lacks one of them is left out of the comparisons)."""
    if args.unseen:
        return {k: r["tests"][k] for k in args.unseen if k in r["tests"]} if all(k in r["tests"] for k in args.unseen) else {}
    seen = {norm(t) for t in r.get("train_sets", [])}
    d = {k: v for k, v in r["tests"].items() if norm(k) not in seen and not k.startswith(("dev_", "x_"))}
    return d or r["tests"]


def m(r, key):
    return 100 * mean([v[key] for v in unseen(r).values()])


def rob(r, pert):
    vals = [100 * v[pert] for t, v in r.get("robust", {}).items() if t in unseen(r) and pert in v]
    return mean(vals)


MAIN = args.main
BASE = args.baseline or "v2_spatial"
supported, inconclusive, rejected, notes = [], [], [], []


def verdict(text, delta):
    if delta != delta:  # nan
        notes.append(f"(skipped, missing data) {text}")
    elif delta >= args.margin:
        supported.append(f"{text}  [+{delta:.2f} pts]")
    elif delta > -args.margin:
        inconclusive.append(f"{text}  [{delta:+.2f} pts, within noise: run 2-3 seeds before claiming]")
    else:
        rejected.append(f"{text}  [{delta:+.2f} pts]")


if MAIN not in runs:
    raise SystemExit(f"main run '{MAIN}' not found in {args.dir}")
main = runs[MAIN]
LABELS = {"dire_resnet50": "DIRE baseline (ResNet-50 on DIRE images, base paper)",
          "v2_spatial": "frozen-CLIP spatial branch alone", "v2_freq": "residual-DCT frequency branch alone",
          "v2_nodeg": "same model without degradation augmentation", "v2_raw": "same model with raw-image DCT (no NPR residual)",
          "v2_nomd": "same model without modality dropout / auxiliary heads",
          "v2_effnet": "same model with a fine-tuned EfficientNet-B4 instead of the frozen CLIP"}
if args.others is not None:
    others = args.others
else:
    others = sorted(t for t in runs if t != MAIN)
for tag in others:
    label = LABELS.get(tag, f"'{tag}'")
    if tag in runs and unseen(runs[tag]) and unseen(main):
        b = runs[tag]
        verdict(f"Unseen-test AP: {MAIN} beats the {label}", m(main, "ap") - m(b, "ap"))
        verdict(f"Unseen-test AUC: {MAIN} beats the {label}", m(main, "auc") - m(b, "auc"))
        verdict(f"Unseen-test ACC@0.5: {MAIN} beats the {label}", m(main, "acc") - m(b, "acc"))
        if all("acc_devthr" in v for v in unseen(main).values()) and all("acc_devthr" in v for v in unseen(b).values()):
            verdict(f"Unseen-test ACC@dev-threshold (threshold picked on a different-generator dev set): {MAIN} beats the {label}",
                    m(main, "acc_devthr") - m(b, "acc_devthr"))
if not unseen(main):
    notes.append(f"{MAIN} lacks some of the --unseen test sets {args.unseen}: no comparisons made")
if BASE in runs and unseen(runs[BASE]) and unseen(main):
    b = runs[BASE]
    gap = lambda r: 100 * (r["val"]["auc"] - mean([v["auc"] for v in unseen(r).values()]))
    verdict(f"Generalization: smaller in-domain -> unseen AUC drop than {BASE}", gap(b) - gap(main))
    for pert, name in [("jpeg_q30", "JPEG q=30"), ("jpeg_q65", "JPEG q=65"), ("blur_s3", "Gaussian blur s=3")]:
        verdict(f"Robustness ({name}), unseen-test AP: {MAIN} beats {BASE}", rob(main, pert) - rob(b, pert))

if unseen(main) and all("acc_oracle" in v for v in unseen(main).values()):
    parts = [f"ACC@0.5 {m(main, 'acc'):.2f}"]
    if all("acc_devthr" in v for v in unseen(main).values()):
        parts.append(f"ACC@dev-threshold {m(main, 'acc_devthr'):.2f}")
    parts.append(f"ACC@oracle {m(main, 'acc_oracle'):.2f} (UPPER BOUND: uses test labels, never report it as the accuracy)")
    notes.append(f"{MAIN}, unseen mean: " + " | ".join(parts) + f" | AUC {m(main, 'auc'):.2f}")

lat = os.path.join(args.dir, "latency.json")
if os.path.exists(lat):
    L = json.load(open(lat))
    ours = L.get("fs_dualnet_v2_ms") if MAIN.startswith("v2") and L.get("fs_dualnet_v2_ms") else L["fs_dualnet_ms"]
    if L.get("dire_ms"):
        sp = L["dire_ms"] / ours
        (supported if sp > 1.5 else inconclusive).append(
            f"Efficiency: {MAIN} {ours:.1f} ms/img vs DIRE {L['dire_ms']:.1f} ms/img = {sp:.1f}x faster (same GPU)")
    else:
        notes.append(f"{MAIN} latency {ours:.1f} ms/img measured; add DIRE's time with "
                     "`benchmark_latency.py --dire_ms <ms/img>` to claim a speed-up.")
else:
    notes.append("No latency.json: run benchmark_latency.py --out <dir>/latency.json")

fmt = lambda xs: "\n".join(f"- {x}" for x in xs) or "- (none)"
uns = ", ".join(args.unseen) if args.unseen else "test sets not used for training (no dev_ / x_ sets)"
out = (f"# Claim check ({args.dir})\n\nMain run: {MAIN}. Unseen = {uns}.\n\n"
       f"## Supported in THIS run (one seed: confirm with aggregate_seeds.py over >= 3 seeds before writing it in the paper)\n{fmt(supported)}\n\n## Do NOT claim yet (inconclusive)\n{fmt(inconclusive)}\n\n"
       f"## Not supported by your data (report honestly, do not hide)\n{fmt(rejected)}\n\n## Notes\n{fmt(notes)}\n\n"
       "Always safe to claim (design facts): no diffusion reconstruction at inference; single forward pass; "
       "Grad-CAM visualisation of decision-relevant regions (fully synthetic faces have no region-level ground truth: do not call it forgery localisation).\n")
print(out)
open(os.path.join(args.dir, "claims.md"), "w").write(out)
