"""
Data-driven claim checker: reads runs/results_*.json (+ latency.json) and tells you which claims
your OWN numbers support. Use it to decide what to write in the abstract/conclusion.

  python claims.py --dir /content/drive/MyDrive/fsdualnet/runs_dff

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
args = ap.parse_args()

runs = {}
for p in glob.glob(os.path.join(args.dir, "results_*.json")):
    r = json.load(open(p))
    if r["tag"] != "debug":
        runs[r["tag"]] = r
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def unseen(r):
    seen = {norm(t) for t in r.get("train_sets", [])}
    d = {k: v for k, v in r["tests"].items() if norm(k) not in seen}
    return d or r["tests"]


def m(r, key):
    return 100 * mean([v[key] for v in unseen(r).values()])


def rob(r, pert):
    vals = [100 * v[pert] for t, v in r.get("robust", {}).items() if t in unseen(r) and pert in v]
    return mean(vals)


MAIN = "both_attn"
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
for tag, label in [("spatial", "plain RGB EfficientNet-B4 baseline (spatial-only)"), ("freq", "frequency-only branch"),
                   ("both_concat", "concatenation fusion (no attention)")]:
    if tag in runs:
        b = runs[tag]
        verdict(f"Unseen-test AP: FS-DualNet beats the {label}", m(main, "ap") - m(b, "ap"))
        verdict(f"Unseen-test AUC: FS-DualNet beats the {label}", m(main, "auc") - m(b, "auc"))
        verdict(f"Unseen-test ACC@0.5: FS-DualNet beats the {label}", m(main, "acc") - m(b, "acc"))
if "spatial" in runs:
    b = runs["spatial"]
    gap = lambda r: 100 * (r["val"]["auc"] - mean([v["auc"] for v in unseen(r).values()]))
    verdict("Generalization: smaller in-domain -> unseen AUC drop than the RGB baseline", gap(b) - gap(main))
    for pert, name in [("jpeg_q30", "JPEG q=30"), ("jpeg_q65", "JPEG q=65"), ("blur_s3", "Gaussian blur s=3")]:
        verdict(f"Robustness ({name}), unseen-test AP: FS-DualNet beats the RGB baseline", rob(main, pert) - rob(b, pert))

lat = os.path.join(args.dir, "latency.json")
if os.path.exists(lat):
    L = json.load(open(lat))
    if L.get("dire_ms"):
        sp = L["dire_ms"] / L["fs_dualnet_ms"]
        (supported if sp > 1.5 else inconclusive).append(
            f"Efficiency: FS-DualNet {L['fs_dualnet_ms']:.1f} ms/img vs DIRE {L['dire_ms']:.1f} ms/img = {sp:.1f}x faster (same GPU)")
    else:
        notes.append(f"FS-DualNet latency {L['fs_dualnet_ms']:.1f} ms/img measured; add DIRE's time with "
                     "`benchmark_latency.py --dire_ms <ms/img>` to claim a speed-up.")
else:
    notes.append("No latency.json: run benchmark_latency.py --out <dir>/latency.json")

fmt = lambda xs: "\n".join(f"- {x}" for x in xs) or "- (none)"
out = (f"# Claim check ({args.dir})\n\nMain run: {MAIN}. Unseen = test sets not used for training.\n\n"
       f"## You CAN claim\n{fmt(supported)}\n\n## Do NOT claim yet (inconclusive)\n{fmt(inconclusive)}\n\n"
       f"## Not supported by your data (report honestly, do not hide)\n{fmt(rejected)}\n\n## Notes\n{fmt(notes)}\n\n"
       "Always safe to claim (design facts): no diffusion reconstruction at inference; single forward pass; "
       "Grad-CAM localisation of forged regions.\n")
print(out)
open(os.path.join(args.dir, "claims.md"), "w").write(out)
