"""Collect runs/results_*.json into one markdown table (main result, ablation, cross-domain, robustness)."""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs")
args = ap.parse_args()

runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, "results_*.json")))]
if not runs:
    raise SystemExit(f"no results_*.json in {args.dir}")
pct = lambda v: f"{100 * v:.2f}"
tests = sorted({t for r in runs for t in r["tests"]})
lines = []

lines += ["## In-domain (validation) and cross-domain AUC (%)", "",
          "| run | val ACC | val AP | val AUC | " + " | ".join(f"{t} AUC" for t in tests) + " | mean cross AUC |",
          "|---|---|---|---|" + "---|" * (len(tests) + 1)]
for r in runs:
    cross = [r["tests"][t]["auc"] for t in tests if t in r["tests"]]
    mean = pct(sum(cross) / len(cross)) if cross else "-"
    cells = [pct(r["tests"][t]["auc"]) if t in r["tests"] else "-" for t in tests]
    lines.append(f"| {r['tag']} | {pct(r['val']['acc'])} | {pct(r['val']['ap'])} | {pct(r['val']['auc'])} | "
                 + " | ".join(cells) + f" | {mean} |")

lines += ["", "## Cross-domain ACC / AP (%)", "", "| run | " + " | ".join(f"{t} ACC/AP" for t in tests) + " |",
          "|---|" + "---|" * len(tests)]
for r in runs:
    cells = [f"{pct(r['tests'][t]['acc'])} / {pct(r['tests'][t]['ap'])}" if t in r["tests"] else "-" for t in tests]
    lines.append(f"| {r['tag']} | " + " | ".join(cells) + " |")

rob = [r for r in runs if r["robust"]]
if rob:
    perts = list(next(iter(rob[0]["robust"].values())).keys())
    lines += ["", "## Robustness: mean AP (%) over test sets", "", "| run | " + " | ".join(perts) + " |",
              "|---|" + "---|" * len(perts)]
    for r in rob:
        vals = [sum(r["robust"][t][p] for t in r["robust"]) / len(r["robust"]) for p in perts]
        lines.append(f"| {r['tag']} | " + " | ".join(pct(v) for v in vals) + " |")

out = "\n".join(lines)
print(out)
open(os.path.join(args.dir, "results.md"), "w").write(out + "\n")
