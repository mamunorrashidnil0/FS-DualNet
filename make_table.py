"""Collect runs/results_*.json into one markdown table (main result, ablation, cross-domain, robustness)."""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs")
ap.add_argument("--unseen", nargs="*", default=None, help="test sets averaged in the 'mean unseen' and robustness columns (default: automatic)")
args = ap.parse_args()

runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, "results_*.json")))]
if not runs:
    raise SystemExit(f"no results_*.json in {args.dir}")
pct = lambda v: f"{100 * v:.2f}"
tests = sorted({t for r in runs for t in r["tests"]})
lines = []

lines += ["## In-domain (validation) and cross-domain AUC (%)", "",
          "| run | val ACC | val AP | val AUC | " + " | ".join(f"{t} AUC" for t in tests) + " | mean unseen AUC |",
          "|---|---|---|---|" + "---|" * (len(tests) + 1)]
for r in runs:
    seen = {re.sub(r"[^a-z0-9]", "", x.lower()) for x in r.get("train_sets", [])}
    cross = [r["tests"][t]["auc"] for t in tests if t in r["tests"] and (t in args.unseen if args.unseen else (
             not t.startswith(("dev_", "x_")) and re.sub(r"[^a-z0-9]", "", t.lower()) not in seen))]
    if args.unseen and len(cross) != len(args.unseen):
        cross = []
    mean = pct(sum(cross) / len(cross)) if cross else "-"
    cells = [pct(r["tests"][t]["auc"]) if t in r["tests"] else "-" for t in tests]
    lines.append(f"| {r['tag']} | {pct(r['val']['acc'])} | {pct(r['val']['ap'])} | {pct(r['val']['auc'])} | "
                 + " | ".join(cells) + f" | {mean} |")

lines += ["", "## Cross-domain ACC / AP (%)", "", "| run | " + " | ".join(f"{t} ACC/AP" for t in tests) + " |",
          "|---|" + "---|" * len(tests)]
for r in runs:
    cells = [f"{pct(r['tests'][t]['acc'])} / {pct(r['tests'][t]['ap'])}" if t in r["tests"] else "-" for t in tests]
    lines.append(f"| {r['tag']} | " + " | ".join(cells) + " |")

cal = [r for r in runs if all("acc_oracle" in m for m in r["tests"].values())]
if cal:
    lines += ["", "## ACC@0.5 vs ACC with a dev-set threshold vs oracle (UPPER BOUND, uses test labels, not a reportable accuracy)", "",
              "| run | " + " | ".join(f"{t} 0.5 / dev-thr / oracle" for t in tests) + " |", "|---|" + "---|" * len(tests)]
    for r in cal:
        cells = []
        for t in tests:
            m = r["tests"].get(t)
            cells.append("-" if m is None else f"{pct(m['acc'])} / {pct(m['acc_devthr']) if 'acc_devthr' in m else 'n/a'} / {pct(m['acc_oracle'])}")
        lines.append(f"| {r['tag']} | " + " | ".join(cells) + " |")
dev = [r for r in runs if r.get("dev")]
if dev:
    dn = sorted({k for r in dev for k in r["dev"]})
    lines += ["", "## DEV sets (different generator than training; use THESE, not the final test sets, to choose between configurations)", "",
              "| run | thr_dev | " + " | ".join(f"{k} ACC/AP/AUC" for k in dn) + " |", "|---|---|" + "---|" * len(dn)]
    for r in dev:
        lines.append(f"| {r['tag']} | {r.get('thr_dev', float('nan')):.3f} | " + " | ".join(
            f"{pct(r['dev'][k]['acc'])} / {pct(r['dev'][k]['ap'])} / {pct(r['dev'][k]['auc'])}" if k in r["dev"] else "-" for k in dn) + " |")

gate = [r for r in runs if any("gate_freq" in m for m in r["tests"].values())]
if gate:
    lines += ["", "## Attention gate: mean weight on the FREQUENCY branch per test set (0 = only spatial, 1 = only frequency)", "",
              "| run | " + " | ".join(tests) + " |", "|---|" + "---|" * len(tests)]
    for r in gate:
        lines.append(f"| {r['tag']} | " + " | ".join(f"{r['tests'][t]['gate_freq']:.2f}" if "gate_freq" in r["tests"].get(t, {}) else "-" for t in tests) + " |")

rob = [r for r in runs if r["robust"]]
if rob:
    perts = list(next(iter(rob[0]["robust"].values())).keys())
    which = f"the sets {', '.join(args.unseen)}" if args.unseen else "all robustness test sets"
    lines += ["", f"## Robustness: mean AP (%) over {which}", "", "| run | " + " | ".join(perts) + " |",
              "|---|" + "---|" * len(perts)]
    for r in rob:
        ts = [t for t in r["robust"] if not args.unseen or t in args.unseen]
        if not ts:
            continue
        vals = [sum(r["robust"][t][p] for t in ts) / len(ts) for p in perts]
        lines.append(f"| {r['tag']} | " + " | ".join(pct(v) for v in vals) + " |")

out = "\n".join(lines)
print(out)
open(os.path.join(args.dir, "results.md"), "w").write(out + "\n")
