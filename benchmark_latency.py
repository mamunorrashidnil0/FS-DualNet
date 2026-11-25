"""
Combine several seeds (one results folder per seed) -> mean +- std, and a seed-consistency check for each comparison.
Only claim a difference when it is >= 1 point AND has the same sign in every seed.

  python aggregate_seeds.py --dirs runs_df runs_df_s1 runs_df_s2
"""
import argparse
import glob
import json
import os
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--main", default="v2")
ap.add_argument("--margin", type=float, default=1.0)
ap.add_argument("--out", default=None, help="where to write aggregate.md (default: first dir)")
ap.add_argument("--unseen", nargs="*", default=None, help="test sets that count as unseen (default: not a training generator, not dev_ / x_)")
args = ap.parse_args()

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
seeds = []  # list of {tag: result}
for d in args.dirs:
    runs = {}
    for p in glob.glob(os.path.join(d, "results_*.json")):
        r = json.load(open(p))
        if r["tag"] != "debug":
            runs[r["tag"]] = r
    seeds.append(runs)
tags = sorted(set.intersection(*[set(s) for s in seeds]))
if args.unseen:
    tags = [t for t in tags if all(all(k in s[t]["tests"] for k in args.unseen) for s in seeds)]
if not tags:
    raise SystemExit("no run tag exists in every folder" + (f" with test sets {args.unseen}" if args.unseen else ""))


def unseen(r):
    if args.unseen:
        if not all(k in r["tests"] for k in args.unseen):
            raise KeyError(f"run '{r['tag']}' has no test set {[k for k in args.unseen if k not in r['tests']]}")
        return {k: r["tests"][k] for k in args.unseen}
    seen = {norm(t) for t in r.get("train_sets", [])}
    d = {k: v for k, v in r["tests"].items() if norm(k) not in seen and not k.startswith(("dev_", "x_"))}
    return d or r["tests"]


def metric(r, key):  # mean over unseen test sets, in %
    v = [100 * m[key] for m in unseen(r).values()]
    return sum(v) / len(v)


def rob(r, pert):
    v = [100 * x[pert] for t, x in r.get("robust", {}).items() if t in unseen(r) and pert in x]
    return sum(v) / len(v) if v else float("nan")


def ms(xs):
    xs = [x for x in xs if x == x]  # drop nan (metric not recorded in that run, e.g. no robustness test)
    if not xs:
        return "-"
    return f"{sum(xs) / len(xs):.2f} +- {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"
n = len(seeds)
uns = ", ".join(args.unseen) if args.unseen else "test sets not used for training (no dev_ / x_ sets)"
L = [f"# Seed aggregate ({n} seeds: {', '.join(args.dirs)})", "", f"Unseen = {uns}. mean +- std over seeds.", "",
     "| run | unseen AUC | unseen AP | unseen ACC@0.5 | unseen ACC@dev-thr | JPEG q30 AP | blur s3 AP |", "|---|---|---|---|---|---|---|"]
for t in tags:
    L.append(f"| {t} | " + " | ".join(ms([f(s[t]) for s in seeds]) for f in (
        lambda r: metric(r, "auc"), lambda r: metric(r, "ap"), lambda r: metric(r, "acc"),
        lambda r: metric(r, "acc_devthr") if all("acc_devthr" in v for v in unseen(r).values()) else float("nan"),
        lambda r: rob(r, "jpeg_q30"), lambda r: rob(r, "blur_s3"))) + " |")

L += ["", "## Per test set AUC (mean +- std)", ""]
names = sorted({k for t in tags for k in seeds[0][t]["tests"]})
L += ["| run | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for t in tags:
    L.append(f"| {t} | " + " | ".join(ms([100 * s[t]["tests"][k]["auc"] for s in seeds]) if all(k in s[t]["tests"] for s in seeds) else "-" for k in names) + " |")

L += ["", f"## Paired comparisons: {args.main} minus other, per seed (unseen AUC / AP / ACC, points)", ""]
verdicts = []
if args.main in tags:
    for t in tags:
        if t == args.main:
            continue
        for key, label in (("auc", "AUC"), ("ap", "AP"), ("acc", "ACC@0.5"), ("acc_devthr", "ACC@dev-thr")):
            try:
                diffs = [metric(s[args.main], key) - metric(s[t], key) for s in seeds]
            except KeyError:
                continue  # metric not recorded in these runs
            pos = sum(d > 0 for d in diffs)
            mean = sum(diffs) / n
            if n >= 2 and all(d >= args.margin for d in diffs):
                v = "SUPPORTED (>= margin in every seed)"
            elif n >= 2 and all(d <= -args.margin for d in diffs):
                v = "OPPOSITE (other is better in every seed)"
            else:
                v = "inconclusive (small or mixed sign)"
            verdicts.append(f"- {args.main} vs {t}, {label}: per-seed {[round(d, 2) for d in diffs]} | mean {mean:+.2f} | positive in {pos}/{n} seeds -> {v}")
L += verdicts or ["(main run not found)"]
if n < 3:
    L += ["", f"NOTE: only {n} seed(s). Use at least 3 before claiming anything."]
out = "\n".join(L)
print(out)
open(os.path.join(args.out or args.dirs[0], "aggregate.md"), "w").write(out + "\n")
