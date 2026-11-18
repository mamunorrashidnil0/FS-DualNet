"""
Paper-style comparison with DIRE (Wang et al., ICCV 2023), Table 2 protocol:
  train on ADM (LSUN-Bedroom), test on every generator, report ACC / AP (%).

Reference rows are the numbers PRINTED in the DIRE paper (Table 2, arXiv v1) -- they are not re-run by us.
Our rows come from runs/results_*.json produced by train_eval.py on the LSUN-Bedroom subset.

  python compare_dire.py --dir /content/drive/MyDrive/fsdualnet/runs_lsun
"""
import argparse
import glob
import json
import os
import re

COLS = ["ADM", "DDPM", "iDDPM", "PNDM", "SD-v1", "SD-v2", "LDM", "VQ-Diffusion"]
# (ACC, AP) per generator, then the paper's printed 8-generator average
REF = {
    "CNNDet* (reproduced in DIRE paper)": ([(100, 100), (87.3, 99.6), (100, 100), (77.8, 99.1), (77.4, 85.8),
                                            (83.4, 98.2), (96.0, 99.9), (70.3, 96.1)], (86.5, 97.3)),
    "Patchfor* (reproduced in DIRE paper)": ([(100, 100), (72.9, 100), (100, 100), (96.6, 100), (63.2, 71.3),
                                              (97.2, 100), (97.3, 100), (100, 100)], (90.9, 96.4)),
    "F3Net* (reproduced in DIRE paper)": ([(94.3, 99.6), (93.7, 99.6), (94.8, 99.9), (94.1, 99.2), (86.1, 95.3),
                                           (83.6, 91.7), (92.5, 97.8), (93.4, 98.7)], (91.6, 97.7)),
    "DIRE (paper, ADM recon., ADM-trained)": ([(100, 100), (100, 100), (100, 100), (99.7, 100), (99.7, 100),
                                               (100, 100), (100, 100), (100, 100)], (99.9, 100)),
}
ALIAS = {"adm": "ADM", "ddpm": "DDPM", "iddpm": "iDDPM", "pndm": "PNDM", "sdv1": "SD-v1", "sd1": "SD-v1",
         "stablediffusionv1": "SD-v1", "sdv2": "SD-v2", "sd2": "SD-v2", "stablediffusionv2": "SD-v2", "ldm": "LDM",
         "vqdiffusion": "VQ-Diffusion", "vqdiff": "VQ-Diffusion", "vq": "VQ-Diffusion"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs")
    args = ap.parse_args()
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, "results_*.json")))]
    runs = [r for r in runs if r["tag"] != "debug"]
    if not runs:
        raise SystemExit(f"no results_*.json in {args.dir}")

    # map our test-set names to table columns
    col_of = {}
    for r in runs:
        for t in r["tests"]:
            if norm(t) in ALIAS:
                col_of[t] = ALIAS[norm(t)]
    matched_cols = [c for c in COLS if c in col_of.values()]
    extra = sorted({t for r in runs for t in r["tests"] if t not in col_of})

    f = lambda a, p: f"{a:.1f}/{p:.1f}"
    lines = ["## Comparison with DIRE (Table 2 protocol: train ADM, test on other generators), ACC/AP (%)", "",
             "Reference rows: numbers printed in the DIRE paper (not re-run). Our rows: this work, LSUN-Bedroom subset.", ""]
    head = "| Method | " + " | ".join(COLS) + " | Avg (paper, 8 gens) | Avg (matched cols only) |"
    lines += [head, "|---|" + "---|" * (len(COLS) + 2)]

    def avg(pairs):
        return (sum(a for a, _ in pairs) / len(pairs), sum(p for _, p in pairs) / len(pairs)) if pairs else None

    for name, (vals, paper_avg) in REF.items():
        m = avg([v for c, v in zip(COLS, vals) if c in matched_cols])
        lines.append(f"| {name} | " + " | ".join(f(*v) for v in vals) + f" | {f(*paper_avg)} | {f(*m) if m else '-'} |")
    for r in runs:
        cell, pairs = {}, []
        for t, m in r["tests"].items():
            if t in col_of:
                cell[col_of[t]] = (100 * m["acc"], 100 * m["ap"])
        for c in COLS:
            if c in cell:
                pairs.append(cell[c])
        a = avg(pairs)
        lines.append(f"| **FS-DualNet [{r['tag']}] (ours)** | " + " | ".join(f(*cell[c]) if c in cell else "-" for c in COLS)
                     + f" | - | {f(*a) if a else '-'} |")

    lines += ["", f"Matched generator columns in our results: {matched_cols or 'none'}. "
              "Compare 'Avg (matched cols only)' of every row; the paper average covers all 8 generators."]
    if extra:
        lines += ["", "## Generators in our data not in DIRE Table 2 (ACC/AP %)", "", "| run | " + " | ".join(extra) + " |",
                  "|---|" + "---|" * len(extra)]
        for r in runs:
            lines.append(f"| {r['tag']} | " + " | ".join(
                f(100 * r['tests'][t]['acc'], 100 * r['tests'][t]['ap']) if t in r['tests'] else "-" for t in extra) + " |")
    lines += ["", "Caveats to state in the paper: our training subset is smaller than DIRE's 40k/class; images are "
              "resized 256 -> crop 224; baselines marked * are the DIRE authors' reproductions on ADM-trained data."]
    out = "\n".join(lines)
    print(out)
    open(os.path.join(args.dir, "comparison_table2.md"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
