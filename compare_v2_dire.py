"""
One table: FS-DualNet v2 vs DIRE under the same conditions (3 seeds each).

  python compare_v2_dire.py --v2_dir <v2 notebook>/v2_results --res <this notebook>/vsdire_results

Reads  v2_dir/cb_s<seed>/results_v2.json                  (v2 on the full test sets, its dev threshold)
       res/dire_off_s<seed>/results_dire_resnet50.json     (DIRE classifier trained like the paper, official maps)
       res/dire_fair_s<seed>/results_dire_fair.json        (same, every PNG map re-saved as JPEG q95 = format-fair)
       res/evals/<model>_s<seed>_<cond>.npz                (eval_ckpt.py outputs on the shared image subset)
       res/dire_speed.json, v2_dir/latency.json, res/dire_validate.json
Writes res/vs_dire.md.  A difference counts only if it is >= 1 point in EVERY seed (same rule as claims.py).
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = (42, 1, 2)
UNSEEN = ("dalle2", "midjourney")
MODELS = [("v2", "FS-DualNet v2"), ("dire_off", "DIRE, trained as published"), ("dire_fair", "DIRE, format-fair training")]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def results(args, model, seed):
    if model == "v2":
        return jload(os.path.join(args.v2_dir, f"cb_s{seed}", "results_v2.json"))
    tag = "dire_resnet50" if model == "dire_off" else "dire_fair"
    return jload(os.path.join(args.res, f"{model}_s{seed}", f"results_{tag}.json"))


def subset_scores(args, model, seed, cond, sets):
    """mean AUC, AP, ACC@dev-threshold (%) over `sets` of one eval file; None if missing"""
    p = os.path.join(args.res, "evals", f"{model}_s{seed}_{cond}.npz")
    r = results(args, model, seed)
    if not os.path.exists(p) or r is None:
        return None
    d, thr = np.load(p), r.get("thr_dev", 0.5)
    out = []
    for s in sets:
        if f"{s}__y" not in d.files:
            return None
        y, pr = d[f"{s}__y"], d[f"{s}__p"]
        out.append((100 * roc_auc_score(y, pr), 100 * average_precision_score(y, pr), 100 * np.mean((pr >= thr) == y)))
    return tuple(np.mean(out, axis=0))


def full_scores(args, model, seed):
    r = results(args, model, seed)
    if r is None or not all(s in r["tests"] for s in UNSEEN):
        return None
    t = [r["tests"][s] for s in UNSEEN]
    return tuple(100 * np.mean([x.get(k, np.nan) for x in t]) for k in ("auc", "ap", "acc", "acc_devthr"))


def ms(vals, i):
    v = [x[i] for x in vals if x is not None and not np.isnan(x[i])]
    if not v:
        return "-"
    return f"{np.mean(v):.2f} +- {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def verdict(per_model, i, margin):
    """v2 vs the better DIRE variant, per seed"""
    v2 = per_model.get("v2")
    dire = [per_model.get(m) for m in ("dire_off", "dire_fair")]
    if v2 is None or not any(dire):
        return "-"
    diffs = []
    for k in range(len(SEEDS)):
        best = [d[k][i] for d in dire if d and d[k] is not None]
        if v2[k] is None or not best:
            continue
        diffs.append(v2[k][i] - max(best))
    if len(diffs) < 2:
        return "-"
    tag = ("v2 BETTER" if all(d >= margin for d in diffs) else "DIRE BETTER" if all(d <= -margin for d in diffs)
           else "no clear difference")
    return f"{tag} ({', '.join(f'{d:+.1f}' for d in diffs)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_dir", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()
    L = ["# FS-DualNet v2 vs DIRE (same images, same protocol, 3 seeds: mean +- std)", "",
         "Verdict: v2 minus the BETTER of the two DIRE variants, per seed; it counts only if >= 1 point in every seed.", ""]

    L += ["## 1. Full test sets, train celebahq sdv2 -> test dalle2 + midjourney (DIRE: official precomputed maps)", "",
          "| model | AUC | AP | ACC@0.5 | ACC@dev-threshold |", "|---|---|---|---|---|"]
    per = {}
    for m, label in MODELS:
        vals = [full_scores(args, m, s) for s in SEEDS]
        per[m] = vals
        L.append(f"| {label} | " + " | ".join(ms(vals, i) for i in range(4)) + " |")
    L += ["", f"Verdict AUC: {verdict(per, 0, args.margin)} | ACC@dev-threshold: {verdict(per, 3, args.margin)}", ""]

    L += ["## 2. Same 400 celebahq images (dalle2 + midjourney, 100 real + 100 fake each), maps recomputed by us", "",
          "DIRE maps of real AND fake images saved as PNG (format-fair). Perturbation applied before DIRE / before v2.", "",
          "| condition | " + " | ".join(f"{label} AUC / ACC@thr" for _, label in MODELS) + " | verdict (AUC) |",
          "|---|" + "---|" * (len(MODELS) + 1)]
    rows = [("official maps (real JPEG, fake PNG)", "official"), ("our maps, official format bias re-created", "asym"),
            ("clean (format-fair)", "clean"), ("JPEG q65", "jpeg_q65"), ("JPEG q30", "jpeg_q30"), ("blur sigma 3", "blur_s3")]
    for label, cond in rows:
        per = {}
        for m, _ in MODELS:
            c = "clean" if m == "v2" and cond in ("official", "asym") else cond  # v2 reads RGB images: clean
            per[m] = [subset_scores(args, m, s, c, UNSEEN) for s in SEEDS]
        cells = [f"{ms(per[m], 0)} / {ms(per[m], 2)}" for m, _ in MODELS]
        L.append(f"| {label} | " + " | ".join(cells) + f" | {verdict(per, 0, args.margin) if cond not in ('official', 'asym') else 'diagnostic'} |")

    L += ["", "## 3. Cross-dataset: celebahq-trained models on DeepFakeFace (real = IMDB-WIKI, fake = SD text2img)", "",
          "| model | AUC | AP | ACC@thr |", "|---|---|---|---|"]
    per = {}
    for m, label in MODELS:
        per[m] = [subset_scores(args, m, s, "clean", ("dff",)) for s in SEEDS]
        L.append(f"| {label} | {ms(per[m], 0)} | {ms(per[m], 1)} | {ms(per[m], 2)} |")
    L += ["", f"Verdict AUC: {verdict(per, 0, args.margin)}", ""]

    sp, lat = jload(os.path.join(args.res, "dire_speed.json")), jload(os.path.join(args.v2_dir, "latency.json"))
    L += ["## 4. Speed (same Kaggle T4 GPU)", ""]
    if sp:
        bk = [k for k in sp if k.startswith("ms_per_img_bs") and k != "ms_per_img_bs1"]
        L.append(f"- DIRE map (ADM-256, 20 + 20 DDIM steps): {sp['ms_per_img_bs1'] / 1000:.2f} s/image at batch 1"
                 + (f", {sp[bk[0]] / 1000:.2f} s/image at batch {sp['bs']}" if bk else "") + " (+ ResNet-50)")
    if lat and "v2" in lat.get("variants", {}):
        v = lat["variants"]["v2"]
        L.append(f"- FS-DualNet v2: {v['ms_per_img_bs1']:.1f} ms/image at batch 1, {1000 / v['img_per_s_bs32']:.1f} ms/image at batch 32")
        if sp:
            L.append(f"- v2 is {sp['ms_per_img_bs1'] / v['ms_per_img_bs1']:.0f}x faster per image (batch 1)")
    val = jload(os.path.join(args.res, "dire_validate.json"))
    if val:
        L += ["", "## 5. Check: our DIRE maps vs the official maps of the same images (mean abs. difference, 0-255)", ""]
        for cls in ("real", "fake"):
            if cls in val and val[cls].get("n"):
                L.append(f"- {cls}: {val[cls]['n']} images, MAE {val[cls]['mae']:.2f} "
                         f"(official {'JPEG q95' if cls == 'real' else 'PNG'}; channel-swapped {val[cls]['mae_rgb_swapped']:.2f})")
    out = "\n".join(L) + "\n"
    open(os.path.join(args.res, "vs_dire.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
