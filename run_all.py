"""
Run ALL experiments of one prepared dataset, one after another, and SKIP the ones that already finished.
After any interruption (Colab disconnect, laptop sleep, VS Code closed) just run the SAME command again:
finished runs are skipped, the interrupted run resumes from its last finished epoch (last_<tag>.pt).

  python run_all.py --data "D:/FS DualNet/data" --train_gen sdv2 --epochs 3 --max_per_class 10000 --out_dir "D:/FS DualNet/runs_local"

Runs: main (proposed) | spatial (RGB baseline) | freq | concat | control (all images re-encoded as JPEG q=90).
Test sets: every folder in <data>/test, or give them explicitly:  --tests name=path name=path ...
"""
import argparse
import os
import subprocess
import sys

RUNS = {  # key: (tag, extra args, robustness test?)
    "main": ("both_attn", [], True),
    "spatial": ("spatial", ["--branch", "spatial"], True),
    "freq": ("freq", ["--branch", "freq"], True),
    "concat": ("both_concat", ["--fusion", "concat"], False),
    "control": ("both_attn_jpeg90", ["--jpeg_all", "90", "--tag", "both_attn_jpeg90"], False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ val/ test/ (made by prep_data.py or prep_dff.py)")
    ap.add_argument("--train_gen", required=True, help="generator folder used for training, e.g. sdv2 / inpainting / adm")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_per_class", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    ap.add_argument("--tests", nargs="*", default=None, help="name=path ... (default: all folders in <data>/test)")
    ap.add_argument("--no_robust", action="store_true")
    args = ap.parse_args()

    tests = args.tests
    if not tests:
        tdir = os.path.join(args.data, "test")
        if not os.path.isdir(tdir):
            raise SystemExit(f"ERROR: {tdir} does not exist. prep_data.py found no test split (see its WARNING lines). "
                             "Give test sets explicitly with --tests name=path, or fix/extract the test folders first.")
        tests = [f"{g}={os.path.join(tdir, g)}" for g in sorted(os.listdir(tdir))]
    print("test sets:", tests)
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)

    for key in args.runs:
        tag, extra, robust = RUNS[key]
        if os.path.exists(os.path.join(args.out_dir, f"results_{tag}.json")):
            print(f"[skip] {key} ('{tag}') already finished")
            continue
        cmd = [sys.executable, os.path.join(here, "train_eval.py"), *extra,
               "--train", os.path.join(args.data, "train", args.train_gen),
               "--val", os.path.join(args.data, "val", args.train_gen),
               "--test", *tests, "--epochs", str(args.epochs), "--max_per_class", str(args.max_per_class),
               "--workers", str(args.workers), "--bs", str(args.bs), "--out_dir", args.out_dir]
        if robust and not args.no_robust:
            cmd.append("--robust")
        print(f"\n[run] {key} ('{tag}')\n  " + " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit(f"run '{key}' stopped/failed. Fix the error (or just re-run this same command to resume).")
    print(f"\nALL DONE. Next:  python make_table.py --dir \"{args.out_dir}\"  and  python claims.py --dir \"{args.out_dir}\"")


if __name__ == "__main__":
    main()
