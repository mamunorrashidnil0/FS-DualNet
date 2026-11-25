"""
Run ALL experiments of one prepared dataset, one after another, and SKIP the ones that already finished.
After any interruption (Colab disconnect, laptop sleep, VS Code closed) just run the SAME command again:
finished runs are skipped, the interrupted run resumes from its last finished epoch (last_<tag>.pt).

  python run_all.py --data "D:/FS DualNet/data" --train_gen sdv2 --epochs 5 --out_dir "D:/FS DualNet/runs_local"

Runs: v2 (proposed FS-DualNet) and its ablations v2_spatial, v2_freq, v2_nodeg, v2_raw, v2_nomd, v2_effnet.
Test sets: every folder in <data>/test, or give them explicitly:  --tests name=path name=path ...
--deadline <unix time>: stop cleanly before that time (Kaggle 12 h limit); the same command later resumes.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

V2 = ["--arch", "v2", "--degrade"]
RUNS = {  # key: (tag, extra args, robustness test?)
    # ---- v2: patch-level frequency-spatial fusion (proposed) + one-change-at-a-time ablations ----
    "v2": ("v2", [*V2, "--tag", "v2"], True),
    "v2_spatial": ("v2_spatial", [*V2, "--branch", "spatial", "--tag", "v2_spatial"], True),  # frozen-CLIP branch only
    "v2_freq": ("v2_freq", [*V2, "--branch", "freq", "--tag", "v2_freq"], True),              # residual-DCT branch only
    "v2_nodeg": ("v2_nodeg", ["--arch", "v2", "--tag", "v2_nodeg"], True),                    # without degradation augmentation
    "v2_raw": ("v2_raw", [*V2, "--freq_input", "raw", "--tag", "v2_raw"], False),             # DCT of the image, no NPR residual
    "v2_nomd": ("v2_nomd", [*V2, "--moddrop", "0", "--aux_weight", "0", "--tag", "v2_nomd"], False),  # no anti-collapse terms
    "v2_effnet": ("v2_effnet", [*V2, "--spatial_backbone", "effnet", "--tag", "v2_effnet"], True),  # fine-tuned EffNet-B4, not CLIP
}
DEFAULT_RUNS = ["v2", "v2_spatial", "v2_freq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ val/ test/ (made by prep_data.py or prep_dff.py)")
    ap.add_argument("--train_gen", required=True, nargs="+", help="generator folder(s) used for training, e.g. sdv2 / inpainting. Several = trained together; the first one provides the validation set")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_per_class", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--runs", nargs="+", choices=list(RUNS), default=DEFAULT_RUNS)
    ap.add_argument("--tests", nargs="*", default=None, help="name=path ... (default: all folders in <data>/test)")
    ap.add_argument("--no_robust", action="store_true")
    ap.add_argument("--dev", nargs="*", default=None, help="DEV sets name=path (a generator NOT used for training): picks the decision threshold")
    ap.add_argument("--extra", default="", help='extra arguments passed to every train_eval.py call, e.g. "--select auc"')
    ap.add_argument("--seed", type=int, default=42, help="random seed (42 = the default of train_eval.py). Use a different --out_dir per seed")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: do not start / stop cleanly before it (resumable)")
    argv = sys.argv[1:]
    if "--extra" in argv[:-1]:  # let  --extra "--no_pretrained"  work (argparse would read the value as an option)
        i = argv.index("--extra")
        argv[i:i + 2] = [f"--extra={argv[i + 1]}"]
    args = ap.parse_args(argv)

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
        if args.deadline and time.time() > args.deadline - 600:
            print(f"TIME BUDGET REACHED: '{key}' and later runs not started. Commit again with this output as input to continue.")
            return
        cmd = [sys.executable, os.path.join(here, "train_eval.py"), *extra,
               "--train", *[os.path.join(args.data, "train", g) for g in args.train_gen],
               "--val", os.path.join(args.data, "val", args.train_gen[0]),
               "--test", *tests, "--epochs", str(args.epochs), "--max_per_class", str(args.max_per_class),
               "--workers", str(args.workers), "--bs", str(args.bs), "--out_dir", args.out_dir, "--seed", str(args.seed)]
        if args.dev:
            cmd += ["--dev", *args.dev]
        if args.deadline:
            cmd += ["--deadline", str(args.deadline)]
        if args.extra:
            cmd += shlex.split(args.extra)
        if robust and not args.no_robust:
            cmd.append("--robust")
        print(f"\n[run] {key} ('{tag}')\n  " + " ".join(cmd), flush=True)
        code = subprocess.run(cmd).returncode
        if code == 3:  # train_eval.TIME_UP: checkpoints saved
            print(f"TIME BUDGET REACHED during '{key}' (saved). Commit again with this output as input to continue.")
            return
        if code != 0:
            raise SystemExit(f"run '{key}' stopped/failed. Fix the error (or just re-run this same command to resume).")
    print(f"\nALL DONE. Next:  python make_table.py --dir \"{args.out_dir}\"  and  python claims.py --dir \"{args.out_dir}\"")


if __name__ == "__main__":
    main()
