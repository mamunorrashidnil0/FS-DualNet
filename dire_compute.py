"""
DIRE maps (Wang et al., ICCV 2023) with the official recipe, for a FORMAT-FAIR comparison with FS-DualNet.

Recipe (from the official compute_dire.py / compute_dire.sh): ADM 256x256 unconditional ImageNet checkpoint
(256x256_diffusion_uncond.pt), fp16, DDIM-20 inversion x0 -> x_T, DDIM-20 reconstruction x_T -> x0',
DIRE = |x0 - x0'| * 255 / 2 as uint8. Images are loaded like the official loader: PIL RGB, center_crop_arr to 256.

Every map is saved as PNG for real AND fake images. In the official DiffusionForensics release the real maps are
JPEG (q95) and the fake maps PNG, and the file format alone can decide the prediction (Ricker et al., CVPR 2024
supplementary, Sec. 9 "Analysis of DIRE").
A perturbation (blur / JPEG) is applied to the 256x256 image BEFORE the diffusion reconstruction.

  run       python dire_compute.py run --jobs jobs.txt --model 256x256_diffusion_uncond.pt --gd_path DIRE/guided-diffusion
            (jobs.txt lines: <source image> TAB <perturbation> TAB <output png>; several workers / GPUs can share one
            list: each claims chunks with a lock file, finished outputs are skipped, so it is resumable)
  bench     time one image (batch 1) and one full batch
  validate  compare our maps with the official maps of the same images (pixel MAE, 0-255)
  asym      rebuild the official format asymmetry on our maps (real -> JPEG q95, fake -> PNG): diagnostic only
"""
import argparse
import inspect
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter

DIRE_FLAGS = dict(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, dropout=0.1, image_size=256,
                  learn_sigma=True, noise_schedule="linear", num_channels=256, num_head_channels=64, num_res_blocks=2,
                  resblock_updown=True, use_fp16=True, use_scale_shift_norm=True, timestep_respacing="ddim20")
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def center_crop_arr(pil_image, image_size):
    """guided-diffusion's center_crop_arr (used by DIRE's load_data_for_reverse)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return arr[cy: cy + image_size, cx: cx + image_size]


def perturb(img, name):
    """Same operations as train_eval.py (Blur = PIL Gaussian blur radius sigma, Jpeg = PIL JPEG quality q)."""
    if name in ("none", "clean"):
        return img
    if name.startswith("blur_s"):
        return img.filter(ImageFilter.GaussianBlur(radius=float(name[6:])))
    if name.startswith("jpeg_q"):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(name[6:]))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"unknown perturbation {name}")


def load_image(path, size, pert):
    pil = Image.open(path)
    pil.load()
    img = Image.fromarray(center_crop_arr(pil.convert("RGB"), size))
    arr = np.asarray(perturb(img, pert), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def import_gd(gd_path):
    """guided_diffusion from DIRE's repo copy (gd_path) or the pip package; a stub stands in for mpi4py, which only
    the training / data utilities need (not imported here)."""
    if gd_path:
        sys.path.insert(0, gd_path)
    import importlib.util
    if importlib.util.find_spec("mpi4py") is None:
        import types
        comm = types.SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1, bcast=lambda x, root=0: x, rank=0, size=1)
        stub = types.ModuleType("mpi4py")
        stub.MPI = types.SimpleNamespace(COMM_WORLD=comm)
        sys.modules["mpi4py"], sys.modules["mpi4py.MPI"] = stub, stub.MPI
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    return create_model_and_diffusion, model_and_diffusion_defaults


def make_dire_fn(args, device):
    """x in [-1, 1], (B,3,H,W) float -> DIRE uint8 (B,3,H,W)."""
    if args.mock:  # code tests only: a cheap stand-in for the diffusion reconstruction
        def mock(x):
            rec = torch.nn.functional.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
            return ((x - rec).abs() * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
        return mock
    create_model_and_diffusion, model_and_diffusion_defaults = import_gd(args.gd_path)
    flags = model_and_diffusion_defaults()
    flags.update(DIRE_FLAGS)
    flags.update(json.loads(args.flags or "{}"))
    flags = {k: v for k, v in flags.items() if k in model_and_diffusion_defaults()}
    model, diffusion = create_model_and_diffusion(**flags)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    if flags.get("use_fp16"):
        model.convert_to_fp16()
    model.to(device).eval()
    rev_loop = getattr(diffusion, "ddim_reverse_sample_loop", None)  # present in DIRE's guided-diffusion copy
    rev_kw = {"real_step": 0} if rev_loop is not None and "real_step" in inspect.signature(rev_loop).parameters else {}
    smp_kw = {"real_step": 0} if "real_step" in inspect.signature(diffusion.ddim_sample_loop).parameters else {}

    def dire(x):
        with torch.no_grad():
            if rev_loop is not None:
                latent = rev_loop(model, tuple(x.shape), noise=x, clip_denoised=True, model_kwargs={}, **rev_kw)
            else:  # openai/guided-diffusion: same loop as DIRE's ddim_reverse_sample_loop (real_step = 0)
                latent = x
                for i in range(diffusion.num_timesteps):
                    t = torch.tensor([i] * x.shape[0], device=x.device)
                    latent = diffusion.ddim_reverse_sample(model, latent, t, clip_denoised=True, model_kwargs={})["sample"]
            rec = diffusion.ddim_sample_loop(model, tuple(x.shape), noise=latent, clip_denoised=True, model_kwargs={}, **smp_kw)
        return ((x - rec).abs() * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
    return dire


def run_batch(fn, x):
    """Run fn on x; on CUDA out-of-memory split the batch in halves."""
    try:
        return fn(x)
    except torch.cuda.OutOfMemoryError:
        if x.shape[0] == 1:
            raise
        torch.cuda.empty_cache()
        h = x.shape[0] // 2
        return torch.cat([run_batch(fn, x[:h]), run_batch(fn, x[h:])])


def save_png(arr_hwc, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.png"
    Image.fromarray(arr_hwc).save(tmp, format="PNG")
    os.replace(tmp, dst)  # atomic: a half-written file never counts as done


def claim(lockdir, k):
    try:
        fd = os.open(os.path.join(lockdir, f"chunk_{k:06d}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def read_jobs(path):
    jobs = []
    for line in open(path):
        if line.strip():
            src, pert, dst = line.rstrip("\n").split("\t")
            jobs.append((src, pert, dst))
    return jobs


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = read_jobs(args.jobs)
    os.makedirs(args.lockdir, exist_ok=True)
    chunks = [jobs[i: i + args.bs] for i in range(0, len(jobs), args.bs)]
    todo = sum(not os.path.exists(d) for _, _, d in jobs)
    print(f"[worker {os.getpid()}] {len(jobs)} jobs, {todo} to do, {len(chunks)} chunks of {args.bs} | device {device}", flush=True)
    fn, done, t0 = None, 0, time.time()
    for k, chunk in enumerate(chunks):
        rest = [j for j in chunk if not os.path.exists(j[2])]
        if not rest or not claim(args.lockdir, k):
            continue
        if args.deadline and time.time() > args.deadline:
            print("time budget reached: stopping (finished maps are kept; run again to continue)", flush=True)
            break
        if fn is None:
            fn = make_dire_fn(args, device)
        x = torch.stack([load_image(s, args.size, p) for s, p, _ in rest]).to(device)
        out = run_batch(fn, x).permute(0, 2, 3, 1).cpu().numpy()
        for (_, _, dst), arr in zip(rest, out):
            save_png(arr, dst)
        done += len(rest)
        rate = (time.time() - t0) / done
        print(f"[worker {os.getpid()}] chunk {k + 1}/{len(chunks)} | {done} maps | {rate:.2f} s/map", flush=True)
    left = sum(not os.path.exists(d) for _, _, d in jobs)
    print(f"[worker {os.getpid()}] finished: {done} maps by this worker, {left} of {len(jobs)} still missing", flush=True)


def cmd_bench(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcs = [l.split("\t")[0] for l in open(args.jobs) if l.strip()][: args.bs]
    fn = make_dire_fn(args, device)
    x = torch.stack([load_image(s, args.size, "none") for s in srcs]).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    fn(x[:1])
    sync()
    t = time.time()
    for i in range(3):
        fn(x[i % len(x): i % len(x) + 1])
    sync()
    ms1 = (time.time() - t) / 3 * 1000
    t = time.time()
    run_batch(fn, x)
    sync()
    msb = (time.time() - t) / len(x) * 1000
    res = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "ms_per_img_bs1": ms1,
           f"ms_per_img_bs{len(x)}": msb, "bs": len(x), "ddim_steps": "20 inversion + 20 reconstruction",
           "note": "DIRE map only; the ResNet-50 classifier on top adds a few ms"}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def list_images(d):
    out = {}
    for dp, _, fs in os.walk(d, followlinks=True):
        for f in fs:
            if f.lower().endswith(EXT):
                out[os.path.splitext(f)[0]] = os.path.join(dp, f)
    return out


def cmd_validate(args):
    """Our maps vs the official maps of the same images, per sub-folder (real / fake), MAE on the 0-255 scale."""
    res = {}
    for cls in ("real", "fake"):
        ours, off = list_images(os.path.join(args.ours, cls)), list_images(os.path.join(args.official, cls))
        maes, flips = [], []
        for stem, p in ours.items():
            if stem in off:
                a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                b = np.asarray(Image.open(off[stem]).convert("RGB"), dtype=np.float32)
                if a.shape == b.shape:
                    maes.append(float(np.abs(a - b).mean()))
                    flips.append(float(np.abs(a - b[..., ::-1]).mean()))
        res[cls] = {"n": len(maes), "mae": float(np.mean(maes)) if maes else None,
                    "mae_rgb_swapped": float(np.mean(flips)) if flips else None,
                    "mean_level_ours": None}
        if maes:
            res[cls]["mean_level_ours"] = float(np.mean([np.asarray(Image.open(ours[s]), dtype=np.float32).mean()
                                                         for s in list(ours)[:50]]))
        print(f"{cls}: {res[cls]}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


def cmd_asym(args):
    """Diagnostic: re-create the official storage asymmetry on our maps (real -> JPEG q95, fake -> unchanged PNG)."""
    n = 0
    for dp, _, fs in os.walk(args.src, followlinks=True):
        rel = os.path.relpath(dp, args.src)
        is_real = "real" in rel.replace("\\", "/").split("/")
        for f in fs:
            if not f.lower().endswith(".png"):
                continue
            src, outd = os.path.join(dp, f), os.path.join(args.dst, rel)
            os.makedirs(outd, exist_ok=True)
            if is_real:
                Image.open(src).convert("RGB").save(os.path.join(outd, os.path.splitext(f)[0] + ".jpg"), quality=args.quality)
            elif not os.path.lexists(os.path.join(outd, f)):
                os.symlink(os.path.abspath(src), os.path.join(outd, f))
            n += 1
    print(f"asym: {n} maps -> {args.dst} (real maps JPEG q{args.quality}, fake maps PNG)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--jobs", required=True)
        p.add_argument("--model", default="", help="256x256_diffusion_uncond.pt")
        p.add_argument("--gd_path", default="", help="folder that contains the guided_diffusion package")
        p.add_argument("--flags", default="", help="JSON overrides of the model flags (tests only)")
        p.add_argument("--size", type=int, default=256)
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--mock", action="store_true", help="code test without a diffusion model")
        if name == "run":
            p.add_argument("--lockdir", required=True)
            p.add_argument("--deadline", type=float, default=0)
        else:
            p.add_argument("--out", required=True)
    p = sub.add_parser("check_import")
    p.add_argument("--gd_path", default="")
    p = sub.add_parser("validate")
    p.add_argument("--ours", required=True)
    p.add_argument("--official", required=True)
    p.add_argument("--out", default="")
    p = sub.add_parser("asym")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    if args.cmd == "check_import":
        import_gd(args.gd_path)
        print("guided_diffusion import OK")
        return
    {"run": cmd_run, "bench": cmd_bench, "validate": cmd_validate, "asym": cmd_asym}[args.cmd](args)


if __name__ == "__main__":
    main()
