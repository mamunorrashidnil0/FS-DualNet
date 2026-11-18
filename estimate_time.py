"""
Measure how fast THIS computer trains FS-DualNet and estimate the total time of your experiments. Needs no dataset.

  python estimate_time.py                       # defaults: 10,000 imgs/class, 3 epochs, 5 runs
  python estimate_time.py --max_per_class 5000 --epochs 3 --runs 5

The estimate covers pure training steps only; data loading, validation and the robustness tests make the real time longer.
"""
import argparse
import os
import platform
import time

import torch
import torch.nn as nn

from fs_dualnet import FSDualNet

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--max_per_class", type=int, default=10000)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--runs", type=int, default=5)
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CPU: {platform.processor()} | logical cores: {os.cpu_count()} | torch threads: {torch.get_num_threads()}")
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
      + (f" | GPU: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
print(f"device used for the test: {dev}")

model = FSDualNet(224, pretrained=False).to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.BCEWithLogitsLoss()
amp = dev.type == "cuda"
x = torch.rand(args.bs, 3, 224, 224, device=dev)
y = (torch.rand(args.bs, device=dev) > 0.5).float()


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev.type, enabled=amp):
        loss = crit(model(x).float(), y)
    loss.backward()
    opt.step()


sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
print(f"warm-up ({args.warm} steps), then timing {args.steps} training steps with batch {args.bs} ...", flush=True)
for _ in range(args.warm):
    step()
sync()
t = time.time()
for _ in range(args.steps):
    step()
sync()
dt = time.time() - t
ips = args.bs * args.steps / dt
imgs_per_epoch = 2 * args.max_per_class
epoch_min = imgs_per_epoch / ips / 60
total_h = epoch_min * args.epochs * args.runs / 60
print(f"\nspeed: {ips:.2f} images/second (training)")
print(f"1 epoch of {imgs_per_epoch} images  ~ {epoch_min:.1f} min")
print(f"{args.runs} runs x {args.epochs} epochs ~ {total_h:.1f} hours (training only; real time is longer)")
