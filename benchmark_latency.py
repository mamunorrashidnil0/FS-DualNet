"""Inference speed of FS-DualNet variants (random input, no data needed).

  python benchmark_latency.py --out runs/latency.json
  python benchmark_latency.py --out runs/latency.json --dire_ms 850     # after timing DIRE yourself

How to get --dire_ms (DIRE's end-to-end cost: DDIM inversion + reconstruction + classifier), on the SAME GPU:
  1. clone DIRE, follow its README, get the pretrained model + ADM checkpoint
  2. time  `python demo.py -f <folder_with_N_images> -m <model>`  for N=100 and for N=1
  3. dire_ms = (time_N100 - time_N1) / 99 * 1000     (removes model-loading time)
Do NOT use numbers from another GPU or from a paper.
Per the DIRE paper, DIRE runs S=20 inversion + S=20 reconstruction steps = 2*S = 40 diffusion-network evaluations per
image (Sec. 3.2, 4.1), versus ONE forward pass here.
"""
import argparse
import json
import time

import torch

from fs_dualnet import FSDualNet

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="latency.json")
ap.add_argument("--dire_ms", type=float, default=None, help="DIRE ms/image measured by you on this GPU")
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = dev.type == "cuda"


@torch.no_grad()
def bench(model, bs, iters=30, warm=10):
    x = torch.rand(bs, 3, 224, 224, device=dev)
    for _ in range(warm):
        with torch.autocast(device_type=dev.type, enabled=amp):
            model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        with torch.autocast(device_type=dev.type, enabled=amp):
            model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    return 1000 * dt / bs, bs / dt


print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}) | AMP: {amp}")
print(f"{'variant':16s} {'params(M)':>9s} {'ms/img (bs=1)':>14s} {'img/s (bs=32)':>14s}")
result = {"device": str(dev), "variants": {}}
for branch, fusion in [("both", "attn"), ("freq", "attn"), ("spatial", "attn")]:
    m = FSDualNet(224, pretrained=False, branch=branch, fusion=fusion).to(dev).eval()
    p = sum(q.numel() for q in m.parameters()) / 1e6
    ms1, _ = bench(m, 1)
    _, ips = bench(m, 32, iters=10, warm=3)
    print(f"{branch + '/' + fusion:16s} {p:9.2f} {ms1:14.1f} {ips:14.1f}")
    result["variants"][f"{branch}/{fusion}"] = {"params_M": p, "ms_per_img_bs1": ms1, "img_per_s_bs32": ips}
    if branch == "both":
        result["fs_dualnet_ms"] = ms1

if args.dire_ms:
    result["dire_ms"] = args.dire_ms
    print(f"\nFS-DualNet {result['fs_dualnet_ms']:.1f} ms/img vs DIRE {args.dire_ms:.1f} ms/img "
          f"-> {args.dire_ms / result['fs_dualnet_ms']:.1f}x faster")
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print("saved", args.out)
