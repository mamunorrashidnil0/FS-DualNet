"""
Predict real/fake for one face image and save a Grad-CAM forged-region visualisation.

  python explain.py --ckpt fs_dualnet_best.pt --image face.png --out explain.png
"""
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from fs_dualnet import FSDualNet, GradCAM, overlay_heatmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="explain.png")
    ap.add_argument("--layer", choices=["auto", "freq"], default="auto", help="auto = spatial conv (face regions)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    S = ck.get("img_size", 224)
    model = FSDualNet(img_size=S, pretrained=False, branch=ck.get("branch", "both"), fusion=ck.get("fusion", "attn")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    R = ck.get("resize", 256)  # same preprocessing as DIRE protocol: resize 256 -> center crop 224
    tf = T.Compose([T.Resize((R, R), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(S), T.ToTensor()])
    x = tf(img).unsqueeze(0).to(device)

    layer = model.freq.encoder if args.layer == "freq" else model.gradcam_layer()
    cam_fn = GradCAM(model, layer)
    cam, prob = cam_fn(x)
    with torch.no_grad():
        _, aux = model(x, return_aux=True)
    bw = aux["branch_weights"]

    label = "FAKE" if prob.item() >= 0.5 else "REAL"
    extra = "" if bw is None else f" | branch weights  freq={bw[0, 0].item():.2f}  spatial={bw[0, 1].item():.2f}"
    print(f"{label}  P(fake)={prob.item():.4f}{extra}")

    orig = (x[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    ov = overlay_heatmap(x[0].cpu(), cam[0].cpu())
    Image.fromarray(np.concatenate([orig, ov], axis=1)).save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
