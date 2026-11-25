"""
Predict real/fake for one face image and save an explanation figure.

  python explain.py --ckpt runs/v2.pt --image face.png --out explain.png

FS-DualNet v2 checkpoint -> 4 panels: input | token Grad-CAM | patch evidence P(fake) per 16x16 patch | modality map
                (per patch: red = decided by the frequency branch, blue = by the spatial branch)
Fully synthetic faces have no region-level ground truth: these maps show decision-relevant regions, not "forged regions".
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from fs_dualnet import build_model, load_model_state, overlay_heatmap, token_gradcam


def label_panel(img, text):
    import cv2
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def patch_map(m, size):
    """(n,n) values in [0,1] -> (size,size), blocky so every patch stays visible."""
    return F.interpolate(m[None, None].float(), size=size, mode="nearest")[0, 0].clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="explain.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)  # our own file; PyTorch >= 2.6 defaults to weights_only=True
    except TypeError:
        ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck).to(device)
    load_model_state(model, ck["model"])
    model.eval()
    S = ck.get("img_size", 224)

    img = Image.open(args.image).convert("RGB")
    R = ck.get("resize", 256)  # same preprocessing as DIRE protocol: resize 256 -> center crop 224
    tf = T.Compose([T.Resize((R, R), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(S), T.ToTensor()])
    x = tf(img).unsqueeze(0).to(device)
    orig = (x[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")

    if ck.get("arch") != "v2":
        raise SystemExit("explain.py needs an FS-DualNet v2 checkpoint (the DIRE baseline has no explanation maps)")
    cam, prob, aux = token_gradcam(model, x)
    p = prob.item()
    panels = [label_panel(orig, f"{'FAKE' if p >= 0.5 else 'REAL'}  P(fake)={p:.3f}"),
              label_panel(overlay_heatmap(x[0].cpu(), cam[0].cpu()), "token Grad-CAM"),
              label_panel(overlay_heatmap(x[0].cpu(), patch_map(torch.sigmoid(aux["patch_logits"][0]), S).cpu()),
                          "patch evidence P(fake)")]
    if "gate_freq" in aux:
        panels.append(label_panel(overlay_heatmap(x[0].cpu(), patch_map(aux["gate_freq"][0], S).cpu()),
                                  "red=frequency blue=spatial"))
    bw = aux["branch_weights"]
    a = aux["patch_attn"][0].flatten()
    top = torch.topk(a, min(3, a.numel())).indices.tolist()
    n = model.grid
    where = ", ".join(f"(row {i // n}, col {i % n}) w={a[i].item():.2f}" for i in top)
    extra = "" if bw is None else f" | branch weights freq={bw[0, 0].item():.2f} spatial={bw[0, 1].item():.2f}"
    print(f"{'FAKE' if p >= 0.5 else 'REAL'}  P(fake)={p:.4f}{extra}\nmost-attended patches: {where}")

    Image.fromarray(np.concatenate(panels, axis=1)).save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
