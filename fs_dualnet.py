"""
FS-DualNet: Frequency-Spatial Dual-Branch Network
for explainable cross-domain detection of diffusion-generated faces.

Pipeline (matches the structure diagram):

    Input face image  (B,3,H,W), values in [0,1]
        |-- Frequency branch:  FFT analysis -> DCT analysis -> Spectral artifact map -> Frequency feature vector
        |-- Spatial branch:    EfficientNet-B4 -> Pixel-level features -> Global avg pooling -> Spatial feature vector
        v
    Attention fusion module
        v
    Binary classification head (Real / Fake)   -> 1 logit, sigmoid(logit) = P(fake)
        v
    Grad-CAM -> forged facial region visualisation

No diffusion reconstruction (DIRE-style) is needed at inference time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Frequency branch
# --------------------------------------------------------------------------- #
def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n). 2-D DCT of x is  D @ x @ D.T"""
    k = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d


class SpectralTransform(nn.Module):
    """FFT analysis + DCT analysis (no learnable parameters).

    For every RGB channel:
      * FFT : centred log-magnitude spectrum  (global periodic / up-sampling artifacts)
      * DCT : log |coefficients|              (energy compaction, JPEG-like block statistics)
    Output: (B, 6, H, W), standardised per sample and channel.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.register_buffer("dct", dct_matrix(img_size), persistent=False)

    @torch.no_grad()
    def _check(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size, (
            f"expected {self.img_size}x{self.img_size} input, got {tuple(x.shape[-2:])}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        # FFT/DCT must run in fp32 (cuFFT half precision only supports power-of-2 sizes)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
            fft_mag = torch.log1p(fft.abs())
            dct = self.dct @ x @ self.dct.t()
            dct_mag = torch.log1p(dct.abs())
            out = torch.cat([fft_mag, dct_mag], dim=1)
            mean = out.mean(dim=(2, 3), keepdim=True)
            std = out.std(dim=(2, 3), keepdim=True)
            return (out - mean) / (std + 1e-6)


def conv_bn_act(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation: re-weights spectral channels that carry artifacts."""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        return x * w


class FrequencyBranch(nn.Module):
    """FFT -> DCT -> light CNN -> spectral artifact map -> GAP -> frequency feature vector."""

    def __init__(self, img_size: int = 224, width: int = 32):
        super().__init__()
        w = width
        self.transform = SpectralTransform(img_size)
        self.encoder = nn.Sequential(
            conv_bn_act(6, w, 2),
            conv_bn_act(w, w, 1),
            conv_bn_act(w, 2 * w, 2),
            conv_bn_act(2 * w, 2 * w, 1),
            conv_bn_act(2 * w, 4 * w, 2),
            conv_bn_act(4 * w, 4 * w, 1),
            conv_bn_act(4 * w, 8 * w, 2),
            ChannelAttention(8 * w),  # output = spectral artifact map
        )
        self.out_dim = 8 * w

    def forward(self, x, return_map=False):
        artifact_map = self.encoder(self.transform(x))
        vec = artifact_map.mean(dim=(2, 3))
        return (vec, artifact_map) if return_map else vec


# --------------------------------------------------------------------------- #
# Spatial branch
# --------------------------------------------------------------------------- #
class SpatialBranch(nn.Module):
    """EfficientNet-B4 -> pixel-level feature map (1792 ch) -> global average pooling."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b4(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1792
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x, return_map=False):
        fmap = self.features((x - self.mean) / self.std)
        vec = self.pool(fmap).flatten(1)
        return (vec, fmap) if return_map else vec


# --------------------------------------------------------------------------- #
# Attention fusion + head
# --------------------------------------------------------------------------- #
class AttentionFusion(nn.Module):
    """Fuses frequency & spatial vectors.

    1. project both to a shared dim, add modality embeddings  -> 2 tokens
    2. multi-head self-attention across the 2 tokens (cross-modal interaction) + FFN
    3. learned softmax gate over tokens -> fused vector.
       The gate weights show how much each branch contributed (useful for explainability).
    """

    def __init__(self, freq_dim, spat_dim, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.proj_f = nn.Sequential(nn.Linear(freq_dim, dim), nn.LayerNorm(dim))
        self.proj_s = nn.Sequential(nn.Linear(spat_dim, dim), nn.LayerNorm(dim))
        self.modality_emb = nn.Parameter(torch.zeros(1, 2, dim))
        nn.init.trunc_normal_(self.modality_emb, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, 1)
        self.out_dim = dim

    def forward(self, f, s):
        t = torch.stack([self.proj_f(f), self.proj_s(s)], dim=1) + self.modality_emb  # (B,2,D)
        a, _ = self.attn(t, t, t, need_weights=False)
        t = self.norm1(t + a)
        t = self.norm2(t + self.ffn(t))
        w = torch.softmax(self.gate(t).squeeze(-1), dim=1)  # (B,2): [freq, spatial]
        fused = (w.unsqueeze(-1) * t).sum(dim=1)
        return fused, w


class ConcatFusion(nn.Module):
    """Ablation baseline: plain concatenation + MLP (no attention)."""

    def __init__(self, freq_dim, spat_dim, dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim + spat_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.out_dim = dim

    def forward(self, f, s):
        return self.mlp(torch.cat([f, s], dim=1)), None


class Projection(nn.Module):
    """Used when only one branch is active (ablation)."""

    def __init__(self, in_dim, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.out_dim = dim

    def forward(self, x):
        return self.net(x)


class FSDualNet(nn.Module):
    """
    branch: "both" (proposed) | "freq" (frequency-only ablation) | "spatial" (RGB EfficientNet-B4 baseline)
    fusion: "attn" (proposed attention fusion) | "concat" (ablation); only used when branch == "both"
    """

    def __init__(self, img_size=224, pretrained=True, fusion_dim=256, dropout=0.3, branch="both", fusion="attn"):
        super().__init__()
        assert branch in ("both", "freq", "spatial") and fusion in ("attn", "concat")
        self.branch, self.fusion_type, self.img_size = branch, fusion, img_size
        self.freq = FrequencyBranch(img_size) if branch in ("both", "freq") else None
        self.spatial = SpatialBranch(pretrained) if branch in ("both", "spatial") else None
        if branch == "both":
            fuse_cls = AttentionFusion if fusion == "attn" else ConcatFusion
            self.fusion = fuse_cls(self.freq.out_dim, self.spatial.out_dim, fusion_dim)
        else:
            enc = self.freq if branch == "freq" else self.spatial
            self.fusion = Projection(enc.out_dim, fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x, return_aux=False):
        """x: (B,3,H,W) in [0,1].  Returns logit (B,) ; sigmoid(logit)=P(fake)."""
        w = None
        if self.branch == "both":
            fused, w = self.fusion(self.freq(x), self.spatial(x))  # w: (B,2) [freq, spatial] for attn fusion
        elif self.branch == "freq":
            fused = self.fusion(self.freq(x))
        else:
            fused = self.fusion(self.spatial(x))
        logit = self.head(fused).squeeze(1)
        return (logit, {"branch_weights": w}) if return_aux else logit

    def gradcam_layer(self):
        """Default Grad-CAM target: last spatial conv block (face-region map), else frequency encoder."""
        return self.spatial.features if self.spatial is not None else self.freq.encoder

    def config(self):
        return {"branch": self.branch, "fusion": self.fusion_type, "img_size": self.img_size}


# --------------------------------------------------------------------------- #
# Grad-CAM (forged facial region visualisation)
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM on any conv feature layer.

    Spatial (recommended, face-region localisation):  GradCAM(model, model.spatial.features)
    Frequency (spectral-domain evidence):             GradCAM(model, model.freq.encoder)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inp, out):
        self.acts = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x: torch.Tensor, target: str = "fake"):
        """Returns (cam (B,H,W) in [0,1], P(fake) (B,))."""
        was_training = self.model.training
        self.model.eval()
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)  # keeps the graph alive even if layers are frozen
            self.model.zero_grad(set_to_none=True)
            logit = self.model(x)
            score = logit if target == "fake" else -logit
            score.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mn, mx = cam.amin(dim=(2, 3), keepdim=True), cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        self.model.train(was_training)
        return cam.squeeze(1).detach(), torch.sigmoid(logit).detach()

    def remove(self):
        self._handle.remove()


def overlay_heatmap(img: torch.Tensor, cam: torch.Tensor, alpha: float = 0.45):
    """img (3,H,W) in [0,1], cam (H,W) in [0,1] -> uint8 RGB overlay (H,W,3)."""
    import cv2
    import numpy as np

    rgb = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    heat = cv2.applyColorMap((cam.cpu().numpy() * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 255).astype("uint8")


if __name__ == "__main__":
    # smoke test over all ablation variants (no pretrained download)
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.rand(2, 3, 224, 224)
    for branch, fusion in [("both", "attn"), ("both", "concat"), ("freq", "attn"), ("spatial", "attn")]:
        model = FSDualNet(224, pretrained=False, branch=branch, fusion=fusion).eval()
        logit, aux = model(x, return_aux=True)
        cam, prob = GradCAM(model, model.gradcam_layer())(x)
        w = None if aux["branch_weights"] is None else tuple(aux["branch_weights"].shape)
        print(f"{branch:8s}/{fusion:6s} params {n(model):6.2f}M | logit {tuple(logit.shape)} | "
              f"branch_w {w} | cam {tuple(cam.shape)}")
