"""
FS-DualNet: patch-level frequency-spatial fusion for explainable cross-domain detection
of diffusion-generated faces.

    Input face image  (B,3,224,224), values in [0,1]
        |-- Frequency branch:  NPR residual -> 16x16 patch DCT -> 196 frequency tokens
        |-- Spatial branch:    frozen CLIP ViT-B/16 -> mix of its 12 layers -> adapter -> 196 spatial tokens
        v
    Bidirectional patch cross-attention + per-patch gate (modality map)
        v
    Per-patch real/fake scores (evidence map) + gated attention pooling -> 1 logit, sigmoid(logit) = P(fake)

No diffusion reconstruction (DIRE-style) is needed at inference time.
Also contains DIREClassifier: the ResNet-50 classifier of the DIRE baseline (trained on DIRE maps).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B4_Weights, ResNet50_Weights, efficientnet_b4, resnet50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# DCT helper
# --------------------------------------------------------------------------- #
def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n). 2-D DCT of x is  D @ x @ D.T"""
    k = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d


# --------------------------------------------------------------------------- #
# CNN backbones (DIRE baseline, EfficientNet ablation)
# --------------------------------------------------------------------------- #
class SpatialBranch(nn.Module):
    """ImageNet CNN -> feature map -> global average pooling.
    arch="resnet50": ResNet-50 (2048 ch), the classifier the DIRE paper uses (DIRE baseline).
    arch="efficientnet_b4": EfficientNet-B4 (1792 ch), used by the v2_effnet ablation."""

    def __init__(self, pretrained: bool = True, arch: str = "efficientnet_b4"):
        super().__init__()
        if arch == "resnet50":
            r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
            self.out_dim = 2048
        else:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.features = efficientnet_b4(weights=weights).features
            self.out_dim = 1792
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x, return_map=False):
        fmap = self.features((x - self.mean) / self.std)
        vec = self.pool(fmap).flatten(1)
        return (vec, fmap) if return_map else vec


class Projection(nn.Module):
    """Linear projection + LayerNorm (DIRE baseline head input)."""

    def __init__(self, in_dim, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.out_dim = dim

    def forward(self, x):
        return self.net(x)


class DIREClassifier(nn.Module):
    """The DIRE baseline: ResNet-50 on DIRE maps -> projection -> binary head (Real / Fake)."""

    def __init__(self, img_size=224, pretrained=True, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.img_size = img_size
        self.spatial = SpatialBranch(pretrained, "resnet50")
        self.fusion = Projection(self.spatial.out_dim, fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x, return_aux=False):
        """x: (B,3,H,W) in [0,1].  Returns logit (B,) ; sigmoid(logit)=P(fake)."""
        logit = self.head(self.fusion(self.spatial(x))).squeeze(1)
        return (logit, {}) if return_aux else logit

    def config(self):
        return {"arch": "dire", "branch": "resnet50", "img_size": self.img_size}


def overlay_heatmap(img: torch.Tensor, cam: torch.Tensor, alpha: float = 0.45):
    """img (3,H,W) in [0,1], cam (H,W) in [0,1] -> uint8 RGB overlay (H,W,3)."""
    import cv2
    import numpy as np

    rgb = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    heat = cv2.applyColorMap((cam.cpu().numpy() * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 255).astype("uint8")


# =========================================================================== #
# FS-DualNet v2: patch-level frequency-spatial fusion
# =========================================================================== #
#   input (B,3,224,224) in [0,1]
#     |-- frequency branch: NPR residual -> per-patch DCT -> local spectral tokens (grid x grid)
#     |-- spatial branch  : frozen CLIP ViT (learned mix of ALL blocks) -> patch tokens -> adapter
#     v
#   patch-wise bidirectional cross-attention + per-patch gate (+ modality dropout in training)
#     v
#   context transformer -> per-patch fake logit (evidence map) -> gated-attention MIL pooling -> image logit
#   explanation: patch evidence map, per-patch frequency-vs-spatial gate, token Grad-CAM
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# vision-tower configs, so the architecture can be built without a download (random init: code tests / timing only)
CLIP_CONFIGS = {
    "openai/clip-vit-base-patch16": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                                         num_attention_heads=12, image_size=224, patch_size=16),
    "openai/clip-vit-large-patch14": dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                                          num_attention_heads=16, image_size=224, patch_size=14),
    "tiny": dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                 image_size=224, patch_size=16),
}


def npr_residual(x: torch.Tensor) -> torch.Tensor:
    """Neighbouring-pixel residual (NPR, Tan et al., CVPR 2024): x - NN_up(NN_down(x)).
    Removes image content and keeps the local 2x2 pixel correlations that generator up-sampling leaves."""
    down = x[..., ::2, ::2]
    up = down.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return x - up[..., : x.shape[-2], : x.shape[-1]]


def _encoder(dim, heads, depth, dropout=0.1):
    layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class PatchSpectralTokens(nn.Module):
    """Frequency branch v2: (NPR residual | raw image) -> 2-D DCT of every P x P patch and colour channel -> log|coef|
    -> one token per patch. Local spectra keep WHERE an artifact is (needed for patch fusion and the evidence map)
    and are not dominated by whole-image statistics such as resize history."""

    def __init__(self, img_size=224, patch=16, dim=256, residual=True, depth=1, heads=4):
        super().__init__()
        self.patch, self.residual, self.grid = patch, residual, img_size // patch
        self.register_buffer("dct", dct_matrix(patch), persistent=False)
        in_dim = 3 * patch * patch
        self.norm_in = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.enc = _encoder(dim, heads, depth) if depth else nn.Identity()

    def spectra(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            r = npr_residual(x) if self.residual else x
            B, C, H, W = r.shape
            P, g = self.patch, self.grid
            p = r[:, :, : g * P, : g * P].reshape(B, C, g, P, g, P).permute(0, 2, 4, 1, 3, 5)  # B,g,g,C,P,P
            c = self.dct @ p @ self.dct.t()
            return torch.log(c.abs() + 1e-4).reshape(B, g * g, C * P * P)

    def forward(self, x):
        t = self.proj(self.norm_in(self.spectra(x))) + self.pos
        return self.enc(t)


class CLIPTokens(nn.Module):
    """Spatial branch v2: FROZEN CLIP ViT (UnivFD, Ojha et al., CVPR 2023). A learned softmax mix of all its blocks is
    used, since intermediate blocks carry lower-level cues (RINE, Koutlis & Papadopoulos, ECCV 2024).
    Only the layer weights and a small adapter are trained, so the backbone cannot memorise dataset specifics."""

    def __init__(self, name="openai/clip-vit-base-patch16", pretrained=True, dim=256):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel
        self.from_pretrained = bool(pretrained and name != "tiny")
        if self.from_pretrained:
            self.clip = CLIPVisionModel.from_pretrained(name).float()  # fp32 master copy; autocast runs it in fp16 on GPU
        else:
            if name not in CLIP_CONFIGS:
                raise ValueError(f"no offline config for {name}; use one of {list(CLIP_CONFIGS)}")
            self.clip = CLIPVisionModel(CLIPVisionConfig(**CLIP_CONFIGS[name]))
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.eval()
        cfg = self.clip.config
        self.patch = cfg.patch_size
        self.layer_logits = nn.Parameter(torch.zeros(cfg.num_hidden_layers))
        self.adapter = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, dim), nn.GELU(),
                                     nn.Linear(dim, dim))
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()  # frozen: never dropout / statistics updates
        return self

    def forward(self, x):
        with torch.no_grad():
            hs = self.clip(pixel_values=(x - self.mean) / self.std, output_hidden_states=True).hidden_states[1:]
            hs = torch.stack(hs, 0)[:, :, 1:]  # (L, B, N, Dc), CLS dropped
        w = torch.softmax(self.layer_logits, 0).to(hs.dtype)
        return self.adapter((w[:, None, None, None] * hs).sum(0))


class EffNetTokens(nn.Module):
    """Ablation spatial branch: a fine-tuned EfficientNet-B4 feature map, resampled to the token grid."""

    def __init__(self, pretrained=True, dim=256, grid=14):
        super().__init__()
        self.backbone = SpatialBranch(pretrained)
        self.grid = grid
        self.adapter = nn.Sequential(nn.LayerNorm(self.backbone.out_dim), nn.Linear(self.backbone.out_dim, dim),
                                     nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        _, fmap = self.backbone(x, return_map=True)
        fmap = F.interpolate(fmap, size=(self.grid, self.grid), mode="nearest")
        return self.adapter(fmap.flatten(2).transpose(1, 2))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention: query tokens of one domain attend to the other domain's tokens, + FFN."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.nq, self.nkv, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))

    def forward(self, q, kv):
        k = self.nkv(kv)
        q = q + self.attn(self.nq(q), k, k, need_weights=False)[0]
        return q + self.ffn(self.n2(q))


class FSDualNetV2(nn.Module):
    """
    branch:           "both" (proposed) | "freq" | "spatial"  (single-branch ablations keep the same MIL head)
    spatial_backbone: "clip" (frozen CLIP ViT, proposed) | "effnet" (fine-tuned EfficientNet-B4, ablation)
    freq_input:       "residual" (NPR residual, proposed) | "raw" (DCT of the image itself)
    moddrop:          probability that a training sample loses one modality (half spatial, half frequency)
    aux:              auxiliary frequency-only and spatial-only heads (training loss only)
    The last two stop the fusion from ignoring one branch.
    """

    def __init__(self, img_size=224, pretrained=True, branch="both", spatial_backbone="clip", freq_input="residual",
                 moddrop=0.3, aux=True, dim=256, clip_name="openai/clip-vit-base-patch16", heads=4, pretrained_effnet=None):
        super().__init__()
        pretrained_effnet = pretrained if pretrained_effnet is None else pretrained_effnet
        assert branch in ("both", "freq", "spatial") and spatial_backbone in ("clip", "effnet")
        assert freq_input in ("residual", "raw")
        self.cfg = dict(arch="v2", img_size=img_size, branch=branch, spatial_backbone=spatial_backbone,
                        freq_input=freq_input, moddrop=float(moddrop if branch == "both" else 0.0),
                        aux=bool(aux and branch == "both"), dim=dim, clip_name=clip_name, pretrained=bool(pretrained))
        self.img_size = img_size
        patch = CLIP_CONFIGS.get(clip_name, {}).get("patch_size", 16) if spatial_backbone == "clip" else 16
        self.spatial = None
        if branch in ("both", "spatial"):
            if spatial_backbone == "clip":
                self.spatial = CLIPTokens(clip_name, pretrained, dim)
                patch = self.spatial.patch
            else:
                self.spatial = EffNetTokens(pretrained_effnet, dim, img_size // patch)
        self.grid = img_size // patch
        self.freq = (PatchSpectralTokens(img_size, patch, dim, residual=freq_input == "residual", heads=heads)
                     if branch in ("both", "freq") else None)
        if branch == "both":
            self.cross_s = CrossBlock(dim, heads)  # spatial tokens attend to frequency tokens
            self.cross_f = CrossBlock(dim, heads)  # frequency tokens attend to spatial tokens
            self.gate = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 2))  # per patch: [freq, spatial]
        self.aux_f = nn.Linear(dim, 1) if self.cfg["aux"] else None
        self.aux_s = nn.Linear(dim, 1) if self.cfg["aux"] else None
        self.context = _encoder(dim, heads, 1)
        self.patch_head = nn.Linear(dim, 1)
        self.mil_V, self.mil_U, self.mil_w = nn.Linear(dim, 128), nn.Linear(dim, 128), nn.Linear(128, 1)

    def backbone_parameters(self):
        """Fine-tuned backbone (gets a smaller learning rate). CLIP is frozen and has no trainable backbone."""
        return list(self.spatial.backbone.parameters()) if isinstance(self.spatial, EffNetTokens) else []

    def forward(self, x, return_aux=False, keep_token_grad=False):
        """x: (B,3,H,W) in [0,1]. Returns logit (B,); sigmoid(logit) = P(fake)."""
        B = x.shape[0]
        f = self.freq(x) if self.freq is not None else None
        s = self.spatial(x) if self.spatial is not None else None
        aux, g = {}, None
        if self.cfg["branch"] == "both":
            keep = torch.ones(B, 2, device=x.device, dtype=f.dtype)  # [freq, spatial]
            if self.training and self.cfg["moddrop"] > 0:
                r = torch.rand(B, device=x.device)
                p = self.cfg["moddrop"]
                keep[r < p / 2, 1] = 0                     # sample loses its spatial tokens
                keep[(r >= p / 2) & (r < p), 0] = 0        # sample loses its frequency tokens
            if self.training and self.aux_f is not None:
                aux["aux_logits"] = [self.aux_f(f.mean(1)).squeeze(1), self.aux_s(s.mean(1)).squeeze(1)]
            kf, ks = keep[:, 0, None, None], keep[:, 1, None, None]
            f_in, s_in = f * kf, s.to(f.dtype) * ks
            s2 = torch.where(kf.bool(), self.cross_s(s_in, f_in), s_in)  # nothing to attend to if freq was dropped
            f2 = torch.where(ks.bool(), self.cross_f(f_in, s_in), f_in)
            g = torch.softmax(self.gate(torch.cat([f2, s2], -1)).float(), -1) * keep[:, None, :].float()
            g = g / g.sum(-1, keepdim=True).clamp_min(1e-6)
            z = g[..., :1].to(f2.dtype) * f2 + g[..., 1:].to(s2.dtype) * s2
        else:
            z = f if f is not None else s
        z = self.context(z)
        if keep_token_grad and z.requires_grad:
            z.retain_grad()
        pl = self.patch_head(z).squeeze(-1).float()  # per-patch fake logit
        att = self.mil_w(torch.tanh(self.mil_V(z)) * torch.sigmoid(self.mil_U(z))).squeeze(-1).float()
        a = torch.softmax(att, dim=1)  # gated-attention MIL (Ilse et al., ICML 2018)
        logit = (a * pl).sum(1)
        if not return_aux:
            return logit
        n = self.grid
        aux.update(patch_logits=pl.view(B, n, n), patch_attn=a.view(B, n, n), tokens=z,
                   branch_weights=None if g is None else (g * a.unsqueeze(-1)).sum(1))
        if g is not None:
            aux["gate_freq"] = g[..., 0].view(B, n, n)
        return logit, aux

    def config(self):
        return dict(self.cfg)


def build_model(cfg):
    """Rebuild a model from the config stored in a checkpoint, ready for load_model_state().
    Downloads only what the checkpoint does not contain (the frozen pretrained CLIP of a v2 model)."""
    cfg = dict(cfg)
    if cfg.get("arch") == "v2":
        keys = ("img_size", "branch", "spatial_backbone", "freq_input", "moddrop", "aux", "dim", "clip_name")
        return FSDualNetV2(pretrained=cfg.get("pretrained", True), pretrained_effnet=False,
                           **{k: cfg[k] for k in keys if k in cfg})
    if cfg.get("branch") == "resnet50":
        return DIREClassifier(img_size=cfg.get("img_size", 224), pretrained=False)
    raise ValueError(f"unsupported checkpoint config: {cfg}")


CLIP_PREFIX = "spatial.clip."


def model_state(model):
    """State dict to save. A pretrained frozen CLIP is NOT saved (it is reloaded from Hugging Face): saves ~350 MB per file."""
    sd = model.state_dict()
    cfg = getattr(model, "cfg", {})
    if cfg.get("arch") == "v2" and cfg.get("pretrained") and cfg.get("clip_name") != "tiny":
        sd = {k: v for k, v in sd.items() if not k.startswith(CLIP_PREFIX)}
    return sd


def load_model_state(model, sd):
    """Load a state dict saved by model_state(). CLIP weights may be absent only if the model downloaded them."""
    missing, unexpected = model.load_state_dict(sd, strict=False)
    clip_ok = getattr(getattr(model, "spatial", None), "from_pretrained", False)
    bad = [k for k in missing if not (clip_ok and k.startswith(CLIP_PREFIX))]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {bad[:5]}, unexpected {list(unexpected)[:5]}"
                           + (" (CLIP weights are not stored: build the model with pretrained=True)"
                              if any(k.startswith(CLIP_PREFIX) for k in bad) else ""))


def token_gradcam(model, x):
    """Grad-CAM on the fused patch tokens of a v2 model -> (B, H, W) in [0,1], P(fake) (B,)."""
    was = model.training
    model.eval()
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logit, aux = model(x, return_aux=True, keep_token_grad=True)
        logit.sum().backward()
        z, gz = aux["tokens"], aux["tokens"].grad
        w = gz.mean(1, keepdim=True)                       # (B,1,D) channel weights
        cam = F.relu((w * z).sum(-1)).detach().float()     # (B,N)
    model.zero_grad(set_to_none=True)
    aux = {k: v.detach() if torch.is_tensor(v) else v for k, v in aux.items()}
    n = model.grid
    cam = F.interpolate(cam.view(-1, 1, n, n), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    mn, mx = cam.amin(dim=(1, 2), keepdim=True), cam.amax(dim=(1, 2), keepdim=True)
    model.train(was)
    return (cam - mn) / (mx - mn + 1e-8), torch.sigmoid(logit).detach(), aux


if __name__ == "__main__":
    # smoke test over all ablation variants (no pretrained download)
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.rand(2, 3, 224, 224)
    model = DIREClassifier(224, pretrained=False).eval()
    print(f"DIRE classifier params {n(model):6.2f}M | logit {tuple(model(x).shape)}")
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        print("v2 skipped: pip install transformers")
    else:
        for kw in [dict(), dict(branch="freq"), dict(branch="spatial"), dict(spatial_backbone="effnet"), dict(freq_input="raw")]:
            model = FSDualNetV2(pretrained=False, clip_name="tiny", **kw).eval()
            cam, prob, aux = token_gradcam(model, x)
            print(f"v2 {str(kw):34s} trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:5.2f}M | "
                  f"evidence {tuple(aux['patch_logits'].shape)} | cam {tuple(cam.shape)}")
