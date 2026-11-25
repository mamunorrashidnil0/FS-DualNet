# FS-DualNet

**Explainable Cross-Domain Detection of Diffusion-Generated Faces Using Frequency-Spatial Feature Fusion**

FS-DualNet detects face images made by diffusion models (Stable Diffusion, DALL-E 2, Midjourney, ...).
It looks at each face in two ways, a frequency view and a spatial view, fuses them patch by patch,
and shows which regions of the face drove its decision.

Base paper: Wang et al., *DIRE for Diffusion-Generated Image Detection*, ICCV 2023.

## Results against DIRE

Train on CelebA-HQ + Stable Diffusion 2, test on generators the models never saw (DALL-E 2, Midjourney).
Mean of 3 seeds (42, 1, 2). A result counts as "better" only if it is at least 1 point ahead in every seed.

| Test (AUC unless noted) | FS-DualNet | DIRE | Verdict |
|---|---|---|---|
| Clean images, full test sets | 98.99 ± 0.86 | 99.82 ± 0.11 | tie |
| Clean images, decision accuracy at the dev threshold | 91.93% | 97.87% | DIRE better |
| JPEG quality 65 | **98.92** | 90.29 | FS-DualNet better |
| JPEG quality 30 | **98.52** | 90.06 | FS-DualNet better |
| Gaussian blur, sigma 3 | **97.66** | 66.92 | FS-DualNet better |
| Second dataset: DeepFakeFace | **64.79** | 53.69 | FS-DualNet better |
| Time per image, T4 GPU, batch 1 | **17.3 ms** | 9.51 s | about 550x faster |
| Explanation maps | yes | no | |

DIRE was reproduced with the authors' code and checkpoint (ADM 256x256, DDIM 20 + 20 steps).
Our maps match the official maps of the same fake images within 0.05 / 255.
The official DIRE maps store real images as JPEG and fake images as PNG; we also trained a format-fair
DIRE (both classes in one format) and always compared against the better of the two DIRE versions.

## Model

```
image 224x224 ─┬─ NPR residual ─ 16x16 patch DCT ─ token encoder ──► 196 frequency tokens
               └─ frozen CLIP ViT-B/16 ─ mix of 12 layers ─ adapter ─► 196 spatial tokens
both ─ bidirectional cross-attention ─ per-patch gate ─ transformer layer
     ─┬─ per-patch real/fake score  (evidence map)
      └─ gated attention pooling    ─► image score (real / fake)
```

- **NPR** (Neighboring Pixel Relationships) residual: `x - up(down(x))` removes the face content and keeps up-sampling traces.
- **Per-patch gate:** for each of the 196 patches the model learns how much to trust the frequency and the spatial branch (modality map).
- **Training:** random resize / blur / JPEG augmentation on real and fake images, modality dropout (0.3),
  auxiliary losses on each branch (weight 0.3), AdamW, 3% warm-up then cosine, gradient clipping 1.0.
- Only 2.76 M of the 88.6 M parameters are trained (CLIP is frozen).
- **Explanations:** evidence map, attention map, modality map, token Grad-CAM.

## Repository layout

| File | Purpose |
|---|---|
| `fs_dualnet.py` | FS-DualNet (NPR residual, patch DCT tokens, CLIP tokens, fusion, token Grad-CAM) and the DIRE ResNet-50 classifier |
| `train_eval.py` | Training and evaluation of one model (augmentation, dev threshold, AUC / AP / ACC, robustness tests); `--branch resnet50` trains the DIRE baseline |
| `run_all.py` | Runs every experiment and ablation, resumable, one output folder per seed |
| `metrics_utils.py` | Shared metric helpers |
| `explain.py` | Explanation figure: input, token Grad-CAM, patch evidence, modality map |
| `aggregate_seeds.py`, `claims.py`, `make_table.py` | Mean ± std over seeds and the "≥ 1 point in every seed" rule |
| `benchmark_latency.py` | Speed per image |
| `dire_compute.py` | DIRE maps with the official recipe (ADM checkpoint, DDIM 20 + 20) |
| `make_dire_subsets.py` | Picks the shared image subsets for the DIRE comparison |
| `eval_ckpt.py` | Evaluates saved checkpoints under clean / JPEG / blur conditions |
| `compare_v2_dire.py` | FS-DualNet vs DIRE table |
| `prep_data.py`, `prep_dff.py`, `make_subset.py`, `extract_first_n.py`, `cache_resize.py` | Dataset preparation |

## Kaggle notebooks

| Notebook | What it does |
|---|---|
| `FS_DualNet_Kaggle_v2.ipynb` | FS-DualNet, 3 seeds, all ablations (23 runs) |
| `FS_DualNet_Kaggle_vsDIRE.ipynb` | Fair comparison with DIRE: recomputed maps, JPEG / blur stress tests, DeepFakeFace, speed |

## Data (not included)

- **DiffusionForensics** (released with DIRE), CelebA-HQ part: real faces and fakes from Stable Diffusion 2,
  DeepFloyd IF, DALL-E 2 and Midjourney. Train: Stable Diffusion 2. Dev (threshold): IF. Test: DALL-E 2, Midjourney.
- **DeepFakeFace** (Hugging Face: OpenRL/DeepFakeFace): real faces from IMDB-WIKI, fakes from Stable Diffusion
  text-to-image, inpainting and InsightFace.

Each dataset has its own license; download it from the source.

## Quick start

```bash
pip install -r requirements.txt     # install PyTorch first: https://pytorch.org/get-started/locally
python fs_dualnet.py                # smoke test
python run_all.py --help            # all options
```

Example (one seed of the v2 cross-generator experiment):

```bash
python run_all.py --data <prepared_celebahq> --train_gen sdv2 --runs v2 \
    --dev if=<path_to_if_images> --seed 42 --out_dir runs/cb_s42
```

The Kaggle notebooks contain the exact commands used for every result above.

## Citation of the base paper

```
@inproceedings{wang2023dire,
  title     = {DIRE for Diffusion-Generated Image Detection},
  author    = {Wang, Zhendong and Bao, Jianmin and Zhou, Wengang and Wang, Weilun and Hu, Hezhen and Chen, Hong and Li, Houqiang},
  booktitle = {ICCV},
  year      = {2023}
}
```
