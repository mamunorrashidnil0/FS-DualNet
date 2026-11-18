# FS-DualNet
FS-DualNet: frequency-spatial dual-branch network (FFT/DCT + EfficientNet-B4, attention fusion) for explainable cross-domain detection of diffusion-generated faces, with Grad-CAM localization.
# FS-DualNet

Frequency-Spatial Dual-Branch Network for explainable cross-domain detection of
diffusion-generated faces.

- **Frequency branch:** FFT + DCT spectra -> CNN -> spectral artifact map
- **Spatial branch:** EfficientNet-B4 -> global average pooling
- **Attention fusion** of both feature vectors -> binary head (real / fake)
- **Grad-CAM** to localize forged facial regions
- No diffusion reconstruction at inference (unlike DIRE)

Base paper: Wang et al., "DIRE for Diffusion-Generated Image Detection", ICCV 2023.

Status: work in progress (thesis). Results will be added after the experiments finish.

## Datasets (not included)
DeepFakeFace (Hugging Face: OpenRL/DeepFakeFace), DiffusionForensics (DIRE authors' release).
Each dataset has its own license; download it from the source.

## Quick start
pip install -r requirements.txt
python fs_dualnet.py            # smoke test
python run_all.py --help        # run all experiments (resumable)