# ---------- Python / environment ----------
.venv/
venv/
env/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/

# ---------- Model weights, checkpoints, archives (too big for GitHub) ----------
*.pt
*.pth
*.ckpt
*.onnx
*.zip
*.7z
*.rar
*.tar
*.gz
*.tgz
*.crdownload
*.part
*.aria2

# ---------- Datasets and experiment outputs (never commit data) ----------
data/
data256/
data_df/
data_df256/
dff/
DiffusionForensics/
DiffusionForensics*/
Images/
runs*/
upload/
backup/
*.log

# ---------- VS Code: keep only the shared settings ----------
.vscode/*
!.vscode/settings.json
!.vscode/extensions.json
!.vscode/launch.json

# ---------- OS junk ----------
Thumbs.db
desktop.ini
.DS_Store
