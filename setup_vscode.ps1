# One-shot setup for VS Code on Windows (CPU build of PyTorch).
# Run from the fs_dualnet folder:
#     powershell -ExecutionPolicy Bypass -File .\setup_vscode.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($msg, [scriptblock]$cmd) {
    Write-Host "`n== $msg" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
}

$py = $null
foreach ($v in @("3.12", "3.11", "3.13")) {
    try {
        & py "-$v" -c "print(1)" | Out-Null
        if ($LASTEXITCODE -eq 0) { $py = $v; break }
    } catch { }
}
if (-not $py) {
    Write-Host "Python 3.12 (or 3.11 / 3.13) was not found." -ForegroundColor Red
    Write-Host "Install Python 3.12 from python.org (tick 'Add python.exe to PATH'), then run this script again."
    exit 1
}
Write-Host "Using Python $py"

Step "create virtual environment (.venv)" { & py "-$py" -m venv .venv }
$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Step "upgrade pip" { & $PY -m pip install --upgrade pip }
Step "install PyTorch (CPU build)" { & $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu }
Step "install the other libraries" { & $PY -m pip install scikit-learn pillow numpy opencv-python-headless matplotlib transformers }
Step "check PyTorch" { & $PY -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" }
Step "smoke test: every model variant builds and runs" { & $PY fs_dualnet.py }
Write-Host "`nSetup finished. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv" -ForegroundColor Green
