#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip --upgrade
fi
# Keep CUDA 12.8 / Blackwell support; Pi3's torch 2.5 pin is not needed for inference.
.venv/bin/python -m pip install --index-url https://pypi.org/simple torch==2.8.0 torchvision==0.23.0
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt
.venv/bin/python - <<'PY'
import torch
import open3d
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Open3D:', open3d.__version__)
if torch.cuda.is_available():
    x = torch.randn(32, 32, device='cuda')
    assert torch.isfinite(x @ x.T).all()
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('CUDA unavailable; inference will use CPU with --device auto.')
PY
# Default all three models; optional arguments: bash tools/setup_env.sh vggt pi3
.venv/bin/python tools/download_models.py --models "${@:-all}"
printf '%s\n' 'Ready: source .venv/bin/activate'
