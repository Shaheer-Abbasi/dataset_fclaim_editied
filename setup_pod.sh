#!/usr/bin/env bash
# RunPod / fresh-env setup for UAP_attack_UBWC.py experiments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> Repo: $REPO_ROOT"
echo "==> Python: $(python3 --version)"

echo "==> Restoring tracked src/ and forest/ (if missing or deleted on disk)"
git restore src forest 2>/dev/null || true
if [[ ! -d src ]] || [[ ! -d forest ]]; then
  echo "ERROR: src/ or forest/ still missing. Run: git checkout HEAD -- src forest"
  exit 1
fi

echo "==> Installing Python dependencies"
# README pins target py3.9; RunPod often has 3.10/3.11 — use flexible versions.
python3 -m pip install -U pip setuptools wheel
python3 -m pip install \
  torch torchvision \
  numpy scipy pillow scikit-learn matplotlib \
  open-clip-torch noise

echo "==> Checking imports"
python3 << 'PY'
import forest
import open_clip
from noise import pnoise2
import torch
print("imports OK, torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "==> Checking CIFAR-100 data layout"
for path in \
  "data/cifar100/train" \
  "data/cifar100/test" \
  "data/cifar100/target(0)/target_model.pth"
do
  if [[ ! -e "$path" ]]; then
    echo "WARNING: missing $path"
  else
    echo "OK: $path"
  fi
done

MODEL="data/cifar100/target(0)/target_model.pth"
if [[ -L "$MODEL" ]] || [[ ! -s "$MODEL" ]]; then
  echo "WARNING: target_model.pth is missing, empty, or a symlink."
  echo "         Retrain with: python3 train_target_model.py --exp_index 0"
  echo "         Or copy a real checkpoint into $MODEL"
fi

mkdir -p logs data/model

if [[ ! -d data/model ]] || [[ -z "$(ls -A data/model 2>/dev/null || true)" ]]; then
  echo "==> Pre-downloading CLIP weights to ./data/model/ (needs network once)"
  python3 << 'PY'
import open_clip
open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='datacomp_xl_s13b_b90k', cache_dir='./data/model/'
)
print('CLIP cache ready under ./data/model/')
PY
  echo "Tip: export HF_HUB_OFFLINE=1 for later runs if downloads are cached."
fi

echo "==> Smoke test (1 trial, dataset-free mode; needs GPU + valid checkpoint)"
echo "Run manually:"
echo "  export HF_HUB_OFFLINE=1   # optional, if CLIP already cached"
echo "  python3 UAP_attack_UBWC.py --net ResNet18 --dataset CIFAR100 \\"
echo "    --attack_method perlin_noaccess --patch_size 8 --num_trials 1 \\"
echo "    --use_tmp --tmp_dir /tmp/uap_run_noaccess"

echo "==> Setup checks finished."
