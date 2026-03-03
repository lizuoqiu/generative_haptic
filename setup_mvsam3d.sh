#!/usr/bin/env bash
set -euo pipefail

# MV-SAM3D full environment setup (CUDA 12.1 / PyTorch 2.5.1+cu121)
# Tested on Linux with NVIDIA driver >= 12.1 support.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROOT_DIR}/MV-SAM3D"
MINICONDA_DIR="${HOME}/miniconda3"
ENV_NAME="sam3d-objects"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-}"  # Auto-detect if empty; override via env, e.g. CUDA_ARCH_LIST=8.6
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
KAOLIN_WHL_INDEX="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
GDOWN_FOLDER_URL="https://drive.google.com/drive/folders/1PXaSvmke0_vZncvR5dFJKTDJvJzRuBSY?usp=drive_link"
GDOWN_OUT_DIR="${ROOT_DIR}/downloads/generative_haptic_dataset"
GDOWN_COOKIES_SRC="${ROOT_DIR}/drive.google.com_cookies.txt"

log() { echo "[setup] $*"; }

DETECTED_GPU_NAME=""
detect_cuda_arch_list() {
  local gpu=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
    DETECTED_GPU_NAME="${gpu}"
  fi

  case "${gpu}" in
    *H100*|*H200*) echo "9.0" ;;
    *A100*) echo "8.0" ;;
    *A6000*|*A5000*|*A40*|*A30*|*A16*|*A10*|*A2*|*RTX\ 30*|*3090*|*3080*|*3070*|*3060*) echo "8.6" ;;
    *4090*|*4080*|*4070*|*4060*|*RTX\ 40*|*L4*|*L40*|*L40S*) echo "8.9" ;;
    *T4*) echo "7.5" ;;
    *V100*) echo "7.0" ;;
    *P100*) echo "6.0" ;;
    *K80*) echo "3.7" ;;
    *) echo "8.6" ;;
  esac
}

ensure_gdown_cookies() {
  if [ -f "${GDOWN_COOKIES_SRC}" ]; then
    log "Installing gdown cookies from ${GDOWN_COOKIES_SRC}"
    mkdir -p "${HOME}/.cache/gdown"
    GDOWN_COOKIES_SRC="${GDOWN_COOKIES_SRC}" python - <<'PY'
from pathlib import Path
import os

src = Path(os.environ["GDOWN_COOKIES_SRC"])
dst = Path.home() / ".cache" / "gdown" / "cookies.txt"

lines = src.read_text().splitlines()
out = []
fixed = 0
skipped = 0

for line in lines:
    if not line or line.startswith("#"):
        out.append(line)
        continue
    parts = line.split("\t")
    if len(parts) != 7:
        skipped += 1
        continue
    domain, include_subdomains, path, secure, expires, name, value = parts
    if include_subdomains.upper() == "TRUE" and not domain.startswith("."):
        domain = "." + domain
        fixed += 1
    out.append("\t".join([domain, include_subdomains, path, secure, expires, name, value]))

dst.write_text("\n".join(out) + "\n")
print(f"[setup] Wrote cookies to {dst} (fixed={fixed}, skipped={skipped})")
PY
  else
    log "Missing cookies file at ${GDOWN_COOKIES_SRC}"
    exit 1
  fi
}

sanitize_gdown_cookies() {
  python - <<'PY'
from pathlib import Path

path = Path.home() / ".cache" / "gdown" / "cookies.txt"
if not path.exists():
    print("[setup] No gdown cookies file to sanitize")
    raise SystemExit(0)

lines = path.read_text().splitlines()
out = []
fixed = 0
skipped = 0

for line in lines:
    if not line or line.startswith("#"):
        out.append(line)
        continue
    parts = line.split("\t")
    if len(parts) != 7:
        skipped += 1
        continue
    domain, include_subdomains, pth, secure, expires, name, value = parts
    if include_subdomains.upper() == "TRUE" and not domain.startswith("."):
        domain = "." + domain
        fixed += 1
    out.append("\t".join([domain, include_subdomains, pth, secure, expires, name, value]))

path.write_text("\n".join(out) + "\n")
print(f"[setup] Sanitized gdown cookies (fixed={fixed}, skipped={skipped})")
PY
}

# 1) System prereqs
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  log "Installing system prerequisites (git, curl)"
  sudo apt-get update
  sudo apt-get install -y git curl
fi

# 1.1) Detect GPU arch (if not provided)
if [ -z "${CUDA_ARCH_LIST}" ]; then
  CUDA_ARCH_LIST="$(detect_cuda_arch_list)"
  log "Detected GPU: ${DETECTED_GPU_NAME:-unknown}; using CUDA_ARCH_LIST=${CUDA_ARCH_LIST}"
else
  log "Using CUDA_ARCH_LIST from env: ${CUDA_ARCH_LIST}"
fi

# 2) Miniconda install
if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
  log "Installing Miniconda to ${MINICONDA_DIR}"
  cd "${HOME}"
  curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_DIR}"
  rm -f Miniconda3-latest-Linux-x86_64.sh
fi

# 3) Init conda for this shell
# Some conda activation scripts reference unset vars; disable nounset temporarily.
# shellcheck disable=SC1090
set +u
source "${MINICONDA_DIR}/bin/activate"
set -u

# 4) Accept Anaconda TOS (avoids non-interactive errors)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 5) Clone repo
if [ ! -d "${REPO_DIR}/.git" ]; then
  log "Cloning MV-SAM3D"
  git clone https://github.com/devinli123/MV-SAM3D.git
else
  log "MV-SAM3D already exists at ${REPO_DIR}"
fi

# 6) Create conda env (skip if exists)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda env ${ENV_NAME} already exists; skipping create"
else
  log "Creating conda env ${ENV_NAME}"
  conda env create -f "${REPO_DIR}/environments/default.yml"
fi

# 7) Activate env
set +u
conda activate "${ENV_NAME}"
set -u

# 8) Install PyTorch (CUDA 12.1)
log "Installing PyTorch 2.5.1+cu121"
pip install --index-url "${TORCH_INDEX_URL}" \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

# 9) Install core requirements
log "Installing base requirements"
pip install -r "${REPO_DIR}/requirements.txt"

# 9.1) Install Google Drive cookies from workspace
ensure_gdown_cookies
sanitize_gdown_cookies

# 9.2) Download Google Drive folder (resume-friendly)
log "Downloading Google Drive folder with gdown"
mkdir -p "${GDOWN_OUT_DIR}"
set +e
attempt=1
while true; do
  sanitize_gdown_cookies
  gdown --folder --remaining-ok --continue -O "${GDOWN_OUT_DIR}/" "${GDOWN_FOLDER_URL}"
  status=$?
  if [ "${status}" -eq 0 ]; then
    break
  fi
  if [ "${attempt}" -ge 3 ]; then
    log "gdown failed after ${attempt} attempts"
    exit "${status}"
  fi
  log "gdown failed (attempt ${attempt}); retrying in 10s..."
  attempt=$((attempt + 1))
  sleep 10
done
set -e

# 10) Build PyTorch3D + FlashAttention
log "Building pytorch3d + flash_attn"
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export CUDA_HOME="${CONDA_PREFIX}"
pip install -r "${REPO_DIR}/requirements.p3d.txt"

# 11) Kaolin + inference requirements
log "Installing kaolin from NVIDIA wheel index"
pip install kaolin==0.17.0 -f "${KAOLIN_WHL_INDEX}"

log "Installing inference requirements"
pip install -r "${REPO_DIR}/requirements.inference.txt"

# 12) Quick sanity check
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('is_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY

log "Setup complete"
