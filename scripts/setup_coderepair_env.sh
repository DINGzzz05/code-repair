#!/usr/bin/env bash
#
# setup_coderepair_env.sh
#
# 在 Ubuntu 服务器上创建 conda 环境 /data/dzz/envs/coderepair 并安装
# CodeRepairRL（agentic-evals-lab）全部依赖，最后做导入自检。
#
# 用法（在服务器上执行）：
#   bash setup_coderepair_env.sh [PROJECT_DIR]
#
#   PROJECT_DIR 项目代码位置，默认 /data/dzz/coderepair；
#               若不存在则从 GIT_URL 克隆。
#
# 可用环境变量覆盖：
#   ENV_DIR         conda 环境路径（默认 /data/dzz/envs/coderepair）
#   PY_VER          Python 版本（默认 3.11，项目要求 >=3.11,<3.13）
#   GIT_URL         项目仓库地址（默认 https://github.com/DINGzzz05/code-repair.git）
#   TORCH_VERSION   强制指定 torch 版本（默认按 NVIDIA 驱动自动选择）
#   TORCH_INDEX     强制指定 torch 下载源（如 .../whl/cu126）
#
set -euo pipefail

ENV_DIR="${ENV_DIR:-/data/dzz/envs/coderepair}"
PY_VER="${PY_VER:-3.11}"
PROJECT_DIR="${1:-/data/dzz/coderepair}"
GIT_URL="${GIT_URL:-https://github.com/DINGzzz05/code-repair.git}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/dzz/pip_cache}"

echo "=============================================="
echo " CodeRepairRL env setup"
echo "   env      : ${ENV_DIR}"
echo "   python   : ${PY_VER}"
echo "   project  : ${PROJECT_DIR}"
echo "   pip cache: ${PIP_CACHE_DIR}"
echo "=============================================="

# ---------- 0. 前置检查 ----------
command -v conda >/dev/null 2>&1 || { echo "[ERROR] conda not found"; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "[ERROR] git not found"; exit 1; }

echo "[1/8] preflight"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "[WARN] nvidia-smi not found; assuming CUDA 12.8 toolchain"
fi
df -h /data/dzz | tail -1

# 按驱动版本选择 torch：CUDA 12.8 需要驱动 >= 570.28；否则退回 12.6
TORCH_VERSION="${TORCH_VERSION:-}"
TORCH_INDEX="${TORCH_INDEX:-}"
LOCK_FLAG="--locked"
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_MM=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | awk -F'.' '{print $1"."$2}')
  echo "  driver: ${DRIVER_MM}"
  if awk "BEGIN{exit !(${DRIVER_MM} >= 570.28)}" 2>/dev/null; then
    TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  else
    TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
    LOCK_FLAG="--no-locked"   # uv.lock 锁的是 torch 2.8.0，需重新解析
    echo "  [WARN] driver < 570.28 -> torch ${TORCH_VERSION}+cu126 (uv.lock 会重新解析)"
  fi
else
  TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
fi
echo "  torch: ${TORCH_VERSION} (${TORCH_INDEX})"

# ---------- 1. 创建 conda 环境 ----------
echo "[2/8] creating conda env ${ENV_DIR}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# conda >= 24.5 要求先接受 Anaconda 默认源（repo.anaconda.com）的 ToS，
# 否则 conda create 会以 CondaToSNonInteractiveError 退出。
# 接受一次即写入配置，之后不再询问；CONDA_PLUGINS_AUTO_ACCEPT_TOS 兜底。
conda tos accept --override-channels --channel defaults >/dev/null 2>&1 || true
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

if [ -f "${ENV_DIR}/conda-meta/history" ]; then
  echo "  env already exists at ${ENV_DIR}; reusing it"
else
  conda create -y -p "${ENV_DIR}" "python=${PY_VER}"
fi
conda activate "${ENV_DIR}"
python -m pip install --upgrade pip
pip install "setuptools>=75.8.0" wheel uv ninja cmake

# ---------- 2. 项目代码 ----------
echo "[3/8] project code"
if [ ! -f "${PROJECT_DIR}/pyproject.toml" ]; then
  echo "  project not found at ${PROJECT_DIR}; cloning ${GIT_URL}"
  mkdir -p "$(dirname "${PROJECT_DIR}")"
  git clone "${GIT_URL}" "${PROJECT_DIR}"
fi
cd "${PROJECT_DIR}"

# ---------- 3. torch（必须先于 flash-attn/flashinfer 安装） ----------
echo "[4/8] installing torch ${TORCH_VERSION}"
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# flash-attn 编译需要 nvcc；优先用 pip 装（绕开 conda nvidia 频道的 EULA/ToS），
# 其次才是 conda 的 cuda-toolkit
if ! command -v nvcc >/dev/null 2>&1; then
  echo "  nvcc not found; installing nvidia-cuda-nvcc via pip"
  pip install "nvidia-cuda-nvcc-cu12"
  NVCC_BIN="$(find "${ENV_DIR}" -path '*/cuda_nvcc/bin/nvcc' -type f 2>/dev/null | head -1)"
  if [ -n "${NVCC_BIN}" ]; then
    export PATH="$(dirname "${NVCC_BIN}"):${PATH}"
  else
    echo "  [WARN] pip nvcc not located; trying conda nvidia channel..."
    CUDA_VER=$(echo "${TORCH_INDEX}" | grep -o 'cu[0-9]*' | head -1 | sed 's/cu//')
    CUDA_DOT="${CUDA_VER:0:2}.${CUDA_VER:2}"
    conda install -y -c nvidia "cuda-toolkit=${CUDA_DOT}"
    export PATH="${ENV_DIR}/bin:${PATH}"
  fi
  command -v nvcc >/dev/null 2>&1 || { echo "[ERROR] nvcc still unavailable"; exit 1; }
fi
echo "  nvcc: $(command -v nvcc) $("${ENV_DIR}/bin/nvcc" --version 2>/dev/null | tail -1 || nvcc --version | tail -1)"

# ---------- 4. fused kernels（对已装 torch 编译，约 10-30 分钟） ----------
echo "[5/8] building flash-attn 2.8.3 + flashinfer 0.5.0"
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3
pip install --no-build-isolation --no-cache-dir flashinfer-python==0.5.0

# ---------- 5. 项目依赖（装进当前 conda 环境） ----------
echo "[6/8] uv sync (${LOCK_FLAG})"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
if ! uv sync --extra gpu ${LOCK_FLAG}; then
  echo "  [WARN] uv sync failed with ${LOCK_FLAG}; retrying without lock flag..."
  uv sync --extra gpu
fi

# ---------- 6. 验证 ----------
echo "[7/8] python import check"
python - <<'PY'
import torch
print("  torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
for mod_name in ("vllm", "flash_attn", "flashinfer", "deepspeed",
                 "trl", "liger_kernel", "accelerate", "peft",
                 "datasets", "swebench"):
    try:
        mod = __import__(mod_name)
        print(f"  {mod_name}:", getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(f"  {mod_name}: FAILED -> {exc}")
        raise
print("  imports OK")
PY

echo "[8/8] project import check"
uv run python -c "from src.rewards.diff import unified_diff_similarity_reward_func; from src.trainers.curriculum import CurriculumConfig; from src.live_difficulty import LiveDifficultyState; print('project imports OK')"

echo ""
echo "=============================================="
echo " DONE. 日常使用："
echo "   conda activate ${ENV_DIR}"
echo "   训练脚本请用 uv run 执行（scripts/4090/ 或 scripts/grpo/）"
echo "=============================================="
