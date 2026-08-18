#!/usr/bin/env bash
# ============================================================
# check_server.sh — 服务器配置一键检查
# 适用：Slurm 登录/计算节点、裸机 Ubuntu GPU 服务器
# 用法：bash scripts/check_server.sh
# 可覆盖环境变量：PROJECT_DIR / ENV_DIR / CRRL_WORKDIR / IMAGE_DIR
# 退出码：0 = 无 FAIL；1 = 存在 FAIL
# ============================================================
set -u

PROJECT_DIR="${PROJECT_DIR:-}"
ENV_DIR="${ENV_DIR:-/data/dzz/envs/coderepair}"
IMAGE_DIR="${IMAGE_DIR:-}"

# 自动探测项目目录（当前目录 → HOME → /data/dzz → /proj/*）
if [ -z "$PROJECT_DIR" ]; then
  for d in "$PWD" "$HOME/coderepair" /data/dzz/coderepair /proj/*/users/*/CodeRepairRL; do
    [ -f "$d/pyproject.toml" ] && PROJECT_DIR="$d" && break
  done
fi
[ -n "$PROJECT_DIR" ] || PROJECT_DIR="/data/dzz/coderepair"
[ -n "$IMAGE_DIR" ] || IMAGE_DIR="$PROJECT_DIR"

PASS=0; WARN=0; FAIL=0

ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "  [INFO] $*"; }
sec()  { echo; echo "===== $* ====="; }

echo "=============================================="
echo " Server config check  $(date '+%Y-%m-%d %H:%M:%S')"
echo " host: $(hostname)  user: ${USER:-$(id -un)}"
echo "=============================================="

sec "1. 系统信息"
info "hostname : $(hostname)"
[ -f /etc/os-release ] && info "os       : $(awk -F= '/PRETTY_NAME/{gsub(/"/,"",$2);print $2}' /etc/os-release)"
info "kernel   : $(uname -r)"
info "uptime   : $(uptime -p 2>/dev/null || uptime)"
info "load     : $(cut -d' ' -f1-3 /proc/loadavg)  (1/5/15 min)"
info "cpu      : $(nproc) cores  $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"

sec "2. 内存"
free -h | awk 'NR==1 || /^Mem:/'
MEM_TOTAL=$(awk '/MemTotal/{print $2}' /proc/meminfo)
MEM_AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "${MEM_TOTAL:-0}" -gt 0 ] 2>/dev/null; then
  PCT=$((100 * MEM_AVAIL / MEM_TOTAL))
  [ "$PCT" -lt 10 ] && fail "可用内存仅 ${PCT}%（${MEM_AVAIL} kB / ${MEM_TOTAL} kB）"
  [ "$PCT" -ge 10 ] && [ "$PCT" -lt 20 ] && warn "可用内存偏低 ${PCT}%"
  [ "$PCT" -ge 20 ] && ok "可用内存 ${PCT}%"
fi

sec "3. 磁盘"
info "检查目录: / , /data , /proj , ${PROJECT_DIR} , ${CRRL_WORKDIR:-未设置}"
for d in / /data /proj "$PROJECT_DIR" "${CRRL_WORKDIR:-}"; do
  [ -d "$d" ] || continue
  AVAIL_GB=$(df -B1G --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
  LINE=$(df -h "$d" 2>/dev/null | tail -1)
  if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt 20 ] 2>/dev/null; then
    fail "磁盘不足: $d 剩余 ${AVAIL_GB}G"
  elif [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt 100 ] 2>/dev/null; then
    warn "磁盘偏少: $d 剩余 ${AVAIL_GB}G（SWE-Gym 镜像建议预留 300-500G）"
  else
    ok "磁盘正常: $d (${LINE})"
  fi
done
SIF_COUNT=$(find "$IMAGE_DIR" -maxdepth 2 -name '*.sif' 2>/dev/null | wc -l)
SIF_SIZE=$(find "$IMAGE_DIR" -maxdepth 2 -name '*.sif' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1073741824}')
[ "$SIF_COUNT" -gt 0 ] && info "SWE-Gym/Apptainer 镜像: ${SIF_COUNT} 个, 共 ${SIF_SIZE} GB (${IMAGE_DIR})" || warn "未在 ${IMAGE_DIR} 找到 .sif 镜像（SWE-Gym 评测需要）"

sec "4. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L | wc -l)
  info "GPU 数量: ${GPU_COUNT}"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
  DRIVER_MM=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1,2)
  if awk "BEGIN{exit !(${DRIVER_MM} >= 570.28)}" 2>/dev/null; then
    ok "NVIDIA 驱动 ${DRIVER_MM} >= 570.28，支持 torch 2.8.0+cu128"
  else
    warn "NVIDIA 驱动 ${DRIVER_MM} < 570.28，建议 torch 2.7.0+cu126（setup_coderepair_env.sh 会自动处理）"
  fi
  MIN_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if [ -n "$MIN_FREE" ]; then
    [ "$MIN_FREE" -ge 2048 ] && ok "GPU 显存空闲充足（最小 ${MIN_FREE} MiB）" || warn "有 GPU 空闲显存 < 2G（最小 ${MIN_FREE} MiB），训练前注意"
  fi
  echo "  [INFO] GPU 占用进程:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/    /'
else
  fail "nvidia-smi 未找到，无法检测 GPU（无 NVIDIA 驱动？）"
fi

sec "5. 软件环境"
command -v conda >/dev/null 2>&1 && ok "conda: $(conda --version)" || warn "conda 未找到"
command -v uv >/dev/null 2>&1 && ok "uv: $(uv --version | cut -d' ' -f2)" || warn "uv 未找到"
command -v git >/dev/null 2>&1 && ok "git: $(git --version)" || fail "git 未找到"
if [ -x "$ENV_DIR/bin/python" ]; then
  info "conda env: ${ENV_DIR}"
  "$ENV_DIR/bin/python" -V 2>/dev/null | sed 's/^/    /'
  if "$ENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
    "$ENV_DIR/bin/python" -c "import torch; print('  [INFO] torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"
    "$ENV_DIR/bin/python" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 && ok "torch CUDA 可用" || fail "torch 检测不到 CUDA"
  else
    fail "torch 未安装或导入失败（env ${ENV_DIR}）"
  fi
  "$ENV_DIR/bin/python" -c "import flash_attn; print('  [INFO] flash-attn', flash_attn.__version__)" >/dev/null 2>&1 && "$ENV_DIR/bin/python" -c "import flash_attn; print('  [INFO] flash-attn', flash_attn.__version__)" || warn "flash-attn 未安装（需要时用 uv sync --extra gpu）"
  VLLM_VER=$("$ENV_DIR/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null)
  [ -n "$VLLM_VER" ] && ok "vllm: ${VLLM_VER}" || warn "vllm 未安装"
else
  warn "未找到 conda env ${ENV_DIR}（可用 ENV_DIR=... 覆盖），跳过 torch/vllm 检查"
fi

sec "6. 容器 / 镜像"
CTR=""
command -v apptainer >/dev/null 2>&1 && CTR="apptainer"
command -v singularity >/dev/null 2>&1 && CTR="${CTR:+$CTR/}singularity"
command -v docker >/dev/null 2>&1 && CTR="${CTR:+$CTR/}docker"
[ -n "$CTR" ] && ok "容器运行时: ${CTR}" || warn "未找到 apptainer/singularity/docker（裸机 conda 环境可忽略）"
CRRL_SIF=$(find "$PROJECT_DIR" "$HOME" -maxdepth 2 -name 'crrl.sif' 2>/dev/null | head -1)
[ -n "$CRRL_SIF" ] && ok "crrl.sif: ${CRRL_SIF}" || warn "未找到 crrl.sif（Slurm 任务需要；裸机 conda 环境可忽略）"

sec "7. 认证 / 密钥"
[ -n "${HF_TOKEN:-}" ] && ok "HF_TOKEN 已设置" || warn "HF_TOKEN 未设置（拉模型/数据集需要）"
[ -n "${WANDB_API_KEY:-}" ] && ok "WANDB_API_KEY 已设置" || warn "WANDB_API_KEY 未设置（wandb 上报需要）"
if command -v huggingface-cli >/dev/null 2>&1; then
  HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
  [ -n "$HF_USER" ] && ok "HF 已登录: ${HF_USER}" || warn "huggingface-cli 未登录（GRPO 入口会强制 whoami）"
else
  warn "huggingface-cli 未安装"
fi

sec "8. 运行中的任务"
PROCS=$(pgrep -af 'vllm|src\.train|trl' 2>/dev/null || true)
if [ -n "$PROCS" ]; then
  warn "检测到训练/推理进程（注意显存占用）:"
  echo "$PROCS" | sed 's/^/    /'
else
  ok "未检测到训练/推理进程"
fi
if command -v sinfo >/dev/null 2>&1; then
  info "Slurm 分区 (sinfo -s):"
  sinfo -s 2>/dev/null | sed 's/^/    /'
  info "我的任务 (squeue -u ${USER:-$(id -un)}):"
  squeue -u "${USER:-$(id -un)}" 2>/dev/null | sed 's/^/    /'
fi

sec "9. 项目 / 路径"
[ -f "$PROJECT_DIR/pyproject.toml" ] && ok "项目代码: ${PROJECT_DIR}" || fail "未找到项目 pyproject.toml（PROJECT_DIR=${PROJECT_DIR}）"
if [ -n "${CRRL_WORKDIR:-}" ]; then
  [ -d "$CRRL_WORKDIR" ] && ok "CRRL_WORKDIR 存在: ${CRRL_WORKDIR}" || warn "CRRL_WORKDIR 未创建: ${CRRL_WORKDIR}"
else
  warn "CRRL_WORKDIR 未设置（Slurm/Apptainer 任务需要）"
fi

echo
echo "----------------------------------------------"
echo "结果汇总: PASS=${PASS}  WARN=${WARN}  FAIL=${FAIL}"
if [ "$FAIL" -gt 0 ]; then
  echo "结论: ❌ 有 ${FAIL} 项需要处理"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo "结论: ⚠️ 基本正常，${WARN} 项建议关注"
  exit 0
else
  echo "结论: ✅ 配置正常"
  exit 0
fi
