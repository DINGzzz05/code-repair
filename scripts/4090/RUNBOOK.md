# 4x RTX 4090 (24GB) 全流程运行手册

本文档对应“4×4090 安全方案”：Qwen3-8B + LoRA 两阶段（SFT → GRPO/GSPO），
不改代码，全部使用仓库已有参数接口。目标是把“数据 → SFT → RL → 评测”闭环跑通。

## 1. 新增文件与作用

| 文件 | 作用 |
|---|---|
| `scripts/4090/01_sft_8b.sh` | Stage 1：SFT（GPU1-2，GPU0 留给 vLLM） |
| `scripts/4090/02_vllm_serve.sh` | Stage 2a：TRL async vLLM（GPU0） |
| `scripts/4090/03_grpo_8b.sh` | Stage 2b：GRPO/GSPO 训练（GPU1-2） |
| `scripts/4090/04_eval_serve.sh` | Stage 3a：评测 vLLM（GPU3） |
| `benchmarks/configs/nano_crrl_8b.yaml` | Stage 3b：nano-agent 评测配置 |
| `src/conf/model/small_qwen_sft.yaml` | GRPO 阶段指向 SFT 合并模型 |
| `src/conf/model/medium_qwen_sft.yaml` | 4×A800/A100-80GB 方案的 14B 配置（备选） |
| `scripts/4090/RUNBOOK.md` | 本运行手册 |

## 2. 服务器准备

```bash
# 1) 拉代码（在服务器上）
git clone <你的仓库地址> agentic-evals-lab && cd agentic-evals-lab
chmod +x scripts/4090/*.sh

# 2) 装依赖（Linux + NVIDIA 驱动/CUDA 12.x；或复用 vllm/vllm-openai:v0.11.0 容器）
uv sync --extra gpu        # 含 flash-attn 2.8.3 / flashinfer / bitsandbytes

# 3) 登录与密钥
huggingface-cli login       # GRPO 入口强制 whoami()，不登录会直接报错
wandb login                 # report_to=wandb 需要；也可传 report_to=none 关闭
export HF_TOKEN=...         # 拉模型/数据集

# 4) SWE-Gym 实例镜像（rollout/eval 需要）
#    apptainer 或 docker 可用，镜像按实例按需拉取，预留 300-500GB 磁盘
```

## 3. 执行顺序

```bash
# Stage 1: SFT（~1-2h，GPU1-2）
scripts/4090/01_sft_8b.sh                      # -> outputs/crrl_8b_sft_v1_merged

# Stage 2a: 终端 1，GPU0 起推理（保持前台运行）
scripts/4090/02_vllm_serve.sh

# Stage 2b: 终端 2，GPU1-2 训练（先冒烟，再全量）
scripts/4090/03_grpo_8b.sh crrl_8b_grpo_v1 1   # 冒烟：max_steps=2
scripts/4090/03_grpo_8b.sh crrl_8b_grpo_v1     # 全量：1 epoch，约 20-30h
# -> grpo_repo_repair_model_merged

# Stage 3a: 训练结束后，终端 3 用 GPU3 起评测推理
scripts/4090/04_eval_serve.sh

# Stage 3b: 评测（先 20 条冒烟，再全量）
uv run python benchmarks/swe_bench/run_nano_eval.py \
  --config benchmarks/configs/nano_crrl_8b.yaml \
  --output-dir nano_crrl_8b_swe-bench/run_0 \
  --slice :20

# Stage 3c: CPU + Docker 判 pass/fail
benchmarks/swe_bench/run_harness_eval.sh \
  --subset verified --split test \
  --preds nano_crrl_8b_swe-bench/run_0/preds.jsonl \
  --run-id crrl_8b_v1_verified --max-workers 16
```

## 4. 为什么是这些参数（24GB 的关键约束）

- **8B 模型**：14B 的 bf16 权重 28GB，24GB 单卡放不下。
- **SFT 关 KL**（`sft.kl_lambda=0.0`）：`KLSFTTrainer` 在非 ZeRO-3 下会深拷贝
  一份完整模型做参考模型（+16GB），24GB 必爆。
- **GRPO 关参考模型**（`grpo.beta=0.0`）：`multi_turn_gspo` 默认 0.02 会触发
  参考模型创建；GSPO 本身是无 KL 设计，置 0 是安全选项。
- **组大小 4**：`num_generations=4`，`2 卡 × per_device 2 = 4`，满足
  “整组单前向”约束；8 的组大小在 24GB 上放不下 batch。
- **上下文压缩**：训练 `1024 + 7168`，vLLM `max-model-len 9216`、`max-num-seqs 4`；
  评测 16k 上下文、`max-num-seqs 2`。

## 5. 显存监控与降级

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 5
```

训练卡峰值接近 23GB 时，按顺序降级：
1. `grpo.max_completion_length=6144`（同时把 vLLM `max-model-len` 降到 8192）；
2. LoRA `model.r=16, model.lora_alpha=32`（GRPO 阶段）；
3. 再不行换 `model.attn_implementation=sdpa`（flash-attn 有问题时的兜底，显存会略升）。

提速选项（进阶，风险自担）：vLLM 卡换 AWQ 4bit 权重，可把并发拉回 8；
需先验证 TRL fork 的权重同步在量化基座上正常（项目记录里提过“8-bit 权重同步错乱”的坑）。

## 6. 已知坑

- `run=repo_repair_multilingual` 当前会报错（`RunConfig.__post_init__` 校验不含该值，
  `nano_rollout_func` 也不接受逗号拼接的 dataset 名），**不要用**，一律 `run=repo_repair`。
- GRPO 必须登录 HuggingFace（`whoami()` 硬校验）。
- 4090 无 NVLink，跨卡走 PCIe；ZeRO-2 只分片 LoRA 的梯度/优化器，量小，影响可忽略。
- 难度测量（`measure_swe_gym_difficulty.py`）在 4090 上太慢，首跑用 `difficulty=all`；
  要课程退火就先 `--max-problems 200` 测子集。

## 7. git 工作流（服务器侧）

```bash
# 本机把改动提交并推送
git add scripts/4090 src/conf/model/medium_qwen_sft.yaml \
        src/conf/model/small_qwen_sft.yaml benchmarks/configs/nano_crrl_8b.yaml
git commit -m "add 4x4090 safe-plan scripts and configs"
git push

# 服务器拉取
git pull
```

## 8. 4×A800/A100-80GB 备选

如果后续换回 4×A800-80GB，可用 14B 方案：`src/conf/model/medium_qwen_sft.yaml` +
原仓库 `scripts/grpo/start_vllm_server.sh` / `start_grpo_train.sh`（默认 medium_qwen），
SFT 把 `model=medium_qwen` 传给 `scripts/sft/small_sft_lora_train_job.sh` 即可。
