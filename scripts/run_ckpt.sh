#!/bin/bash
#SBATCH -J ckpt11372-server
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/Qwen3_8B_ckpt11372_%j.out
#SBATCH --error=logs/Qwen3_8B_ckpt11372_%j.err

echo "🚀 启动 Qwen3-8B checkpoint-11372 vLLM 服务..."

# ========== 环境初始化 ==========
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

export VLLM_USE_MODELSCOPE=true

# ========== 模型路径 ==========
MODEL_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/output_model/Qwen3-8B-1031/checkpoint-11372"

# ========== 启动 vLLM 服务 ==========
vllm serve $MODEL_PATH \
    --served-model-name Qwen3-8B-ckpt11372 \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 4

echo "✅ Qwen3-8B checkpoint-11372 vLLM 服务已启动。"
