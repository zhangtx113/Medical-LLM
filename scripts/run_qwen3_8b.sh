#!/bin/bash
#SBATCH -J qwen3-8b-server
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/Qwen3_8B_%j.out
#SBATCH --error=logs/Qwen3_8B_%j.err

echo "🚀 启动 Qwen3-8B vLLM 服务..."

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm


export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/Models/Qwen3-8B"

vllm serve $MODEL_PATH  --served-model-name Qwen3-8B --port 8000 --host 0.0.0.0 --tensor-parallel-size 4