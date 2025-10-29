#!/bin/bash
#SBATCH -J qwen3-0.6b-server
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qwen3_0.6b_%j.out
#SBATCH --error=logs/qwen3_0.6b_%j.err

echo "🚀 启动 Qwen3-0.6B vLLM 服务..."

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm


export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/Models/Qwen3-0.6B"

vllm serve $MODEL_PATH  --served-model-name Qwen3-0.6B --port 8000 --host 0.0.0.0 --tensor-parallel-size 1