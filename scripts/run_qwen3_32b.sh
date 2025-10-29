#!/bin/bash
#SBATCH -J qwen3-32b-server
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/qwen3_server_%j.out
#SBATCH --error=logs/qwen3_server_%j.err

echo "🚀 启动 Qwen3-32B vLLM 服务..."

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate esys


export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/models/Qwen3-32b"

vllm serve $MODEL_PATH  --served-model-name Qwen3-32b --port 8000 --host 0.0.0.0 --tensor-parallel-size 8