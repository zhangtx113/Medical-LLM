#!/bin/bash
#SBATCH -J LoRA_Train_Model                         # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                           # 使用 8 张 GPU
#SBATCH --cpus-per-task=48                     # 分配 48 个 CPU 核心
#SBATCH --output=logs/LoRA_%j.out       # 标准输出日志
#SBATCH --error=logs/LoRA_%j.err        # 错误输出日志

echo "🚀 开始训练模型..."

# === 加载环境 ===
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

# === 设置 SwanLab API Key ===
export SWANLAB_API_KEY="zZt33jJxQnffzLZB18XvZ"

# === 切换到项目目录 ===
cd /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/sft

# === 运行训练脚本 ===
python train_lora.py

echo "✅ 模型训练任务完成"
