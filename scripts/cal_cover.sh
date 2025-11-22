#!/bin/bash
#SBATCH -J Coverage_Calculation                     # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                                # 使用 8 张 GPU
#SBATCH --cpus-per-task=48                          # 分配 48 个 CPU 核心
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/Coverage_%j.out              # 标准输出日志
#SBATCH --error=logs/Coverage_%j.err               # 错误输出日志

echo "🚀 开始计算文本覆盖度..."

# === 加载 Conda 环境 ===
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

# === 切换到项目目录 ===
cd /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/coverage

# === 运行覆盖度计算脚本 ===
python coverage_calculator.py

echo "✅ 文本覆盖度计算任务完成"
