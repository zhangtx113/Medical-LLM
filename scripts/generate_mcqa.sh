#!/bin/bash
#SBATCH -J Generate_MCQA              # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # ✅ 推理调用只需 1 张 GPU
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/Generate_MCQA_%j.out
#SBATCH --error=logs/Generate_MCQA_%j.err

echo "📘 启动医学问答数据集生成任务..."

# === 加载 Conda 环境 ===
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

# === 切换到项目目录 ===
cd /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/dataset

# === 运行 Python 脚本 ===
python generate_mcqa.py

echo "✅ 医学问答数据集生成任务完成！"
