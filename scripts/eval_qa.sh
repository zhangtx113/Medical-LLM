#!/bin/bash
#SBATCH -J Eval_QA              # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # ✅ 推理调用只需 1 张 GPU
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/Eval_QA_%j.out
#SBATCH --error=logs/Eval_QA_%j.err

echo "📘 启动比较原文和问答对质量任务..."

# === 加载 Conda 环境 ===
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

# === 切换到项目目录 ===
cd /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/dataset

# === 运行 Python 脚本 ===
python eval_qa.py

echo "✅ 任务完成！"
