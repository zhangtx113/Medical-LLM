#!/bin/bash
#SBATCH -J eval-medxpertqa              # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # ✅ 推理调用只需 1 张 GPU
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/eval_medxpertqa_%j.out
#SBATCH --error=logs/eval_medxpertqa_%j.err

echo "🚀 启动 MedXpertQA 模型评估..."

# === 激活环境 ===
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

# === 基础变量设置 ===
MODEL_NAME="Qwen3-8B"
DATA_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/eval/medxpertqa_text.jsonl"

# === 启动评估 ===
python /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/eval/eval_medxpertqa.py \
  --data-path $DATA_PATH \
  --model local \
  --medical-task "Diagnosis","Treatment","Basic Medicine" \
  --body-system Cardiovascular \
  --question-type Reasoning,Understanding \
  --output-path /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/eval/results/predictions_${MODEL_NAME}.jsonl

echo "✅ 评估完成：结果已保存至 results/predictions_${MODEL_NAME}.jsonl"
