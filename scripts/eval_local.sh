#!/bin/bash
#SBATCH -J eval-local          # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # ✅ 单卡即可运行评估
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/eval_local_%j.out
#SBATCH --error=logs/eval_local_%j.err

echo "🚀 启动本地 Qwen3-8B 模型评估 (checkpoint-8000)..."

# ========== 环境初始化 ==========
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm

# ========== 目录设置 ==========
PROJECT_DIR="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM"
DATA_PATH="$PROJECT_DIR/eval/medxpertqa_text.jsonl"
OUTPUT_PATH="$PROJECT_DIR/eval/results/predictions_qwen3_8b_local.jsonl"
SCRIPT_PATH="$PROJECT_DIR/eval/eval_medxpertqa_local.py"

mkdir -p $PROJECT_DIR/results

# ========== 运行评估 ==========
python $SCRIPT_PATH \
  --data-path $DATA_PATH \
  --medical-task "Diagnosis","Treatment","Basic Medicine" \
  --body-system "Cardiovascular" \
  --question-type "Reasoning","Understanding" \
  --output-path $OUTPUT_PATH

# ========== 任务结束 ==========
echo "✅ 评估完成！结果保存在：$OUTPUT_PATH"
