#!/bin/bash
#SBATCH -J PPO_Qwen3_8B_Train              # 作业名
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                       # 使用 8 张 GPU
#SBATCH --cpus-per-task=48                 # 每个任务使用的 CPU 数量
#SBATCH -o logs/ppo_train_%j.out           # 标准输出日志
#SBATCH -e logs/ppo_train_%j.err           # 错误日志

# ======= 环境设置 =======
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate medical-llm
cd /mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/PPO

# ======= 参数配置 =======
MODEL_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/output_model/Qwen3-8B-1031/checkpoint-11372"
DATA_PATH="/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/dataset/merged_data.jsonl"
OUTPUT_DIR="../output_model/qwen3-8b-ppo"
EPOCHS=3
BATCH_SIZE=1
LR=3e-6

export SWANLAB_API_KEY="zZt33jJxQnffzLZB18XvZ"

# ======= 日志文件夹 =======
mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# ======= 启动 PPO 训练 =======
echo "Starting PPO training with model: $MODEL_PATH"
python ppo_train.py \
    --model ${MODEL_PATH} \
    --dataset ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR}

echo "PPO training finished."
