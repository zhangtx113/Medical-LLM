import os, re, json, argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from sentence_transformers import SentenceTransformer, util
import swanlab


# ---------------------------------------------------------
# 加载 JSONL 数据
# ---------------------------------------------------------
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]


# ---------------------------------------------------------
# 提取 <think> 与 answer
# ---------------------------------------------------------
def parse_think_answer(resp_text: str):
    text = resp_text.strip()
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        think = match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    else:
        think, answer = "", text
    return think, answer


# ---------------------------------------------------------
# 奖励函数（Accuracy + Consistency）
# ---------------------------------------------------------
def reward_fn(resp_text, gold_think, gold_answer, keywords, sbert_model):
    pred_think, pred_answer = parse_think_answer(resp_text)

    # ---------- Accuracy ----------
    acc_score = 0.0
    if keywords:
        if isinstance(keywords, str):
            keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        total_kw = len(keywords)
        matched = sum(1 for kw in keywords if kw.lower() in pred_answer.lower())
        acc_score = min(5.0, 5.0 * matched / max(1, total_kw))

    # ---------- Consistency ----------
    try:
        emb_pred = sbert_model.encode(pred_think, convert_to_tensor=True)
        emb_ref = sbert_model.encode(gold_think, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb_pred, emb_ref).item()
        cons_score = max(0.0, (sim + 1) / 2 * 5.0)
    except Exception:
        cons_score = 0.0

    total_reward = acc_score + cons_score
    return total_reward, acc_score, cons_score, pred_think, pred_answer


# ---------------------------------------------------------
# 主训练函数 (TRL 0.24.0 风格)
# ---------------------------------------------------------
def main(args):
    # ✅ 初始化 SwanLab
    swanlab.init(
        project="ppo-rlvr",
        experiment_name=f"PPO_{args.model.split('/')[-1]}",
        config={
            "model": args.model,
            "dataset": args.dataset,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
    )

    # ✅ 加载数据
    raw_data = load_jsonl(args.dataset)
    dataset = Dataset.from_list([{"prompt": f"Question: {d['question']}\nPlease think step-by-step and answer.",
                                  "gold_think": d["think"],
                                  "gold_answer": d["answer"],
                                  "keywords": d["keyword"]}
                                 for d in raw_data])

    # ✅ 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    # ✅ 奖励模型（Sentence-BERT）
    sbert_model = SentenceTransformer("/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/Models/all-MiniLM-L6-v2")

    # ✅ PPOConfig
    training_args = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=1,
        output_dir=args.output_dir
    )

    # ✅ 初始化 PPOTrainer
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        reward_model=sbert_model,
        value_model=sbert_model,
    )

    # ✅ 自定义训练循环
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch} =====")

        for batch in dataset:
            prompt = batch["prompt"]
            gold_think = batch["gold_think"]
            gold_answer = batch["gold_answer"]
            keywords = batch["keywords"]

            # 生成模型响应
            response_tensors = trainer.generate([prompt], max_new_tokens=256)
            response = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

            # 奖励计算
            total_reward, acc_score, cons_score, pred_think, pred_answer = reward_fn(
                resp_text=response,
                gold_think=gold_think,
                gold_answer=gold_answer,
                keywords=keywords,
                sbert_model=sbert_model
            )

            # 进行 PPO 更新
            stats = trainer.step([prompt], [response], [total_reward])
            global_step += 1

            # 记录日志
            swanlab.log({
                "reward/total": total_reward,
                "reward/accuracy": acc_score,
                "reward/consistency": cons_score,
                "ppo/loss": stats["ppo/loss/total_loss"],
                "train/step": global_step
            })

            if global_step % 20 == 0:
                print(f"\n[Step {global_step}]")
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print(f"Reward={total_reward:.2f} (Acc={acc_score:.2f}, Cons={cons_score:.2f})")

        # 每个 epoch 保存模型
        save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"✅ Epoch {epoch} done, model saved to {save_path}")
        swanlab.log({"train/epoch": epoch})

    swanlab.finish()
    print("🎯 RLVR PPO Training complete!")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="dataset/merged_data.jsonl")
    p.add_argument("--output_dir", default="models/qwen3-rlvr")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-6)
    args = p.parse_args()
    main(args)
