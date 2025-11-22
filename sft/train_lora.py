import os
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model
import swanlab


# === 全局配置 ===
PROJECT_NAME = "qwen3-sft-medical-lora"
MODEL_NAME = "Qwen/Qwen3-32B"
PROMPT = "You are a medical expert have read a textbook. You should provide thoughtful answers in English."
MAX_LENGTH = 4096


def setup_swanlab():
    """初始化 SwanLab 配置"""
    os.environ["SWANLAB_PROJECT"] = PROJECT_NAME
    swanlab.config.update(
        {
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "data_max_length": MAX_LENGTH,
        }
    )


def dataset_jsonl_transfer(origin_path, new_path):
    """将原始 JSONL 转换为微调所需格式"""
    messages = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            input_text = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think>\n{answer}"
            message = {
                "instruction": PROMPT,
                "input": input_text,
                "output": output,
            }
            messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """将数据集转为 input_ids / attention_mask / labels"""
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(example["output"], add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids[:MAX_LENGTH],
        "attention_mask": attention_mask[:MAX_LENGTH],
        "labels": labels[:MAX_LENGTH],
    }


def predict(messages, model, tokenizer):
    """推理预测"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=MAX_LENGTH)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_model_and_tokenizer():
    """下载并加载模型与 tokenizer"""
    # model_dir = snapshot_download(MODEL_NAME, cache_dir="./", revision="master")
    model_dir = "/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/Models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token  # ✅ 修改点：显式设置 pad_token
    model.config.pad_token_id = tokenizer.eos_token_id  # ✅ 修改点：显式设置 pad_token_id
    model.enable_input_require_grads()
    return model, tokenizer


def add_lora(model):
    """为模型添加 LoRA 配置"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    return get_peft_model(model, config)


def load_datasets():
    """加载并处理训练/验证集"""
    # train_path, val_path = "train.jsonl", "val.jsonl"
    # train_fmt, val_fmt = "train_format.jsonl", "val_format.jsonl"
    # train_path = "../dataset/sft1031/train.jsonl"
    # val_path = "../dataset/sft1031/val.jsonl"

    train_fmt = "../dataset/sft1031/train.jsonl"
    val_fmt = "../dataset/sft1031/val.jsonl"


    train_df = pd.read_json(train_fmt, lines=True)
    eval_df = pd.read_json(val_fmt, lines=True)

    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    return (
        train_ds.map(process_func, remove_columns=train_ds.column_names),
        eval_ds.map(process_func, remove_columns=eval_ds.column_names),
        val_fmt,
    )


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """训练模型"""
    args = TrainingArguments(
        output_dir="../output_model/Qwen3-8B-1031",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=400,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="swanlab",
        run_name="qwen3-8B",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()


def evaluate_model(model, tokenizer, val_path):
    """主观测试模型（取验证集前3条）"""
    test_df = pd.read_json(val_path, lines=True)[:3]
    test_text_list = []

    for _, row in test_df.iterrows():
        instruction, input_value = row["instruction"], row["input"]
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_value},
        ]
        response = predict(messages, model, tokenizer)
        response_text = f"Question: {input_value}\n\nLLM: {response}"
        test_text_list.append(swanlab.Text(response_text))
        print(response_text)

    swanlab.log({"Prediction": test_text_list})


def main():
    setup_swanlab()
    model, tokenizer = load_model_and_tokenizer()
    globals()["tokenizer"] = tokenizer  # 给 process_func 使用
    model = add_lora(model)
    train_dataset, eval_dataset, val_path = load_datasets()
    train_model(model, tokenizer, train_dataset, eval_dataset)
    evaluate_model(model, tokenizer, val_path)
    swanlab.finish()


if __name__ == "__main__":
    main()
