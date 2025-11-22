import argparse
import json
import re
from tqdm import tqdm
from time import sleep
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ Local model path ============
LOCAL_MODEL_PATH = "/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/output_model/Qwen3-8B-1031/checkpoint-9200"

PROMPT_TEMPLATE = """You are a medical expert.
Read the question and select the most appropriate answer.
Only reply with the letter (A, B, C, ...).

Question:
{question}

Options:
{options}

Answer:"""


# ============ Helper functions ============
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


def extract_answer(text: str):
    if not text:
        return None
    for pattern in [r"Answer[:：]?\s*([A-J])\b", r"\(([A-J])\)", r"\b([A-J])\b"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def format_options(options):
    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])
    elif isinstance(options, list):
        return "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    else:
        return str(options)


# ============ Local model inference ============
def load_local_model():
    print(f"🧠 Loading local model from {LOCAL_MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def local_infer(model, tokenizer, prompt):
    """Generate model output from local checkpoint"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 截取模型在 “Answer:” 之后的部分
    answer = text.split("Answer:")[-1].strip()
    return answer


# ============ Evaluation main ============
def evaluate(data_path,
             medical_task=None, body_system=None, question_type=None,
             max_samples=None, output_path="predictions_local.jsonl"):

    data = load_jsonl(data_path)

    def match(val, allowed):
        if allowed is None:
            return True
        return str(val).lower() in [a.lower() for a in allowed if a]

    filtered = [
        ex for ex in data
        if match(ex.get("medical_task"), medical_task)
        and match(ex.get("body_system"), body_system)
        and match(ex.get("question_type"), question_type)
    ]
    print(f"📊 After filtering: {len(filtered)} samples")

    if max_samples:
        filtered = filtered[:max_samples]

    # 加载本地模型
    model, tokenizer = load_local_model()

    y_true, y_pred, results = [], [], []

    for item in tqdm(filtered, desc="Evaluating"):
        question = item["question"]
        options = format_options(item["options"])
        gold = item["label"]
        prompt = PROMPT_TEMPLATE.format(question=question, options=options)

        try:
            output = local_infer(model, tokenizer, prompt)
        except Exception as e:
            output = f"ERROR: {e}"

        pred = extract_answer(output)
        y_true.append(gold)
        y_pred.append(pred)

        results.append({
            "id": item.get("id"),
            "medical_task": item.get("medical_task"),
            "body_system": item.get("body_system"),
            "question_type": item.get("question_type"),
            "gold": gold,
            "pred": pred,
            "raw_output": output,
        })

    valid_idx = [i for i, p in enumerate(y_pred) if p]
    acc = accuracy_score([y_true[i] for i in valid_idx],
                         [y_pred[i] for i in valid_idx]) if valid_idx else 0.0

    print("\n=== ✅ Evaluation Summary ===")
    print(f"Samples evaluated: {len(filtered)}")
    print(f"Accuracy: {acc:.3f}")

    # 保存预测结果
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"📝 Saved predictions to {output_path}")


# ============ Entry Point ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="medxpertqa_text.jsonl")
    parser.add_argument("--medical-task", default=None)
    parser.add_argument("--body-system", default=None)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-path", default="predictions_local.jsonl")
    args = parser.parse_args()

    med_task = args.medical_task.split(",") if args.medical_task else None
    body_sys = args.body_system.split(",") if args.body_system else None
    qtype = args.question_type.split(",") if args.question_type else None

    evaluate(
        data_path=args.data_path,
        medical_task=med_task,
        body_system=body_sys,
        question_type=qtype,
        max_samples=args.max_samples,
        output_path=args.output_path,
    )
