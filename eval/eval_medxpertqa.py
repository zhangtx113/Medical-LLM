import argparse
import json
import re
from tqdm import tqdm
from time import sleep
from sklearn.metrics import accuracy_score, classification_report
from openai import OpenAI

# ============ Local Qwen3 configuration ============
LOCAL_BASE_URL = "http://gpu35:8000/v1"
LOCAL_MODEL_PATH = "Qwen3-8B"

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
    """Load dataset from local JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


def extract_answer(text: str):
    """Extract letter (A-J) from model output."""
    if not text:
        return None
    for pattern in [r"Answer[:：]?\s*([A-J])\b", r"\(([A-J])\)", r"\b([A-J])\b"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def format_options(options):
    """Format options into readable text."""
    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])
    elif isinstance(options, list):
        return "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    else:
        return str(options)


def local_qwen_call(prompt):
    """Call local Qwen3-32B model via OpenAI-compatible API."""
    client = OpenAI(api_key="EMPTY", base_url=LOCAL_BASE_URL)
    completion = client.chat.completions.create(
        model=LOCAL_MODEL_PATH,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        extra_body={"enable_thinking": True},
    )
    return completion.choices[0].message.content


def openai_call(prompt, model="gpt-4o-mini"):
    """Call OpenAI API."""
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    return resp.choices[0].message.content


def filter_data(data, medical_task=None, body_system=None, question_type=None):
    """Filter dataset by attributes."""
    def match(val, allowed):
        if allowed is None:
            return True
        return str(val).lower() in [a.lower() for a in allowed if a]
    return [
        ex for ex in data
        if match(ex.get("medical_task"), medical_task)
        and match(ex.get("body_system"), body_system)
        and match(ex.get("question_type"), question_type)
    ]


# ============ Evaluation main ============
def evaluate(data_path, model_type="local", openai_model="gpt-4o-mini",
             medical_task=None, body_system=None, question_type=None,
             max_samples=None, sleep_time=0.0, output_path="predictions_local.jsonl"):
    """Evaluate model performance on filtered dataset."""
    data = load_jsonl(data_path)
    filtered = filter_data(data, medical_task, body_system, question_type)
    print(f"📊 After filtering: {len(filtered)} samples")

    if max_samples:
        filtered = filtered[:max_samples]

    y_true, y_pred, results = [], [], []

    for item in tqdm(filtered, desc="Evaluating"):
        question = item["question"]
        options = format_options(item["options"])
        gold = item["label"]
        prompt = PROMPT_TEMPLATE.format(question=question, options=options)

        try:
            if model_type == "local":
                output = local_qwen_call(prompt)
            else:
                output = openai_call(prompt, model=openai_model)
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

        if sleep_time > 0:
            sleep(sleep_time)

    valid_idx = [i for i, p in enumerate(y_pred) if p]
    acc = accuracy_score([y_true[i] for i in valid_idx], [y_pred[i] for i in valid_idx]) if valid_idx else 0.0
    # report = classification_report([y_true[i] for i in valid_idx], [y_pred[i] for i in valid_idx], zero_division=0)

    print("\n=== ✅ Evaluation Summary ===")
    print(f"Samples evaluated: {len(filtered)}")
    print(f"Accuracy: {acc:.3f}")
    # print(report)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"📝 Saved predictions to {output_path}")


# ============ Entry Point ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="medxpertqa_text.jsonl")
    parser.add_argument("--model", choices=["local", "openai"], default="local")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--medical-task", default=None)
    parser.add_argument("--body-system", default=None)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--output-path", default="predictions_local.jsonl")
    args = parser.parse_args()

    med_task = args.medical_task.split(",") if args.medical_task else None
    body_sys = args.body_system.split(",") if args.body_system else None
    qtype = args.question_type.split(",") if args.question_type else None

    evaluate(
        data_path=args.data_path,
        model_type=args.model,
        openai_model=args.openai_model,
        medical_task=med_task,
        body_system=body_sys,
        question_type=qtype,
        max_samples=args.max_samples,
        sleep_time=args.sleep,
        output_path=args.output_path,
    )
