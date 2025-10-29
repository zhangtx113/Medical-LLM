from datasets import load_dataset
import json
import pandas as pd

SAVE_JSONL = "medxpertqa_text.jsonl"
SAVE_CSV = "medxpertqa_text.csv"

def main():
    print("🔹 Downloading dataset: TsinghuaC3I/MedXpertQA (Text)...")
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text")

    # 获取 train split
    data = ds["test"]
    print(f"✅ Loaded {len(data)} samples.")

    # 保存为 JSONL
    print(f"💾 Saving JSONL -> {SAVE_JSONL}")
    with open(SAVE_JSONL, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 保存为 CSV（可选）
    # print(f"💾 Saving CSV -> {SAVE_CSV}")
    # df = pd.DataFrame(data)
    # df.to_csv(SAVE_CSV, index=False, encoding="utf-8")

    print("✅ Done! Dataset saved locally.")


if __name__ == "__main__":
    main()