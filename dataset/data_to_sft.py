import json
import os


def convert_merged_sft(input_file, output_file):
    """
    将 merged_sft.jsonl 转换为 instruction 格式：
    {
      "instruction": "You are a medical expert. You should provide thoughtful answers in English.",
      "input": "<question>",
      "output": "<think>...</think>\\n<answer>"
    }
    """
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return 0

    total = 0

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无法解析的行 line {line_num}")
                continue

            # 检查字段
            if not all(k in data for k in ("question", "think", "answer")):
                print(f"⚠️ 跳过缺字段的行 line {line_num}: {list(data.keys())}")
                continue

            # 构造新格式
            new_entry = {
                "instruction": "You are a medical expert. You should provide thoughtful answers in English.",
                "input": data["question"].strip(),
                "output": f"<think>{data['think'].strip()}</think>\n{data['answer'].strip()}"
            }

            f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            total += 1

    print(f"\n✅ 转换完成，输出文件: {output_file}，共生成 {total} 条数据。")
    return total


def main():
    input_file = "merged_data.jsonl"
    output_file = "sft_data.jsonl"
    convert_merged_sft(input_file, output_file)


if __name__ == "__main__":
    main()
