import json
import os


def merge_jsonl_files(input_files, output_file):
    total = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"⚠️ 文件不存在: {input_file}")
                continue

            print(f"📄 正在合并文件: {input_file}")

            with open(input_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"跳过无法解析的行 in {input_file} line {line_num}")
                        continue

                    if not isinstance(data, dict):
                        continue

                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    total += 1

    print(f"\n✅ 合并完成，结果已保存到 {output_file}，共合并 {total} 条数据。")
    return total


def main():
    input_files = [f"output/qa_dataset{i}.jsonl" for i in range(1, 11)]
    output_file = "merged_data.jsonl"

    merge_jsonl_files(input_files, output_file)


if __name__ == "__main__":
    main()
