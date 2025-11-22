import json
import random
import os


def split_jsonl_dataset(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):

    # 检查比例合法性
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为 1.0，但当前为 {total_ratio}")

    # 固定随机种子
    random.seed(seed)

    # 读取数据
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无法解析的行: {line[:100]}...")
                continue

    print(f"📊 总数据量: {len(data)}")

    if not data:
        print("❌ 无可用数据，终止。")
        return

    # 打乱
    random.shuffle(data)

    # 划分索引
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存函数
    def save_jsonl(path, dataset):
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ 已保存 {path} ({len(dataset)} 条样本)")

    # 输出路径
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    # 保存文件
    save_jsonl(train_path, train_data)
    save_jsonl(val_path, val_data)
    save_jsonl(test_path, test_data)

    # 打印统计
    print(f"\n📁 数据划分完成：")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    return {"train": len(train_data), "val": len(val_data), "test": len(test_data)}


def main():
    """示例入口：划分 merged_sft.jsonl 为 train/val/test"""
    input_file = "sft_data.jsonl"
    output_dir = "sft1031"
    split_jsonl_dataset(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)


if __name__ == "__main__":
    main()
