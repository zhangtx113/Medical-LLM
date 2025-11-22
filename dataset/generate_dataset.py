import os
import json
import re
from openai import OpenAI


# ==============================
# 工具函数
# ==============================

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_text_to_chunks(text, min_len=1500, max_len=2000):
    sections, current = [], ""
    for line in text.splitlines():
        if line.startswith("#"):
            if current.strip():
                sections.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        sections.append(current.strip())

    merged = []
    buf = ""
    for sec in sections:
        if len(sec) < min_len:
            buf += sec + "\n"
            if len(buf) >= min_len:
                merged.append(buf.strip())
                buf = ""
        else:
            if buf:
                merged.append((buf + sec).strip())
                buf = ""
            else:
                merged.append(sec.strip())
    if buf:
        merged.append(buf.strip())

    final_chunks = []
    for sec in merged:
        if len(sec) > max_len:
            for i in range(0, len(sec), max_len):
                final_chunks.append(sec[i:i + max_len])
        else:
            final_chunks.append(sec)
    return final_chunks


# ==============================
# 模型接口
# ==============================

def init_client():
    return OpenAI(api_key="EMPTY", base_url="http://gpu50:8000/v1")


def call_model(client, messages):
    completion = client.chat.completions.create(
        model="Qwen3-32b",
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=False
    )
    raw_text = completion.choices[0].message.content.strip()

    # 清理 <think> 标签
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return cleaned_text

# ==============================
# 核心流程
# ==============================

def extract_knowledge_points(client, text_chunk, system_prompt, kp_prompt):
    """步骤1：提取知识点"""
    prompt = kp_prompt.replace("{{TEXT}}", text_chunk)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return call_model(client, messages)


def generate_questions(client, text_chunk, knowledge_points, system_prompt, qg_prompt):
    """步骤2：基于知识点生成问题和类型"""
    prompt = (
        qg_prompt
        .replace("{{KNOWLEDGE}}", knowledge_points)
        .replace("{{TEXT}}", text_chunk)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return call_model(client, messages)


def generate_answers(client, text_chunk, question, q_type, system_prompt, ans_prompt):
    """步骤3：生成问题答案、关键词"""
    prompt = (
        ans_prompt
        .replace("{{TEXT}}", text_chunk)
        .replace("{{QUESTION}}", question)
        .replace("{{TYPE}}", q_type)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return call_model(client, messages)


def process_markdown_file(volume_index, system_prompt, kp_prompt, qg_prompt, ans_prompt):
    """主流程：知识点 → 问题 → 答案"""
    input_path = f"original_data/volume{volume_index}.md"
    output_path = f"output/qa_dataset{volume_index}.jsonl"

    if not os.path.exists(input_path):
        print(f"⚠️ 文件 {input_path} 不存在，跳过。")
        return

    client = init_client()
    md_text = read_file(input_path)
    chunks = split_text_to_chunks(md_text)

    all_entries = []
    print(f"\n📘 开始处理文件：{input_path}，共 {len(chunks)} 块。")

    for idx, chunk in enumerate(chunks, 1):
        print(f"\n🧩 第 {idx}/{len(chunks)} 块")

        try:
            # Step 1: 知识点
            kp_text = extract_knowledge_points(client, chunk, system_prompt, kp_prompt)
            print("✅ 知识点提取完成：")
            # print(kp_text)

            # Step 2: 问题
            q_text = generate_questions(client, chunk, kp_text, system_prompt, qg_prompt)
            print("✅ 问题生成完成")
            # print(q_text)

            questions = []
            for line in q_text.strip().splitlines():
                try:
                    data = json.loads(line)
                    if all(k in data for k in ("question", "type")):
                        questions.append(data)
                except json.JSONDecodeError:
                    continue

            # Step 3: 答案
            for q_obj in questions:
                q = q_obj["question"]
                q_type = q_obj["type"]
                ans_text = generate_answers(client, chunk, q, q_type, system_prompt, ans_prompt)
                # print(ans_text)

                try:
                    ans_data = json.loads(ans_text)
                    entry = {
                        "question": q,
                        "think": ans_data.get("think", ""),
                        "answer": ans_data.get("answer", ""),
                        "type": q_type,
                        "keyword": ans_data.get("keyword", "")
                    }
                    all_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"⚠️ 非 JSON 格式答案：{ans_text}")
                print("✅ 答案生成完成")

        except Exception as e:
            print(f"❌ 第 {idx} 块处理失败：{e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ volume{volume_index} 完成，共生成问答：{len(all_entries)}")


def main():
    system_prompt = read_file("prompts/system.txt")
    kp_prompt = read_file("prompts/extract_knowledge.txt")
    qg_prompt = read_file("prompts/generate_question.txt")
    ans_prompt = read_file("prompts/generate_answer.txt")

    for vol in range(1, 11):
        process_markdown_file(vol, system_prompt, kp_prompt, qg_prompt, ans_prompt)


if __name__ == "__main__":
    main()
