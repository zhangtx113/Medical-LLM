import os
import json
import re
from openai import OpenAI
import random
import time

# ------------------------------
# 配置
# ------------------------------
MODEL_NAME = "Qwen3-32b"
API_BASE = "http://gpu44:8000/v1"
API_KEY = "EMPTY"

# ------------------------------
# 工具函数
# ------------------------------
def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_text_to_chunks(text, min_len=3000, max_len=4000):
    sections, current = [], ""
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("##"):
            if current.strip():
                sections.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        sections.append(current.strip())

    merged, buf = [], ""
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

# ------------------------------
# 模型接口
# ------------------------------
def init_client():
    return OpenAI(api_key=API_KEY, base_url=API_BASE)

def call_model(client, messages, model=MODEL_NAME):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    raw = completion.choices[0].message.content.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return cleaned

# ------------------------------
# 两个主要步骤
# ------------------------------
def generate_questions(client, text_chunk, system_prompt, qg_prompt):
    """直接从教材文本生成多选题"""
    prompt = qg_prompt.replace("{{TEXT}}", text_chunk)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return call_model(client, messages)

def generate_think_and_answer(client, text_chunk, question_text, system_prompt, ans_prompt):
    """根据题目生成答案与思考"""
    prompt = (
        ans_prompt
        .replace("{{TEXT}}", text_chunk)
        .replace("{{QUESTION}}", question_text)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return call_model(client, messages)

# ------------------------------
# 主流程
# ------------------------------
def process_markdown_file(volume_index, system_prompt, qg_prompt, ans_prompt):
    input_path = f"en_data/En_Part{volume_index}.mmd"
    output_path = f"output/mcq_volume{volume_index}.jsonl"

    if not os.path.exists(input_path):
        print(f"⚠️ 文件 {input_path} 不存在，跳过。")
        return

    client = init_client()
    md_text = read_file(input_path)
    chunks = split_text_to_chunks(md_text)

    kept = []
    print(f"\n处理文件：{input_path}，共 {len(chunks)} 块")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n块 {idx}/{len(chunks)} 开始")

        try:
            # Step1: 直接生成问题
            q_text = generate_questions(client, chunk, system_prompt, qg_prompt)
            print("问题候选生成完成。开始解析问题。")

            q_lines = [ln for ln in q_text.strip().splitlines() if ln.strip()]
            questions = []
            for ln in q_lines:
                try:
                    obj = json.loads(ln)
                    if "question" in obj and "answer" in obj:
                        questions.append(obj)
                except json.JSONDecodeError:
                    continue

            print(f"解析到 {len(questions)} 道题（候选）。")

            # Step2: 生成思考与答案一致性检查
            for q in questions:
                q_text_field = q["question"]
                q_answer_declared = q["answer"].strip().upper()[:1]

                ans_text = generate_think_and_answer(client, chunk, q_text_field, system_prompt, ans_prompt)
                try:
                    ans_obj = json.loads(ans_text)
                except json.JSONDecodeError:
                    print("⚠️ 第三步返回非 JSON：", ans_text[:200])
                    continue

                ans_answer = ans_obj.get("answer", "").strip().upper()[:1]
                think = ans_obj.get("think", "")

                if q_answer_declared and ans_answer and q_answer_declared == ans_answer:
                    record = {
                        "question": q_text_field,
                        "think": think,
                        "answer": ans_answer
                    }
                    kept.append(record)
                    print(f"✅ 保留题目（答案一致）：{q_answer_declared}")
                else:
                    print(f"❌ 丢弃题目（答案不一致）：declared={q_answer_declared} vs generated={ans_answer}")

        except Exception as e:
            print(f"❌ 块 {idx} 处理失败：{e}")

    write_jsonl(output_path, kept)
    print(f"\n完成 volume{volume_index}，共保留题目：{len(kept)} -> {output_path}")

# ------------------------------
# main
# ------------------------------
def main():
    system_prompt = read_file("prompts/multichoice/system.txt")
    qg_prompt = read_file("prompts/multichoice/generate_question.txt")
    ans_prompt = read_file("prompts/multichoice/generate_answer.txt")

    for vol in range(1, 11):
        process_markdown_file(vol, system_prompt, qg_prompt, ans_prompt)

if __name__ == "__main__":
    main()
