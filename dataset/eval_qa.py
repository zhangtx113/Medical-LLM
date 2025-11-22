import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from openai import OpenAI

# ============================
# 配置部分
# ============================
MODEL_NAME = "Qwen3-32b"
API_BASE = "http://gpu43:8000/v1"
# API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "EMPTY"
# API_KEY = "sk-7e7e921795e54f18bcf4291658d2ca75"

TOP_K_ORIGINAL = 1
TOP_K_QA = 5
MAX_CONTEXT_CHARS = 200000
SPILT_LENGTH = 4000

# —— 修改为你的本地 bge-large-en-v1.5 路径 ——
EMB_MODEL_PATH = "/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/dataset/models/bge-large-en-v1.5"

CHAPTER_FILE = "./en_data/merge_section1.md"
QA_FILE = "./output/merge_mcqa1.jsonl"
TEST_FILE = "./test_section/eval_section1.jsonl"

# OUT_ORIGINAL = "./results/results_original5.jsonl"
# OUT_QA = "./results/results_qa5.jsonl"
# OUT_DIRECT = "./results/results_direct5.jsonl"
# OUT_QA_USAGE = "./results/qa_usage_count5.json"

OUT_ORIGINAL = "./results/test_original.jsonl"
OUT_QA = "./results/test_qa.jsonl"
OUT_DIRECT = "./results/test_direct.jsonl"
OUT_QA_USAGE = "./results/test_usage_count.json"



# ============ LLM client (vLLM, OpenAI API format) ============
client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
)


def llm_answer(prompt):
    """调用 vLLM 得到模型预测的 ABCDE 选项"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw_text = response.choices[0].message.content.strip()
        # print(raw_text)

        # 清理 <think> 标签
        cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # print(cleaned_text)

        # 只提取 A/B/C/D/E
        for ch in reversed(cleaned_text):
            if ch.upper() in ["A", "B", "C", "D", "E"]:
                return ch.upper()
        return "UNKNOWN"

    except Exception as e:
        print("LLM ERROR:", e)
        return "UNKNOWN"


# ============================ Utils ============================

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def embed_texts(model, texts, batch_size=32):
    """稳定的 batch embedding"""
    return model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True  # bge 推荐 cosine 标准化
    )


def top_k_cosine(query_vec, matrix, k):
    """用余弦相似度取 top-k"""
    sims = matrix @ query_vec
    idx = np.argsort(sims)[::-1][:k]
    return idx


def build_prompt_original(chunks, question):
    context = "\n\n".join(chunks)
    context = context[:MAX_CONTEXT_CHARS]

    return f"""You are a medical expert. Given the following context, answer the question. 

Context:
{context}

Question:
{question}

Answer with only one letter (A/B/C/D/E)."""


def build_prompt_qa(qas, question):
    qa_text = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qas])
    qa_text = qa_text[:MAX_CONTEXT_CHARS]

    return f"""You are a medical expert. Given the following context, answer the question. 

Reference QA:
{qa_text}

Question:
{question}

Answer with only one letter (A/B/C/D/E)."""


def build_prompt_direct(question):
    """新增：直接回答，不提供任何信息"""
    return f"""You are a medical expert.

Answer the question.

Question:
{question}

Answer with only one letter (A/B/C/D/E)."""


# ============================ 主流程 ============================

def main():

    print("Loading local embedding model...")
    emb_model = SentenceTransformer(
        EMB_MODEL_PATH, device="cuda", trust_remote_code=True
    )

    # ------------------------
    # Load 1 chapter
    # ------------------------
    print("Loading chapter...")
    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        chapter_text = f.read()

    print("Chunking...")
    chunks = [chapter_text[i:i+SPILT_LENGTH] for i in range(0, len(chapter_text), SPILT_LENGTH)]
    print("Total chunks:", len(chunks))

    print("Embedding chapter chunks...")
    chunk_vecs = embed_texts(emb_model, chunks)

    # ------------------------
    # Load QA dataset
    # ------------------------
    print("Loading QA pairs...")
    qa_list = load_jsonl(QA_FILE)
    for i, qa in enumerate(qa_list):
        qa["qa_id"] = f"QA-{i+1}"

    qa_questions = [qa["question"] for qa in qa_list]

    print("Embedding QA questions...")
    qa_vecs = embed_texts(emb_model, qa_questions)

    qa_usage = {qa["qa_id"]: 0 for qa in qa_list}

    # ------------------------
    # Load test questions
    # ------------------------
    print("Loading test questions...")
    tests = load_jsonl(TEST_FILE)

    f_orig = open(OUT_ORIGINAL, "w", encoding="utf-8")
    f_qa = open(OUT_QA, "w", encoding="utf-8")
    f_dir = open(OUT_DIRECT, "w", encoding="utf-8")

    correct_orig = 0
    correct_qa = 0
    correct_dir = 0

    print("Evaluating...")
    for test in tqdm(tests):

        q_text = test["question"]

        # ---- embed question ----
        q_vec = embed_texts(emb_model, [q_text])[0]

        # ======================
        # A) Direct（模型直接作答）
        # ======================
        prompt_d = build_prompt_direct(q_text)
        ans_d = llm_answer(prompt_d)

        f_dir.write(json.dumps({
            "id": test["id"],
            "pred": ans_d,
            "gold": test["answer"],
            "correct": ans_d == test["answer"]
        }) + "\n")

        if ans_d == test["answer"]:
            correct_dir += 1

        # =======================
        # B) 原文检索 + in-context
        # =======================
        idx = top_k_cosine(q_vec, chunk_vecs, TOP_K_ORIGINAL)
        selected_chunks = [chunks[i] for i in idx]

        prompt_o = build_prompt_original(selected_chunks, q_text)
        ans_o = llm_answer(prompt_o)

        f_orig.write(json.dumps({
            "id": test["id"],
            "pred": ans_o,
            "gold": test["answer"],
            "correct": ans_o == test["answer"],
            "chunks_used": idx.tolist()
        }) + "\n")

        if ans_o == test["answer"]:
            correct_orig += 1

        # =======================
        # C) QA 检索 + in-context
        # =======================
        idx2 = top_k_cosine(q_vec, qa_vecs, TOP_K_QA)
        retrieved_qas = [qa_list[i] for i in idx2]

        # count usage
        for qa in retrieved_qas:
            qa_usage[qa["qa_id"]] += 1

        prompt_q = build_prompt_qa(retrieved_qas, q_text)
        ans_q = llm_answer(prompt_q)

        f_qa.write(json.dumps({
            "id": test["id"],
            "pred": ans_q,
            "gold": test["answer"],
            "correct": ans_q == test["answer"],
            "used_qas": [qa["qa_id"] for qa in retrieved_qas]
        }) + "\n")

        if ans_q == test["answer"]:
            correct_qa += 1

    f_orig.close()
    f_qa.close()
    f_dir.close()

    # ------------------------
    # 保存 QA 使用次数
    # ------------------------
    with open(OUT_QA_USAGE, "w", encoding="utf-8") as f:
        json.dump(qa_usage, f, indent=2, ensure_ascii=False)

    print("========== SUMMARY ==========")
    print("Direct answer accuracy:", correct_dir, "/", len(tests))
    print("Original context accuracy:", correct_orig, "/", len(tests))
    print("QA context accuracy:", correct_qa, "/", len(tests))
    print("QA usage saved to:", OUT_QA_USAGE)


if __name__ == "__main__":
    main()
