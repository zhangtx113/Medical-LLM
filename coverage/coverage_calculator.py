import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
from tqdm import tqdm

# ================= 配置 =================
LOCAL_MODEL_PATH = "/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/Models/deberta-large-mnli"
device = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= 加载模型 =================
print(f"Loading DeBERTa model from: {LOCAL_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True).to(device).eval()

# ================= 核心语义提取 =================
def extract_core_semantics(text):
    """
    简单核心语义提取，去掉停用词和非字母字符
    """
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    from spacy.lang.en.stop_words import STOP_WORDS
    return [w for w in words if w not in STOP_WORDS]

# ================= 批量嵌入 + 缓存 =================
@torch.no_grad()
def batch_encode_texts(texts, prefix="context", layer_index=-2):
    all_embeddings = []

    for text in tqdm(texts, desc=f"Encoding {prefix}"):
        cache_file = os.path.join(CACHE_DIR, f"{prefix}_{hash(text)}.pt")
        if os.path.exists(cache_file):
            emb = torch.load(cache_file, map_location=device)
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index][0]  # [seq_len, hidden_size]
            emb = hidden_states.mean(dim=0, keepdim=True)          # 平均池化 -> [1, H]
            emb = F.normalize(emb, p=2, dim=1)                     # 向量归一化
            torch.save(emb, cache_file)
        all_embeddings.append(emb)

    return torch.cat(all_embeddings, dim=0)  # [num_texts, hidden_size]

# ================= 覆盖度计算 =================
def compute_coverage(contexts, qa_pairs):
    """
    直接平均段落与 QA 相似度，不加权
    """
    qa_texts = [qa["question"] + " " + qa.get("think", "") + " " + qa.get("answer", "") for qa in qa_pairs]

    # 批量嵌入
    ctx_emb = batch_encode_texts(contexts, prefix="context")
    qa_emb = batch_encode_texts(qa_texts, prefix="qa")

    # 归一化
    ctx_emb = F.normalize(ctx_emb, p=2, dim=1)
    qa_emb = F.normalize(qa_emb, p=2, dim=1)

    # 矩阵化余弦相似度
    sim_matrix = torch.mm(ctx_emb, qa_emb.T)
    sim_matrix = torch.clamp(sim_matrix, min=0.0)  # 避免负值

    max_sim, _ = sim_matrix.max(dim=1)  # 每段落最大相似度
    coverage = max_sim.mean()           # 平均得到总覆盖度

    return coverage.item()

# ================= 文件读取 =================
def read_markdown_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_qa_jsonl(path):
    qa_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qa_list.append(json.loads(line))
    return qa_list

# ================= 主程序 =================
if __name__ == "__main__":
    context_path = "/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/dataset/en_data/En_Part6.mmd"
    qa_path = "/mnt/inaisfs/data/home/zhaozc_criait/zhangtx/Medical_LLM/dataset/output/qa_dataset6.jsonl"

    print("Reading context 6 and QA data...")
    context_text = read_markdown_text(context_path)
    qa_pairs = read_qa_jsonl(qa_path)

    print("Calculating coverage...")
    coverage = compute_coverage([context_text], qa_pairs)

    print(f"\n=== Weighted Coverage: {coverage:.4f} ===")

