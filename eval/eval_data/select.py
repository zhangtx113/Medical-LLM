import re
import json


def read_md_file(file_path):
    """读取Markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到文件: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件时出错: {str(e)}")


def clean_text(raw_content):
    """清洗文本内容，移除无关标记"""
    cleaned = re.sub(r'<--- Page Split --->', '', raw_content)
    cleaned = re.sub(r'!\[\]\(images/.*?\)', '', cleaned)
    cleaned = re.sub(r'<center>.*?</center>', '', cleaned, flags=re.DOTALL)
    return cleaned


def extract_question_blocks(cleaned_content):
    """从清洗后的内容中提取题目块（标记+内容）"""
    blocks = re.split(r'(## QUESTION \d+)', cleaned_content)
    question_blocks = []
    for i in range(1, len(blocks), 2):
        if i + 1 < len(blocks):
            question_blocks.append((blocks[i], blocks[i + 1]))
    return question_blocks


def process_question_block(mark, content, answer_dict):
    """处理单个题目块，结合答案字典生成标准化题目"""
    # 提取题号
    num_match = re.search(r'QUESTION (\d+)', mark)
    if not num_match:
        return None
    question_num = num_match.group(1)
    question_id = f"QUESTION-{question_num}"

    # 检查是否包含图片引用
    if 'Fig.' in content:
        return None

    # 处理内容
    content = content.strip()
    if not content:
        return None

    # 分离问题文本和选项
    options_split = re.split(r'(?=[A-E]\. )', content, 1)
    if len(options_split) < 2:
        return None
    question_text, options_text = options_split

    # 提取选项
    option_pattern = re.compile(r'([A-E])\. (.*?)(?=\s*[A-E]\. |$)', re.DOTALL)
    options = {}
    for match in option_pattern.finditer(options_text):
        key = match.group(1)
        value = re.sub(r'\s+', ' ', match.group(2).strip())
        options[key] = value

    if not options:
        return None

    # 处理问题文本
    question_text = re.sub(r'\s+', ' ', question_text).strip()

    # 获取正确答案（默认A如果未找到对应答案）
    correct_answer = answer_dict.get(question_id, "A")
    if question_id not in answer_dict:  # 严格检查键是否存在
        print(question_id)

    return {
        "id": question_id,
        "question": question_text,
        "options": options,
        "answer": correct_answer
    }


def process_all_questions(question_blocks, answer_dict):
    """处理所有题目块，结合答案字典返回有效题目列表"""
    valid_questions = []
    for mark, content in question_blocks:
        question = process_question_block(mark, content, answer_dict)
        if question:
            valid_questions.append(question)
    return valid_questions


def extract_answer_dict(answer_md_content):
    """从answer1.md内容中提取答案字典{QUESTION-ID: 答案}"""
    answer_dict = {}
    # 匹配所有答案块
    answer_pattern = re.compile(
        r'## ANSWER TO QUESTION (\d+)\s+## ([A-E])',
        re.DOTALL
    )
    # 查找所有匹配
    for match in answer_pattern.finditer(answer_md_content):
        question_num = match.group(1)
        answer = match.group(2).strip()
        question_id = f"QUESTION-{question_num}"
        answer_dict[question_id] = answer
    return answer_dict


def save_to_jsonl(questions, output_file):
    """将题目列表保存为JSONL文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for q in questions:
                json.dump(q, f, ensure_ascii=False)
                f.write('\n')
        print(f"成功保存 {len(questions)} 个题目到 {output_file}")
    except Exception as e:
        raise Exception(f"保存文件时出错: {str(e)}")


def main(input_md, answer_md, output_jsonl):
    """主函数：协调各步骤执行"""
    # 读取题目和答案内容
    raw_question_content = read_md_file(input_md)
    answer_content = read_md_file(answer_md)

    # 提取答案字典
    answer_dict = extract_answer_dict(answer_content)
    # print(answer_dict)
    print(f"成功提取 {len(answer_dict)} 个问题的答案")

    # 处理题目
    cleaned_content = clean_text(raw_question_content)
    question_blocks = extract_question_blocks(cleaned_content)
    valid_questions = process_all_questions(question_blocks, answer_dict)

    # 保存结果
    save_to_jsonl(valid_questions, output_jsonl)


if __name__ == "__main__":
    # 调用示例
    main(
        input_md="section1.md",  # 题目文件
        answer_md="answer1.md",  # 答案文件
        output_jsonl="eval_section1.jsonl"  # 输出文件
    )