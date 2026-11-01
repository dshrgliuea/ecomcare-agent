
from pathlib import Path
import re


# 获取知识库文件路径
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "after_sales.txt"


def load_knowledge():
    """读取售后知识库文件。"""

    if not KNOWLEDGE_FILE.exists():
        return []

    content = KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )

    # 按章节分割知识库
    sections = re.split(
        r"(?=^[一二三四五六]、)",
        content,
        flags=re.MULTILINE
    )

    return [
        section.strip()
        for section in sections
        if section.strip()
    ]


def search_knowledge(question: str, top_k: int = 3):
    """
    使用关键词匹配检索知识库。

    当前是简单的本地检索版本，
    后续升级为向量检索。
    """

    sections = load_knowledge()

    if not sections:
        return []

    keywords = {
        "退货": ["退货", "七天", "无理由", "申请售后"],
        "退款": ["退款", "到账", "支付", "退回"],
        "运费": ["运费", "邮费", "快递费", "运费险"],
        "物流": ["物流", "快递", "配送", "运单"],
        "人工": ["人工", "客服", "投诉", "争议"],
        "质量": ["质量", "破损", "损坏", "瑕疵"],
    }

    expanded_terms = set()

    for term, related in keywords.items():
        if term in question or any(
            word in question for word in related
        ):
            expanded_terms.update(related)

    # 添加问题中的常见词
    for word in keywords:
        if word in question:
            expanded_terms.add(word)

    scored_sections = []

    for section in sections:
        score = 0

        for term in expanded_terms:
            if term in section:
                score += 1

        scored_sections.append(
            (score, section)
        )

    scored_sections.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        section
        for score, section in scored_sections[:top_k]
        if score > 0
    ]


if __name__ == "__main__":

    question = "买了五天的衣服可以退货吗？"

    results = search_knowledge(question)

    print("用户问题：", question)
    print()

    for index, result in enumerate(results, start=1):
        print(f"检索结果 {index}：")
        print(result)
        print("-" * 30)
