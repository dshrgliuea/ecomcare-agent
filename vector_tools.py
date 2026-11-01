
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# =====================================
# 1. 项目配置
# =====================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env",
    override=True
)

API_KEY = os.getenv(
    "DASHSCOPE_API_KEY", ""
).strip()

BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()

DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ecommerce_after_sales"

EMBEDDING_MODEL = "text-embedding-v3"


# =====================================
# 2. 语义检索工具
# =====================================

def search_vector_knowledge(question, top_k=3):
    """
    根据用户问题进行向量检索。

    返回最相关的售后政策及相似度。
    """

    if not API_KEY:
        raise ValueError(
            "未读取到 DASHSCOPE_API_KEY"
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "找不到 Chroma 数据库，"
            "请先运行 build_vector_db.py"
        )

    # 连接已有的向量数据库
    chroma_client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        return []

    # 连接百炼向量模型
    ai_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=1
    )

    # 将用户问题转换成向量
    response = ai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question]
    )

    query_embedding = response.data[0].embedding

    # Chroma 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            top_k,
            collection.count()
        ),
        include=[
            "documents",
            "distances"
        ]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # cosine 距离转换为相似度
    return [
        {
            "content": document,
            "similarity": round(
                1 - distance,
                4
            )
        }
        for document, distance in zip(
            documents,
            distances
        )
    ]


# =====================================
# 3. 独立测试
# =====================================

if __name__ == "__main__":

    question = (
        "衣服买回来不想要了，"
        "能不能寄回去？"
    )

    print("用户问题：", question)
    print()

    results = search_vector_knowledge(
        question,
        top_k=3
    )

    if not results:
        print("没有找到相关知识。")

    for index, result in enumerate(
        results,
        start=1
    ):

        print(f"检索结果 {index}")
        print(
            f"相似度：{result['similarity']}"
        )
        print(result["content"])
        print("-" * 40)
