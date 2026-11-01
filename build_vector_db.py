
import os
import re
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

KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge" / "after_sales.txt"
)

DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ecommerce_after_sales"

EMBEDDING_MODEL = "text-embedding-v3"


# =====================================
# 2. 读取售后知识库
# =====================================

def load_documents():

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"找不到知识库文件：{KNOWLEDGE_FILE}"
        )

    content = KNOWLEDGE_FILE.read_text(
        encoding="utf-8-sig"
    )

    # 按章节分割售后政策
    sections = re.split(
        r"(?=^[一二三四五六]、)",
        content,
        flags=re.MULTILINE
    )

    documents = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    if not documents:
        raise ValueError(
            "知识库文件没有可用内容"
        )

    return documents


# =====================================
# 3. 构建向量数据库
# =====================================

def build_database():

    if not API_KEY:
        raise ValueError(
            "没有读取到 DASHSCOPE_API_KEY"
        )

    print("正在读取售后知识库...")

    documents = load_documents()

    print(
        f"成功读取 {len(documents)} 个知识片段。"
    )

    # 创建百炼客户端
    print("正在连接阿里云百炼向量模型...")

    ai_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=1
    )

    # 生成文本向量
    print("正在生成文本向量...")

    response = ai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=documents
    )

    embeddings = [
        item.embedding
        for item in sorted(
            response.data,
            key=lambda item: item.index
        )
    ]

    # 创建 Chroma 客户端
    print("正在写入 Chroma 数据库...")

    chroma_client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # 获取或创建集合
    # 不再删除旧集合，避免 NotFoundError
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    # 使用固定 ID
    # 已存在则更新，不存在则添加
    ids = [
        f"policy_{index}"
        for index in range(len(documents))
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print()
    print("向量知识库创建成功！")
    print(
        f"知识片段数量：{collection.count()}"
    )
    print(
        f"数据库位置：{DB_PATH}"
    )


# =====================================
# 4. 运行程序
# =====================================

if __name__ == "__main__":
    build_database()
