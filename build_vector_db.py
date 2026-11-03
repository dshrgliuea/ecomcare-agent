
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. 项目配置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
DB_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env", override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY")

BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

EMBEDDING_MODEL = "text-embedding-v3"

COLLECTION_NAME = "ecommerce_after_sales"

BATCH_SIZE = 10

# 只读取这六份正式知识库文件。
# 原来的 after_sales.txt 保留作为备份，但不会参与本次构建。
KNOWLEDGE_FILES = [
    {
        "filename": "return_policy.txt",
        "policy_type": "return",
        "policy_name": "个人原因退货",
        "policy_id": "RETURN-001",
        "version": "demo-1.0",
        "verified": "partial",
    },
    {
        "filename": "quality_policy.txt",
        "policy_type": "quality",
        "policy_name": "商品质量问题",
        "policy_id": "QUALITY-001",
        "version": "demo-1.0",
        "verified": "partial",
    },
    {
        "filename": "refund_policy.txt",
        "policy_type": "refund",
        "policy_name": "退款与仅退款",
        "policy_id": "REFUND-001",
        "version": "demo-1.0",
        "verified": "partial",
    },
    {
        "filename": "shipping_policy.txt",
        "policy_type": "shipping",
        "policy_name": "运费与退货物流",
        "policy_id": "SHIPPING-001",
        "version": "demo-1.0",
        "verified": "pending",
    },
    {
        "filename": "exchange_policy.txt",
        "policy_type": "exchange",
        "policy_name": "商品换货",
        "policy_id": "EXCHANGE-001",
        "version": "demo-1.0",
        "verified": "pending",
    },
    {
        "filename": "human_service.txt",
        "policy_type": "human_service",
        "policy_name": "人工客服",
        "policy_id": "SERVICE-001",
        "version": "demo-1.0",
        "verified": "partial",
    },
]


# ============================================================
# 2. 初始化百炼 Embedding 客户端
# ============================================================

def create_embedding_client():

    if not API_KEY:
        raise ValueError(
            "未读取到 DASHSCOPE_API_KEY。"
            "请检查项目根目录下的 .env 文件。"
        )

    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


# ============================================================
# 3. 读取知识库文件
# ============================================================

def read_knowledge_files() -> List[Dict[str, Any]]:

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}"
        )

    documents = []

    for config in KNOWLEDGE_FILES:

        file_path = KNOWLEDGE_DIR / config["filename"]

        if not file_path.exists():
            raise FileNotFoundError(
                f"缺少知识库文件：{file_path}"
            )

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            raise ValueError(
                f"知识库文件内容为空：{file_path}"
            )

        documents.append({
            "config": config,
            "content": content,
        })

        print(
            f"已读取：{config['filename']} "
            f"({len(content)} 个字符)"
        )

    return documents


# ============================================================
# 4. 按段落切分知识库
# ============================================================

def split_text(
    text: str,
    max_chars: int = 700,
) -> List[str]:
    """
    按自然段落切分，尽量保持规则内容完整。
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(paragraph) > max_chars:

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # 对特别长的段落按字符长度切分
            for start in range(0, len(paragraph), max_chars):
                chunks.append(
                    paragraph[start:start + max_chars]
                )

            continue

        candidate = (
            current_chunk + "\n" + paragraph
            if current_chunk
            else paragraph
        )

        if len(candidate) <= max_chars:
            current_chunk = candidate

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ============================================================
# 5. 构建文档及元数据
# ============================================================

def build_documents(
    source_documents: List[Dict[str, Any]]
):
    documents = []
    metadatas = []
    ids = []

    for source in source_documents:

        config = source["config"]
        content = source["content"]

        chunks = split_text(content)

        print(
            f"{config['filename']}："
            f"切分为 {len(chunks)} 个知识片段"
        )

        for index, chunk in enumerate(chunks):

            unique_text = (
                config["policy_id"]
                + "_"
                + str(index)
                + "_"
                + chunk
            )

            chunk_hash = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()[:16]

            document_id = (
                config["policy_id"]
                + "_"
                + str(index)
                + "_"
                + chunk_hash
            )

            metadata = {
                "policy_type": config["policy_type"],
                "policy_name": config["policy_name"],
                "policy_id": config["policy_id"],
                "source_file": config["filename"],
                "version": config["version"],
                "verified": config["verified"],
                "chunk_index": index,
            }

            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(document_id)

    return documents, metadatas, ids


# ============================================================
# 6. 调用百炼生成向量
# ============================================================

def create_embeddings(
    client: OpenAI,
    texts: List[str],
) -> List[List[float]]:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # 按照原始输入顺序排列
    sorted_data = sorted(
        response.data,
        key=lambda item: item.index,
    )

    return [
        item.embedding
        for item in sorted_data
    ]


# ============================================================
# 7. 重建 Chroma Collection
# ============================================================

def rebuild_collection(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
):

    print("\n正在初始化 Chroma 数据库……")

    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chroma_client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    # 只清空当前知识库 Collection，不删除整个数据库目录。
    # orders.db 等其他项目数据不会受到影响。
    existing = collection.get()

    old_ids = existing.get("ids", [])

    if old_ids:

        print(
            f"正在清除旧知识片段：{len(old_ids)} 条"
        )

        collection.delete(
            ids=old_ids
        )

    print("\n开始生成向量并写入 Chroma……")

    total = len(documents)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch_documents = documents[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = ids[start:end]

        print(
            f"正在处理 {start + 1} - {end} / {total}"
        )

        embeddings = create_embeddings(
            embedding_client,
            batch_documents,
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
            embeddings=embeddings,
        )

    print("\nChroma 写入完成！")

    return collection


# ============================================================
# 8. 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("电商智能客服知识库重建程序")
    print("=" * 60)

    try:

        print("\n第一步：初始化 Embedding 客户端")

        embedding_client = create_embedding_client()

        print(
            f"Embedding 模型：{EMBEDDING_MODEL}"
        )

        print("\n第二步：读取六份知识库文件")

        source_documents = read_knowledge_files()

        print("\n第三步：切分知识片段")

        documents, metadatas, ids = build_documents(
            source_documents
        )

        if not documents:
            raise ValueError(
                "没有生成任何知识片段，停止构建。"
            )

        print(
            f"\n共生成 {len(documents)} 个知识片段"
        )

        print("\n第四步：重建 Chroma 数据库")

        # 提供给 rebuild_collection 使用
        collection = rebuild_collection(
            documents,
            metadatas,
            ids,
        )

        print("\n第五步：验证构建结果")

        print(
            f"Collection：{COLLECTION_NAME}"
        )

        print(
            f"知识片段总数：{collection.count()}"
        )

        print(
            f"数据库位置：{DB_DIR}"
        )

        print("\n知识库重建成功！")

    except Exception as exc:

        print("\n知识库重建失败！")
        print(
            f"错误类型：{type(exc).__name__}"
        )
        print(f"错误信息：{exc}")

        raise
