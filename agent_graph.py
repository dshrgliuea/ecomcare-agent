
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from order_tools import query_order
from vector_tools import search_vector_knowledge


# 读取项目根目录下的 .env
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if not API_KEY:
    raise ValueError(
        "没有读取到 DASHSCOPE_API_KEY，请检查项目根目录下的 .env 文件。"
    )


SYSTEM_PROMPT = """
你是一个专业、友好的电商智能客服助手。

你可以使用以下工具：
1. 订单查询工具：查询订单状态、物流等订单信息。
2. 售后知识库工具：检索退货、退款、换货等售后政策。

工作要求：
- 用户询问具体订单状态或物流时，优先调用订单查询工具。
- 用户询问退货、退款、换货或售后政策时，调用售后知识库工具。
- 如果一个问题同时涉及订单和售后政策，可以调用两个工具。
- 根据工具返回的真实结果回答，不得编造订单状态、物流信息或政策。
- 如果缺少订单号，向用户询问订单号。
- 如果工具没有找到相关信息，诚实说明无法确认。
- 使用简洁、自然的中文回答用户。
"""


@tool
def lookup_order(order_id: str) -> str:
    """根据订单号查询电商订单状态和物流信息。"""
    result = query_order(order_id)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def search_after_sales_knowledge(question: str) -> str:
    """检索退货、退款、换货和其他售后政策知识。"""
    results = search_vector_knowledge(question, top_k=3)

    if not results:
        return "知识库中没有找到相关内容。"

    formatted_results = []

    for item in results:
        content = item.get("content", "")
        similarity = item.get("similarity")

        formatted_results.append(
            f"知识内容：{content}\n"
            f"相似度：{similarity}"
        )

    return "\n\n".join(formatted_results)


# 使用阿里云百炼兼容 OpenAI 接口的 Qwen 模型
model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
    timeout=60,
    max_retries=1,
)


# 创建 LangGraph Agent
agent = create_react_agent(
    model=model,
    tools=[
        lookup_order,
        search_after_sales_knowledge,
    ],
    prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    print("正在测试 LangGraph Agent……")

    test_question = "帮我查询订单10001的物流信息"

    print(f"\n用户问题：{test_question}\n")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": test_question,
                }
            ]
        }
    )

    print("Agent 执行过程：")

    for message in result["messages"]:
        message_type = type(message).__name__

        if message_type == "ToolMessage":
            print(f"\n[工具结果] {message.content}")

        elif message_type == "AIMessage":
            if message.content:
                print(f"\n[AI 回复] {message.content}")

    print("\n测试结束。")
