
# ============================================================
# ecomcare-agent：LangGraph 智能客服核心
# 保留现有 app.py 调用接口
# ============================================================

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain.agents import create_agent

from order_tools import query_order
from vector_tools import search_vector_knowledge


# ============================================================
# 1. 项目配置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen-plus").strip()
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()

if not API_KEY:
    raise ValueError(
        "未读取到 DASHSCOPE_API_KEY，请检查项目根目录下的 .env 文件。"
    )


# ============================================================
# 2. 初始化模型
# ============================================================

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
    timeout=60,
    max_retries=1,
)


# ============================================================
# 3. 订单查询工具
# ============================================================

@tool
def lookup_order(order_id: str) -> str:
    """根据订单号查询订单状态、商品名称、物流公司、运单号和退款状态。"""

    try:
        result = query_order(str(order_id).strip())

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"订单查询失败：{type(exc).__name__}: {exc}",
            },
            ensure_ascii=False,
        )


# ============================================================
# 4. 售后知识库工具
# ============================================================

@tool
def search_after_sales_knowledge(question: str) -> str:
    """检索七天无理由退货、质量问题、退款、换货、运费及人工客服政策。"""

    try:
        results = search_vector_knowledge(question, top_k=3)

        if not results:
            return "知识库没有检索到相关内容，无法确认对应政策。"

        formatted_results = []

        for index, item in enumerate(results, start=1):
            content = str(item.get("content", "")).strip()

            if not content:
                continue

            similarity = item.get("similarity", "未知")

            formatted_results.append(
                f"【知识片段 {index}】\n"
                f"{content}\n"
                f"检索相似度：{similarity}"
            )

        if not formatted_results:
            return "知识库没有返回有效内容，无法确认对应政策。"

        return "\n\n".join(formatted_results)

    except Exception as exc:
        return (
            f"知识库检索失败：{type(exc).__name__}: {exc}"
        )


# ============================================================
# 5. Agent 系统提示词
# ============================================================

SYSTEM_PROMPT = """
你是一个严谨、诚实的电商智能客服 Agent。

你可以使用以下工具：

1. lookup_order：查询订单事实。
2. search_after_sales_knowledge：查询售后政策。

【工具使用规则】

1. 用户询问订单、物流、发货、签收或退款状态时，
   应调用 lookup_order。
2. 用户询问退货、退款、换货、质量问题、运费或售后时，
   应调用 search_after_sales_knowledge。
3. 用户说不喜欢、不想要、不合适、款式不满意时，
   应检索个人原因退货规则。
4. 用户提到瑕疵、破损、开线、污渍或质量问题时，
   应检索质量问题规则。
5. 同时提到个人原因和瑕疵时，应分别检索两类规则。
6. 同时涉及订单和售后时，应调用两个工具。
7. 可以从历史对话中获取订单号，不要要求用户重复提供已有信息。

【事实约束】

1. 订单状态、商品、物流公司、运单号和退款状态必须来自订单工具。
2. 售后政策必须来自知识库。
3. 不得编造物流轨迹、订单状态、退款结果、时间、金额或政策。
4. 不得自行增加知识库没有记载的退货条件。
5. 不得承诺退款、换货、免运费或具体到账时间。
6. 不得声称已经替用户提交售后。
7. 知识库标注待确认的规则，必须明确说明尚待确认。
8. 不能因为用户不方便寄回商品，就推断支持仅退款。

【回答风格】

使用简体中文。
礼貌、自然、简洁。
先回答用户最关心的问题。
信息不足时，明确询问缺少的信息。
不要把个人原因退货自动判定为质量问题。
"""


# 使用 LangChain 的 create_agent 构建工具 Agent。
# create_agent 基于 LangGraph 的 Agent 执行机制。
tool_agent = create_agent(
    model=model,
    tools=[
        lookup_order,
        search_after_sales_knowledge,
    ],
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 6. 消息格式处理
# ============================================================

def get_message_content(content: Any) -> str:
    """兼容字符串、文本块列表等消息内容格式。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                value = item.get("text")
                if value is not None:
                    parts.append(str(value))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    if content is None:
        return ""

    return str(content)


def normalize_messages(messages: list) -> list:
    """将 app.py 传入的字典消息转换为 LangChain 消息对象。"""

    normalized = []

    for message in messages:
        if isinstance(message, BaseMessage):
            normalized.append(message)
            continue

        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).lower()
        content = get_message_content(
            message.get("content", "")
        )

        if role in ("user", "human"):
            normalized.append(HumanMessage(content=content))

        elif role in ("assistant", "ai"):
            normalized.append(AIMessage(content=content))

        elif role == "system":
            normalized.append(SystemMessage(content=content))

        elif role == "tool":
            # 外部传入的工具消息不直接重放，避免缺少 tool_call_id。
            continue

    return normalized


def get_latest_user_question(messages: list) -> str:
    """获取最近一条用户消息。"""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return get_message_content(message.content)

        if isinstance(message, dict):
            if message.get("role") in ("user", "human"):
                return get_message_content(
                    message.get("content", "")
                )

    return ""


def get_history_text(messages: list) -> str:
    """生成便于最终回答模型理解的历史对话文本。"""

    parts = []

    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            content = message.get("content", "")
        else:
            role = getattr(message, "type", "")
            content = getattr(message, "content", "")

        if role in ("user", "human"):
            role_name = "用户"
        elif role in ("assistant", "ai"):
            role_name = "客服"
        else:
            continue

        parts.append(
            f"{role_name}：{get_message_content(content)}"
        )

    return "\n".join(parts)


# ============================================================
# 7. 多轮订单上下文
# ============================================================

def extract_order_id(text: str) -> str:
    """提取演示系统中的五位订单号。"""

    text = get_message_content(text)

    patterns = [
        r"订单\s*[：:]?\s*(\d{5})",
        r"订单号\s*[：:]?\s*(\d{5})",
        r"\b(\d{5})\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

    return ""


def get_context_order_id(messages: list) -> str:
    """从最近的用户历史消息中获取订单号。"""

    for message in reversed(messages):

        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "type", "")
            content = getattr(message, "content", "")

        if role not in ("user", "human"):
            continue

        order_id = extract_order_id(
            get_message_content(content)
        )

        if order_id:
            return order_id

    return ""


# ============================================================
# 8. 售后意图识别
# ============================================================

PERSONAL_KEYWORDS = [
    "不喜欢",
    "不想要",
    "不合适",
    "不满意",
    "穿着不舒服",
    "穿起来不合适",
    "款式不满意",
    "颜色不喜欢",
    "买错了",
    "不符合预期",
    "七天无理由",
    "无理由退货",
]

QUALITY_KEYWORDS = [
    "质量问题",
    "有瑕疵",
    "瑕疵",
    "破损",
    "开线",
    "脱线",
    "污渍",
    "损坏",
    "质量缺陷",
    "做工问题",
    "破洞",
    "裂开",
]

ORDER_KEYWORDS = [
    "订单",
    "物流",
    "快递",
    "运单",
    "发货",
    "签收",
    "退款状态",
    "订单状态",
    "查单",
    "查一下",
]

POLICY_KEYWORDS = [
    "退货",
    "退款",
    "换货",
    "售后",
    "运费",
    "申请",
    "怎么处理",
    "怎么办",
    "可以退吗",
    "能退吗",
    "仅退款",
    "只退款",
    "人工客服",
]


def classify_intent(question: str) -> dict:
    """根据当前问题识别需要查询的业务类型。"""

    text = question.lower()

    personal = any(
        keyword in text
        for keyword in PERSONAL_KEYWORDS
    )

    quality = any(
        keyword in text
        for keyword in QUALITY_KEYWORDS
    )

    order = (
        any(keyword in text for keyword in ORDER_KEYWORDS)
        or bool(extract_order_id(text))
    )

    policy = (
        personal
        or quality
        or any(keyword in text for keyword in POLICY_KEYWORDS)
    )

    mixed = personal and quality

    return {
        "personal": personal,
        "quality": quality,
        "mixed": mixed,
        "order": order,
        "policy": policy,
    }


def build_retrieval_question(
    question: str,
    intent: dict,
) -> str:
    """将用户表达转换为更明确的知识库检索问题。"""

    if intent["mixed"]:
        return (
            "请分别检索两类售后规则：\n"
            "第一，个人原因、不喜欢、穿着不合适时的"
            "七天无理由退货条件、期限、商品要求和申请流程；\n"
            "第二，商品存在瑕疵或质量问题时的"
            "申请流程、审核方式和相关限制。\n"
            "不要将两种情形混为一谈。\n"
            f"用户原始问题：{question}"
        )

    if intent["personal"]:
        return (
            "请检索个人原因退货及七天无理由退货规则，"
            "包括申请期限、商品状态要求、申请流程和限制条件。"
            "不要自行增加知识库没有记载的条件。\n"
            f"用户问题：{question}"
        )

    if intent["quality"]:
        return (
            "请检索商品质量问题售后规则，"
            "包括申请流程、问题描述及凭证要求、审核方式、"
            "退货换货和运费规则。"
            "只返回知识库明确记载的内容。\n"
            f"用户问题：{question}"
        )

    return question


# ============================================================
# 9. 工具结果处理
# ============================================================

def extract_tool_results(agent_messages: list) -> list:
    """提取 Agent 执行期间产生的工具结果。"""

    results = []

    for message in agent_messages:
        if not isinstance(message, ToolMessage):
            continue

        tool_name = (
            getattr(message, "name", None)
            or "unknown_tool"
        )

        results.append({
            "tool": tool_name,
            "content": get_message_content(message.content),
        })

    return results


def has_tool_result(
    tool_results: list,
    tool_name: str,
) -> bool:
    return any(
        result.get("tool") == tool_name
        for result in tool_results
    )


# ============================================================
# 10. 必要工具补充调用
# ============================================================

def ensure_required_tools(
    question: str,
    messages: list,
    tool_results: list,
) -> list:
    """
    检查必要工具是否已经调用。
    如果 Agent 漏掉必要工具，则补充调用。
    """

    intent = classify_intent(question)

    # ---------- 订单工具 ----------

    if intent["order"]:

        order_id = (
            extract_order_id(question)
            or get_context_order_id(messages)
        )

        if order_id and not has_tool_result(
            tool_results,
            "lookup_order",
        ):
            result = lookup_order.invoke({
                "order_id": order_id
            })

            tool_results.append({
                "tool": "lookup_order",
                "content": get_message_content(result),
            })

    # ---------- 售后知识库工具 ----------

    if intent["policy"]:

        if not has_tool_result(
            tool_results,
            "search_after_sales_knowledge",
        ):
            retrieval_question = build_retrieval_question(
                question,
                intent,
            )

            result = search_after_sales_knowledge.invoke({
                "question": retrieval_question
            })

            tool_results.append({
                "tool": "search_after_sales_knowledge",
                "content": get_message_content(result),
            })

    return tool_results


# ============================================================
# 11. 最终回答生成
# ============================================================

FINAL_ANSWER_PROMPT = """
你是电商智能客服的最终回复生成器。

你必须严格依据本次实际工具返回结果回答。
工具没有提供的信息，不得自行补充为事实。

【个人原因退货】

1. 用户说不喜欢、不想要、不合适、款式不满意时，
   按个人原因退货情形说明。
2. 使用知识库中的七天无理由退货规则。
3. 不要将个人原因直接说成商品质量问题。
4. 如果规则要求从签收日起计算期限，但没有签收日期，
   必须说明还需要确认签收日期。
5. 不得自行判断用户一定符合退货条件。
6. 不得自行增加吊牌、水洗标等知识库没有记载的要求。

【质量问题】

1. 用户提到瑕疵、破损或质量问题时，
   使用知识库中的质量问题规则。
2. 不得擅自承诺退款、换货或审核通过。
3. 不得编造凭证要求、质量判定标准或处理时效。
4. 如果用户只说“有点瑕疵”，但没有描述具体情况，
   应先询问瑕疵表现。

【混合原因】

1. 如果用户既不喜欢商品，又提到瑕疵，
   分别说明个人原因退货与质量问题售后是不同情形。
2. 不要擅自认定商品存在质量缺陷。
3. 具体瑕疵不明确时，先请用户描述问题。

【仅退款】

1. 如果知识库没有明确确认仅退款政策，
   必须明确说明目前无法确认是否支持仅退款。
2. 不得因为用户不方便寄回商品就承诺可以仅退款。
3. 不得将普通退货退款流程说成仅退款。

【订单事实】

1. 订单状态、商品名称、物流公司、运单号和退款状态，
   必须来自订单工具。
2. 没有实时物流轨迹时，说明目前只能确认工具返回的订单信息。
3. 不得推测物流位置或预计送达时间。
4. 如果没有找到订单，明确说明没有查询到对应订单。

【通用限制】

1. 不得创造知识库没有记载的政策。
2. 不得创造操作入口、时间、金额或审核结果。
3. 不得声称已经替用户提交售后。
4. 待确认的政策必须明确标注为待确认。
5. 使用简体中文，表达自然、简洁、礼貌。
"""


def generate_final_answer(
    user_question: str,
    history_text: str,
    tool_results: list,
) -> str:

    if not tool_results:
        return (
            "您好，我可以帮您查询订单和核实售后政策。"
            "为了准确处理您的问题，请补充订单号或具体售后情况。"
        )

    evidence = []

    for index, result in enumerate(tool_results, start=1):
        evidence.append(
            f"【工具结果 {index}：{result['tool']}】\n"
            f"{result['content']}"
        )

    evidence_text = "\n\n".join(evidence)

    user_content = f"""
【此前对话】
{history_text or "无"}

【用户当前问题】
{user_question}

【本次实际工具返回结果】
{evidence_text}

请严格依据本次工具结果，结合必要的对话上下文，
生成最终客服回复。
"""

    response = model.invoke([
        SystemMessage(content=FINAL_ANSWER_PROMPT),
        HumanMessage(content=user_content),
    ])

    answer = get_message_content(
        response.content
    ).strip()

    return answer or "抱歉，目前无法生成可靠的客服回复。"


# ============================================================
# 12. 兼容 app.py 的 Agent 接口
# ============================================================

class CustomerServiceAgent:
    """
    兼容现有 app.py：

    result = agent.invoke({
        "messages": messages
    })

    最终回答位于 result["messages"][-1]
    """

    def invoke(
        self,
        input_data,
        config=None,
        **kwargs,
    ):

        raw_messages = input_data.get("messages", [])

        if not isinstance(raw_messages, list):
            raise ValueError("messages 必须是列表。")

        user_question = get_latest_user_question(
            raw_messages
        )

        if not user_question:
            raise ValueError("没有找到有效的用户问题。")

        normalized_messages = normalize_messages(
            raw_messages
        )

        history_text = get_history_text(
            raw_messages
        )

        # 第一步：让 LangGraph Agent 根据上下文自主选择工具。
        agent_result = tool_agent.invoke(
            {
                "messages": normalized_messages
            },
            config=config,
            **kwargs,
        )

        agent_messages = agent_result.get(
            "messages",
            []
        )

        # 第二步：提取 Agent 已经调用的工具结果。
        tool_results = extract_tool_results(
            agent_messages
        )

        # 第三步：检查是否遗漏必要工具。
        tool_results = ensure_required_tools(
            question=user_question,
            messages=raw_messages,
            tool_results=tool_results,
        )

        # 第四步：基于工具证据生成最终回答。
        final_answer = generate_final_answer(
            user_question=user_question,
            history_text=history_text,
            tool_results=tool_results,
        )

        # 第五步：整理输出消息，兼容 app.py。
        final_messages = list(agent_messages)

        existing_tool_messages = {
            (
                getattr(message, "name", None),
                get_message_content(message.content),
            )
            for message in final_messages
            if isinstance(message, ToolMessage)
        }

        for result in tool_results:

            key = (
                result["tool"],
                result["content"],
            )

            if key not in existing_tool_messages:
                final_messages.append(
                    ToolMessage(
                        content=result["content"],
                        tool_call_id=(
                            "fallback_"
                            + str(len(final_messages))
                        ),
                        name=result["tool"],
                    )
                )

                existing_tool_messages.add(key)

        final_messages.append(
            AIMessage(content=final_answer)
        )

        return {
            "messages": final_messages
        }


# 保持 app.py 使用的变量名称
agent = CustomerServiceAgent()


# ============================================================
# 13. 本地测试
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("电商智能客服 LangGraph Agent 测试")
    print("=" * 65)

    test_questions = [
        "收到商品后发现衣服有质量问题，应该如何申请售后？",
        "帮我查询订单10001的物流信息",
        "我的订单10001已经发货了，如果收到后发现衣服有质量问题，应该怎么处理？",
        "我不喜欢这条牛仔裤，穿起来不合适，可以退吗？",
        "我不喜欢这个东西了，我觉得有点瑕疵，不适合我。",
        "我不方便寄回去，可以只退款吗？",
    ]

    for test_question in test_questions:

        print("\n" + "=" * 65)
        print("用户问题：" + test_question)
        print("=" * 65)

        try:
            result = agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": test_question,
                    }
                ]
            })

            print("\n[工具结果]")

            for message in result["messages"]:
                if isinstance(message, ToolMessage):
                    print("\n工具名称：" + str(message.name))
                    print(
                        get_message_content(
                            message.content
                        )
                    )

            print("\n[最终客服回复]")

            print(
                get_message_content(
                    result["messages"][-1].content
                )
            )

        except Exception as exc:
            print(
                f"\n测试失败：{type(exc).__name__}: {exc}"
            )

    print("\n测试结束。")
