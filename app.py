import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent_graph import (
    agent,
    lookup_order,
    search_after_sales_knowledge,
)


# =====================================================
# 1. 基础配置
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen-plus").strip()

MIN_SIMILARITY = 0.50

WELCOME_MESSAGE = (
    "您好！欢迎来到电商智能客服。\n\n"
    "我可以帮您查询演示订单、了解物流状态，"
    "也可以解答退货、退款、换货及商品质量问题。"
)

ORDER_FALLBACK = (
    "目前没有获得有效的订单查询结果，"
    "暂时无法确认订单的具体情况。"
)

KNOWLEDGE_FALLBACK = (
    "目前没有获得足够的售后政策信息，"
    "暂时无法确认具体规则。建议查看商品页面的售后说明，"
    "或联系商家客服进一步核实。"
)


# =====================================================
# 2. 页面配置
# =====================================================

st.set_page_config(
    page_title="电商智能客服 Agent",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ 电商智能客服 Agent")
st.caption("通义千问 + LangGraph + SQLite + Chroma")

if not API_KEY:
    st.error("未检测到百炼 API Key，请检查项目根目录下的 .env 文件。")
    st.stop()


# =====================================================
# 3. 通用数据解析
# =====================================================

def parse_json(content):
    """兼容字符串、字典和其他工具返回格式。"""

    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            result = json.loads(content)

            if isinstance(result, dict):
                return result

        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def message_type(message):
    """兼容 LangChain 消息对象。"""

    return getattr(message, "type", "")


def message_content(message):
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


# =====================================================
# 4. 知识库结果解析
# =====================================================

def parse_knowledge_result(content):

    data = parse_json(content)

    if data:
        references = (
            data.get("references")
            or data.get("results")
            or []
        )

        if isinstance(references, list):
            cleaned = []

            for item in references:
                if not isinstance(item, dict):
                    continue

                text = str(
                    item.get("content", "")
                ).strip()

                try:
                    similarity = float(
                        item.get("similarity", 1)
                    )
                except (ValueError, TypeError):
                    similarity = 0

                if text and similarity >= MIN_SIMILARITY:
                    cleaned.append({
                        "content": text,
                        "similarity": similarity,
                    })

            return {
                "success": bool(cleaned),
                "references": cleaned,
            }

    if not isinstance(content, str):
        return {
            "success": False,
            "references": [],
        }

    references = []

    patterns = [
        re.compile(
            r"【知识片段\s*\d+】\s*"
            r"([\s\S]*?)"
            r"检索相似度：\s*([0-9.]+)"
        ),
        re.compile(
            r"知识内容：\s*"
            r"([\s\S]*?)"
            r"相似度：\s*([0-9.]+)"
        ),
    ]

    for pattern in patterns:

        for match in pattern.finditer(content):

            knowledge_text = match.group(1).strip()

            try:
                similarity = float(match.group(2))
            except ValueError:
                continue

            if (
                knowledge_text
                and similarity >= MIN_SIMILARITY
            ):
                references.append({
                    "content": knowledge_text,
                    "similarity": similarity,
                })

        if references:
            break

    # 去重
    unique = []
    seen = set()

    for ref in references:
        key = ref["content"]

        if key not in seen:
            seen.add(key)
            unique.append(ref)

    return {
        "success": bool(unique),
        "references": unique,
    }


# =====================================================
# 5. 提取本轮工具结果
# =====================================================

def collect_tool_data(messages):

    orders = []
    references = []
    used_tools = set()
    tool_messages = []

    for message in messages:

        if message_type(message) != "tool":
            continue

        name = getattr(message, "name", "")
        content = message_content(message)

        used_tools.add(name)
        tool_messages.append(message)

        if name == "lookup_order":

            data = parse_json(content)

            if (
                data.get("success") is True
                and isinstance(data.get("order"), dict)
            ):
                orders.append(data["order"])

        elif name == "search_after_sales_knowledge":

            data = parse_knowledge_result(content)

            if data["success"]:
                references.extend(data["references"])

    return orders, references, used_tools, tool_messages


# =====================================================
# 6. 问题识别
# =====================================================

def extract_order_id(text):

    match = re.search(
        r"(?<!\d)\d{5}(?!\d)",
        text,
    )

    return match.group() if match else None


def refers_to_previous_order(text):

    phrases = [
        "这个订单",
        "该订单",
        "这笔订单",
        "刚才的订单",
        "上一个订单",
        "这个单",
        "这笔单",
        "刚才那笔",
        "这单",
        "它",
        "这条裤子",
        "这件商品",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def classify_question(text):

    order_words = [
        "订单",
        "发货",
        "发了吗",
        "物流",
        "快递",
        "签收",
        "配送",
        "运单号",
    ]

    policy_words = [
        "退货",
        "退款",
        "换货",
        "售后",
        "运费",
        "七天",
        "无理由",
        "审核",
        "到账",
        "寄回",
        "不想要",
        "能退",
        "质量问题",
        "质量",
        "瑕疵",
        "破损",
        "退钱",
        "仅退款",
        "不方便寄",
    ]

    needs_order = (
        extract_order_id(text) is not None
        or any(word in text for word in order_words)
        or refers_to_previous_order(text)
    )

    needs_policy = any(
        word in text
        for word in policy_words
    )

    return needs_order, needs_policy


def build_retrieval_question(text):

    if any(
        word in text
        for word in [
            "仅退款",
            "不方便寄",
            "不想寄回",
            "不用寄回",
        ]
    ):
        return (
            "退款与仅退款政策。"
            "商家是否支持仅退款？"
            "已签收商品是否支持仅退款？"
            "用户不方便寄回商品能否仅退款？"
            "仅退款的适用条件和审核规则是什么？"
        )

    if any(
        word in text
        for word in [
            "质量",
            "瑕疵",
            "破损",
            "做工",
        ]
    ):
        return (
            "商品质量问题售后政策。"
            "商品存在瑕疵、破损或做工问题如何申请售后？"
            "需要提供什么问题描述或凭证？"
            "商家如何审核？是否可以退货、换货或退款？"
        )

    if any(
        word in text
        for word in [
            "运费",
            "运费险",
        ]
    ):
        return (
            "退货运费和运费险政策。"
            "质量问题退货运费由谁承担？"
            "个人原因退货运费如何处理？"
            "运费险适用规则是什么？"
        )

    if any(
        word in text
        for word in [
            "不喜欢",
            "不合适",
            "无理由",
            "七天",
            "不想要",
        ]
    ):
        return (
            "个人原因退货和七天无理由退货政策。"
            "签收日起几天内可以申请？"
            "商品完好、不影响二次销售的条件是什么？"
            "如何申请普通退货退款？"
        )

    return text


# =====================================================
# 7. 订单回答
# =====================================================

def build_order_answer(order):

    if not order:
        return ORDER_FALLBACK

    order_id = order.get("order_id", "未知")
    status = order.get("order_status", "未知")

    lines = [
        f"订单 **{order_id}** 当前记录的状态为：**{status}**。"
    ]

    fields = [
        ("商品名称", "product_name"),
        ("物流公司", "logistics_company"),
        ("物流单号", "tracking_number"),
        ("退款状态", "refund_status"),
    ]

    for label, key in fields:

        value = order.get(key)

        if value not in (None, ""):
            lines.append(f"- {label}：{value}")

    if status == "已发货":

        lines.append(
            "\n目前订单记录为已发货，"
            "但系统没有提供实时物流轨迹，"
            "因此暂时无法确认最新配送位置。"
        )

    return "\n".join(lines)


# =====================================================
# 8. 售后政策可信回答
# =====================================================

def build_policy_answer(references, question):

    if not references:
        return KNOWLEDGE_FALLBACK

    contents = "\n".join(
        str(ref.get("content", ""))
        for ref in references
    )

    lines = []

    # -------------------------------------------------
    # 仅退款
    # -------------------------------------------------

    only_refund = any(
        word in question
        for word in [
            "仅退款",
            "不方便寄",
            "不想寄回",
            "不用寄回",
        ]
    )

    if only_refund:

        if (
            "尚未明确确认" in contents
            or "是否支持仅退款" in contents
            or "不能直接回答支持" in contents
        ):
            return (
                "关于您提出的仅退款需求，目前知识库尚未明确确认"
                "商家是否支持仅退款，也没有明确不方便寄回商品时"
                "是否可以直接退款。\n\n"
                "因此，我暂时不能承诺可以免退货退款。"
                "具体是否支持、适用条件及审核要求，"
                "需要以商家实际售后政策和审核结果为准。\n\n"
                "目前知识库记录的普通退货退款流程是："
                "申请售后 → 等待商家审核 → "
                "审核通过后按照系统提供的地址寄回商品 → "
                "商家收货检查后处理退款。"
            )

        return (
            "目前检索到的知识内容不足以确认该商品是否支持仅退款。"
            "建议进一步向商家核实，不能将普通退货退款"
            "直接理解为仅退款。"
        )

    # -------------------------------------------------
    # 质量问题
    # -------------------------------------------------

    quality_related = any(
        word in question
        for word in [
            "质量",
            "瑕疵",
            "破损",
            "做工",
        ]
    )

    if quality_related:

        if (
            "商品质量问题" in contents
            or "质量问题售后" in contents
        ):

            lines.append(
                "**商品质量问题售后申请流程：**"
            )

            lines.append(
                "1. 进入「我的订单」，选择对应商品并申请售后。"
            )

            lines.append(
                "2. 选择与商品质量问题相关的售后原因。"
            )

            lines.append(
                "3. 按页面提示填写问题描述，"
                "并根据实际情况提供图片或视频凭证。"
            )

            lines.append(
                "4. 提交后等待商家审核。"
            )

            lines.append(
                "商家审核后，根据实际情况处理退货、换货或退款。"
            )

            lines.append(
                "\n**需要注意：**仅凭用户认为商品有瑕疵，"
                "不能直接认定为质量问题；"
                "是否属于质量问题以及最终处理方式，"
                "均需以实际情况和商家审核结果为准。"
            )

            lines.append(
                "目前知识库尚未明确质量问题的具体认定标准、"
                "申请期限、退货运费承担规则和审核时效。"
            )

            return "\n\n".join(lines)

    # -------------------------------------------------
    # 个人原因退货
    # -------------------------------------------------

    return_related = any(
        word in question
        for word in [
            "退货",
            "退回",
            "寄回",
            "不喜欢",
            "不合适",
            "不想要",
            "无理由",
            "七天",
            "能退",
        ]
    )

    if return_related:

        if (
            ("7天" in contents or "七天" in contents)
            and (
                "不影响二次销售" in contents
                or "商品完好" in contents
            )
        ):

            lines.append(
                "根据当前演示知识库，七天无理由退货期限"
                "从商品签收之日起计算，应在签收之日起7天内申请。"
            )

            lines.append(
                "商品应保持完好，不影响二次销售。"
                "已洗涤、严重污损或人为损坏的服装，"
                "不符合当前演示知识库记录的无理由退货条件。"
            )

            lines.append(
                "具体商品是否支持七天无理由退货，"
                "以商品页面标注及商家实际规则为准。"
            )

            lines.append(
                "目前尚未确认签收日期或商品状态时，"
                "不能直接判断是否符合退货条件。"
            )

        if (
            "申请售后" in contents
            and "退货退款" in contents
        ):

            lines.append(
                "\n普通退货退款流程：进入「我的订单」→ "
                "选择对应商品 →「申请售后」→ "
                "选择「退货退款」→ 等待商家审核。"
                "审核通过后，按照系统提供的地址寄回商品。"
            )

        if lines:
            return "\n\n".join(lines)

    # -------------------------------------------------
    # 普通退款
    # -------------------------------------------------

    refund_related = any(
        word in question
        for word in [
            "退款",
            "退钱",
            "到账",
        ]
    )

    if refund_related:

        if "原支付渠道" in contents:

            lines.append(
                "当前知识库记录：商家收到退货并完成检查后"
                "开始处理退款，原则上退回原支付渠道。"
                "具体到账时间取决于支付渠道及银行，"
                "目前不能承诺具体天数。"
            )

            return "\n\n".join(lines)

    # -------------------------------------------------
    # 运费
    # -------------------------------------------------

    if any(
        word in question
        for word in [
            "运费",
            "运费险",
        ]
    ):

        if (
            "待商家确认" in contents
            or "尚未明确" in contents
        ):
            return (
                "目前知识库尚未明确确认该场景下的退货运费"
                "由哪一方承担，也无法确认运费险的具体适用条件。"
                "建议以商家实际售后政策和订单页面说明为准。"
            )

    return KNOWLEDGE_FALLBACK


# =====================================================
# 9. 生成最终可信回答
# =====================================================

def build_verified_answer(
    user_input,
    orders,
    references,
    previous_order=None,
    model_reply="",
):

    needs_order, needs_policy = classify_question(user_input)

    if not needs_order and not needs_policy:
        return model_reply or (
            "您好，请告诉我您需要查询的订单或想了解的售后问题。"
        )

    requested_id = extract_order_id(user_input)

    effective_order = None

    if requested_id:

        effective_order = next(
            (
                order
                for order in orders
                if str(order.get("order_id")) == requested_id
            ),
            None,
        )

    elif (
        refers_to_previous_order(user_input)
        and previous_order
    ):

        effective_order = previous_order

        effective_order = next(
            (
                order
                for order in orders
                if str(order.get("order_id"))
                == str(previous_order.get("order_id"))
            ),
            effective_order,
        )

    elif orders:

        effective_order = orders[-1]

    parts = []

    if needs_order:

        parts.append(
            build_order_answer(effective_order)
        )

    if needs_policy:

        parts.append(
            build_policy_answer(
                references,
                user_input,
            )
        )

    return "\n\n".join(parts)


# =====================================================
# 10. Session State
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]

if "current_order" not in st.session_state:
    st.session_state.current_order = None

if "last_tool_debug" not in st.session_state:
    st.session_state.last_tool_debug = []


# =====================================================
# 11. 侧边栏
# =====================================================

with st.sidebar:

    st.header("⚙️ 智能客服系统")

    st.success("百炼 API Key 已读取")

    st.write("当前模型：", MODEL_NAME)

    st.success("LangGraph Agent 已接入")

    st.divider()

    st.subheader("📦 演示订单")

    st.write("10001：已发货")
    st.write("10002：待发货")
    st.write("10003：已签收")

    st.divider()

    st.subheader("📚 向量知识库")

    st.success("Chroma 语义检索已接入")

    st.caption(
        f"相似度过滤阈值：{MIN_SIMILARITY:.2f}"
    )

    st.divider()

    if st.button("🗑️ 清空聊天记录"):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]

        st.session_state.current_order = None
        st.session_state.last_tool_debug = []

        st.rerun()

    st.info(
        "演示系统，不连接真实商城，"
        "不执行真实退款或售后操作。"
    )


# =====================================================
# 12. 显示历史聊天
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# =====================================================
# 13. 用户输入
# =====================================================

user_input = st.chat_input(
    "例如：订单10001发货了吗？"
)


# =====================================================
# 14. Agent 主流程
# =====================================================

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Agent 正在分析并查询工具..."):

                previous_order = (
                    st.session_state.current_order
                )

                needs_order, needs_policy = (
                    classify_question(user_input)
                )

                requested_id = extract_order_id(user_input)

                # -------------------------------------
                # 14.1 准备历史消息
                # 只传递用户和助手的对话内容
                # 不把旧工具消息重复传入 Agent
                # -------------------------------------

                history = []

                for item in st.session_state.messages[-21:]:

                    if item["role"] not in [
                        "user",
                        "assistant",
                    ]:
                        continue

                    history.append({
                        "role": item["role"],
                        "content": item["content"],
                    })

                # 当前用户问题之前的消息数量
                input_message_count = len(history)

                # -------------------------------------
                # 14.2 调用 LangGraph Agent
                # -------------------------------------

                result = agent.invoke({
                    "messages": history
                })

                result_messages = result.get(
                    "messages",
                    [],
                )

                # 只处理本轮新增消息
                new_messages = result_messages[
                    input_message_count:
                ]

                (
                    orders,
                    references,
                    used_tools,
                    tool_messages,
                ) = collect_tool_data(new_messages)

                debug_results = []

                # -------------------------------------
                # 14.3 订单工具兜底
                # -------------------------------------

                context_order_id = requested_id

                if (
                    not context_order_id
                    and previous_order
                    and refers_to_previous_order(user_input)
                ):

                    context_order_id = str(
                        previous_order.get("order_id")
                    )

                if (
                    needs_order
                    and context_order_id
                    and "lookup_order" not in used_tools
                ):

                    try:

                        raw = lookup_order.invoke({
                            "order_id": context_order_id
                        })

                        data = parse_json(raw)

                        if (
                            data.get("success") is True
                            and isinstance(data.get("order"), dict)
                        ):

                            orders.append(data["order"])

                            used_tools.add("lookup_order")

                            debug_results.append({
                                "name": "lookup_order",
                                "content": raw,
                            })

                    except Exception as tool_error:

                        debug_results.append({
                            "name": "lookup_order",
                            "content": (
                                "订单查询失败："
                                + str(tool_error)
                            ),
                        })

                # -------------------------------------
                # 14.4 知识库工具兜底
                # -------------------------------------

                if (
                    needs_policy
                    and not references
                ):

                    retrieval_question = (
                        build_retrieval_question(user_input)
                    )

                    try:

                        raw = search_after_sales_knowledge.invoke({
                            "question": retrieval_question
                        })

                        knowledge_data = (
                            parse_knowledge_result(raw)
                        )

                        if knowledge_data["success"]:

                            references.extend(
                                knowledge_data["references"]
                            )

                            used_tools.add(
                                "search_after_sales_knowledge"
                            )

                        debug_results.append({
                            "name": "search_after_sales_knowledge",
                            "content": raw,
                        })

                    except Exception as tool_error:

                        debug_results.append({
                            "name": "search_after_sales_knowledge",
                            "content": (
                                "知识库检索失败："
                                + str(tool_error)
                            ),
                        })

                # -------------------------------------
                # 14.5 保存订单上下文
                # -------------------------------------

                if orders:

                    st.session_state.current_order = (
                        orders[-1]
                    )

                # -------------------------------------
                # 14.6 提取本轮模型回答
                # -------------------------------------

                ai_messages = [
                    message
                    for message in new_messages
                    if message_type(message) == "ai"
                    and message_content(message).strip()
                ]

                model_reply = (
                    message_content(ai_messages[-1])
                    if ai_messages
                    else ""
                )

                # -------------------------------------
                # 14.7 生成最终回答
                # -------------------------------------

                ai_reply = build_verified_answer(
                    user_input=user_input,
                    orders=orders,
                    references=references,
                    previous_order=previous_order,
                    model_reply=model_reply,
                )

                st.session_state.last_tool_debug = (
                    debug_results
                )

        except Exception as error:

            st.error("智能客服运行失败。")

            error_message = str(error)

            if API_KEY:
                error_message = error_message.replace(
                    API_KEY,
                    "[已隐藏密钥]",
                )

            st.code(
                f"{type(error).__name__}: {error_message}"
            )

            st.stop()

        # ---------------------------------------------
        # 14.8 显示工具执行记录
        # ---------------------------------------------

        if tool_messages or debug_results:

            with st.expander("🔧 查看 Agent 工具执行结果"):

                for message in tool_messages:

                    st.markdown(
                        "**工具："
                        f"{getattr(message, 'name', '未知工具')}**"
                    )

                    content = message_content(message)

                    st.code(content)

                for item in debug_results:

                    st.markdown(
                        f"**补充调用：{item['name']}**"
                    )

                    content = item["content"]

                    if not isinstance(content, str):

                        content = json.dumps(
                            content,
                            ensure_ascii=False,
                            default=str,
                        )

                    st.code(content)

        st.write(ai_reply)

    # ---------------------------------------------
    # 14.9 保存最终回答
    # ---------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply,
    })