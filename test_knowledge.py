
from vector_tools import search_vector_knowledge

TEST_CASES = [
    ("个人原因退货", "我不喜欢这条牛仔裤，穿起来不合适，可以退吗？"),
    ("质量问题", "收到的商品有瑕疵，应该如何申请质量问题售后？"),
    ("仅退款", "我不方便寄回商品，可以只退款不退货吗？"),
    ("退货运费", "个人原因退货和商品质量问题退货的运费由谁承担？"),
    ("换货政策", "尺码不合适，我可以申请换货吗？"),
    ("人工客服", "我想联系人工客服处理售后问题。"),
]

if __name__ == "__main__":
    for category, question in TEST_CASES:
        print("\n" + "=" * 65)
        print(f"测试类别：{category}")
        print(f"用户问题：{question}")
        print("-" * 65)

        try:
            results = search_vector_knowledge(question, top_k=3)

            if not results:
                print("没有检索到知识片段")
                continue

            for index, item in enumerate(results, start=1):
                print(f"\n知识片段 {index}")
                print(f"相似度：{item.get('similarity', '未知')}")
                print(item.get("content", item))

        except Exception as error:
            print(f"检索失败：{type(error).__name__}: {error}")
