
import sqlite3

# 创建数据库文件
conn = sqlite3.connect("orders.db")
cursor = conn.cursor()

# 创建订单表
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_name TEXT,
    product_name TEXT,
    order_status TEXT,
    logistics_company TEXT,
    tracking_number TEXT,
    refund_status TEXT
)
""")

# 插入模拟订单数据
orders = [
    (
        "10001",
        "张三",
        "蓝色牛仔裤",
        "已发货",
        "顺丰速运",
        "SF123456789",
        "未申请退款"
    ),
    (
        "10002",
        "李四",
        "白色运动鞋",
        "待发货",
        "暂无",
        "暂无",
        "未申请退款"
    ),
    (
        "10003",
        "王五",
        "黑色外套",
        "已签收",
        "中通快递",
        "ZT987654321",
        "退款处理中"
    )
]

cursor.executemany("""
INSERT OR REPLACE INTO orders
VALUES (?, ?, ?, ?, ?, ?, ?)
""", orders)

conn.commit()
conn.close()

print("订单数据库创建成功！")
print("已添加 3 条模拟订单。")
