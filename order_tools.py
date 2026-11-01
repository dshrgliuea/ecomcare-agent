
import sqlite3


def query_order(order_id: str) -> dict:
    """根据订单号查询订单，只返回允许顾客查看的订单业务信息。"""

    conn = sqlite3.connect("orders.db")
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                product_name,
                order_status,
                logistics_company,
                tracking_number,
                refund_status
            FROM orders
            WHERE order_id = ?
            """,
            (str(order_id),)
        )

        row = cursor.fetchone()

        if row is None:
            return {
                "success": False,
                "message": f"没有找到订单 {order_id}"
            }

        return {
            "success": True,
            "order": dict(row)
        }

    except sqlite3.Error as e:
        return {
            "success": False,
            "message": "订单查询暂时不可用，请稍后重试。",
            "error": str(e)
        }

    finally:
        conn.close()


# 单独运行此文件时，测试订单查询
if __name__ == "__main__":
    print(query_order("10001"))
