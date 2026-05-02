import sqlite3
import os
import asyncio
from pathlib import Path
from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent.parent / "data" / "mock_data.db"

def _run_sql(query: str) -> str:
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows: return "查询成功，但结果为空。"
            columns = [desc[0] for desc in cursor.description]
            header = " | ".join(columns)
            body = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
            return f"{header}\n{'-' * len(header)}\n{body}"
    except Exception as e:
        return f"ERROR: SQL 执行失败: {str(e)}"

@tool
async def execute_sql(query: str) -> str:
    """在 mock_data.db 上执行 SELECT SQL 查询。参数 query: SQL 语句。"""
    return await asyncio.to_thread(_run_sql, query)

@tool
def search_knowledge_base(query: str) -> str:
    """查询金融风控专业术语、公司政策、操作手册。"""
    # 🚨 修正：从 core 导入单例
    from src.core.vector_store import get_kb_instance
    print(f"\n[Agent 动作] 🕵️‍♂️ 正在翻阅风控手册: {query}...")
    return get_kb_instance().query(query)