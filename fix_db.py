# -*- coding: utf-8 -*-
import sqlite3
import os
from pathlib import Path


def force_inject_data():
    # 1. 确定路径：确保数据放在 data 文件夹下
    root_dir = Path(__file__).resolve().parent
    db_dir = root_dir / "data"
    db_path = db_dir / "mock_data.db"

    print("=" * 50)
    print(f"🎯 [数据中心] 正在定位金库：\n{db_path}")
    print("=" * 50)

    # 2. 确保文件夹存在
    if not db_dir.exists():
        db_dir.mkdir(parents=True)
        print(f"📁 已创建缺失的文件夹: {db_dir}")

    # 3. 连接并操作数据库
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 4. 强制初始化表结构
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS risk_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_name TEXT NOT NULL,
        credit_score INTEGER,
        dpd INTEGER,
        recent_status TEXT,
        total_debt REAL
    )
    ''')

    # 5. 清空旧数据并注入新“案情”
    cursor.execute('DELETE FROM risk_indicators')

    # 💡 探长请看：我给“星际重工”安排了一个 95 天的严重逾期，看它怎么狡辩！
    test_data = [
        ('Jack', 780, 0, '正常还款', 50000.0),
        ('Alice', 450, 45, '催收中', 120000.0),
        ('Bob', 610, 5, '关注类', 15000.0),
        ('星际贸易公司', 820, 0, '经营稳健', 5000000.0),
        ('月球矿业集团', 580, 12, '资金链偏紧', 18000000.0),
        # 🚨 关键目标已上线！
        ('星际重工', 320, 95, '高危违约', 88000000.0)
    ]

    cursor.executemany('''
    INSERT INTO risk_indicators (target_name, credit_score, dpd, recent_status, total_debt)
    VALUES (?, ?, ?, ?, ?)
    ''', test_data)

    conn.commit()
    conn.close()

    print("\n✅ [物理注入成功] 探长，数据已入库！")
    print("🕵️‍♂️ 现在的数据库情况：")
    print("   - 星际重工：逾期 95 天 (高危)")
    print("   - 月球矿业：逾期 12 天 (偏紧)")
    print("=" * 50)


if __name__ == "__main__":
    force_inject_data()