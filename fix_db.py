# fix_db.py (必须放在项目根目录下，和 main.py 挨着)
import sqlite3
from pathlib import Path


def force_inject_data():
    # 强制获取当前脚本所在的绝对路径（即项目根目录）
    root_dir = Path(__file__).resolve().parent
    db_path = root_dir / "data" / "mock_data.db"

    print("==================================================")
    print(f"🎯 [物理定位] 准备写入的绝对路径为：\n{db_path}")
    print("==================================================")

    # 确保 data 文件夹存在
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 连接数据库
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 强制建表
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

    # 清空并注入数据
    cursor.execute('DELETE FROM risk_indicators')
    test_data = [
        ('Jack', 780, 0, '正常还款', 50000.0),
        ('Alice', 450, 45, '催收中', 120000.0),
        ('Bob', 610, 5, '关注类', 15000.0),
        ('星际贸易公司', 820, 0, '经营稳健', 5000000.0),
        ('月球矿业集团', 580, 12, '资金链偏紧', 18000000.0)
    ]
    cursor.executemany('''
    INSERT INTO risk_indicators (target_name, credit_score, dpd, recent_status, total_debt)
    VALUES (?, ?, ?, ?, ?)
    ''', test_data)

    conn.commit()
    conn.close()

    print("✅ [注入成功] 探长，真金白银已经放入金库，您可以让探员去查了！")


if __name__ == "__main__":
    force_inject_data()