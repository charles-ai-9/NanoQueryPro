# -*- coding: utf-8 -*-
import os, sys, asyncio, uuid, warnings, logging
from pathlib import Path
from dotenv import load_dotenv

# ── 消音配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
for _lib in ["httpx", "sentence_transformers", "httpcore", "huggingface_hub"]:
    logging.getLogger(_lib).setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(dotenv_path=ROOT_DIR / ".env")
warnings.filterwarnings("ignore")

from langchain_core.messages import AIMessage
from src.core.llm_client import get_llm
from src.agent.engine import AgentEngine
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore


async def main() -> None:
    print("\n" + "═" * 54 + "\n║   🤖 星际金融风控系统 - V5.0.0 (Engine 架构版)   ║\n" + "═" * 54)
    print("\n[System] 正在初始化探员大脑...")
    get_llm()
    print("[System] 大脑初始化完毕 ✅")

    db_path = str(ROOT_DIR / "data" / "memory" / "nanoquery.db")

    async with AsyncSqliteStore.from_conn_string(db_path) as global_store:
        await global_store.setup()
        async with AsyncSqliteSaver.from_conn_string(db_path) as memory:

            # ── 实例化 AgentEngine（编译图一次，全程复用）──────────────────────
            engine = AgentEngine(memory=memory, store=global_store)

            while True:
                print("")
                custom_id = input("🔌 Session ID (回车生成, exit 退出): ").strip()
                if custom_id.lower() == "exit":
                    break
                thread_id = custom_id if custom_id else str(uuid.uuid4())[:8]
                config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

                while True:
                    question = input(f"\n💬 [{thread_id}] 探长请提问: ").strip()
                    if not question or question.lower() in ["q", "quit"]:
                        break

                    try:
                        print("\n" + "─" * 20 + " ⚙️ Agent 实时流 " + "─" * 20)

                        # ── 第一轮：发起提问，接收流式输出 ────────────────────
                        hitl_signal = None
                        async for chunk in engine.stream_run(question, config):
                            if chunk["type"] == "text":
                                print(chunk["content"], end="", flush=True)
                            elif chunk["type"] == "tool_start":
                                print(f"\n\033[96m[⚙️ 系统播报: 探员正在启动工具 {chunk['name']}...]\033[0m\n")
                            elif chunk["type"] == "HITL_REQUIRED":
                                hitl_signal = chunk

                        print("\n" + "─" * 68)

                        # ── HITL 审批循环：每轮审批后继续接收新的流式输出 ────────
                        while hitl_signal:
                            print("\n" + "⏸️ " * 15 + "\n⚠️ [HITL 审批中]：等待授权！")
                            for tc in hitl_signal["tool_calls"]:
                                print(f"🔍 任务: \033[93m{tc['name']}\033[0m | 参数: \033[93m{tc['args']}\033[0m")

                            action = input("⚖️ 审批 (y: 放行 / n: 驳回 / f: 指示): ").strip().lower()
                            feedback = ""
                            if action == "f":
                                feedback = input("🗣️ 输入口谕: ").strip()

                            # 在重置前保存本轮的关键信息
                            current_tool_calls = hitl_signal["tool_calls"]
                            current_target_node = hitl_signal["target_node"]
                            hitl_signal = None  # 重置，准备接收下一轮拦截信号

                            print("\n" + "─" * 20 + " ⚙️ Agent 实时流 " + "─" * 20)

                            async for chunk in engine.handle_approval(
                                config,
                                action,
                                current_tool_calls,
                                current_target_node,
                                feedback,
                            ):
                                if chunk["type"] == "text":
                                    print(chunk["content"], end="", flush=True)
                                elif chunk["type"] == "tool_start":
                                    print(f"\n\033[96m[⚙️ 系统播报: 探员正在启动工具 {chunk['name']}...]\033[0m\n")
                                elif chunk["type"] == "HITL_REQUIRED":
                                    hitl_signal = chunk  # 捕获新的拦截信号，继续审批循环

                            print("\n" + "─" * 68)

                    except Exception as e:
                        print(f"\n\033[91m❌ 办案过程中出现异常：\n   错误类型: {type(e).__name__}\n   错误详情: {e}\033[0m")

                    # 输出最终报告
                    try:
                        final_state = await engine.graph.aget_state(config)
                        final_ans = next(
                            (m.content for m in reversed(final_state.values["messages"])
                             if isinstance(m, AIMessage) and m.content and not m.tool_calls),
                            ""
                        )
                        if final_ans:
                            print(f"\n\033[92m✅ 【最终报告】：\n{final_ans}\033[0m")
                    except Exception:
                        pass


if __name__ == "__main__":
    asyncio.run(main())