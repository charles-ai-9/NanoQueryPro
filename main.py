# -*- coding: utf-8 -*-
import os, sys, asyncio, uuid, warnings, importlib, logging
from pathlib import Path
from dotenv import load_dotenv

# 消音配置
logging.basicConfig(level=logging.INFO)
for lib in ["httpx", "sentence_transformers", "httpcore", "huggingface_hub"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(dotenv_path=ROOT_DIR / ".env")
warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.core.llm_client import get_llm
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore


async def process_stream(graph_obj, state_input, run_config):
    print("\n" + "─" * 20 + " ⚙️ Agent 实时流 " + "─" * 20)
    async for event in graph_obj.astream_events(state_input, run_config, version="v2"):
        kind = event["event"]
        node_name = event.get("metadata", {}).get("langgraph_node", "")
        if kind == "on_tool_start":
            print(f"\n\033[96m[⚙️ 系统播报: 探员正在启动工具 {event['name']}...]\033[0m\n")
        elif kind == "on_chat_model_stream":
            if node_name not in ["intent", "check_freshness"]:
                if event["data"]["chunk"].content: print(event["data"]["chunk"].content, end="", flush=True)
    print("\n" + "─" * 68)


async def main() -> None:
    print("\n" + "═" * 54 + "\n║   🤖 星际金融风控系统 - V4.9.2 (协议全兼容版)   ║\n" + "═" * 54)
    print("\n[System] 正在初始化探员大脑...")
    get_llm()
    print("[System] 大脑初始化完毕 ✅")

    db_path = str(ROOT_DIR / "data" / "memory" / "nanoquery.db")
    async with AsyncSqliteStore.from_conn_string(db_path) as global_store:
        await global_store.setup()
        async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
            while True:
                print("")
                custom_id = input("🔌 Session ID (回车生成, exit 退出): ").strip()
                if custom_id.lower() == "exit": break
                thread_id = custom_id if custom_id else str(uuid.uuid4())[:8]
                config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

                while True:
                    question = input(f"\n💬 [{thread_id}] 探长请提问: ").strip()
                    if not question or question.lower() in ["q", "quit"]: break

                    # 重新加载模块
                    import src.agent.nodes, src.agent.graph
                    importlib.reload(src.agent.nodes);
                    importlib.reload(src.agent.graph)
                    graph = src.agent.graph.build_graph_with_deps(memory, store=global_store)

                    await process_stream(graph, {"messages": [HumanMessage(content=question)]}, config)
                    current_state = await graph.aget_state(config)

                    while current_state.next and any("tools" in str(node) for node in current_state.next):
                        print("\n" + "⏸️ " * 15 + "\n⚠️ [HITL 审批中]：等待授权！")
                        last_msg = current_state.values["messages"][-1]
                        for tc in last_msg.tool_calls:
                            print(f"🔍 任务: \033[93m{tc['name']}\033[0m | 参数: \033[93m{tc['args']}\033[0m")

                        action = input("⚖️ 审批 (y: 放行 / edit: 修改 / n: 驳回 / f: 指示): ").strip().lower()
                        target_node = current_state.next[0]

                        if action == "y":
                            await process_stream(graph, None, config)
                        elif action == "f":
                            feedback = input("🗣️ 输入口谕: ")
                            # 🚨 协议闭环：先为每个 tool_call 生成对应 ToolMessage，关闭 tool_calls 链
                            # 再追加 HumanMessage 作为探长的指导意见，确保 Qwen 协议完整
                            fb_tool_msgs = [
                                ToolMessage(
                                    tool_call_id=tc["id"],
                                    name=tc["name"],
                                    content=f"【已收到探长口谕，暂停执行原计划】"
                                )
                                for tc in last_msg.tool_calls
                            ]
                            fb_human_msg = HumanMessage(content=f"【探长口谕】：{feedback} 请根据此指示重新规划并执行。")
                            await graph.aupdate_state(
                                config,
                                {"messages": fb_tool_msgs + [fb_human_msg]},
                                as_node=target_node
                            )
                            await process_stream(graph, None, config)
                        elif action == "n":
                            rejects = [ToolMessage(tool_call_id=tc["id"], name=tc["name"], content="已驳回") for tc in
                                       last_msg.tool_calls]
                            await graph.aupdate_state(config, {"messages": rejects}, as_node=target_node)
                            await process_stream(graph, None, config)
                        current_state = await graph.aget_state(config)

                    final_state = await graph.aget_state(config)
                    final_ans = next((m.content for m in reversed(final_state.values["messages"]) if
                                      isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
                    if final_ans: print(f"\n\033[92m✅ 【最终报告】：\n{final_ans}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())