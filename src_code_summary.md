# src 目录代码整合

---

**__init__.py**
```python

```

---

**agent/__init__.py**


---

**agent/graph.py**
```python
"""
graph.py - NanoQuery 拓扑升级：支持工具执行循环与物理装甲
"""
import sys
import os
from src.agent.nodes import (
    check_data_freshness_node,
    generate_sql_node,
    intent_node,
    parallel_detective_node,  # 🌟 确保引入
    distribute_tasks,
    aggregate_reports_node
)

# langgraph dev 加载此模块时 stdout 可能为 ASCII 编码
# 在此处强制重绑定为 UTF-8，防止任何中文/emoji 输出触发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from src.agent.nodes import check_data_freshness_node, generate_sql_node, intent_node
from src.agent.state import MessagesState
from src.tools.sql_tools import execute_sql, search_knowledge_base
from src.agent.subgraphs.rca_graph import rca_graph

# 意图路由器，根据当前会话状态决定流程走向
# state: 当前的消息状态对象，包含上下文信息，封装好了后续用到
def intent_router(state: MessagesState):
    """[交通警察]：根据 route 字段决定流程分支"""
    route = state.route

    # 🌟 集团篇核心逻辑：如果是并行模式，直接调用分发函数
    # distribute_tasks 会返回一个包含多个 Send 对象的列表 [Send(...), Send(...)]
    # LangGraph 看到列表里是 Send 对象，就会自动开启“影分身”并发模式
    if route == "parallel":
        return distribute_tasks(state)

    # --- 以下是原有的单兵路由逻辑 ---
    if route == "chat":
        return END
    if route == "meta":
        return "generate_sql"
    if route == "analysis":
        return "rca_subgraph"
    # 其他情况（默认 business），进入数据新鲜度检查节点
    return "check_freshness"


def _build_builder() -> StateGraph:
    builder = StateGraph(MessagesState)

    network_armor = RetryPolicy(
        initial_interval=2.0,
        backoff_factor=2.0,
        max_attempts=3
    )

    # ============== 步骤 A：注册节点 (关键修正) ==============
    builder.add_node("intent", intent_node, retry_policy=network_armor)
    builder.add_node("check_freshness", check_data_freshness_node)
    builder.add_node("generate_sql", generate_sql_node, retry_policy=network_armor)
    builder.add_node("tools", ToolNode([execute_sql, search_knowledge_base]))
    builder.add_node("rca_subgraph", rca_graph)

    # 🌟 必须注册分身探员和主编节点，否则报错 NodeNotFound
    builder.add_node("parallel_detective", parallel_detective_node)
    builder.add_node("aggregate_reports", aggregate_reports_node)

    # ============== 步骤 B：布置连线 (关键修正) ==============
    builder.add_edge(START, "intent")

    # 意图路由
    builder.add_conditional_edges(
        "intent",
        intent_router,
        {
            END: END,
            "generate_sql": "generate_sql",
            "check_freshness": "check_freshness",
            "rca_subgraph": "rca_subgraph",
            "parallel_detective": "parallel_detective"
        }
    )

    # 常规业务流
    builder.add_edge("check_freshness", "generate_sql")
    builder.add_edge("rca_subgraph", END)

    # 🌟 并行协作流：Map-Reduce 的精髓
    # 1. 所有分身干完活，必须去汇总节点集合，不能连向 END
    builder.add_edge("parallel_detective", "aggregate_reports")
    # 2. 汇总节点写完综述后，正式结案
    builder.add_edge("aggregate_reports", END)

    builder.add_conditional_edges(
        "generate_sql",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "generate_sql")
    return builder

def build_graph():
    """[供 LangGraph Studio/Server 框架调用]。
    新版 langgraph-api >= 0.7.95 要求工厂函数必须无参数，
    checkpointer 和 store 由框架在运行时自动注入。
    """
    return _build_builder().compile(interrupt_before=["tools"])

def build_graph_with_deps(memory=None, store=None):
    """[供 main.py 本地运行调用]。
    手动传入 checkpointer (memory) 和 store，实现持久化与长期记忆。
    """
    return _build_builder().compile(
        checkpointer=memory,
        store=store,
        interrupt_before=["tools"]
    )
```

---

**agent/nodes.py**
```python
import logging
import traceback
from functools import lru_cache
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from .state import MessagesState, IntentOutput, SupervisorDecision, Plan
from src.core.llm_client import get_llm
from src.tools.sql_tools import execute_sql, search_knowledge_base

logger = logging.getLogger(__name__)


def safe_get(state, key, default=None):
    if isinstance(state, dict): return state.get(key, default)
    return getattr(state, key, default)


def sanitize_history(messages):
    """
    🚨 绝对无菌协议洗白器 (Rebuild Everything)
    彻底销毁原有的 AIMessage 和 ToolMessage 对象，提纯纯文本重新实例化。
    绝不保留任何 additional_kwargs 隐藏属性，让 Qwen 只看到纯粹的文字对话历史。
    """
    clean_msgs = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = msg.content if msg.content else "【系统日志】：发起了一次数据检索请求。"
            if not isinstance(content, str):
                content = str(content)
            clean_msgs.append(AIMessage(content=content))
        elif isinstance(msg, ToolMessage):
            content = msg.content if msg.content else "无结果"
            clean_msgs.append(HumanMessage(content=f"📊 [工具检索结果]:\n{content}"))
        else:
            clean_msgs.append(msg)
    return clean_msgs


@lru_cache(maxsize=1)
def get_llm_with_tools():
    _llm = get_llm()
    return _llm.bind_tools([execute_sql, search_knowledge_base])


async def intent_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    【意图中心】：返璞归真，使用最稳健的文本匹配
    """
    _llm = get_llm()
    messages = safe_get(state, "messages", [])
    if not messages: return {"route": "chat"}

    system_prompt = """
你是指挥中心的意图路由器。请根据探长的输入，判断意图并【严格只输出以下英文单词之一】，绝不要有任何标点符号或其他废话：
    - PLAN: 包含先后顺序的复合任务（例如“先查...再查...最后写...”）
    - CHAT: 闲聊或打招呼
    - BUSINESS: 常规的业务查询、查数或查手册
    """

    try:
        res = await _llm.ainvoke([SystemMessage(content=system_prompt)] + messages[-1:])
        res_text = res.content.strip().upper()
        if "PLAN" in res_text:
            route = "plan"
        elif "CHAT" in res_text:
            route = "chat"
        else:
            route = "business"
        if route == "chat":
            return {"route": "chat", "messages": [AIMessage(content="收到，探长！")]}
        return {"route": route, "messages": [AIMessage(content=f"📝 确认意图: {route.upper()}")]}
    except Exception as e:
        logger.error(f"意图识别故障: {e}")
        return {"route": "business"}


async def planner_node(state: MessagesState):
    _llm = get_llm()
    planner_brain = _llm.with_structured_output(Plan)
    messages = sanitize_history(safe_get(state, "messages", []))
    system_prompt = "你是军师。请将任务拆解为步骤列表 JSON。例如: ['查询逾期', '查手册', '写报告']。"
    try:
        plan_result = await planner_brain.ainvoke([SystemMessage(content=system_prompt)] + messages[-5:])
        return {"plan": plan_result.steps, "messages": [AIMessage(content=f"摸底计划已制定。")]}
    except:
        return {"plan": ["查询数据", "查阅政策", "输出建议"]}


def task_dispatcher_node(state: MessagesState):
    plan = safe_get(state, "plan", [])
    if not plan: return {"messages": [AIMessage(content="✅ 计划执行完毕。")]}
    current_task = plan[0]
    return {
        "plan": plan[1:],
        "messages": [HumanMessage(content=f"🎯 【当前阶段任务】：{current_task}\n请局长立刻派人执行。")]
    }


async def supervisor_node(state: MessagesState):
    """
    【铁血局长】：抛弃脆弱的 JSON，使用硬核文本匹配
    """
    print("\n\033[91m🚨 [DEBUG] 探长，我是新的纯文本局长！如果您没看到这句话，说明旧代码还在跑！\033[0m")
    _llm = get_llm()
    messages = sanitize_history(safe_get(state, "messages", []))
    plan = safe_get(state, "plan", [])
    system_prompt = """
你是风控局指挥官。请根据最新的【当前阶段任务】指示，决定下一步由谁来干活。
你【必须且只能】回复以下三个英文单词之一，绝不允许包含任何标点符号或额外解释：
    - sql_detective : 如果任务需要查询数据库、底层业务数据
    - knowledge_agent : 如果任务需要查阅风控操作手册、处罚标准
    - FINISH : 如果当前阶段任务已经完成，或者无事可做
    """
    try:
        res = await _llm.ainvoke([SystemMessage(content=system_prompt)] + messages[-6:])
        res_text = res.content.strip().lower()
        if "sql" in res_text:
            next_action = "sql_detective"
            instruction = "请调取数据库执行核查。"
        elif "knowledge" in res_text:
            next_action = "knowledge_agent"
            instruction = "请翻阅风控操作手册。"
        else:
            next_action = "FINISH"
            instruction = "当前阶段任务结束，准备流转。"
        logger.info(f"👔 [局长决策]：去向 -> {next_action}")
        return {
            "route": next_action,
            "messages": [AIMessage(content=f"👔 【局长指令】：{instruction}", name="supervisor")]
        }
    except Exception as e:
        logger.warning(f"局长发生不可抗力网络故障 ({type(e).__name__}: {e})，启动容错降级...")
        if plan:
            return {"route": "task_dispatcher", "messages": [AIMessage(content="【系统容错】强行推入下一项计划。")]}
        return {"route": "FINISH", "messages": [AIMessage(content="【系统容错】收队。")]}


async def generate_sql_node(state: MessagesState):
    _llm_with_tools = get_llm_with_tools()
    messages = sanitize_history(safe_get(state, "messages", []))
    sys = SystemMessage(content="你是 SQL 探员。查完后直接文本汇报结果。")
    response = await _llm_with_tools.ainvoke([sys] + messages[-8:])
    return {"messages": [response]}


async def knowledge_node(state: MessagesState):
    _llm_with_tools = get_llm_with_tools()
    messages = sanitize_history(safe_get(state, "messages", []))
    sys = SystemMessage(content="你是知识库专家。请通过工具查阅事实，严禁脑补。")
    response = await _llm_with_tools.ainvoke([sys] + messages[-5:])
    return {"messages": [response]}


async def check_data_freshness_node(state: MessagesState):
    return {"messages": [SystemMessage(content="水位：2024-12-23")], "data_freshness": "2024-12-23"}

# src/agent/subgraphs/rca_graph.py
import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.core.llm_client import get_llm

class RcaState(BaseModel):
    sql_result: str = Field(default="", description="传入的 SQL 查询结果")
    analysis: str = Field(default="", description="输出的归因分析报告")

async def rca_analyse_node(state: RcaState) -> Dict[str, Any]:
    _llm = get_llm()
    sql_data = state.sql_result.strip()
    if len(sql_data) > 5000:
        sql_data = sql_data[:2500] + "\n...[数据过长已截断]...\n" + sql_data[-2500:]
    if not sql_data or "空" in sql_data:
        return {"analysis": "未发现异常数据，无需归因。"}
    messages = [
        SystemMessage(content=(
            "你是一名金融风控专家。当用户要求'核查'异常数据时：\n"
            "1. 首先锁定异常发生的具体日期和维度。\n"
            "2. 必须生成 SQL 来查询该异常点背后的【明细数据】（如具体流水），而不是去看无关的表。\n"
            "3. 对比该异常点与前后日期的分布差异。"
        )),
        HumanMessage(content=f"用户要求核查异常，已知前置汇总数据为：{sql_data}。请开始下钻分析。")
    ]
    response = await _llm.ainvoke(messages)
    return {"analysis": response.content}

def build_rca_subgraph() -> CompiledStateGraph:
    sg = StateGraph(RcaState)
    sg.add_node("rca_analyse_node", rca_analyse_node)
    sg.set_entry_point("rca_analyse_node")
    sg.add_edge("rca_analyse_node", END)
    return sg.compile()

rca_graph = build_rca_subgraph()

# src/agent/subgraphs/__init__.py

# src/agent/__init__.py

# src/agent/graph.py
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.agent.state import MessagesState
from src.tools.sql_tools import execute_sql, search_knowledge_base
from src.agent.nodes import (
    intent_node, supervisor_node, generate_sql_node, knowledge_node,
    planner_node, task_dispatcher_node, safe_get
)

def intent_router(state: MessagesState):
    """
    【交通警察】：精准分发路由
    """
    route = safe_get(state, "route", "chat")
    if route == "plan":
        return "planner"
    if route == "chat":
        return END
    return "supervisor"

def supervisor_router(state: MessagesState):
    """
    【局长传送门】：拦截早退
    """
    next_action = safe_get(state, "route", "FINISH")
    if next_action == "FINISH":
        if safe_get(state, "plan", []):
            return "task_dispatcher"
        return END
    return next_action

def _build_builder() -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node("intent", intent_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sql_detective", generate_sql_node)
    builder.add_node("knowledge_agent", knowledge_node)
    builder.add_node("planner", planner_node)
    builder.add_node("task_dispatcher", task_dispatcher_node)
    builder.add_node("sql_tools", ToolNode([execute_sql]))
    builder.add_node("knowledge_tools", ToolNode([search_knowledge_base]))
    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        intent_router,
        {
            "planner": "planner",
            "supervisor": "supervisor",
            END: END
        }
    )
    builder.add_edge("planner", "task_dispatcher")
    builder.add_edge("task_dispatcher", "supervisor")
    builder.add_conditional_edges("supervisor", supervisor_router, {
        "sql_detective": "sql_detective",
        "knowledge_agent": "knowledge_agent",
        "task_dispatcher": "task_dispatcher",
        END: END
    })
    builder.add_conditional_edges("sql_detective", tools_condition, {"tools": "sql_tools", END: "supervisor"})
    builder.add_edge("sql_tools", "sql_detective")
    builder.add_conditional_edges("knowledge_agent", tools_condition, {"tools": "knowledge_tools", END: "supervisor"})
    builder.add_edge("knowledge_tools", "knowledge_agent")
    return builder

def build_graph_with_deps(memory=None, store=None):
    return _build_builder().compile(checkpointer=memory, store=store, interrupt_before=["sql_tools", "knowledge_tools"])

# src/__init__.py

# src/core/llm_client.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    mode = os.getenv("LLM_MODE", "local").lower()
    try:
        if mode == "cloud":
            from langchain_community.chat_models.tongyi import ChatTongyi
            api_key = os.getenv("DASHSCOPE_API_KEY")
            model_name = os.getenv("CLOUD_MODEL_NAME", "qwen-max")
            if not api_key:
                raise ValueError("未配置 DASHSCOPE_API_KEY，请检查 .env 文件")
            _llm_instance = ChatTongyi(
                model=model_name,
                dashscope_api_key=api_key,
                temperature=0.7,
                top_p=0.9,
                streaming=True
            )
            logger.info("已切换至云端模式：通义千问 %s (已开启 Streaming)", model_name)
        else:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("OPENAI_API_BASE")
            model_name = os.getenv("MODEL_NAME")
            if not api_key:
                raise ValueError("未配置 OPENAI_API_KEY，请检查 .env 文件")
            _llm_instance = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=api_base,
                temperature=0.7,
                streaming=True,
                model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            )
            logger.info("已切换至本地模式：自建模型 %s (已开启 Streaming)", model_name)
        return _llm_instance
    except Exception as e:
        logger.error("LLM 初始化失败 (模式: %s): %s", mode, str(e))
        return None

# src/core/instances.py
# -*- coding: utf-8 -*-
"""
src/core/instances.py

全局单例注册中心：负责管理跨模块共享的重量级对象（如知识库）。
将此类实例化逻辑放在最底层的 core 包中，彻底消除 agent -> tools -> agent 的循环依赖。
"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_kb_instance():
    """
    知识库单例工厂。
    使用 lru_cache 确保全局只初始化一次 KnowledgeBase 实例。
    """
    from src.core.vector_store import KnowledgeBase
    try:
        kb = KnowledgeBase()
        kb.load_index()
        logger.info("✅ [知识库] 索引加载成功")
        return kb
    except Exception as e:
        logger.error(f"❌ [知识库] 索引加载失败: {e}")
        raise

# src/core/vector_store.py
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

_kb_instance = None

@lru_cache(maxsize=1)
def get_kb_instance():
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
        if not _kb_instance.load_index():
            _kb_instance.build_index()
    return _kb_instance

class EnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: List[float]
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        all_results = [r.invoke(query) for r in self.retrievers]
        scores, doc_map = {}, {}
        rrf_k = 60
        for retriever_docs, weight in zip(all_results, self.weights):
            for rank, doc in enumerate(retriever_docs):
                doc_id = doc.page_content
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (rrf_k + rank + 1))
                doc_map[doc_id] = doc
        sorted_docs = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_docs]

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
load_dotenv(dotenv_path=ROOT_DIR / ".env")
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder=str(ROOT_DIR / "data" / "models")
        )
        self.db_path = ROOT_DIR / "data" / "vector_db"
        self.vector_db = None
        self.ensemble_retriever = None
    def _setup_ensemble(self, chunks):
        faiss_retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 2
        self.ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    def build_index(self):
        knowledge_dir = ROOT_DIR / "data" / "knowledge"
        documents = []
        for file in knowledge_dir.glob("*.md"):
            try:
                loader = TextLoader(str(file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"加载失败: {e}")
        if not documents: return False
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(str(self.db_path))
        self._setup_ensemble(chunks)
        return True
    def load_index(self):
        if self.db_path.exists():
            try:
                self.vector_db = FAISS.load_local(str(self.db_path), self.embeddings, allow_dangerous_deserialization=True)
                self._setup_ensemble(list(self.vector_db.docstore._dict.values()))
                return True
            except: return False
        return False
    def query(self, question: str):
        if not self.ensemble_retriever: return "❌ 检索系统未初始化"
        docs = self.ensemble_retriever.invoke(question)
        return "\n---\n".join([doc.page_content for doc in docs])

# src/tools/__init__.py

# src/tools/sql_tools.py
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
    """
    在 mock_data.db 上执行 SELECT SQL 查询。参数 query: SQL 语句。
    """
    return await asyncio.to_thread(_run_sql, query)

@tool
def search_knowledge_base(query: str) -> str:
    """
    查询金融风控专业术语、公司政策、操作手册。
    """
    from src.core.vector_store import get_kb_instance
    print(f"\n[Agent 动作] 🕵️‍♂️ 正在翻阅风控手册: {query}...")
    return get_kb_instance().query(query)
```

main.py
```python
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


langgraph.json
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:build_graph"
  },
  "env": ".env"
}
```

requirements.txt
```
# ==========================================
# NanoQuery 金融侦探项目 - 最终环境依赖清单
# ==========================================

# 1. AI 逻辑与图编排框架
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=1.0.0
langgraph>=0.3.0
langchain_community>=0.0.30
langchain_text_splitters>=0.0.1

# 2. 检索与嵌入（Hybrid RAG 必需）
faiss-cpu>=1.7.4
transformers>=4.36.2
sentence-transformers>=2.2.2
rank-bm25>=0.2.2

# 3. 数据库与 Web 通信
SQLAlchemy>=2.0.0
requests
python-dotenv

# 4. 数据校验与异步底层
pydantic>=2.0.0
typing_extensions>=4.0.0
anyio
tqdm

# 5. 辅助增强
langchain-github-copilot

# 6. 监控与可视化（可选）
langsmith>=0.0.80

# 7. 兼容性与性能优化（可选）
huggingface-hub>=0.20.3

# 8. 其他（如有需要可补充）
# ...
