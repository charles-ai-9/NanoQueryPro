# src 目录代码整合

---

**__init__.py**
```python

```

---

**agent/__init__.py**
```python

```

---

**agent/graph.py**
```python
"""
graph.py - NanoQuery 拓扑升级：支持工具执行循环与物理装甲
"""
import sys
import os

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
    # 如果 route 是 chat，流程直接结束
    if route == "chat":
        return END
    # 如果 route 是 meta，进入 SQL 生成节点
    if route == "meta":
        return "generate_sql"
    # 如果 route 是 analysis，进入 root cause analysis 子图
    if route == "analysis":
        return "rca_subgraph"
    # 其他情况（默认 business），进入数据新鲜度检查节点
    return "check_freshness"

def _build_builder() -> StateGraph:
    """[内部共用函数]：构建并返回未 compile 的 StateGraph builder，供两个入口共用"""
    builder = StateGraph(MessagesState)
    network_armor = RetryPolicy(
        initial_interval=2.0,
        backoff_factor=2.0,
        max_attempts=3
    )
    builder.add_node("intent", intent_node, retry_policy=network_armor)
    builder.add_node("check_freshness", check_data_freshness_node)
    builder.add_node("generate_sql", generate_sql_node, retry_policy=network_armor)
    builder.add_node("tools", ToolNode([execute_sql,search_knowledge_base]))
    builder.add_node("rca_subgraph", rca_graph)
    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        intent_router,
        {
            END: END,
            "generate_sql": "generate_sql",
            "check_freshness": "check_freshness",
            "rca_subgraph": "rca_subgraph"
        }
    )
    builder.add_edge("check_freshness", "generate_sql")
    builder.add_edge("rca_subgraph", END)
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
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command
from .state import MessagesState
from src.tools.sql_tools import execute_sql
from functools import lru_cache
from src.core.llm_client import get_llm
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from src.tools.sql_tools  import search_knowledge_base

logger = logging.getLogger(__name__)

llm = None
llm_with_tools = None

@lru_cache(maxsize=1)
def get_kb_instance():
    from src.core.vector_store import KnowledgeBase
    kb = KnowledgeBase()
    if not kb.load_index():
        kb.build_index()
    return kb

@lru_cache(maxsize=1)
def get_llm_with_tools():
    _llm = get_llm()
    if _llm is None:
        raise ValueError("❌ 大模型初始化失败，请检查环境变量配置！")
    return _llm.bind_tools([execute_sql, search_knowledge_base])

def initialize_llm(llm_instance):
    global llm, llm_with_tools
    llm = llm_instance
    llm_with_tools = llm.bind_tools([execute_sql])

class UserMemory(BaseModel):
    has_preference: bool = Field(description="用户是否在这句话中明确表达了个人喜好、习惯、身份或人物特征？")
    preference_content: str = Field(description="如果表达了特征，请提取具体内容(精简为短语，如'喜欢喝咖啡'、'我是审计部的')；如果没有，返回空字符串。")

async def intent_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_name = config.get("configurable", {}).get("user_name", "Jack")
    user_role = config.get("configurable", {}).get("role", "admin")
    if not state.messages:
        logger.warning("intent_node: 消息列表为空，直接路由到 chat")
        return {"route": "chat"}
    last_msg_content = state.messages[-1].content.strip()
    _llm = get_llm()
    if _llm is None:
        error_msg = "❌ 大模型初始化失败，请检查 .env 文件配置。"
        logger.error("intent_node: %s", error_msg)
        return {"messages": [AIMessage(content=error_msg)], "route": "chat"}
    namespace = ("user_profiles", user_name)
    memory_extractor = _llm.with_structured_output(UserMemory)
    try:
        memory_result = await memory_extractor.ainvoke([
            SystemMessage(
                content="你是一个心理分析师，任务是从用户的日常对话中提取他们的长期偏好或个人特征。如果没有明确特征，不要凭空捏造。"),
            HumanMessage(content=last_msg_content)
        ])
        if memory_result and memory_result.has_preference and memory_result.preference_content:
            await store.aput(namespace, "preference", {"likes": memory_result.preference_content})
            logger.info(f"💾 [Store API]: 智能提取并持久化特征 -> [{memory_result.preference_content}]")
    except Exception as e:
        logger.warning(f"记忆提取环节发生异常 (非致命，跳过): {e}")
    profile = await store.aget(namespace, "preference")
    known_preference = profile.value.get("likes") if profile else None
    META_KEYWORDS = ["表", "字段", "结构", "元数据", "有哪些表", "schema"]
    ANALYSIS_KEYWORDS = ["为什么", "原因", "分析", "归因", "排查"]
    for kw in META_KEYWORDS:
        if kw in last_msg_content:
            logger.info("intent_node: 关键词[%s]触发物理拦截 → meta", kw)
            return {"messages": [AIMessage(content="【META】")], "route": "meta"}
    for kw in ANALYSIS_KEYWORDS:
        if kw in last_msg_content:
            logger.info("intent_node: 关键词[%s]触发物理拦截 → analysis", kw)
            return {"messages": [AIMessage(content="【ANALYSIS】")], "route": "analysis"}
    system_prompt = """
你是一个极其严谨的星际金融风控局前台接待员（意图路由器）。
    请严格根据用户的输入，将其划分到以下四个意图之一：

    1. 【BUSINESS】（业务数据与知识查询🎯）：
       - 当用户询问具体的业务数据（如“有多少客户”、“逾期金额是多少”）。
       - 当用户询问金融风控术语（如“什么是 DPD”、“解释一下 M1/M2”）。
       - 当用户询问公司内部政策、规章制度、催收操作手册（如“M2的惩罚策略是什么”）。
       ⚠️ 极其重要：所有名词解释、政策查询，一律归为 BUSINESS！

    2. 【ANALYSIS】（根因与异常分析📉）：
       - 当用户发现某个指标发生异动，要求查明原因时（如“为什么上个月的坏账率突然升高了”、“帮我排查一下订单量下降的归因”）。
       ⚠️ 注意：不要把简单的名词“解释”归类为“分析”！

    3. 【META】（数据库元数据📊）：
       - 当用户询问数据库表结构、有哪些表、字段代表什么意思时。

    4. 【CHAT】（闲聊与问候☕）：
       - 日常打招呼、夸奖、或者与金融风控无关的闲聊。

    回复格式要求：如果是 CHAT，回复 【CHAT】+ 一句符合探员身份的幽默回应；如果是其他三类，请严格只回复【标签名】（如 【BUSINESS】），绝不要输出任何其他字符！"""
    res = await _llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=last_msg_content)])
    res_text = res.content.upper()
    logger.info("intent_node: LLM 分类结果 → %s", res_text[:50])
    if "CHAT" in res_text:
        reply = res.content.replace("【CHAT】", "").strip()
        if known_preference:
            personalized_reply = f"[权限: {user_role}] 敬礼！{user_name}！我知道您【{known_preference}】！{reply}"
        else:
            personalized_reply = f"[权限: {user_role}] 敬礼！{user_name}！{reply}"
        return {"messages": [AIMessage(content=personalized_reply)], "route": "chat"}
    elif "META" in res_text:
        return {"messages": [res], "route": "meta"}
    elif "ANALYSIS" in res_text:
        return {"messages": [res], "route": "analysis"}
    else:
        return {"messages": [res], "route": "business"}

async def check_data_freshness_node(state: MessagesState):
    date = "2024-12-23"
    return {"messages": [SystemMessage(content=f"当前数据截止到 {date}。")], "data_freshness": date}

async def generate_sql_node(state: MessagesState, config: RunnableConfig):
    _llm_with_tools = get_llm_with_tools()
    user_name = config.get("configurable", {}).get("user_name", "未知员工")
    user_role = config.get("configurable", {}).get("role", "user")
    messages = state.messages
    last_msg = messages[-1] if messages else None
    last_msg_content = last_msg.content if last_msg else ""
    correction_prompt = ""
    if isinstance(last_msg, HumanMessage) and len(messages) > 1:
        correction_prompt = (
            "\n[👨‍💼 人类导师反馈]\n"
            f"反馈内容：{last_msg_content}\n"
            "请仔细阅读上述人类反馈修正 SQL。如提供了完整 SQL 则原封不动执行。"
        )
    elif "ERROR" in last_msg_content:
        correction_prompt = (
            "\n[🚩 紧急纠错指令]\n"
            f"错误信息为: {last_msg_content}\n"
            "请分析原因修正 SQL 后再次调用。"
        )
    role_instruction = ""
    if user_role == "admin":
        role_instruction = f"3. 【权限最高级】：当前操作者是 {user_name} (Admin)，拥有所有数据库表的无限制查询权限。"
    else:
        role_instruction = f"3. 【权限受限】：当前操作者是 {user_name} ({user_role})，生成的 SQL 必须严格限制范围，严禁查询薪酬、密码等高管敏感表！"
    sys_instruction = SystemMessage(content=(
        "你是一个严谨的金融 SQL 侦探。\n"
        "1. 严禁幻觉：必须且只能使用 `execute_sql` 获取数据。\n"
        "2. 命名规范：严格遵守数据库表名，SQLite 中不要随意加复数 's'。\n"
        f"{role_instruction}\n"
        f"{correction_prompt}"
    ))
    input_msgs = [sys_instruction] + messages[-10:]
    try:
        logger.info(f"generate_sql_node: 正在调度 AI... [当前权限: {user_role}]")
        response = await _llm_with_tools.ainvoke(input_msgs)
        return {"messages": [response]}
    except Exception as e:
        logger.error("大脑节点崩溃: %s", str(e))
        return {"messages": [AIMessage(content=f"❌ 侦探大脑思考时发生意外: {str(e)}")]}
```

---

**agent/state.py**
```python
from typing import Annotated, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class MessagesState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    route: str = Field(default="", description="路由标签:chat, meta, business, analysis")
    data_freshness: str = Field(default="", description="数据水位日期")
    sql_result: str = Field(default="", description="给子图用的输入数据")
    analysis: str = Field(default="", description="子图返回的分析结论")
```

---

**agent/subgraphs/__init__.py**
```python

```

---

**agent/subgraphs/rca_graph.py**
```python
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
```

---

**core/llm_client.py**
```python
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
```

---

**core/vector_store.py**
```python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class EnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: List[float]
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        all_results: list[list[Document]] = [r.invoke(query) for r in self.retrievers]
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        rrf_k = 60
        for retriever_docs, weight in zip(all_results, self.weights):
            for rank, doc in enumerate(retriever_docs):
                doc_id = doc.page_content
                rrf_score = weight * (1.0 / (rrf_k + rank + 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
                doc_map[doc_id] = doc
        sorted_docs = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_docs]

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
load_dotenv(dotenv_path=ROOT_DIR / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        logger.info("🤖 正在启动 LangChain 原生双模检索系统...")
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
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
    def build_index(self):
        knowledge_dir = ROOT_DIR / "data" / "knowledge"
        documents = []
        for file in knowledge_dir.glob("*.md"):
            try:
                loader = TextLoader(str(file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"加载失败: {e}")
        if not documents:
            logger.error("❌ 没找到手册！")
            return False
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(str(self.db_path))
        self._setup_ensemble(chunks)
        logger.info(f"✅ 官方版混合索引构建完成！")
        return True
    def load_index(self):
        if self.db_path.exists():
            try:
                self.vector_db = FAISS.load_local(
                    str(self.db_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                doc_store = self.vector_db.docstore._dict
                chunks = list(doc_store.values())
                self._setup_ensemble(chunks)
                return True
            except Exception as e:
                logger.error(f"磁盘加载失败: {e}")
        return False
    def query(self, question: str):
        if not self.ensemble_retriever:
            if not self.load_index():
                if not self.build_index():
                    return "❌ 引擎故障"
        docs = self.ensemble_retriever.invoke(question)
        return "\n---\n".join([doc.page_content for doc in docs])
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.build_index()
    print("\n🔍 测试 DPD:")
    print(kb.query("DPD 是什么？"))
```

---

**tools/__init__.py**
```python

```

---

**tools/sql_tools.py**
```python
import sqlite3
import os
import asyncio
from pathlib import Path
from langchain_core.tools import tool
DB_PATH = Path(__file__).parent.parent.parent / "data" / "mock_data.db"
def _run_sql(query: str) -> str:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                return "查询成功，但结果为空。"
            columns = [desc[0] for desc in cursor.description]
            header = " | ".join(columns)
            body = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
            return f"{header}\n{'-' * len(header)}\n{body}"
    except Exception as e:
        return f"ERROR: SQL 执行失败: {str(e)}"
@tool
async def execute_sql(query: str) -> str:
    """
    在 mock_data.db 上执行只读 SELECT SQL 查询并返回结果。
    参数 query: 完整的 SQL 查询语句。
    """
    return await asyncio.to_thread(_run_sql, query)
@tool
def search_knowledge_base(query: str) -> str:
    """
    当用户询问金融风控专业术语（如 DPD, M1, M2）、催收政策、内部操作手册、
    公司规章制度或业务逻辑时，请务必调用此工具。
    输入应是一个具体的业务搜索问题，例如 "M1级别的催收惩罚策略是什么？" 或 "DPD的定义"。
    """
    from src.agent.nodes import get_kb_instance
    print(f"\n[Agent 动作] 🕵️‍♂️ 正在翻阅内部风控手册，检索: {query}...")
    result = get_kb_instance().query(query)
    return result
```

