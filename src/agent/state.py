# src/agent/state.py
from typing import Annotated, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator

class MessagesState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    route: str = Field(default="", description="路由标签:chat, meta, business, analysis")
    data_freshness: str = Field(default="", description="数据水位日期")
    sql_result: str = Field(default="", description="给子图用的输入数据")
    analysis: str = Field(default="", description="子图返回的分析结论")

    # 并发控制字段
    targets: List[str] = Field(default_factory=list)
    parallel_reports: Annotated[List[str], operator.add] = Field(default_factory=list)

class IntentOutput(BaseModel):
    """【意图分发表】"""
    route: Literal["business", "analysis", "meta", "chat", "parallel"] = Field(
        description="意图标签"
    )
    targets: List[str] = Field(default_factory=list)
    chat_reply: Optional[str] = Field(None)

class SupervisorDecision(BaseModel):
    """【局长决策表】：监督者必须严格按此格式下发指令"""
    next_action: Literal["sql_detective", "knowledge_agent", "FINISH"] = Field(
        description="下一步该由谁接手？查询数据选 sql_detective，查规章选 knowledge_agent，结案选 FINISH。"
    )
    instruction: str = Field(
        description="给下属的具体指令，或者选 FINISH 时的结案总结。"
    )

class WorkerState(BaseModel):
    """🌟 分身探员的单兵任务卡"""
    target: str = Field(description="当前分身探员负责的具体审计对象")