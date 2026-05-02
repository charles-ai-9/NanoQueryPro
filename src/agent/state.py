# src/agent/state.py
from typing import Annotated, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator

class MessagesState(BaseModel):
    # 🚨 核心：必须有 add_messages，否则对话记录无法累加
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    route: str = Field(default="", description="路由标签")
    data_freshness: str = Field(default="", description="数据水位日期")
    sql_result: str = Field(default="", description="给子图用的输入数据")
    analysis: str = Field(default="", description="子图返回的分析结论")

    # 🚨 军师的小黑板：存入类内部
    plan: List[str] = Field(default_factory=list)

    # 🚨 记事本：记录已完成的 [(步骤名, 执行结果), ...]，使用 operator.add 实现追加
    past_steps: Annotated[List[Tuple[str, str]], operator.add] = Field(default_factory=list)

    # 并发控制字段
    targets: List[str] = Field(default_factory=list)
    parallel_reports: Annotated[List[str], operator.add] = Field(default_factory=list)

class IntentOutput(BaseModel):
    route: Literal["business", "analysis", "meta", "chat", "parallel", "plan"] = Field(
        description="意图分类的路由标签"
    )
    targets: Optional[List[str]] = Field(default=[], description="如果涉及具体实体，提取出来")
    chat_reply: Optional[str] = Field(default="", description="如果是闲聊，直接在这里生成回复")

class SupervisorDecision(BaseModel):
    """【局长决策表】"""
    next_action: Literal["sql_detective", "knowledge_agent", "FINISH"] = Field(
        description="下一步该由谁接手？"
    )
    instruction: str = Field(description="给下属的具体指令")

class WorkerState(BaseModel):
    target: str = Field(description="单兵任务卡")

class Plan(BaseModel):
    """军师输出的计划"""
    steps: List[str] = Field(description="执行步骤清单")