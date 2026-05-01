from typing import Annotated, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator  # 🌟 新增


class MessagesState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    route: str = Field(default="", description="路由标签")
    data_freshness: str = Field(default="", description="数据水位日期")
    sql_result: str = Field(default="", description="给子图用的输入数据")
    analysis: str = Field(default="", description="子图返回的分析结论")

    # ================= 🌟 集团篇新增：并发协作字段 =================
    # targets: 存放分身任务的目标清单，比如 ["Jack", "Alice", "Bob"]
    targets: List[str] = Field(default_factory=list)

    # parallel_reports: 分身探员交差的信箱。
    # Annotated[..., operator.add] 确保多个人同时塞入报告时，列表会自动拼接，绝不丢件。
    parallel_reports: Annotated[List[str], operator.add] = Field(default_factory=list)


# 🌟 新增：分身探员的“单兵任务卡”
class WorkerState(BaseModel):
    target: str = Field(description="当前分身探员负责的具体审计对象")