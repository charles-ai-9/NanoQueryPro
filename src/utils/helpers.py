from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def safe_get(state, key, default=None):
    """安全读取 state 中的字段，兼容 dict 和对象两种形式"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def sanitize_history(messages):
    """
    🚨 绝对无菌协议洗白器 (Rebuild Everything)
    彻底销毁原有的 AIMessage 和 ToolMessage 对象，提取纯文本重新实例化。
    绝不保留任何 additional_kwargs 隐藏属性，让 Qwen 只看到纯粹的文字对话历史。
    """
    clean_msgs = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            # 🔪 提取纯文本，如果为空则给占位符。创建一个崭新的 AIMessage。
            content = msg.content if msg.content else "【系统日志】：发起了一次数据检索请求。"
            # 确保内容是字符串格式
            if not isinstance(content, str):
                content = str(content)
            clean_msgs.append(AIMessage(content=content))

        elif isinstance(msg, ToolMessage):
            # 🔪 彻底抛弃 ToolMessage，转生为 HumanMessage
            content = msg.content if msg.content else "无结果"
            clean_msgs.append(HumanMessage(content=f"📊 [工具检索结果]:\n{content}"))

        else:
            # HumanMessage 和 SystemMessage 直接保留
            clean_msgs.append(msg)

    return clean_msgs

