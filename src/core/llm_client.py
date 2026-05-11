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
                streaming=True  # 👈 核心修改：开启云端大模型的流式输出
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
                streaming=True, # 👈 核心修改：开启本地/兼容大模型的流式输出
                model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            )
            logger.info("已切换至本地模式：自建模型 %s (已开启 Streaming)", model_name)

        return _llm_instance

    except Exception as e:
        logger.error("LLM 初始化失败 (模式: %s): %s", mode, str(e))
        return None


from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm_with_tools():
    """绑定 SQL 和知识库工具的 LLM 实例（单例缓存）"""
    from src.tools.sql_tools import execute_sql, search_knowledge_base
    _llm = get_llm()
    return _llm.bind_tools([execute_sql, search_knowledge_base])
