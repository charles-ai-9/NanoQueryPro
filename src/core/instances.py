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

