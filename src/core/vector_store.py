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

# --- 全局单例定义 ---
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