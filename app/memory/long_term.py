import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.config import get_settings

settings = get_settings()

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


class LongTermMemory:
    def __init__(self):
        self.base_path = Path("./faiss_indexes")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_index_path(self, user_id: uuid.UUID) -> Path:
        return self.base_path / str(user_id)

    def _get_or_create_index(self, user_id: uuid.UUID) -> FAISS:
        """Get or create FAISS index for a user."""
        index_path = self._get_index_path(user_id)

        if index_path.exists() and (index_path / "index.faiss").exists():
            return FAISS.load_local(
                str(index_path),
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )

        texts = ["This is a placeholder for conversation memory."]
        metadatas = [{"source": "init"}]
        docstore = FAISS.from_texts(texts, _get_embeddings(), metadatas=metadatas)

        docstore.save_local(str(index_path))
        return docstore

    async def add_memory(self, user_id: uuid.UUID, text: str) -> None:
        """Embed and store a conversation exchange."""
        index = self._get_or_create_index(user_id)

        metadatas = [{"source": "conversation", "timestamp": str(uuid.uuid4())}]
        index.add_texts([text], metadatas=metadatas)

        index.save_local(str(self._get_index_path(user_id)))

    async def search(self, user_id: uuid.UUID, query: str, k: int = 5) -> str:
        """Semantic search, returns top-k joined as context string."""
        try:
            index = self._get_or_create_index(user_id)
            docs = index.similarity_search(query, k=k)

            if not docs:
                return ""

            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return ""


long_term_memory = LongTermMemory()