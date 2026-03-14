import os
import uuid
from pathlib import Path
from typing import Any, Optional

import chromadb
import structlog
from chromadb.api import ClientAPI
from chromadb.config import Settings
from fastembed import TextEmbedding

from app.core.config import get_settings

settings = get_settings()
logger = structlog.get_logger()

_embedder: Optional[TextEmbedding] = None
_chroma_client: Optional[ClientAPI] = None


def _get_embedder() -> TextEmbedding:
    """Get or create fastembed text embedder."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedder


def _get_chroma_client() -> ClientAPI:
    """Get or create ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = Path("./chroma_db")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Use PersistentClient explicitly for disk persistence
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
            )
        )
        logger.info("ChromaDB client created", persist_dir=str(persist_dir))
    return _chroma_client


async def init_chroma() -> None:
    """Initialize ChromaDB - ensures directory exists and is accessible."""
    client = _get_chroma_client()
    # Ping to verify connection
    client.heartbeat()
    logger.info("ChromaDB initialized successfully")
    
    # List existing collections
    collections = client.list_collections()
    logger.info("Existing collections", count=len(collections), names=[c.name for c in collections])


class LongTermMemory:
    """ChromaDB-based long-term memory with fastembed embeddings."""

    def __init__(self):
        self.client = _get_chroma_client()

    def _get_collection(self, user_id: uuid.UUID):
        """Get or create collection for user."""
        collection_name = f"user_{str(user_id)}"
        try:
            collection = self.client.get_collection(name=collection_name)
            logger.debug("Got existing collection", name=collection_name)
            return collection
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist, create it
            logger.info("Creating new collection", name=collection_name)
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error("Error getting/creating collection", name=collection_name, error=str(e))
            raise

    async def add_memory(self, user_id: uuid.UUID, text: str) -> None:
        """Store a conversation exchange in ChromaDB."""
        try:
            collection = self._get_collection(user_id)
            embedder = _get_embedder()

            # Generate embedding
            embedding = list(embedder.embed([text]))[0]

            # Generate unique ID
            memory_id = str(uuid.uuid4())

            collection.add(
                ids=[memory_id],
                embeddings=[embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)],
                documents=[text],
                metadatas=[{"timestamp": str(uuid.uuid4()), "source": "conversation"}],
            )
            
            logger.info(
                "Memory added to ChromaDB",
                user_id=str(user_id),
                memory_id=memory_id,
                text_length=len(text),
                collection_name=f"user_{user_id}",
            )
        except Exception as e:
            logger.error(
                "Failed to add memory to ChromaDB",
                user_id=str(user_id),
                error=str(e),
                exc_info=True,
            )
            raise

    async def search(self, user_id: uuid.UUID, query: str, k: int = 5) -> str:
        """Semantic search, returns top-k joined as context string."""
        try:
            collection = self._get_collection(user_id)
            embedder = _get_embedder()

            # Generate query embedding
            query_embedding = list(embedder.embed([query]))[0]

            results = collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)],
                n_results=k,
            )

            if not results["documents"] or not results["documents"][0]:
                return ""

            return "\n\n".join(results["documents"][0])
        except Exception:
            return ""


long_term_memory = LongTermMemory()