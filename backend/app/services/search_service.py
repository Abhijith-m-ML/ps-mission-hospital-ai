import time
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging_config import logger
from app.ingestion.embedder import DocumentEmbedder
from app.retrieval.vector_store import PineconeVectorStore


class SearchService:
    """Handles query embedding and semantic nearest-neighbor retrieval from Pinecone without an LLM."""

    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        embedder: Optional[DocumentEmbedder] = None,
    ):
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedder = embedder or DocumentEmbedder()

    def search_query(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Embeds natural-language query and searches Pinecone for top-K matching chunks.
        
        Args:
            query: User search query (e.g. "What department treats heart problems?").
            top_k: Number of most similar results to return (max 50).
            namespace: Hospital namespace (defaults to settings.PINECONE_NAMESPACE).
            filter_dict: Optional metadata filter (e.g. {"section": "Cardiology"}).
            index_name: Optional index override.
            
        Returns:
            Dict containing query, namespace, count, and ranked results.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty.")

        if len(clean_query) > 1000:
            raise ValueError("Query exceeds maximum allowed length of 1000 characters.")

        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50.")

        target_namespace = namespace or self.vector_store.namespace
        start_time = time.time()

        # 1. Embed query into 384-dimensional vector using existing embedding model
        query_vector = self.embedder.embed_text(clean_query)

        # 2. Search Pinecone vector store
        matches = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=target_namespace,
            filter_dict=filter_dict,
            index_name=index_name,
        )

        duration = time.time() - start_time
        return {
            "query": clean_query,
            "index": index_name or self.vector_store.index_name,
            "namespace": target_namespace,
            "top_k": top_k,
            "total_found": len(matches),
            "search_duration_seconds": round(duration, 3),
            "results": matches,
        }
