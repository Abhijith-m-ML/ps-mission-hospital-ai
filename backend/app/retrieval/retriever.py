import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.core.logging_config import logger
from app.ingestion.embedder import DocumentEmbedder
from app.models.schemas import (
    ContextResult,
    RetrievalChunkItem,
    RetrievalDebugInfo,
    RetrievalPreviewResponse,
)
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.vector_store import PineconeVectorStore


class Retriever:
    """Formalized retrieval layer coordinating query vectorization, Pinecone candidate search,
    score-based relevance filtering, deduplication, ranking, and context assembly.
    
    Architecture:
    User Query -> Query Embedding (384-d) -> Pinecone Search -> Candidates
               -> Relevance Filtering (min_score) -> Deduplication -> Ordering
               -> Context Assembly (ContextBuilder) -> Ready for Stage 8 LLM
    """

    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        embedder: Optional[DocumentEmbedder] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedder = embedder or DocumentEmbedder()
        self.context_builder = context_builder or ContextBuilder()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        max_context_chunks: Optional[int] = None,
        max_context_characters: Optional[int] = None,
    ) -> RetrievalPreviewResponse:
        """Executes the complete retrieval pipeline for a natural-language user query.
        
        Args:
            query: User's question or search phrase.
            top_k: Candidate count to fetch from Pinecone (defaults to RETRIEVAL_TOP_K).
            min_score: Minimum similarity score cutoff (defaults to RETRIEVAL_MIN_SCORE).
            namespace: Hospital tenant partition (defaults to PINECONE_NAMESPACE).
            filter_dict: Optional metadata filtering constraint for Pinecone.
            max_context_chunks: Maximum chunks to assemble into final context.
            max_context_characters: Maximum character budget for assembled context.
            
        Returns:
            RetrievalPreviewResponse: Structured candidate chunks, assembled context, and debug metrics.
        """
        start_time = time.time()

        # 1. Validate Query
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty or blank.")

        if len(clean_query) > 1000:
            raise ValueError(
                f"Query length ({len(clean_query)} chars) exceeds maximum allowed limit of 1000 characters."
            )

        k_limit = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        if k_limit < 1 or k_limit > 50:
            raise ValueError("top_k must be an integer between 1 and 50.")

        score_threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
        if score_threshold < 0.0 or score_threshold > 1.0:
            raise ValueError("min_score must be a float between 0.0 and 1.0.")

        target_namespace = namespace or getattr(self.vector_store, "namespace", settings.PINECONE_NAMESPACE)

        # 2. Query Embedding using Stage 5 DocumentEmbedder
        query_vector = self.embedder.embed_text(clean_query)
        expected_dim = self.embedder.embedding_dimension()
        if len(query_vector) != expected_dim:
            raise ValueError(
                f"Generated query vector dimension ({len(query_vector)}) does not match "
                f"expected embedding dimension ({expected_dim})."
            )

        # 3. Pinecone Candidate Search
        candidates = self.vector_store.search(
            query_vector=query_vector,
            top_k=k_limit,
            namespace=target_namespace,
            filter_dict=filter_dict,
        )

        candidates_count = len(candidates)
        logger.info(
            "Retriever fetched %d candidates from Pinecone (top_k=%d, namespace='%s').",
            candidates_count,
            k_limit,
            target_namespace,
        )

        # 4. Deduplication & Relevance Filtering
        seen_chunk_keys: Set[Tuple[str, str]] = set()
        filtered_candidates: List[Dict[str, Any]] = []
        removed_duplicates = 0
        removed_by_threshold = 0

        for cand in candidates:
            # Extract metadata
            meta = cand.get("metadata", {}) or {}
            c_id = str(meta.get("chunk_id") or cand.get("id", ""))
            doc_id = str(meta.get("document_id", ""))
            score = float(cand.get("score", 0.0))

            # Deduplication key: (document_id, chunk_id) or chunk_id
            dedup_key = (doc_id, c_id) if doc_id else (c_id, c_id)

            if dedup_key in seen_chunk_keys:
                removed_duplicates += 1
                logger.debug("Deduplicator removed duplicate chunk: %s", dedup_key)
                continue

            seen_chunk_keys.add(dedup_key)

            # Relevance Score Threshold Filtering
            if score < score_threshold:
                removed_by_threshold += 1
                logger.debug(
                    "Threshold filter excluded chunk '%s' with score %.4f < %.4f",
                    c_id,
                    score,
                    score_threshold,
                )
                continue

            filtered_candidates.append(cand)

        # 5. Result Ordering (Strict descending similarity score)
        filtered_candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        # 6. Format Structured Retrieval Items with 1-indexed Ranks
        results: List[RetrievalChunkItem] = []
        for idx, cand in enumerate(filtered_candidates, start=1):
            meta = cand.get("metadata", {}) or {}
            c_id = str(meta.get("chunk_id") or cand.get("id", ""))
            doc_id = str(meta.get("document_id", ""))
            content = str(cand.get("content") or meta.get("content", ""))
            section = str(cand.get("section") or meta.get("section", "General"))
            url = str(cand.get("url") or meta.get("url", ""))
            title = cand.get("title") or meta.get("title")
            source = str(cand.get("source") or meta.get("source", "website"))
            score = round(float(cand.get("score", 0.0)), 4)

            results.append(
                RetrievalChunkItem(
                    rank=idx,
                    chunk_id=c_id,
                    document_id=doc_id,
                    score=score,
                    content=content,
                    section=section,
                    url=url,
                    title=title,
                    source=source,
                    metadata=meta,
                )
            )

        # 7. Context Assembly & Budgeting
        context_result: ContextResult = self.context_builder.build_context(
            chunks=results,
            max_chunks=max_context_chunks,
            max_characters=max_context_characters,
        )

        has_context = context_result.has_context and len(results) > 0

        # 8. Diagnostics & Debug Metrics
        duration = time.time() - start_time
        debug_info = RetrievalDebugInfo(
            pinecone_top_k=k_limit,
            candidates_retrieved=candidates_count,
            removed_by_threshold=removed_by_threshold,
            removed_duplicates=removed_duplicates,
            final_chunks_count=len(results),
            context_character_count=context_result.total_characters,
        )

        logger.info(
            "Retrieval completed in %.3fs: %d final chunks (has_context=%s, context_chars=%d).",
            duration,
            len(results),
            has_context,
            context_result.total_characters,
        )

        return RetrievalPreviewResponse(
            query=clean_query,
            has_context=has_context,
            results=results,
            context=context_result,
            debug=debug_info,
            retrieval_duration_seconds=round(duration, 3),
        )
