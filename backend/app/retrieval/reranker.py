"""Lightweight Reranking engine for hospital context passages.

Reranks candidate chunks using a hybrid composite score combining dense semantic
similarity with lexical keyword overlap and entity relevance boosts (doctors,
departments, facilities), avoiding heavyweight models for fast, free deployment.
"""
import re
from typing import Any, Dict, List, Optional, Set

from app.core.logging_config import logger
from app.models.schemas import RetrievalChunkItem


class LightweightReranker:
    """Fast, memory-efficient reranker for hospital retrieval chunks."""

    def __init__(
        self,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.20,
        entity_weight: float = 0.15,
    ):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.entity_weight = entity_weight

    def rerank(
        self,
        query: str,
        chunks: List[RetrievalChunkItem],
        entities: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None,
    ) -> List[RetrievalChunkItem]:
        """Reranks candidate chunks by composite score and deduplicates."""
        if not chunks:
            return []

        ent = entities or {}
        target_doc = (ent.get("doctor") or "").lower()
        target_dept = (ent.get("department") or "").lower()
        target_fac = (ent.get("facility") or "").lower()

        # Extract significant query tokens (length > 2, non-stopwords)
        stopwords = {
            "the", "and", "for", "with", "what", "which", "where", "when",
            "who", "how", "are", "is", "was", "were", "hospital", "mission",
        }
        tokens = [
            t for t in re.findall(r"\b[A-Za-z0-9]{3,}\b", query.lower())
            if t not in stopwords
        ]

        scored_items: List[tuple[float, RetrievalChunkItem]] = []
        seen_contents: Set[str] = set()

        for chunk in chunks:
            # Simple content deduplication
            norm_content = re.sub(r"\s+", " ", (chunk.content or "").strip().lower())
            content_sig = norm_content[:150]
            if content_sig in seen_contents:
                continue
            seen_contents.add(content_sig)

            # 1. Base semantic score (normalized 0.0 - 1.0)
            base_score = max(0.0, min(1.0, float(chunk.score or 0.5)))

            # 2. Lexical token overlap score
            lexical_matches = 0
            if tokens and norm_content:
                for tok in tokens:
                    if tok in norm_content or tok in (chunk.title or "").lower():
                        lexical_matches += 1
                lexical_score = min(1.0, lexical_matches / max(1, len(tokens)))
            else:
                lexical_score = 0.0

            # 3. Entity relevance boost
            entity_score = 0.0
            meta = chunk.metadata or {}
            chunk_doc = str(meta.get("doctor_name", "")).lower()
            chunk_dept = str(chunk.section or meta.get("department", "")).lower()

            if target_doc and (target_doc in chunk_doc or target_doc in norm_content):
                entity_score += 0.6
            if target_dept and (target_dept in chunk_dept or target_dept in norm_content):
                entity_score += 0.4
            if target_fac and (target_fac in norm_content):
                entity_score += 0.4
            entity_score = min(1.0, entity_score)

            # Composite final score
            composite = (
                (self.semantic_weight * base_score)
                + (self.lexical_weight * lexical_score)
                + (self.entity_weight * entity_score)
            )

            scored_items.append((composite, chunk))

        # Sort descending by composite score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        limit = top_n if top_n is not None else len(scored_items)
        reranked_chunks: List[RetrievalChunkItem] = []

        for idx, (composite_score, chunk) in enumerate(scored_items[:limit], start=1):
            reranked_chunks.append(
                RetrievalChunkItem(
                    rank=idx,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=round(composite_score, 4),
                    content=chunk.content,
                    section=chunk.section,
                    url=chunk.url,
                    title=chunk.title,
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            )

        logger.info(
            "Reranked %d chunks into %d top candidates (query: '%s')",
            len(chunks),
            len(reranked_chunks),
            query[:60],
        )
        return reranked_chunks
