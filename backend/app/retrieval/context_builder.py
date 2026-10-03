from typing import Any, Dict, List, Optional, Union
from app.core.config import settings
from app.core.logging_config import logger
from app.models.schemas import ContextResult, RetrievalChunkItem


class ContextBuilder:
    """Assembles retrieved and ordered hospital chunks into a structured context text
    with explicit source demarcations, respecting chunk count and character budgets.
    
    NOTE: Does NOT generate an LLM prompt or answer; this layer only formats the
    retrieved factual background context for later LLM synthesis in Stage 8.
    """

    def __init__(
        self,
        default_max_chunks: Optional[int] = None,
        default_max_characters: Optional[int] = None,
    ):
        self.default_max_chunks = default_max_chunks or settings.RETRIEVAL_MAX_CONTEXT_CHUNKS
        self.default_max_characters = default_max_characters or settings.RETRIEVAL_MAX_CONTEXT_CHARACTERS

    def build_context(
        self,
        chunks: List[Union[RetrievalChunkItem, Dict[str, Any]]],
        max_chunks: Optional[int] = None,
        max_characters: Optional[int] = None,
    ) -> ContextResult:
        """Assembles structured context from ranked candidate chunks.
        
        Args:
            chunks: Ranked chunks sorted by descending relevance score.
            max_chunks: Maximum number of chunks to include (defaults to RETRIEVAL_MAX_CONTEXT_CHUNKS).
            max_characters: Maximum character budget for the context string.
            
        Returns:
            ContextResult: Structured context with boundary markers, source tracking, and metadata.
        """
        if not chunks:
            logger.info("ContextBuilder received empty chunks list. Returning empty context.")
            return ContextResult(
                has_context=False,
                context_text="",
                total_chunks=0,
                total_characters=0,
                reason="No sufficiently relevant information was found.",
            )

        chunk_limit = max_chunks if max_chunks is not None else self.default_max_chunks
        char_budget = max_characters if max_characters is not None else self.default_max_characters

        assembled_blocks: List[str] = []
        current_char_count = 0
        chunks_included = 0

        for chunk in chunks:
            if chunks_included >= chunk_limit:
                logger.debug(
                    "ContextBuilder reached max_chunks limit (%d). Skipping remaining %d chunks.",
                    chunk_limit,
                    len(chunks) - chunks_included,
                )
                break

            # Normalize chunk fields whether passed as Pydantic model or dictionary
            if isinstance(chunk, dict):
                rank = chunk.get("rank", chunks_included + 1)
                chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{chunks_included + 1}"
                section = chunk.get("section", "General")
                url = chunk.get("url", "")
                title = chunk.get("title", "")
                content = (chunk.get("content") or "").strip()
            else:
                rank = getattr(chunk, "rank", chunks_included + 1)
                chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None) or f"chunk_{chunks_included + 1}"
                section = getattr(chunk, "section", "General")
                url = getattr(chunk, "url", "")
                title = getattr(chunk, "title", "")
                content = (getattr(chunk, "content", "") or "").strip()

            if not content:
                continue

            # Build source header with explicit SOURCE ID for LLM citations
            header_lines = [f"--- SOURCE {rank} ---", f"Source ID: {chunk_id}"]
            if section:
                header_lines.append(f"Section: {section}")
            if title:
                header_lines.append(f"Title: {title}")
            if url:
                header_lines.append(f"URL: {url}")

            source_header = "\n".join(header_lines)
            block = f"{source_header}\n\n{content}"

            # Calculate additional characters (including separator between blocks)
            separator_len = 2 if assembled_blocks else 0  # "\n\n"
            projected_len = current_char_count + separator_len + len(block)

            if projected_len > char_budget:
                if not assembled_blocks:
                    # If even the very first chunk exceeds the budget, truncate it gracefully
                    allowed_content_len = char_budget - (len(source_header) + 4)
                    if allowed_content_len > 50:
                        truncated_content = content[:allowed_content_len].rstrip() + "..."
                        truncated_block = f"{source_header}\n\n{truncated_content}"
                        assembled_blocks.append(truncated_block)
                        current_char_count = len(truncated_block)
                        chunks_included = 1
                    logger.warning(
                        "First chunk exceeded max_characters budget (%d). Truncated content.",
                        char_budget,
                    )
                else:
                    logger.info(
                        "Context character budget (%d) reached at chunk rank %d. "
                        "Preserving top %d chunks without cutting.",
                        char_budget,
                        rank,
                        chunks_included,
                    )
                break

            assembled_blocks.append(block)
            current_char_count = projected_len
            chunks_included += 1

        if not assembled_blocks:
            return ContextResult(
                has_context=False,
                context_text="",
                total_chunks=0,
                total_characters=0,
                reason="No chunks could be included within the character budget.",
            )

        final_context_text = "\n\n".join(assembled_blocks)
        return ContextResult(
            has_context=True,
            context_text=final_context_text,
            total_chunks=chunks_included,
            total_characters=len(final_context_text),
            reason=None,
        )
