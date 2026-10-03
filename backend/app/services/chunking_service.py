from typing import Optional
from app.crawler.fetcher import Fetcher, FetchResponse
from app.ingestion.document_parser import DocumentParser, CleanedDocument
from app.ingestion.chunker import DocumentChunker, Chunk
from app.models.schemas import (
    ChunkPreviewRequest,
    ChunkPreviewResponse,
    DocumentBrief,
    ChunkItem,
)
from app.core.logging_config import logger


class ChunkingService:
    """Coordinates fetching, cleaning, and structure-aware chunking for document preview."""

    @staticmethod
    def preview_chunks(request: ChunkPreviewRequest) -> ChunkPreviewResponse:
        logger.info("Executing chunking preview for: %s", request.url)
        
        # 1. Fetch webpage using existing fetcher
        fetcher = Fetcher()
        fetch_result: FetchResponse = fetcher.fetch(request.url)

        if not fetch_result.is_success:
            logger.warning("Chunk preview fetch failed for %s: %s", request.url, fetch_result.error)
            return ChunkPreviewResponse(
                document=DocumentBrief(url=request.url, title="Fetch Error"),
                chunk_count=0,
                chunks=[],
            )

        # 2. Extract clean structured document using existing DocumentParser
        parser = DocumentParser()
        cleaned_doc: CleanedDocument = parser.parse_document(
            html=fetch_result.html or "",
            url=request.url,
        )

        # 3. Apply DocumentChunker
        chunker = DocumentChunker(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        chunks: list[Chunk] = chunker.chunk_document(cleaned_doc)

        logger.info(
            "Chunking preview completed for %s: Generated %d chunks",
            request.url,
            len(chunks),
        )

        return ChunkPreviewResponse(
            document=DocumentBrief(
                url=cleaned_doc.url,
                title=cleaned_doc.title,
            ),
            chunk_count=len(chunks),
            chunks=[
                ChunkItem(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    metadata=c.metadata,
                )
                for c in chunks
            ],
        )
