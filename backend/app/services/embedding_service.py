from typing import List, Optional
from app.core.config import settings
from app.core.logging_config import logger
from app.crawler.fetcher import Fetcher
from app.ingestion.cleaner import HTMLCleaner
from app.ingestion.document_parser import DocumentParser, CleanedDocument
from app.ingestion.chunker import DocumentChunker, Chunk
from app.ingestion.embedder import DocumentEmbedder, EmbeddedChunk
from app.models.schemas import (
    EmbeddingTextPreviewRequest,
    EmbeddingTextItem,
    EmbeddingTextPreviewResponse,
    EmbeddingChunkPreviewRequest,
    ChunkEmbeddingItem,
    ChunkEmbeddingPreviewResponse,
    ChunkItem,
)


class EmbeddingService:
    """Service handling text and chunk embedding operations, caching, and inspection previews."""

    @staticmethod
    def preview_text_embeddings(request: EmbeddingTextPreviewRequest) -> EmbeddingTextPreviewResponse:
        """Embeds raw text inputs (single or batch) and returns dimensions with previews."""
        embedder = DocumentEmbedder()
        dim = embedder.embedding_dimension()
        preview_dims = min(request.preview_dims, dim)

        texts_to_process: List[str] = []
        if request.text is not None:
            texts_to_process.append(request.text)
        if request.texts:
            texts_to_process.extend(request.texts)

        if not texts_to_process:
            raise ValueError("No text provided. Please provide 'text' or 'texts'.")

        # Security check: limit batch count for preview to avoid denial of service
        if len(texts_to_process) > 50:
            raise ValueError("Cannot preview more than 50 texts in a single preview request.")

        # Batch embed
        vectors = embedder.embed_texts(texts_to_process, as_numpy=False)

        items: List[EmbeddingTextItem] = []
        for text, vec in zip(texts_to_process, vectors):
            items.append(
                EmbeddingTextItem(
                    text_preview=text[:120] + ("..." if len(text) > 120 else ""),
                    embedding_dimension=dim,
                    preview=[round(val, 5) for val in vec[:preview_dims]],
                )
            )

        first_preview = items[0].preview if items else []

        return EmbeddingTextPreviewResponse(
            model=embedder.model_name,
            embedding_dimension=dim,
            count=len(items),
            preview=first_preview,
            items=items,
        )

    @staticmethod
    def preview_chunk_embeddings(request: EmbeddingChunkPreviewRequest) -> ChunkEmbeddingPreviewResponse:
        """Chunks a target webpage or uses provided chunks, generates embeddings,
        and returns sample results for inspection.
        """
        chunks_to_embed: List[Chunk] = []

        if request.chunks:
            # Use chunks provided in request
            for c in request.chunks:
                chunks_to_embed.append(
                    Chunk(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        content=c.content,
                        metadata=c.metadata,
                    )
                )
        elif request.url:
            # Pipeline: Fetch -> Extract/Clean -> Chunk
            fetcher = Fetcher()
            fetch_res = fetcher.fetch(request.url)
            if not fetch_res.is_success:
                raise ValueError(f"Failed to fetch webpage at {request.url}: {fetch_res.error}")

            parser = DocumentParser()
            cleaned_doc: CleanedDocument = parser.parse_document(fetch_res.html or "", request.url)

            chunker = DocumentChunker(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )
            chunks_to_embed = chunker.chunk_document(cleaned_doc)
        else:
            raise ValueError("Either 'url' or 'chunks' must be provided in the request.")

        if not chunks_to_embed:
            embedder = DocumentEmbedder()
            dim = embedder.embedding_dimension()
            return ChunkEmbeddingPreviewResponse(
                model=embedder.model_name,
                total_chunks_processed=0,
                preview_chunks_count=0,
                embedding_dimension=dim,
                sample_results=[],
            )

        embedder = DocumentEmbedder()
        dim = embedder.embedding_dimension()
        preview_dims = min(request.preview_dims, dim)

        # Batch embed all chunks preserving data
        embedded_chunks: List[EmbeddedChunk] = embedder.embed_chunks(chunks_to_embed, as_numpy=False)

        # Sample at most max_preview_chunks for preview response
        max_samples = min(request.max_preview_chunks, len(embedded_chunks))
        sample_results: List[ChunkEmbeddingItem] = []

        for ec in embedded_chunks[:max_samples]:
            sample_results.append(
                ChunkEmbeddingItem(
                    chunk_id=ec.chunk_id,
                    document_id=ec.document_id,
                    section=ec.metadata.get("section", "General"),
                    content_preview=ec.content[:150] + ("..." if len(ec.content) > 150 else ""),
                    embedding_dimension=dim,
                    embedding_preview=[round(val, 5) for val in ec.embedding[:preview_dims]],
                    metadata=ec.metadata,
                )
            )

        return ChunkEmbeddingPreviewResponse(
            model=embedder.model_name,
            total_chunks_processed=len(embedded_chunks),
            preview_chunks_count=len(sample_results),
            embedding_dimension=dim,
            sample_results=sample_results,
        )
