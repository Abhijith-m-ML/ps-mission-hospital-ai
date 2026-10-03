import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging_config import logger
from app.crawler.fetcher import Fetcher
from app.ingestion.cleaner import HTMLCleaner
from app.ingestion.document_parser import DocumentParser, CleanedDocument
from app.ingestion.chunker import DocumentChunker, Chunk
from app.ingestion.embedder import DocumentEmbedder, EmbeddedChunk
from app.retrieval.vector_store import PineconeVectorStore


class IndexingService:
    """Coordinates document ingestion, chunking, vector embedding, and deterministic
    batch upsertion into the Pinecone serverless vector database.
    """

    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        embedder: Optional[DocumentEmbedder] = None,
    ):
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedder = embedder or DocumentEmbedder()

    def index_chunks_with_embeddings(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Indexes pre-computed chunks and embeddings into Pinecone after strict validation.
        
        Requirements:
        1. Validate that number of chunks equals number of embeddings.
        2. Validate that every vector has dimension 384.
        3. Construct Pinecone records with deterministic chunk_id as id.
        4. Upsert records.
        5. Return indexing statistics.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        target_dim = self.embedder.embedding_dimension()
        target_namespace = namespace or self.vector_store.namespace
        target_index = index_name or self.vector_store.index_name

        records: List[Dict[str, Any]] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            if len(vec) != target_dim:
                raise ValueError(
                    f"Vector dimension mismatch at index {idx}: expected {target_dim}, found {len(vec)}."
                )

            meta = chunk.metadata or {}
            # Flatten/clean metadata for Pinecone compatibility (strings, numbers, booleans, lists of strings)
            clean_metadata = {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "content": str(chunk.content),
                "url": str(meta.get("url", "")),
                "title": str(meta.get("title", "")),
                "section": str(meta.get("section", "General")),
                "source": str(meta.get("source", "website")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "total_chunks": int(meta.get("total_chunks", len(chunks))),
                "content_type": str(meta.get("content_type", "webpage")),
            }

            # Preserve rich doctor and facility metadata fields if present
            for extra_key in (
                "doctor_name",
                "qualification",
                "department",
                "schedule_text",
                "image_url",
                "source_url",
                "facility_name",
                "description",
            ):
                val = meta.get(extra_key)
                if val is not None and val != "":
                    clean_metadata[extra_key] = str(val)


            records.append(
                {
                    "id": str(chunk.chunk_id),
                    "values": vec,
                    "metadata": clean_metadata,
                }
            )

        # Batch upsert into Pinecone
        upserted = self.vector_store.upsert_vectors(
            vectors=records,
            namespace=target_namespace,
            index_name=target_index,
        )

        return {
            "indexed": upserted,
            "failed": 0,
            "dimension": target_dim,
            "namespace": target_namespace,
            "index": target_index,
        }

    def index_chunks(
        self,
        chunks: List[Chunk],
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> int:
        """Embeds and indexes a list of chunks into Pinecone."""
        if not chunks:
            logger.info("No chunks provided for indexing.")
            return 0

        embedded_chunks: List[EmbeddedChunk] = self.embedder.embed_chunks(chunks)
        embeddings = [ec.embedding for ec in embedded_chunks]

        res = self.index_chunks_with_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            namespace=namespace,
            index_name=index_name,
        )
        return res["indexed"]

    def index_from_url(
        self,
        url: str,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the entire pipeline (Fetch -> Clean -> Chunk -> Embed -> Pinecone Upsert) for a URL."""
        start_time = time.time()
        logger.info("Executing URL indexing pipeline for: %s", url)

        # 1. Fetch webpage
        fetcher = Fetcher()
        fetch_res = fetcher.fetch(url)
        if not fetch_res.is_success:
            raise ValueError(f"Failed to fetch webpage at {url}: {fetch_res.error}")

        # 2. Extract clean structured document
        parser = DocumentParser()
        cleaned_doc: CleanedDocument = parser.parse_document(fetch_res.html or "", url)

        # 3. Chunk document
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk_document(cleaned_doc)

        if max_chunks and len(chunks) > max_chunks:
            logger.info("Limiting chunks to %d (out of %d total)", max_chunks, len(chunks))
            chunks = chunks[:max_chunks]

        # 4. Embed & Index
        stats = self.index_chunks_with_embeddings(
            chunks=chunks,
            embeddings=[ec.embedding for ec in self.embedder.embed_chunks(chunks)],
            namespace=namespace,
            index_name=index_name,
        )

        duration = time.time() - start_time
        return {
            "index_name": stats["index"],
            "namespace": stats["namespace"],
            "url": url,
            "title": cleaned_doc.title,
            "indexed_count": stats["indexed"],
            "failed_count": stats["failed"],
            "dimension": stats["dimension"],
            "duration_seconds": round(duration, 3),
        }
