import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.core.config import settings
from app.core.logging_config import logger
from app.ingestion.chunker import Chunk

# Global in-memory cache to guarantee the model is loaded only once across the application
_MODEL_CACHE: Dict[str, Any] = {}


@dataclass
class EmbeddedChunk:
    """Represents a chunk bundled with its computed numerical embedding."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }


class DocumentEmbedder:
    """Embedding service that converts texts and document chunks into numerical vectors
    using sentence-transformers models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._dimension: Optional[int] = None

    def _get_model(self) -> Any:
        """Retrieves or loads the SentenceTransformer model singleton."""
        global _MODEL_CACHE
        if self.model_name not in _MODEL_CACHE:
            logger.info("Loading embedding model: %s ...", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                logger.error("sentence-transformers package not installed: %s", err)
                raise ImportError(
                    "sentence-transformers is not installed. Please install it via 'pip install sentence-transformers'."
                ) from err

            # Load model instance and cache it
            model_instance = SentenceTransformer(self.model_name)
            _MODEL_CACHE[self.model_name] = model_instance
            logger.info("Successfully loaded and cached embedding model: %s", self.model_name)

        return _MODEL_CACHE[self.model_name]

    def embedding_dimension(self) -> int:
        """Reports the vector dimension dynamically from the loaded model without hardcoding."""
        if self._dimension is not None:
            return self._dimension

        model = self._get_model()
        if hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
            if dim and isinstance(dim, int):
                self._dimension = dim
                return self._dimension

        # Fallback: encode a short probe string to determine dimensionality
        probe_vector = model.encode("probe", show_progress_bar=False)
        self._dimension = int(len(probe_vector))
        return self._dimension

    def embed_text(
        self,
        text: str,
        as_numpy: bool = False,
    ) -> Union[List[float], np.ndarray]:
        """Generates an embedding vector for a single string.
        
        Args:
            text: Input text string.
            as_numpy: If True, returns a 1D numpy array; otherwise returns a Python list of floats.
            
        Returns:
            Vector of floats with dimension equal to embedding_dimension().
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be str, got {type(text).__name__}")

        clean_text = text.strip()
        if not clean_text:
            # Handle empty text gracefully: return zero-vector
            dim = self.embedding_dimension()
            logger.debug("Empty text received; returning zero-vector of dimension %d", dim)
            zero_vec = np.zeros(dim, dtype=np.float32)
            return zero_vec if as_numpy else zero_vec.tolist()

        embeddings = self.embed_texts([clean_text], as_numpy=as_numpy)
        return embeddings[0]

    def embed_texts(
        self,
        texts: List[str],
        as_numpy: bool = False,
    ) -> Union[List[List[float]], np.ndarray]:
        """Generates embedding vectors for a batch of strings.
        
        Args:
            texts: List of text strings to embed.
            as_numpy: If True, returns a 2D numpy array; otherwise returns a list of lists.
            
        Returns:
            Batch vectors of shape (len(texts), embedding_dimension).
        """
        if not isinstance(texts, list):
            raise TypeError(f"Expected texts to be list, got {type(texts).__name__}")

        if not texts:
            dim = self.embedding_dimension()
            return np.empty((0, dim), dtype=np.float32) if as_numpy else []

        model = self._get_model()
        dim = self.embedding_dimension()

        # Identify empty texts to replace with zero-vectors while encoding non-empty texts in batches
        valid_indices: List[int] = []
        valid_texts: List[str] = []

        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"Item at index {idx} is not a string: {type(t).__name__}")
            t_clean = t.strip()
            if t_clean:
                valid_indices.append(idx)
                valid_texts.append(t_clean)

        total_count = len(texts)
        result_array = np.zeros((total_count, dim), dtype=np.float32)

        if valid_texts:
            logger.debug(
                "Batch encoding %d texts (batch_size=%d, model=%s)",
                len(valid_texts),
                self.batch_size,
                self.model_name,
            )
            encoded = model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            for i, original_idx in enumerate(valid_indices):
                result_array[original_idx] = encoded[i]

        if as_numpy:
            return result_array

        return [row.tolist() for row in result_array]

    def embed_chunks(
        self,
        chunks: List[Union[Chunk, Dict[str, Any]]],
        as_numpy: bool = False,
    ) -> List[EmbeddedChunk]:
        """Processes chunks from Stage 4, generating embeddings while completely
        preserving the original chunk id, document id, content, and metadata.
        
        Args:
            chunks: List of Chunk dataclasses or dictionaries containing chunk_id, document_id, content, metadata.
            as_numpy: If True, embeddings within the results are preserved as numpy arrays (otherwise Python lists).
            
        Returns:
            List of EmbeddedChunk objects containing both original chunk data and numerical embedding vectors.
        """
        if not chunks:
            return []

        # Extract contents and normalize chunk representation
        chunk_items: List[Dict[str, Any]] = []
        texts_to_embed: List[str] = []

        for idx, item in enumerate(chunks):
            if isinstance(item, Chunk):
                c_dict = item.to_dict()
            elif isinstance(item, dict):
                c_dict = dict(item)
            else:
                raise TypeError(f"Expected Chunk or dict at index {idx}, got {type(item).__name__}")

            chunk_items.append(c_dict)
            texts_to_embed.append(c_dict.get("content", ""))

        # Batch embed all chunk texts
        raw_embeddings = self.embed_texts(texts_to_embed, as_numpy=False)

        embedded_chunks: List[EmbeddedChunk] = []
        for c_dict, emb in zip(chunk_items, raw_embeddings):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=c_dict.get("chunk_id", ""),
                    document_id=c_dict.get("document_id", ""),
                    content=c_dict.get("content", ""),
                    metadata=c_dict.get("metadata", {}),
                    embedding=emb,
                )
            )

        logger.info(
            "Successfully embedded %d chunks (dimension=%d)",
            len(embedded_chunks),
            self.embedding_dimension(),
        )
        return embedded_chunks
