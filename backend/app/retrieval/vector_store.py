import time
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging_config import logger


class PineconeVectorStore:
    """Manages Pinecone serverless index lifecycle, vector upserting,
    namespace routing, and semantic similarity search.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.namespace = namespace or settings.PINECONE_NAMESPACE
        self.cloud = cloud or settings.PINECONE_CLOUD
        self.region = region or settings.PINECONE_REGION
        self._client = client
        self._index = None

    @property
    def client(self) -> Any:
        """Returns or lazily initializes the Pinecone client instance."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "Pinecone API key is missing. Please set PINECONE_API_KEY in backend/.env "
                "or provide it during initialization."
            )

        try:
            from pinecone import Pinecone
            self._client = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully.")
            return self._client
        except Exception as err:
            logger.error("Failed to initialize Pinecone client: %s", err)
            raise

    def get_index(self, index_name: Optional[str] = None) -> Any:
        """Connects to the specified Pinecone index."""
        target_name = index_name or self.index_name
        if self._index is not None and getattr(self._index, "name", None) == target_name:
            return self._index

        pc = self.client
        self._index = pc.Index(target_name)
        return self._index

    def create_index_if_not_exists(
        self,
        index_name: Optional[str] = None,
        dimension: int = 384,
        metric: str = "cosine",
    ) -> bool:
        """Checks if a Pinecone index exists; if not, creates a serverless index with
        specified dimension and metric. Verifies configuration if index already exists.
        
        Args:
            index_name: Target index name.
            dimension: Embedding vector size (e.g. 384 for all-MiniLM-L6-v2).
            metric: Distance metric ('cosine', 'dotproduct', 'euclidean').
            
        Returns:
            bool: True if created, False if already exists and verified.
        """
        target_name = index_name or self.index_name
        pc = self.client

        # Discover existing indexes
        index_list = pc.list_indexes()
        if hasattr(index_list, "names"):
            existing_names = list(index_list.names())
        elif hasattr(index_list, "__iter__"):
            existing_names = [getattr(idx, "name", str(idx)) for idx in index_list]
        else:
            existing_names = []

        if target_name in existing_names:
            logger.info("Pinecone index '%s' already exists. Verifying configuration...", target_name)
            desc = pc.describe_index(target_name)
            if hasattr(desc, "dimension") and desc.dimension != dimension:
                raise ValueError(
                    f"Pinecone index '{target_name}' dimension mismatch: "
                    f"expected {dimension}, found {desc.dimension}."
                )
            if hasattr(desc, "metric") and desc.metric.lower() != metric.lower():
                raise ValueError(
                    f"Pinecone index '{target_name}' metric mismatch: "
                    f"expected {metric}, found {desc.metric}."
                )
            logger.info(
                "Pinecone index '%s' verified (dimension=%d, metric=%s).",
                target_name,
                getattr(desc, "dimension", dimension),
                getattr(desc, "metric", metric),
            )
            return False

        # Create new serverless index
        logger.info(
            "Creating serverless Pinecone index '%s' (dimension=%d, metric=%s, cloud=%s, region=%s)...",
            target_name,
            dimension,
            metric,
            self.cloud,
            self.region,
        )
        from pinecone import ServerlessSpec
        spec = ServerlessSpec(cloud=self.cloud, region=self.region)
        pc.create_index(
            name=target_name,
            dimension=dimension,
            metric=metric,
            spec=spec,
        )
        logger.info("Successfully initiated creation of Pinecone index '%s'.", target_name)
        return True

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        index_name: Optional[str] = None,
    ) -> int:
        """Batch upserts vector records into Pinecone.
        
        Args:
            vectors: List of dicts formatted as {"id": chunk_id, "values": [...], "metadata": {...}}.
            namespace: Hospital namespace (defaults to settings.PINECONE_NAMESPACE).
            batch_size: Batch size for network requests.
            index_name: Optional index override.
            
        Returns:
            int: Number of vectors upserted.
        """
        if not vectors:
            logger.info("No vectors provided for upsert.")
            return 0

        target_namespace = namespace if namespace is not None else self.namespace
        batch_limit = batch_size or settings.PINECONE_UPSERT_BATCH_SIZE
        index = self.get_index(index_name)

        start_time = time.time()
        total_vectors = len(vectors)

        for i in range(0, total_vectors, batch_limit):
            batch = vectors[i : i + batch_limit]
            index.upsert(vectors=batch, namespace=target_namespace)

        duration = time.time() - start_time
        logger.info(
            "Successfully upserted %d vectors to index '%s' (namespace='%s') in %.3fs (batch_size=%d).",
            total_vectors,
            index_name or self.index_name,
            target_namespace,
            duration,
            batch_limit,
        )
        return total_vectors

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes vector similarity search in Pinecone and returns ranked matches.
        
        Args:
            query_vector: 384-dimensional query vector.
            top_k: Number of most similar results to retrieve.
            namespace: Hospital namespace.
            filter_dict: Optional metadata filtering dictionary (e.g. {"section": "Cardiology"}).
            index_name: Optional index override.
            
        Returns:
            List of matching records with id, score, content, and metadata.
        """
        if not query_vector:
            raise ValueError("Query vector cannot be empty.")

        target_namespace = namespace if namespace is not None else self.namespace
        index = self.get_index(index_name)

        start_time = time.time()
        query_kwargs = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": target_namespace,
        }
        if filter_dict:
            query_kwargs["filter"] = filter_dict

        response = index.query(**query_kwargs)
        duration = time.time() - start_time

        matches = getattr(response, "matches", []) or response.get("matches", [])
        logger.info(
            "Pinecone search in '%s' (namespace='%s') returned %d matches for top_k=%d in %.3fs.",
            index_name or self.index_name,
            target_namespace,
            len(matches),
            top_k,
            duration,
        )

        results: List[Dict[str, Any]] = []
        for match in matches:
            if isinstance(match, dict):
                m_id = match.get("id", "")
                m_score = match.get("score", 0.0)
                metadata = match.get("metadata", {}) or {}
            else:
                m_id = getattr(match, "id", "")
                m_score = getattr(match, "score", 0.0)
                metadata = getattr(match, "metadata", {}) or {}

            results.append(
                {
                    "id": m_id,
                    "score": round(float(m_score), 4),
                    "content": metadata.get("content", ""),
                    "section": metadata.get("section", "General"),
                    "url": metadata.get("url", ""),
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", "website"),
                    "metadata": metadata,
                }
            )

        return results

    def get_index_stats(
        self,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves index and namespace statistics."""
        target_name = index_name or self.index_name
        target_namespace = namespace if namespace is not None else self.namespace
        index = self.get_index(target_name)

        stats = index.describe_index_stats()
        # Handle dict or object response from Pinecone SDK
        total_vectors = getattr(stats, "total_vector_count", None)
        if total_vectors is None:
            total_vectors = stats.get("total_vector_count", 0)

        dim = getattr(stats, "dimension", None)
        if dim is None:
            dim = stats.get("dimension", 384)

        namespaces_dict = getattr(stats, "namespaces", None)
        if namespaces_dict is None:
            namespaces_dict = stats.get("namespaces", {})

        ns_info = namespaces_dict.get(target_namespace, {}) if namespaces_dict else {}
        ns_count = getattr(ns_info, "vector_count", None)
        if ns_count is None and isinstance(ns_info, dict):
            ns_count = ns_info.get("vector_count", 0)

        return {
            "index": target_name,
            "namespace": target_namespace,
            "dimension": dim or 384,
            "total_vector_count": total_vectors or 0,
            "namespace_vector_count": ns_count or 0,
        }
