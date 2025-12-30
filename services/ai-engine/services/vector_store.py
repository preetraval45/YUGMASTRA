import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """
    High-performance vector storage using FAISS
    """

    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension

        # Initialize FAISS indices for different data types
        self.indices: Dict[str, faiss.IndexFlatL2] = {}
        self.document_store: Dict[str, List[str]] = {}

        # Create indices
        self._initialize_indices()

        logger.info("Vector Store initialized with FAISS")

    def _initialize_indices(self):
        """
        Initialize FAISS indices for different document types
        """
        index_types = ["threats", "vulnerabilities", "attack_patterns", "defense_strategies"]

        for index_type in index_types:
            self.indices[index_type] = faiss.IndexFlatL2(self.dimension)
            self.document_store[index_type] = []

        logger.info(f"Initialized {len(index_types)} FAISS indices")

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text data
        """
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings

        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise

    async def store_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        index_type: str = "threats"
    ):
        """
        Store embeddings in FAISS index
        """
        try:
            if index_type not in self.indices:
                raise ValueError(f"Index type {index_type} not found")

            # Add to FAISS index
            self.indices[index_type].add(embeddings.astype('float32'))

            # Store documents
            self.document_store[index_type].extend(texts)

            logger.info(f"Stored {len(texts)} embeddings in {index_type} index")

        except Exception as e:
            logger.error(f"Storage error: {str(e)}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        index_type: str = "threats"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        """
        try:
            if index_type not in self.indices:
                raise ValueError(f"Index type {index_type} not found")

            # Generate query embedding
            query_embedding = await self.embed_texts([query])

            # Search FAISS index
            distances, indices = self.indices[index_type].search(
                query_embedding.astype('float32'),
                top_k
            )

            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_store[index_type]):
                    results.append({
                        "document": self.document_store[index_type][idx],
                        "distance": float(distances[0][i]),
                        "similarity": 1 / (1 + float(distances[0][i]))  # Convert distance to similarity
                    })

            return results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def search_all_indices(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all indices
        """
        results = {}

        for index_type in self.indices.keys():
            results[index_type] = await self.search(query, top_k, index_type)

        return results

    def save_index(self, index_type: str, path: str):
        """
        Save FAISS index to disk
        """
        try:
            if index_type not in self.indices:
                raise ValueError(f"Index type {index_type} not found")

            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.indices[index_type], f"{path}.index")

            # Save document store
            with open(f"{path}.docs", "wb") as f:
                pickle.dump(self.document_store[index_type], f)

            logger.info(f"Saved {index_type} index to {path}")

        except Exception as e:
            logger.error(f"Save error: {str(e)}")
            raise

    def load_index(self, index_type: str, path: str):
        """
        Load FAISS index from disk
        """
        try:
            # Load FAISS index
            self.indices[index_type] = faiss.read_index(f"{path}.index")

            # Load document store
            with open(f"{path}.docs", "rb") as f:
                self.document_store[index_type] = pickle.load(f)

            logger.info(f"Loaded {index_type} index from {path}")

        except Exception as e:
            logger.error(f"Load error: {str(e)}")
            raise

    def is_ready(self) -> bool:
        """
        Check if vector store is ready
        """
        return len(self.indices) > 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about vector stores
        """
        stats = {}

        for index_type, index in self.indices.items():
            stats[index_type] = {
                "total_vectors": index.ntotal,
                "total_documents": len(self.document_store[index_type]),
                "dimension": self.dimension
            }

        return stats
