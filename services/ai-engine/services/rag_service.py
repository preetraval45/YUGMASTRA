import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import logging
import hashlib

logger = logging.getLogger(__name__)

class RAGService:
    """
    Retrieval-Augmented Generation service for cybersecurity knowledge
    """

    def __init__(self):
        # Initialize ChromaDB for vector storage (new API)
        self.client = chromadb.PersistentClient(path="./data/chroma")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create collections for different knowledge domains
        self.collections = {
            "threats": self._get_or_create_collection("threats"),
            "vulnerabilities": self._get_or_create_collection("vulnerabilities"),
            "mitre_attack": self._get_or_create_collection("mitre_attack"),
            "cve_database": self._get_or_create_collection("cve_database"),
            "general_security": self._get_or_create_collection("general_security")
        }

        logger.info("RAG Service initialized with ChromaDB")

    def _get_or_create_collection(self, name: str):
        """
        Get or create a ChromaDB collection
        """
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"description": f"Knowledge base for {name}"}
            )
        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            raise

    async def ingest_documents(
        self,
        documents: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = "general_security"
    ):
        """
        Ingest documents into RAG system
        """
        try:
            logger.info(f"Ingesting {len(documents)} documents into {collection_name}")

            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection {collection_name} not found")

            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()

            # Generate IDs
            ids = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]

            # Prepare metadata
            metadatas = [metadata or {} for _ in documents]

            # Add to collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

            logger.info(f"Successfully ingested {len(documents)} documents")

        except Exception as e:
            logger.error(f"Document ingestion error: {str(e)}")
            raise

    async def query(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = "general_security",
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query RAG system for relevant context
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection {collection_name} not found")

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()

            # Query collection
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=filter_metadata
            )

            # Format results
            formatted_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "document": doc,
                        "distance": results["distances"][0][i] if "distances" in results else None,
                        "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            raise

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context to augment LLM prompt
        """
        try:
            # Query all relevant collections
            all_results = []

            for collection_name in self.collections.keys():
                results = await self.query(query, top_k=2, collection_name=collection_name)
                all_results.extend(results)

            # Sort by relevance (distance)
            all_results.sort(key=lambda x: x.get("distance", float("inf")))

            # Build context string
            context_parts = []
            total_length = 0

            for result in all_results:
                doc = result["document"]
                doc_length = len(doc.split())

                if total_length + doc_length > max_tokens:
                    break

                context_parts.append(doc)
                total_length += doc_length

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return ""

    def is_ready(self) -> bool:
        """
        Check if RAG service is ready
        """
        return self.client is not None and len(self.collections) > 0

    async def load_mitre_attack_data(self):
        """
        Load MITRE ATT&CK framework data into RAG
        """
        try:
            logger.info("Loading MITRE ATT&CK data...")

            # Sample MITRE ATT&CK techniques
            mitre_data = [
                {
                    "technique": "T1059.001 - PowerShell",
                    "description": "Adversaries may abuse PowerShell commands and scripts for execution. PowerShell is a powerful interactive command-line interface and scripting environment included in the Windows operating system.",
                    "tactics": ["Execution"],
                    "mitigation": "Disable or remove PowerShell if not required, use AppLocker or Windows Defender Application Control to restrict PowerShell execution"
                },
                {
                    "technique": "T1003 - OS Credential Dumping",
                    "description": "Adversaries may attempt to dump credentials to obtain account login and credential material in the form of a hash or a clear text password from the operating system and software.",
                    "tactics": ["Credential Access"],
                    "mitigation": "Monitor for unusual processes accessing LSASS memory, implement credential guard, use privileged access management"
                },
                {
                    "technique": "T1190 - Exploit Public-Facing Application",
                    "description": "Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program using software, data, or commands in order to cause unintended or unanticipated behavior.",
                    "tactics": ["Initial Access"],
                    "mitigation": "Regularly update and patch applications, implement WAF, use network segmentation"
                },
                # Add more techniques...
            ]

            documents = [
                f"{item['technique']}\n{item['description']}\nTactics: {', '.join(item['tactics'])}\nMitigation: {item['mitigation']}"
                for item in mitre_data
            ]

            await self.ingest_documents(
                documents=documents,
                metadata={"source": "MITRE ATT&CK"},
                collection_name="mitre_attack"
            )

            logger.info("MITRE ATT&CK data loaded successfully")

        except Exception as e:
            logger.error(f"Error loading MITRE data: {str(e)}")
