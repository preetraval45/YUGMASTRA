"""
Vector Store for RAG (Retrieval-Augmented Generation)
Supports multiple vector databases: ChromaDB, Pinecone, Weaviate, FAISS
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss


@dataclass
class Document:
    """Document with metadata for vector storage"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """Advanced vector store with hybrid search capabilities"""

    def __init__(
        self,
        store_type: str = "chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "yugmastra_threats"
    ):
        self.store_type = store_type
        self.collection_name = collection_name

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Initialize vector store
        if store_type == "chromadb":
            self._init_chromadb()
        elif store_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported store type: {store_type}")

    def _init_chromadb(self):
        """Initialize ChromaDB for persistent vector storage"""
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chromadb"
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "YUGMASTRA threat intelligence and security knowledge"}
        )
        print(f"ChromaDB collection '{self.collection_name}' initialized")

    def _init_faiss(self):
        """Initialize FAISS for high-performance similarity search"""
        # Use IVF (Inverted File Index) with PQ (Product Quantization) for large-scale
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.embedding_dim,
            100,  # nlist - number of clusters
            8,    # M - number of sub-quantizers
            8     # nbits - bits per sub-quantizer
        )
        self.documents = []
        print("FAISS index initialized with IVF-PQ")

    def add_documents(self, documents: List[Document]):
        """Add documents to vector store with embeddings"""
        # Generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self.embedder.encode(
            contents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        if self.store_type == "chromadb":
            # Add to ChromaDB
            self.collection.add(
                ids=[doc.id for doc in documents],
                embeddings=embeddings.tolist(),
                documents=contents,
                metadatas=[doc.metadata for doc in documents]
            )
        elif self.store_type == "faiss":
            # Train FAISS index if needed
            if not self.index.is_trained:
                self.index.train(embeddings)

            # Add to FAISS
            self.index.add(embeddings)
            self.documents.extend(documents)

        print(f"Added {len(documents)} documents to vector store")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic similarity search using vector embeddings

        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of (Document, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        if self.store_type == "chromadb":
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                where=filter_metadata
            )

            # Convert to Document objects
            documents = []
            for i in range(len(results['ids'][0])):
                doc = Document(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                documents.append((doc, score))

            return documents

        elif self.store_type == "faiss":
            # FAISS search
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    score = 1.0 / (1.0 + dist)  # Convert L2 distance to similarity
                    results.append((doc, score))

            return results

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining semantic similarity and keyword matching

        Args:
            query: Search query
            k: Number of results
            alpha: Weight for semantic search (1-alpha for keyword)
        """
        # Semantic search
        semantic_results = self.similarity_search(query, k=k*2)

        # Keyword search (simple BM25-style)
        query_terms = set(query.lower().split())
        keyword_scores = {}

        for doc, _ in semantic_results:
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            keyword_scores[doc.id] = overlap / len(query_terms) if query_terms else 0

        # Combine scores
        combined = []
        for doc, sem_score in semantic_results:
            kw_score = keyword_scores.get(doc.id, 0)
            final_score = alpha * sem_score + (1 - alpha) * kw_score
            combined.append((doc, final_score))

        # Sort by combined score and return top k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]


class ThreatIntelligenceRAG:
    """RAG system for cybersecurity threat intelligence"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._load_threat_data()

    def _load_threat_data(self):
        """Load cybersecurity knowledge base into vector store"""

        # MITRE ATT&CK Techniques
        mitre_techniques = [
            Document(
                id="T1566.001",
                content="Spearphishing Attachment: Adversaries send spearphishing emails with malicious attachments to gain initial access. Common file types include Office documents with macros, PDFs with exploits, and executables disguised as legitimate files.",
                metadata={
                    "category": "initial_access",
                    "platform": "windows,linux,macos",
                    "data_sources": "email_gateway,edr,network_traffic"
                }
            ),
            Document(
                id="T1059.001",
                content="PowerShell: Adversaries abuse PowerShell for execution. Common techniques include download cradles, encoded commands, reflective DLL injection, and AMSI bypass. Detection focuses on suspicious parent processes, encoded commands, and network connections.",
                metadata={
                    "category": "execution",
                    "platform": "windows",
                    "data_sources": "process_monitoring,powershell_logs,network_traffic"
                }
            ),
            Document(
                id="T1003.001",
                content="LSASS Memory Dumping: Adversaries dump LSASS memory to extract credentials. Tools like Mimikatz, ProcDump, and Comsvcs.dll are commonly used. Detection includes monitoring for access to LSASS process, suspicious tool execution, and memory dump file creation.",
                metadata={
                    "category": "credential_access",
                    "platform": "windows",
                    "data_sources": "edr,sysmon,windows_event_logs"
                }
            ),
            Document(
                id="T1071.001",
                content="Web Protocols for C2: Adversaries use HTTP/HTTPS for command and control to blend with normal traffic. Detection includes analyzing traffic patterns, checking for known C2 domains, and identifying beaconing behavior through regular intervals.",
                metadata={
                    "category": "command_and_control",
                    "platform": "windows,linux,macos",
                    "data_sources": "network_traffic,proxy_logs,firewall"
                }
            ),
            Document(
                id="T1548.002",
                content="Bypass User Account Control: Adversaries bypass UAC to elevate privileges without user interaction. Common methods include DLL hijacking, CMSTP abuse, and eventvwr.exe hijacking. Detection focuses on suspicious process behavior and registry modifications.",
                metadata={
                    "category": "privilege_escalation",
                    "platform": "windows",
                    "data_sources": "registry,process_monitoring,windows_event_logs"
                }
            )
        ]

        # CVE Examples with exploitation details
        cve_examples = [
            Document(
                id="CVE-2021-44228",
                content="Log4Shell (CVE-2021-44228): Critical RCE vulnerability in Apache Log4j. Exploitation via JNDI lookup in log messages allows arbitrary code execution. Affected versions: 2.0-beta9 to 2.14.1. Mitigation: Upgrade to 2.17.1+, disable JNDI, set formatMsgNoLookups=true.",
                metadata={
                    "category": "vulnerability",
                    "cvss_score": 10.0,
                    "attack_vector": "network",
                    "exploitation_status": "active"
                }
            ),
            Document(
                id="CVE-2017-0144",
                content="EternalBlue (CVE-2017-0144): Critical SMB vulnerability used by WannaCry and NotPetya ransomware. Allows remote code execution via specially crafted packets to SMBv1. Detection: Monitor SMB traffic anomalies, disable SMBv1, apply MS17-010 patch.",
                metadata={
                    "category": "vulnerability",
                    "cvss_score": 8.1,
                    "attack_vector": "network",
                    "exploitation_status": "widespread"
                }
            ),
            Document(
                id="CVE-2014-0160",
                content="Heartbleed (CVE-2014-0160): OpenSSL vulnerability allowing memory disclosure from servers. Attackers can read up to 64KB of memory per request, potentially exposing private keys, passwords, and sensitive data. Mitigation: Upgrade OpenSSL to 1.0.1g+, regenerate keys and certificates.",
                metadata={
                    "category": "vulnerability",
                    "cvss_score": 7.5,
                    "attack_vector": "network",
                    "exploitation_status": "historical"
                }
            )
        ]

        # Detection Rules and Best Practices
        detection_rules = [
            Document(
                id="RULE-001",
                content="Detect PowerShell Download Cradle: Monitor for PowerShell execution with patterns like 'IEX', 'Invoke-WebRequest', 'DownloadString', or 'wget' combined with URL patterns. Check for encoded commands using -encodedcommand flag. Alert on suspicious parent processes.",
                metadata={
                    "category": "detection_rule",
                    "technique": "T1059.001",
                    "severity": "high"
                }
            ),
            Document(
                id="RULE-002",
                content="Lateral Movement via Pass-the-Hash: Detect abnormal NTLM authentication patterns including authentication from non-standard source IPs, multiple authentication attempts across many hosts, and privileged account usage from workstations.",
                metadata={
                    "category": "detection_rule",
                    "technique": "T1550.002",
                    "severity": "critical"
                }
            ),
            Document(
                id="RULE-003",
                content="Data Exfiltration via DNS: Identify DNS tunneling through analysis of query patterns including unusually long subdomain names, high volume of queries to single domain, non-standard query types (TXT records), and entropy analysis of domain names.",
                metadata={
                    "category": "detection_rule",
                    "technique": "T1048.003",
                    "severity": "high"
                }
            )
        ]

        # Threat Actor Profiles
        threat_actors = [
            Document(
                id="APT28",
                content="APT28 (Fancy Bear): Russian state-sponsored APT group attributed to GRU. Known for targeting government, military, and political organizations. TTPs include spearphishing, credential harvesting, X-Agent malware, and exploitation of zero-days. Active since 2004.",
                metadata={
                    "category": "threat_actor",
                    "origin": "russia",
                    "sophistication": "advanced",
                    "targets": "government,military,political"
                }
            ),
            Document(
                id="LAZARUS",
                content="Lazarus Group: North Korean APT conducting espionage and financially-motivated attacks. Responsible for Sony Pictures breach, WannaCry ransomware, and cryptocurrency exchange hacks. Uses custom malware families, living-off-the-land techniques, and supply chain compromises.",
                metadata={
                    "category": "threat_actor",
                    "origin": "north_korea",
                    "sophistication": "advanced",
                    "targets": "financial,cryptocurrency,media"
                }
            )
        ]

        # Combine all documents
        all_documents = (
            mitre_techniques +
            cve_examples +
            detection_rules +
            threat_actors
        )

        self.vector_store.add_documents(all_documents)
        print(f"Loaded {len(all_documents)} threat intelligence documents")

    def query(self, question: str, k: int = 3) -> Dict:
        """
        Query the threat intelligence knowledge base

        Args:
            question: Natural language question
            k: Number of relevant documents to retrieve

        Returns:
            Dict containing retrieved documents and generated answer
        """
        # Retrieve relevant documents
        results = self.vector_store.hybrid_search(question, k=k)

        # Format context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}] (Relevance: {score:.2f})\n"
                f"ID: {doc.id}\n"
                f"Content: {doc.content}\n"
                f"Metadata: {doc.metadata}\n"
            )

        context = "\n".join(context_parts)

        return {
            "question": question,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                }
                for doc, score in results
            ],
            "context": context,
            "num_results": len(results)
        }


if __name__ == "__main__":
    # Example usage
    print("Initializing YUGMASTRA RAG System...")

    # Create vector store
    vector_store = VectorStore(
        store_type="chromadb",
        embedding_model="all-MiniLM-L6-v2"
    )

    # Initialize RAG system
    rag = ThreatIntelligenceRAG(vector_store)

    # Example queries
    queries = [
        "How can I detect PowerShell attacks?",
        "What is the Log4Shell vulnerability?",
        "Tell me about APT28 tactics",
        "How to prevent credential dumping?",
        "What are indicators of DNS tunneling?"
    ]

    print("\n" + "="*80)
    print("TESTING RAG QUERIES")
    print("="*80 + "\n")

    for query in queries:
        print(f"\nQUERY: {query}")
        print("-" * 80)
        result = rag.query(query, k=2)

        for doc in result['retrieved_documents']:
            print(f"\n📄 {doc['id']} (Score: {doc['relevance_score']:.3f})")
            print(f"   {doc['content'][:200]}...")

        print("\n" + "="*80)
