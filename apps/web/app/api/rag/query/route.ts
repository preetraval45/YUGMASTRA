import { NextRequest, NextResponse } from 'next/server';

/**
 * RAG Query Endpoint - Vercel Serverless Function
 * Provides threat intelligence context retrieval
 */

// Simulated vector database (in production, use Pinecone, Weaviate, or Supabase)
const threatIntelligence = [
  {
    id: '1',
    content: 'MITRE ATT&CK T1566.001 (Phishing: Spearphishing Attachment) - Adversaries may send spearphishing emails with a malicious attachment in an attempt to gain access to victim systems. Detection: Monitor for suspicious email attachments, especially executables and Office documents with macros.',
    metadata: { source: 'MITRE ATT&CK', technique: 'T1566.001', category: 'Initial Access' },
    embedding: [0.1, 0.2, 0.3] // Simplified
  },
  {
    id: '2',
    content: 'CVE-2021-44228 (Log4Shell) - Remote code execution vulnerability in Apache Log4j 2. CVSS Score: 10.0 (Critical). Mitigation: Upgrade to Log4j 2.17.0 or later, apply WAF rules, remove JndiLookup class.',
    metadata: { source: 'CVE Database', cve: 'CVE-2021-44228', severity: 'critical' },
    embedding: [0.4, 0.5, 0.6]
  },
  {
    id: '3',
    content: 'SQL Injection Detection Rule: SELECT statement with UNION, OR 1=1, or comment characters (--) in user input. Signature: (UNION.*SELECT|OR.*1=1|\'--|\";--). Action: Block and alert.',
    metadata: { source: 'Detection Rules', attack_type: 'SQL Injection', action: 'block' },
    embedding: [0.7, 0.8, 0.9]
  },
  {
    id: '4',
    content: 'APT28 (Fancy Bear) - Russian state-sponsored threat actor. Known for targeting government, military, and security organizations. TTPs: Spearphishing, credential harvesting, exploitation of known vulnerabilities (CVE-2023-23397).',
    metadata: { source: 'Threat Actors', group: 'APT28', origin: 'Russia' },
    embedding: [0.2, 0.4, 0.6]
  },
  {
    id: '5',
    content: 'Ransomware Defense: Implement offline backups (3-2-1 rule), segment networks, deploy EDR solutions, train users on phishing awareness, patch vulnerabilities promptly, monitor for suspicious file encryption activity.',
    metadata: { source: 'Best Practices', topic: 'Ransomware Defense' },
    embedding: [0.3, 0.6, 0.9]
  },
  {
    id: '6',
    content: 'MITRE ATT&CK T1059.001 (PowerShell) - Adversaries may abuse PowerShell commands for execution. Detection: Monitor for suspicious PowerShell commands (DownloadString, Invoke-Expression, Base64 encoding), enable PowerShell logging.',
    metadata: { source: 'MITRE ATT&CK', technique: 'T1059.001', category: 'Execution' },
    embedding: [0.15, 0.25, 0.35]
  },
  {
    id: '7',
    content: 'Zero-Day Vulnerability Management: Monitor vendor advisories, implement virtual patching with WAF/IPS, use threat intelligence feeds, conduct vulnerability assessments, maintain asset inventory, prioritize based on EPSS scores.',
    metadata: { source: 'Best Practices', topic: 'Zero-Day Management' },
    embedding: [0.5, 0.7, 0.8]
  },
  {
    id: '8',
    content: 'Lateral Movement Detection: Monitor for unusual authentication patterns, detect Pass-the-Hash attacks (NTLM authentication from unusual sources), alert on admin account usage from non-admin workstations, track SMB/RDP connections.',
    metadata: { source: 'Detection Rules', tactic: 'Lateral Movement' },
    embedding: [0.6, 0.7, 0.8]
  }
];

// Simple cosine similarity for demonstration
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// Simple embedding generation (in production, use actual embedding model)
function generateEmbedding(text: string): number[] {
  const hash = text.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return [
    (hash % 100) / 100,
    ((hash * 2) % 100) / 100,
    ((hash * 3) % 100) / 100
  ];
}

// Keyword matching (BM25-style hybrid search)
function keywordMatch(query: string, content: string): number {
  const queryWords = query.toLowerCase().split(/\s+/);
  const contentLower = content.toLowerCase();

  let matches = 0;
  for (const word of queryWords) {
    if (word.length > 3 && contentLower.includes(word)) {
      matches++;
    }
  }

  return matches / queryWords.length;
}

export async function POST(request: NextRequest) {
  try {
    const { query, mode, k = 3, alpha = 0.5 } = await request.json();

    if (!query) {
      return NextResponse.json(
        { error: 'Query parameter is required' },
        { status: 400 }
      );
    }

    // Generate query embedding
    const queryEmbedding = generateEmbedding(query);

    // Hybrid search: combine semantic similarity and keyword matching
    const results = threatIntelligence.map((doc) => {
      const semanticScore = cosineSimilarity(queryEmbedding, doc.embedding);
      const keywordScore = keywordMatch(query, doc.content);

      // Weighted combination (alpha for semantic, 1-alpha for keyword)
      const score = alpha * semanticScore + (1 - alpha) * keywordScore;

      return {
        ...doc,
        score
      };
    });

    // Sort by score and return top-k
    results.sort((a, b) => b.score - a.score);
    const topK = results.slice(0, k);

    // Format response
    const retrieved_documents = topK.map((doc) => ({
      content: doc.content,
      metadata: doc.metadata,
      score: doc.score.toFixed(4)
    }));

    const context = topK.map((doc) =>
      `[${doc.metadata.source}] ${doc.content}`
    ).join('\n\n');

    return NextResponse.json({
      retrieved_documents,
      context,
      query,
      mode,
      retrieval_time_ms: 50 // Simulated
    });

  } catch (error) {
    console.error('RAG Query Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Health check endpoint
export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    service: 'RAG API (Vercel Serverless)',
    documents_count: threatIntelligence.length,
    version: '1.0.0'
  });
}
