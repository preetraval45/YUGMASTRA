/**
 * YUGMĀSTRA AI Client
 * Cross-platform AI communication layer
 */

import axios, { AxiosInstance } from 'axios';
import { io, Socket } from 'socket.io-client';
import type {
  AIRequest,
  AIResponse,
  Battle,
  BattleEvent,
  ModelInfo,
  SIEMRule,
  ThreatIntel,
  SecurityMetrics,
  WebSocketEvent,
} from './types';

export class YugmastraAIClient {
  private api: AxiosInstance;
  private socket: Socket | null = null;
  private baseURL: string;
  private wsURL: string;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor(config: {
    apiURL?: string;
    wsURL?: string;
    apiKey?: string;
  } = {}) {
    this.baseURL = config.apiURL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
    this.wsURL = config.wsURL || process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3000';

    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(config.apiKey && { Authorization: `Bearer ${config.apiKey}` }),
      },
    });
  }

  // ==================== AI Chat ====================

  async chat(request: AIRequest): Promise<AIResponse> {
    const response = await this.api.post<AIResponse>('/api/ai/chat', request);
    return response.data;
  }

  // ==================== Battle System ====================

  async createBattle(config: {
    redTeamAgent: string;
    blueTeamAgent: string;
    scenario?: string;
  }): Promise<Battle> {
    const response = await this.api.post<Battle>('/api/battles', config);
    return response.data;
  }

  async getBattle(id: string): Promise<Battle> {
    const response = await this.api.get<Battle>(`/api/battles/${id}`);
    return response.data;
  }

  async listBattles(filters?: {
    status?: string;
    limit?: number;
  }): Promise<Battle[]> {
    const response = await this.api.get<Battle[]>('/api/battles', { params: filters });
    return response.data;
  }

  // ==================== Knowledge Graph ====================

  async getGraph(filters?: {
    nodeType?: string;
    limit?: number;
  }): Promise<{ nodes: any[]; edges: any[] }> {
    const response = await this.api.get('/api/knowledge-graph', { params: filters });
    return response.data;
  }

  async queryGraph(query: string): Promise<any> {
    const response = await this.api.post('/api/knowledge-graph/query', { query });
    return response.data;
  }

  // ==================== Model Management ====================

  async listModels(): Promise<ModelInfo[]> {
    const response = await this.api.get<{ models: ModelInfo[] }>('/api/ai/models');
    return response.data.models;
  }

  async trainModel(config: {
    modelType: string;
    datasetPath: string;
    epochs?: number;
    batchSize?: number;
  }): Promise<{ status: string; taskId: string }> {
    const response = await this.api.post('/api/ai/train', config);
    return response.data;
  }

  // ==================== RAG System ====================

  async queryRAG(query: string, topK: number = 5): Promise<any> {
    const response = await this.api.post('/api/ai/rag/query', null, {
      params: { query, top_k: topK },
    });
    return response.data;
  }

  async ingestDocuments(documents: string[], metadata?: Record<string, any>): Promise<any> {
    const response = await this.api.post('/api/ai/ingest', { documents, metadata });
    return response.data;
  }

  // ==================== SIEM Rules ====================

  async generateSIEMRule(config: {
    attackPattern: string;
    format: 'sigma' | 'splunk' | 'elastic' | 'suricata';
  }): Promise<SIEMRule> {
    const response = await this.api.post<SIEMRule>('/api/siem/generate-rule', config);
    return response.data;
  }

  async listSIEMRules(filters?: {
    format?: string;
    severity?: string;
  }): Promise<SIEMRule[]> {
    const response = await this.api.get<SIEMRule[]>('/api/siem/rules', { params: filters });
    return response.data;
  }

  // ==================== Threat Intelligence ====================

  async getThreatIntel(filters?: {
    type?: string;
    confidence?: number;
  }): Promise<ThreatIntel[]> {
    const response = await this.api.get<ThreatIntel[]>('/api/threat-intel', { params: filters });
    return response.data;
  }

  // ==================== Analytics ====================

  async getSecurityMetrics(timeRange?: {
    start: string;
    end: string;
  }): Promise<SecurityMetrics[]> {
    const response = await this.api.get<SecurityMetrics[]>('/api/analytics/metrics', {
      params: timeRange,
    });
    return response.data;
  }

  // ==================== Zero-Day Discovery ====================

  async trainZeroDayModels(config: {
    normalBehaviorData: number[][];
    normalSequences?: number[][][];
  }): Promise<{ status: string; results: any }> {
    const response = await this.api.post('/api/zero-day/train', {
      normal_behavior_data: config.normalBehaviorData,
      normal_sequences: config.normalSequences,
    });
    return response.data;
  }

  async analyzeSystemBehavior(config: {
    behaviorData: number[][];
    metadata: Record<string, any>[];
  }): Promise<{ anomalies: any[]; count: number }> {
    const response = await this.api.post('/api/zero-day/analyze/behavior', {
      behavior_data: config.behaviorData,
      metadata: config.metadata,
    });
    return response.data;
  }

  async analyzeExecutionSequences(config: {
    sequenceData: number[][][];
    metadata: Record<string, any>[];
  }): Promise<{ anomalies: any[]; count: number }> {
    const response = await this.api.post('/api/zero-day/analyze/sequences', {
      sequence_data: config.sequenceData,
      metadata: config.metadata,
    });
    return response.data;
  }

  async discoverVulnerabilities(attackPatterns: Record<string, any>[]): Promise<{
    vulnerabilities: any[];
    count: number;
  }> {
    const response = await this.api.post('/api/zero-day/discover', {
      attack_patterns: attackPatterns,
    });
    return response.data;
  }

  async getZeroDayVulnerabilities(filters?: {
    minSeverity?: number;
    minConfidence?: number;
  }): Promise<{ vulnerabilities: any[]; count: number }> {
    const response = await this.api.get('/api/zero-day/vulnerabilities', {
      params: {
        min_severity: filters?.minSeverity,
        min_confidence: filters?.minConfidence,
      },
    });
    return response.data;
  }

  async getZeroDayStatistics(): Promise<{ statistics: any }> {
    const response = await this.api.get('/api/zero-day/statistics');
    return response.data;
  }

  // ==================== SIEM Rule Generator ====================

  async generateSIEMRule(config: {
    attackPattern: Record<string, any>;
    format?: 'sigma' | 'splunk' | 'elastic' | 'suricata' | 'snort' | 'yara';
    enhanceWithLLM?: boolean;
  }): Promise<{ rule: any }> {
    const response = await this.api.post('/api/siem/generate-rule', null, {
      params: {
        attack_pattern: config.attackPattern,
        format: config.format || 'sigma',
        enhance_with_llm: config.enhanceWithLLM || false,
      },
    });
    return response.data;
  }

  async batchGenerateSIEMRules(config: {
    attackPatterns: Record<string, any>[];
    formats?: string[];
  }): Promise<{ results: any; totalRules: number }> {
    const response = await this.api.post('/api/siem/batch-generate', {
      attack_patterns: config.attackPatterns,
      formats: config.formats,
    });
    return response.data;
  }

  async getSIEMStatistics(): Promise<{ statistics: any }> {
    const response = await this.api.get('/api/siem/statistics');
    return response.data;
  }

  async getSupportedSIEMFormats(): Promise<{ formats: string[]; severities: string[] }> {
    const response = await this.api.get('/api/siem/formats');
    return response.data;
  }

  // ==================== Knowledge Graph (Enhanced) ====================

  async addGraphNode(config: {
    id: string;
    type: string;
    properties: Record<string, any>;
  }): Promise<{ status: string; nodeId: string }> {
    const response = await this.api.post('/api/knowledge-graph/nodes', null, {
      params: config,
    });
    return response.data;
  }

  async addGraphEdge(config: {
    source: string;
    target: string;
    relationship: string;
    weight?: number;
    properties?: Record<string, any>;
  }): Promise<{ status: string; edge: string }> {
    const response = await this.api.post('/api/knowledge-graph/edges', null, {
      params: config,
    });
    return response.data;
  }

  async getAttackChains(techniqueId: string, maxDepth?: number): Promise<{
    chains: string[][];
    count: number;
  }> {
    const response = await this.api.get(`/api/knowledge-graph/attack-chains/${techniqueId}`, {
      params: { max_depth: maxDepth },
    });
    return response.data;
  }

  async getMitigations(attackId: string): Promise<{
    mitigations: any[];
    count: number;
  }> {
    const response = await this.api.get(`/api/knowledge-graph/mitigations/${attackId}`);
    return response.data;
  }

  async getSimilarNodes(nodeId: string, topK?: number): Promise<{
    similarNodes: Array<{ id: string; similarity: number }>;
    count: number;
  }> {
    const response = await this.api.get(`/api/knowledge-graph/similar/${nodeId}`, {
      params: { top_k: topK },
    });
    return response.data;
  }

  async ingestMITREData(mitreData: Record<string, any>): Promise<{
    status: string;
    nodesAdded: number;
  }> {
    const response = await this.api.post('/api/knowledge-graph/ingest/mitre', mitreData);
    return response.data;
  }

  async ingestBattleResults(battleData: Record<string, any>): Promise<{
    status: string;
    nodesAdded: number;
  }> {
    const response = await this.api.post('/api/knowledge-graph/ingest/battle', battleData);
    return response.data;
  }

  async ingestCVEData(cveData: Record<string, any>[]): Promise<{
    status: string;
    nodesAdded: number;
  }> {
    const response = await this.api.post('/api/knowledge-graph/ingest/cve', cveData);
    return response.data;
  }

  async getGraphStatistics(): Promise<{ statistics: any }> {
    const response = await this.api.get('/api/knowledge-graph/statistics');
    return response.data;
  }

  // ==================== WebSocket ====================

  connect(): void {
    if (this.socket?.connected) {
      return;
    }

    this.socket = io(this.wsURL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
    });

    this.socket.on('connect', () => {
      console.log('[YUGMĀSTRA] WebSocket connected');
    });

    this.socket.on('disconnect', () => {
      console.log('[YUGMĀSTRA] WebSocket disconnected');
    });

    this.socket.on('message', (event: WebSocketEvent) => {
      this.emit(event.type, event.data);
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  on(event: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(event);
      if (listeners) {
        listeners.delete(callback);
      }
    };
  }

  private emit(event: string, data: any): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach((callback) => callback(data));
    }
  }

  // ==================== Health Check ====================

  async healthCheck(): Promise<{
    status: string;
    services: Record<string, boolean>;
  }> {
    const response = await this.api.get('/health');
    return response.data;
  }
}

// Singleton instance
let defaultClient: YugmastraAIClient | null = null;

export function getAIClient(config?: Parameters<typeof YugmastraAIClient>[0]): YugmastraAIClient {
  if (!defaultClient || config) {
    defaultClient = new YugmastraAIClient(config);
  }
  return defaultClient;
}
