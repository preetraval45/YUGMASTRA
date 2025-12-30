/**
 * Shared TypeScript types for AI SDK
 */

// AI Agent Types
export type AgentMode = 'red-team' | 'blue-team' | 'evolution' | 'analyst';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
}

export interface AIRequest {
  message: string;
  mode: AgentMode;
  history?: Message[];
  context?: Record<string, any>;
}

export interface AIResponse {
  response: string;
  mode: AgentMode;
  confidence: number;
  sources: string[];
  timestamp: string;
  metadata?: {
    model?: string;
    tokens_used?: number;
    reasoning_chain?: string[];
  };
}

// Battle System Types
export interface Battle {
  id: string;
  redTeamAgent: string;
  blueTeamAgent: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  winner?: 'red' | 'blue' | 'draw';
  score: {
    red: number;
    blue: number;
  };
  events: BattleEvent[];
}

export interface BattleEvent {
  id: string;
  timestamp: string;
  type: 'attack' | 'defense' | 'detection' | 'breach' | 'block';
  agent: 'red' | 'blue';
  action: string;
  target?: string;
  success: boolean;
  metadata?: Record<string, any>;
}

// Knowledge Graph Types
export interface GraphNode {
  id: string;
  type: 'attack' | 'defense' | 'vulnerability' | 'asset' | 'threat_actor' | 'technique';
  label: string;
  connections: string[];
  position: [number, number, number];
  velocity: { x: number; y: number; z: number };
  severity?: 'critical' | 'high' | 'medium' | 'low';
  metadata: {
    mitre?: string;
    cve?: string;
    score?: number;
    description?: string;
  };
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

// ML Model Types
export interface ModelInfo {
  id: string;
  name: string;
  type: 'llm' | 'rl' | 'classifier' | 'embedding';
  version: string;
  status: 'loaded' | 'loading' | 'error';
  metrics?: {
    accuracy?: number;
    latency?: number;
    memory_usage?: number;
  };
}

// Training Types
export interface TrainingConfig {
  modelType: string;
  datasetPath: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  eta: number; // seconds
}

// SIEM Rule Types
export interface SIEMRule {
  id: string;
  name: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  format: 'sigma' | 'splunk' | 'elastic' | 'suricata' | 'yara';
  rule: string;
  mitre_techniques: string[];
  tags: string[];
  confidence: number;
  created_at: string;
  tested: boolean;
  false_positive_rate?: number;
}

// Threat Intelligence Types
export interface ThreatIntel {
  id: string;
  type: 'indicator' | 'campaign' | 'actor' | 'technique';
  value: string;
  confidence: number;
  source: string;
  first_seen: string;
  last_seen: string;
  tags: string[];
  related_threats: string[];
}

// Analytics Types
export interface SecurityMetrics {
  timestamp: string;
  attacks_detected: number;
  attacks_blocked: number;
  false_positives: number;
  mean_time_to_detect: number; // seconds
  mean_time_to_respond: number; // seconds
  security_score: number; // 0-100
}

// WebSocket Events
export type WebSocketEvent =
  | { type: 'battle_started'; data: Battle }
  | { type: 'battle_event'; data: BattleEvent }
  | { type: 'battle_completed'; data: Battle }
  | { type: 'model_update'; data: ModelInfo }
  | { type: 'training_progress'; data: TrainingProgress }
  | { type: 'alert'; data: { severity: string; message: string } };
