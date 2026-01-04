/**
 * React Hooks for AI SDK (Web, Mobile, Desktop)
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getAIClient } from './client';
import type { AIRequest, AIResponse, Battle, ModelInfo, SIEMRule } from './types';

// ==================== useAIChat ====================
export function useAIChat() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const client = getAIClient();

  const chat = useCallback(async (request: AIRequest): Promise<AIResponse | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await client.chat(request);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { chat, loading, error };
}

// ==================== useBattles ====================
export function useBattles() {
  const [battles, setBattles] = useState<Battle[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const client = getAIClient();

  const fetchBattles = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await client.listBattles();
      setBattles(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setLoading(false);
    }
  }, [client]);

  const createBattle = useCallback(async (config: Parameters<typeof client.createBattle>[0]) => {
    setLoading(true);
    setError(null);

    try {
      const battle = await client.createBattle(config);
      setBattles(prev => [battle, ...prev]);
      return battle;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
      return null;
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchBattles();
  }, [fetchBattles]);

  return { battles, loading, error, fetchBattles, createBattle };
}

// ==================== useWebSocket ====================
export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const client = useRef(getAIClient());

  useEffect(() => {
    client.current.connect();
    setConnected(true);

    return () => {
      client.current.disconnect();
      setConnected(false);
    };
  }, []);

  const subscribe = useCallback((event: string, callback: (data: any) => void) => {
    return client.current.on(event, callback);
  }, []);

  return { connected, subscribe };
}

// ==================== useModels ====================
export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const client = getAIClient();

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await client.listModels();
      setModels(data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return { models, loading, fetchModels };
}

// ==================== useSIEMRules ====================
export function useSIEMRules() {
  const [rules, setRules] = useState<SIEMRule[]>([]);
  const [loading, setLoading] = useState(false);
  const client = getAIClient();

  const fetchRules = useCallback(async () => {
    setLoading(true);
    try {
      const data = await client.listSIEMRules();
      setRules(data);
    } catch (err) {
      console.error('Failed to fetch SIEM rules:', err);
    } finally {
      setLoading(false);
    }
  }, [client]);

  const generateRule = useCallback(async (config: Parameters<typeof client.generateSIEMRule>[0]) => {
    setLoading(true);
    try {
      const response = await client.generateSIEMRule(config);
      const newRule: SIEMRule = {
        id: Math.random().toString(36).substring(7),
        name: 'Generated Rule',
        description: 'Auto-generated SIEM rule',
        rule: response.rule,
        format: config.format || 'sigma',
        severity: 'medium',
        mitre_techniques: [],
        tags: [],
        confidence: 0.8,
        created_at: new Date().toISOString(),
        tested: false,
        false_positive_rate: 0
      };
      setRules(prev => [newRule, ...prev]);
      return newRule;
    } catch (err) {
      console.error('Failed to generate SIEM rule:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchRules();
  }, [fetchRules]);

  return { rules, loading, fetchRules, generateRule };
}
