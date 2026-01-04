/**
 * Advanced WebSocket Client for Real-Time Updates
 * Supports auto-reconnection, heartbeat, message queuing
 */

export type WebSocketMessage = {
  type: string;
  payload: any;
  timestamp: number;
};

export type WebSocketConfig = {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  debug?: boolean;
};

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export class RealtimeWebSocket {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private reconnectAttempts = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private statusListeners: Set<(status: ConnectionStatus) => void> = new Set();
  private currentStatus: ConnectionStatus = 'disconnected';

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectInterval: 3000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      debug: false,
      ...config,
    };
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.log('Already connected');
      return;
    }

    this.updateStatus('connecting');
    this.log(`Connecting to ${this.config.url}...`);

    try {
      this.ws = new WebSocket(this.config.url);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
    } catch (error) {
      this.log('Connection error:', error);
      this.updateStatus('error');
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.log('Disconnecting...');
    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.updateStatus('disconnected');
  }

  /**
   * Send message to server
   */
  send(type: string, payload: any): void {
    const message: WebSocketMessage = {
      type,
      payload,
      timestamp: Date.now(),
    };

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      this.log('Sent:', message);
    } else {
      // Queue message for later
      this.messageQueue.push(message);
      this.log('Message queued (not connected):', message);
    }
  }

  /**
   * Subscribe to specific message type
   */
  on(type: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.listeners.get(type)?.delete(callback);
    };
  }

  /**
   * Subscribe to connection status changes
   */
  onStatus(callback: (status: ConnectionStatus) => void): () => void {
    this.statusListeners.add(callback);
    // Immediately call with current status
    callback(this.currentStatus);

    return () => {
      this.statusListeners.delete(callback);
    };
  }

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return this.currentStatus;
  }

  private handleOpen(): void {
    this.log('Connected successfully');
    this.updateStatus('connected');
    this.reconnectAttempts = 0;
    this.startHeartbeat();
    this.flushMessageQueue();
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.log('Received:', message);

      // Handle heartbeat response
      if (message.type === 'pong') {
        return;
      }

      // Notify listeners
      const callbacks = this.listeners.get(message.type);
      if (callbacks) {
        callbacks.forEach(callback => {
          try {
            callback(message.payload);
          } catch (error) {
            this.log('Listener error:', error);
          }
        });
      }

      // Notify wildcard listeners
      const wildcardCallbacks = this.listeners.get('*');
      if (wildcardCallbacks) {
        wildcardCallbacks.forEach(callback => callback(message));
      }
    } catch (error) {
      this.log('Failed to parse message:', error);
    }
  }

  private handleError(event: Event): void {
    this.log('WebSocket error:', event);
    this.updateStatus('error');
  }

  private handleClose(event: CloseEvent): void {
    this.log(`Connection closed (code: ${event.code}, reason: ${event.reason})`);
    this.stopHeartbeat();
    this.updateStatus('disconnected');

    // Attempt reconnection
    if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    } else {
      this.log('Max reconnection attempts reached');
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.config.reconnectInterval * Math.min(this.reconnectAttempts, 5);

    this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.send('ping', { timestamp: Date.now() });
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private flushMessageQueue(): void {
    if (this.messageQueue.length > 0) {
      this.log(`Flushing ${this.messageQueue.length} queued messages`);

      while (this.messageQueue.length > 0) {
        const message = this.messageQueue.shift()!;
        this.send(message.type, message.payload);
      }
    }
  }

  private updateStatus(status: ConnectionStatus): void {
    if (this.currentStatus !== status) {
      this.currentStatus = status;
      this.statusListeners.forEach(callback => callback(status));
    }
  }

  private log(...args: any[]): void {
    if (this.config.debug) {
      console.log('[WebSocket]', ...args);
    }
  }
}


/**
 * React Hook for WebSocket connection
 */
import { useEffect, useState, useCallback, useRef } from 'react';

export function useWebSocket(url: string, options?: Partial<WebSocketConfig>) {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<RealtimeWebSocket | null>(null);

  // Initialize WebSocket
  useEffect(() => {
    const ws = new RealtimeWebSocket({
      url,
      debug: true,
      ...options,
    });

    wsRef.current = ws;

    // Subscribe to status changes
    const unsubscribeStatus = ws.onStatus(setStatus);

    // Subscribe to all messages
    const unsubscribeMessages = ws.on('*', (message) => {
      setLastMessage(message);
    });

    // Connect
    ws.connect();

    // Cleanup
    return () => {
      unsubscribeStatus();
      unsubscribeMessages();
      ws.disconnect();
    };
  }, [url]);

  // Send message function
  const sendMessage = useCallback((type: string, payload: any) => {
    wsRef.current?.send(type, payload);
  }, []);

  // Subscribe to specific message type
  const subscribe = useCallback((type: string, callback: (data: any) => void) => {
    return wsRef.current?.on(type, callback) || (() => {});
  }, []);

  return {
    status,
    lastMessage,
    sendMessage,
    subscribe,
  };
}


/**
 * Battle Arena WebSocket Client
 */
export class BattleArenaWebSocket extends RealtimeWebSocket {
  constructor(battleId: string) {
    super({
      url: `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080'}/battle/${battleId}`,
      debug: process.env.NODE_ENV === 'development',
    });
  }

  /**
   * Subscribe to attack events
   */
  onAttack(callback: (attack: any) => void): () => void {
    return this.on('attack', callback);
  }

  /**
   * Subscribe to defense events
   */
  onDefense(callback: (defense: any) => void): () => void {
    return this.on('defense', callback);
  }

  /**
   * Subscribe to score updates
   */
  onScoreUpdate(callback: (scores: any) => void): () => void {
    return this.on('score_update', callback);
  }

  /**
   * Subscribe to system health changes
   */
  onHealthUpdate(callback: (health: number) => void): () => void {
    return this.on('health_update', callback);
  }

  /**
   * Join battle as spectator
   */
  joinSpectator(username: string): void {
    this.send('join_spectator', { username });
  }

  /**
   * Send manual defense command
   */
  executeDefense(command: string, target?: string): void {
    this.send('manual_defense', { command, target });
  }
}
