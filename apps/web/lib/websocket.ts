import { io, Socket } from 'socket.io-client';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';

type EventCallback = (data: any) => void;

interface ReconnectionConfig {
  maxAttempts: number;
  delay: number;
  backoffMultiplier: number;
  maxDelay: number;
}

class WebSocketClient {
  private socket: Socket | null = null;
  private listeners: Map<string, Set<EventCallback>> = new Map();
  private reconnectAttempts: number = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private isManualDisconnect: boolean = false;

  private reconnectionConfig: ReconnectionConfig = {
    maxAttempts: 10,
    delay: 1000,
    backoffMultiplier: 1.5,
    maxDelay: 30000,
  };

  connect() {
    if (this.socket?.connected) return;

    this.isManualDisconnect = false;

    this.socket = io(WS_URL, {
      transports: ['websocket'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'connected' });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connection', { status: 'disconnected', reason });

      // Attempt manual reconnection if not a manual disconnect
      if (!this.isManualDisconnect && reason === 'io server disconnect') {
        this.attemptReconnection();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.emit('connection', { status: 'error', error: error.message });
      this.attemptReconnection();
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log('WebSocket reconnected after', attemptNumber, 'attempts');
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'reconnected', attempts: attemptNumber });
    });

    this.socket.on('reconnect_attempt', (attemptNumber) => {
      console.log('WebSocket reconnection attempt:', attemptNumber);
      this.emit('connection', { status: 'reconnecting', attempt: attemptNumber });
    });

    this.socket.on('reconnect_failed', () => {
      console.error('WebSocket reconnection failed');
      this.emit('connection', { status: 'failed' });
    });

    // Register event handlers
    this.socket.on('update', (data: any) => {
      this.emit('update', data);
    });

    this.socket.on('attack', (data: any) => {
      this.emit('attack', data);
    });

    this.socket.on('defense', (data: any) => {
      this.emit('defense', data);
    });

    this.socket.on('metrics', (data: any) => {
      this.emit('metrics', data);
    });
  }

  private attemptReconnection() {
    if (this.isManualDisconnect) return;

    if (this.reconnectAttempts >= this.reconnectionConfig.maxAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('connection', { status: 'failed', reason: 'max_attempts' });
      return;
    }

    const delay = Math.min(
      this.reconnectionConfig.delay * Math.pow(this.reconnectionConfig.backoffMultiplier, this.reconnectAttempts),
      this.reconnectionConfig.maxDelay
    );

    this.reconnectAttempts++;
    console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.reconnectionConfig.maxAttempts} in ${delay}ms`);

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.reconnectTimer = setTimeout(() => {
      if (!this.socket?.connected && !this.isManualDisconnect) {
        this.socket?.connect();
      }
    }, delay);
  }

  disconnect() {
    this.isManualDisconnect = true;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.reconnectAttempts = 0;
  }

  subscribe(event: string, callback: EventCallback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    return () => this.unsubscribe(event, callback);
  }

  unsubscribe(event: string, callback: EventCallback) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(callback);
    }
  }

  private emit(event: string, data: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((callback) => callback(data));
    }
  }

  send(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    }
  }
}

export const wsClient = new WebSocketClient();
