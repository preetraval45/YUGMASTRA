'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface SocketContextType {
  socket: Socket | null;
  isConnected: boolean;
}

const SocketContext = createContext<SocketContextType>({
  socket: null,
  isConnected: false,
});

export function useSocketContext() {
  return useContext(SocketContext);
}

export function SocketProvider({ children }: { children: React.ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Only attempt WebSocket connection in development or when explicitly enabled
    const shouldConnect = process.env.NODE_ENV === 'development' ||
                         process.env.NEXT_PUBLIC_ENABLE_WEBSOCKET === 'true';

    if (!shouldConnect) {
      console.log('[WebSocket] WebSocket disabled for this environment');
      return;
    }

    try {
      const socketInstance = io({
        path: '/api/socket/io',
        addTrailingSlash: false,
        reconnection: true,
        reconnectionAttempts: 3,
        reconnectionDelay: 2000,
        timeout: 5000,
      });

      socketInstance.on('connect', () => {
        console.log('[WebSocket] Connected');
        setIsConnected(true);
      });

      socketInstance.on('disconnect', () => {
        console.log('[WebSocket] Disconnected');
        setIsConnected(false);
      });

      socketInstance.on('connect_error', (error) => {
        console.warn('[WebSocket] Connection failed, WebSocket features disabled:', error.message);
        setIsConnected(false);
      });

      socketInstance.on('error', (error) => {
        console.warn('[WebSocket] Error, WebSocket features disabled:', error);
        setIsConnected(false);
      });

      setSocket(socketInstance);

      return () => {
        socketInstance.disconnect();
      };
    } catch (error) {
      console.warn('[WebSocket] Failed to initialize WebSocket, features disabled:', error);
      setIsConnected(false);
    }
  }, []);

  return (
    <SocketContext.Provider value={{ socket, isConnected }}>
      {children}
    </SocketContext.Provider>
  );
}
