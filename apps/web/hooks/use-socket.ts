import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { BattleUpdate, AttackEvent, DefenseEvent } from '@/lib/socket';

export function useSocket() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize socket connection
    const socketInstance = io({
      path: '/api/socket/io',
      addTrailingSlash: false,
    });

    socketInstance.on('connect', () => {
      console.log('Socket connected');
      setIsConnected(true);
    });

    socketInstance.on('disconnect', () => {
      console.log('Socket disconnected');
      setIsConnected(false);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, []);

  return { socket, isConnected };
}

export function useBattleSocket(battleId: string | null) {
  const { socket, isConnected } = useSocket();
  const [attacks, setAttacks] = useState<AttackEvent[]>([]);
  const [defenses, setDefenses] = useState<DefenseEvent[]>([]);
  const [battleData, setBattleData] = useState<any>(null);

  useEffect(() => {
    if (!socket || !battleId) return;

    // Join battle room
    socket.emit('join-battle', battleId);

    // Listen for battle updates
    socket.on('battle-update', (update: BattleUpdate) => {
      if (update.battleId === battleId) {
        switch (update.type) {
          case 'attack':
            setAttacks((prev) => [update.data, ...prev].slice(0, 50));
            break;
          case 'defense':
            setDefenses((prev) => [update.data, ...prev].slice(0, 50));
            break;
          case 'score':
          case 'status':
            setBattleData((prev: any) => ({ ...prev, ...update.data }));
            break;
        }
      }
    });

    socket.on('attack-event', (attack: AttackEvent) => {
      setAttacks((prev) => [attack, ...prev].slice(0, 50));
    });

    socket.on('defense-event', (defense: DefenseEvent) => {
      setDefenses((prev) => [defense, ...prev].slice(0, 50));
    });

    return () => {
      socket.emit('leave-battle', battleId);
      socket.off('battle-update');
      socket.off('attack-event');
      socket.off('defense-event');
    };
  }, [socket, battleId]);

  const emitAttack = (attackData: Partial<AttackEvent>) => {
    if (socket && battleId) {
      socket.emit('attack-event', { ...attackData, battleId });
    }
  };

  const emitDefense = (defenseData: Partial<DefenseEvent>) => {
    if (socket && battleId) {
      socket.emit('defense-event', { ...defenseData, battleId });
    }
  };

  return {
    isConnected,
    attacks,
    defenses,
    battleData,
    emitAttack,
    emitDefense,
  };
}
