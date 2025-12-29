import { Server as NetServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import { NextApiResponse } from 'next';

export type NextApiResponseServerIO = NextApiResponse & {
  socket: {
    server: NetServer & {
      io: SocketIOServer;
    };
  };
};

export interface BattleUpdate {
  battleId: string;
  type: 'attack' | 'defense' | 'score' | 'status';
  data: any;
}

export interface AttackEvent {
  id: string;
  battleId: string;
  type: string;
  target: string;
  severity: string;
  success: boolean;
  detected: boolean;
  timestamp: string;
}

export interface DefenseEvent {
  id: string;
  battleId: string;
  action: string;
  ruleType: string;
  effectiveness: number;
  timestamp: string;
}
