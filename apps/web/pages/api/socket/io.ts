import { Server as NetServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import { NextApiRequest } from 'next';
import { NextApiResponseServerIO } from '@/lib/socket';

export const config = {
  api: {
    bodyParser: false,
  },
};

const ioHandler = (req: NextApiRequest, res: NextApiResponseServerIO) => {
  if (!res.socket.server.io) {
    console.log('Initializing Socket.IO server...');

    const httpServer: NetServer = res.socket.server as any;
    const io = new SocketIOServer(httpServer, {
      path: '/api/socket/io',
      addTrailingSlash: false,
      cors: {
        origin: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
        methods: ['GET', 'POST'],
      },
    });

    // Socket.IO connection handler
    io.on('connection', (socket) => {
      console.log('Client connected:', socket.id);

      // Join a battle room
      socket.on('join-battle', (battleId: string) => {
        socket.join(`battle-${battleId}`);
        console.log(`Socket ${socket.id} joined battle-${battleId}`);
      });

      // Leave a battle room
      socket.on('leave-battle', (battleId: string) => {
        socket.leave(`battle-${battleId}`);
        console.log(`Socket ${socket.id} left battle-${battleId}`);
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
      });
    });

    res.socket.server.io = io;
  } else {
    console.log('Socket.IO server already running');
  }

  res.end();
};

export default ioHandler;
