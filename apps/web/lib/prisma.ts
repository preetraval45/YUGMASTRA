import { PrismaClient } from '@prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';
import pg from 'pg';

const { Pool } = pg;

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
  pool: pg.Pool | undefined;
};

let prismaClient: PrismaClient;

function getPrismaClient() {
  if (!prismaClient) {
    // Create connection pool only when needed
    const pool = globalForPrisma.pool ?? new Pool({
      connectionString: process.env.DATABASE_URL,
    });

    // Create Prisma adapter
    const adapter = new PrismaPg(pool);

    // Create Prisma client
    prismaClient = globalForPrisma.prisma ?? new PrismaClient({
      adapter,
      log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
    });

    if (process.env.NODE_ENV !== 'production') {
      globalForPrisma.prisma = prismaClient;
      globalForPrisma.pool = pool;
    }
  }

  return prismaClient;
}

// Export prisma as a getter to avoid initialization during module load
export const prisma = (() => getPrismaClient())();
