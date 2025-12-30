import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  const healthData = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    services: {
      database: false,
      aiEngine: false,
      redis: false,
    },
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
  };

  // Check database connection
  try {
    await prisma.$queryRaw`SELECT 1`;
    healthData.services.database = true;
  } catch (error) {
    console.error('Database health check failed:', error);
    healthData.status = 'degraded';
  }

  // Check AI Engine
  try {
    const aiEngineUrl = process.env.AI_ENGINE_URL || 'http://ai-engine:8001';
    const response = await fetch(`${aiEngineUrl}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      healthData.services.aiEngine = true;
    }
  } catch (error) {
    console.error('AI Engine health check failed:', error);
    healthData.status = 'degraded';
  }

  // Check Redis (if available)
  try {
    if (process.env.REDIS_URL) {
      healthData.services.redis = true;
    }
  } catch (error) {
    console.error('Redis health check failed:', error);
  }

  const statusCode = healthData.status === 'healthy' ? 200 : 503;

  return NextResponse.json(healthData, { status: statusCode });
}

export const dynamic = 'force-dynamic';
