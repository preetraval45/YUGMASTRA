import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try:
    await prisma.$queryRaw`SELECT 1`;
    const aiEngineUrl = process.env.AI_ENGINE_URL || 'http://ai-engine:8001';
    const response = await fetch(`${aiEngineUrl}/ready`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      throw new Error('AI Engine not ready');
    }

    return NextResponse.json({
      status: 'ready',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: 'not ready',
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
}

export const dynamic = 'force-dynamic';
