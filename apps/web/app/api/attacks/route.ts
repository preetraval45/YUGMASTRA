import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

// GET /api/attacks - Get all attacks
export async function GET(request: NextRequest) {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(request.url);
    const battleId = searchParams.get('battleId');
    const severity = searchParams.get('severity');
    const detected = searchParams.get('detected');
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');

    const where: any = {
      battle: { userId: session.userId },
    };

    if (battleId) {
      where.battleId = battleId;
    }

    if (severity) {
      where.severity = severity;
    }

    if (detected !== null && detected !== undefined) {
      where.detected = detected === 'true';
    }

    const attacks = await prisma.attack.findMany({
      where,
      include: {
        battle: {
          select: {
            id: true,
            status: true,
          },
        },
      },
      orderBy: {
        timestamp: 'desc',
      },
      take: limit,
      skip: offset,
    });

    const total = await prisma.attack.count({ where });

    // Calculate statistics
    const stats = await prisma.attack.aggregate({
      where: {
        battle: { userId: session.userId },
      },
      _count: true,
      _avg: {
        impact: true,
      },
    });

    const successfulCount = await prisma.attack.count({
      where: {
        battle: { userId: session.userId },
        success: true,
      },
    });

    const detectedCount = await prisma.attack.count({
      where: {
        battle: { userId: session.userId },
        detected: true,
      },
    });

    return NextResponse.json({
      attacks,
      pagination: {
        total,
        limit,
        offset,
      },
      stats: {
        total: stats._count,
        successful: successfulCount,
        detected: detectedCount,
        avgImpact: stats._avg.impact || 0,
      },
    });
  } catch (error) {
    console.error('Get attacks error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/attacks - Create a new attack
export async function POST(request: NextRequest) {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const body = await request.json();
    const {
      battleId,
      type,
      technique,
      target,
      severity,
      success,
      detected,
      impact,
    } = body;

    // Verify battle ownership
    const battle = await prisma.battle.findUnique({
      where: { id: battleId },
    });

    if (!battle || battle.userId !== session.userId) {
      return NextResponse.json(
        { error: 'Battle not found or access denied' },
        { status: 403 }
      );
    }

    const attack = await prisma.attack.create({
      data: {
        battleId,
        type,
        technique,
        target,
        severity,
        success: success || false,
        detected: detected || false,
        impact: impact || 0,
      },
    });

    // Update battle metrics
    await prisma.battleMetrics.upsert({
      where: { battleId },
      create: {
        battleId,
        totalAttacks: 1,
        successfulAttacks: success ? 1 : 0,
        detectedAttacks: detected ? 1 : 0,
      },
      update: {
        totalAttacks: { increment: 1 },
        successfulAttacks: { increment: success ? 1 : 0 },
        detectedAttacks: { increment: detected ? 1 : 0 },
      },
    });

    return NextResponse.json(
      { message: 'Attack recorded successfully', attack },
      { status: 201 }
    );
  } catch (error) {
    console.error('Create attack error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
