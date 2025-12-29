import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

// GET /api/defenses - Get all defenses
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
    const ruleType = searchParams.get('ruleType');
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');

    const where: any = {
      battle: { userId: session.userId },
    };

    if (battleId) {
      where.battleId = battleId;
    }

    if (ruleType) {
      where.ruleType = ruleType;
    }

    const defenses = await prisma.defense.findMany({
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

    const total = await prisma.defense.count({ where });

    // Calculate statistics
    const stats = await prisma.defense.aggregate({
      where: {
        battle: { userId: session.userId },
      },
      _count: true,
      _sum: {
        attacksBlocked: true,
        falsePositives: true,
      },
      _avg: {
        effectiveness: true,
      },
    });

    return NextResponse.json({
      defenses,
      pagination: {
        total,
        limit,
        offset,
      },
      stats: {
        total: stats._count,
        totalBlocked: stats._sum.attacksBlocked || 0,
        totalFalsePositives: stats._sum.falsePositives || 0,
        avgEffectiveness: stats._avg.effectiveness || 0,
      },
    });
  } catch (error) {
    console.error('Get defenses error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/defenses - Create a new defense
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
      action,
      ruleType,
      effectiveness,
      attacksBlocked,
      falsePositives,
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

    const defense = await prisma.defense.create({
      data: {
        battleId,
        action,
        ruleType,
        effectiveness: effectiveness || 0,
        attacksBlocked: attacksBlocked || 0,
        falsePositives: falsePositives || 0,
      },
    });

    // Update battle metrics
    const isEffective = effectiveness > 0.7;
    await prisma.battleMetrics.upsert({
      where: { battleId },
      create: {
        battleId,
        totalDefenses: 1,
        effectiveDefenses: isEffective ? 1 : 0,
      },
      update: {
        totalDefenses: { increment: 1 },
        effectiveDefenses: { increment: isEffective ? 1 : 0 },
      },
    });

    return NextResponse.json(
      { message: 'Defense recorded successfully', defense },
      { status: 201 }
    );
  } catch (error) {
    console.error('Create defense error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
