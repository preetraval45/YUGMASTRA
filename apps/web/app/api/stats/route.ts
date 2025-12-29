import { NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

// GET /api/stats - Get overall statistics
export async function GET() {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Get battle statistics
    const totalBattles = await prisma.battle.count({
      where: { userId: session.userId },
    });

    const activeBattles = await prisma.battle.count({
      where: {
        userId: session.userId,
        status: 'active',
      },
    });

    // Get attack statistics
    const attackStats = await prisma.attack.aggregate({
      where: {
        battle: { userId: session.userId },
      },
      _count: true,
      _avg: {
        impact: true,
      },
    });

    const successfulAttacks = await prisma.attack.count({
      where: {
        battle: { userId: session.userId },
        success: true,
      },
    });

    const detectedAttacks = await prisma.attack.count({
      where: {
        battle: { userId: session.userId },
        detected: true,
      },
    });

    // Get defense statistics
    const defenseStats = await prisma.defense.aggregate({
      where: {
        battle: { userId: session.userId },
      },
      _count: true,
      _sum: {
        attacksBlocked: true,
      },
      _avg: {
        effectiveness: true,
      },
    });

    // Get recent battles with metrics
    const recentBattles = await prisma.battle.findMany({
      where: { userId: session.userId },
      include: {
        metrics: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
      take: 10,
    });

    // Calculate average Nash equilibrium
    const nashValues = recentBattles
      .map((b) => b.nashEquilibrium)
      .filter((n): n is number => n !== null);
    const avgNashEquilibrium =
      nashValues.length > 0
        ? nashValues.reduce((a, b) => a + b, 0) / nashValues.length
        : 0;

    return NextResponse.json({
      battles: {
        total: totalBattles,
        active: activeBattles,
        completed: totalBattles - activeBattles,
      },
      attacks: {
        total: attackStats._count || 0,
        successful: successfulAttacks,
        detected: detectedAttacks,
        undetected: (attackStats._count || 0) - detectedAttacks,
        avgImpact: attackStats._avg.impact || 0,
        successRate:
          attackStats._count > 0
            ? (successfulAttacks / attackStats._count) * 100
            : 0,
        detectionRate:
          attackStats._count > 0
            ? (detectedAttacks / attackStats._count) * 100
            : 0,
      },
      defenses: {
        total: defenseStats._count || 0,
        totalBlocked: defenseStats._sum.attacksBlocked || 0,
        avgEffectiveness: defenseStats._avg.effectiveness || 0,
      },
      evolution: {
        avgNashEquilibrium,
        currentGeneration: recentBattles[0]?.coevolutionGen || 1,
      },
    });
  } catch (error) {
    console.error('Get stats error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
