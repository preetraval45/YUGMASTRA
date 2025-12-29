import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

// GET /api/battles - Get all battles for the current user
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
    const status = searchParams.get('status');
    const limit = parseInt(searchParams.get('limit') || '10');
    const offset = parseInt(searchParams.get('offset') || '0');

    const where: any = { userId: session.userId };
    if (status) {
      where.status = status;
    }

    const battles = await prisma.battle.findMany({
      where,
      include: {
        metrics: true,
        _count: {
          select: {
            attacks: true,
            defenses: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
      take: limit,
      skip: offset,
    });

    const total = await prisma.battle.count({ where });

    return NextResponse.json({
      battles,
      pagination: {
        total,
        limit,
        offset,
      },
    });
  } catch (error) {
    console.error('Get battles error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/battles - Create a new battle
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
    const { coevolutionGen } = body;

    const battle = await prisma.battle.create({
      data: {
        userId: session.userId,
        coevolutionGen: coevolutionGen || 1,
        status: 'active',
      },
      include: {
        metrics: true,
      },
    });

    // Create initial battle metrics
    await prisma.battleMetrics.create({
      data: {
        battleId: battle.id,
      },
    });

    return NextResponse.json(
      { message: 'Battle created successfully', battle },
      { status: 201 }
    );
  } catch (error) {
    console.error('Create battle error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
