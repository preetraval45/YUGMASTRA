import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

// GET /api/battles/:id - Get a specific battle
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const battle = await prisma.battle.findUnique({
      where: { id: params.id },
      include: {
        metrics: true,
        attacks: {
          orderBy: { timestamp: 'desc' },
          take: 50,
        },
        defenses: {
          orderBy: { timestamp: 'desc' },
          take: 50,
        },
      },
    });

    if (!battle) {
      return NextResponse.json(
        { error: 'Battle not found' },
        { status: 404 }
      );
    }

    if (battle.userId !== session.userId) {
      return NextResponse.json(
        { error: 'Forbidden' },
        { status: 403 }
      );
    }

    return NextResponse.json({ battle });
  } catch (error) {
    console.error('Get battle error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// PATCH /api/battles/:id - Update a battle
export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const battle = await prisma.battle.findUnique({
      where: { id: params.id },
    });

    if (!battle) {
      return NextResponse.json(
        { error: 'Battle not found' },
        { status: 404 }
      );
    }

    if (battle.userId !== session.userId) {
      return NextResponse.json(
        { error: 'Forbidden' },
        { status: 403 }
      );
    }

    const body = await request.json();
    const { status, redScore, blueScore, duration, nashEquilibrium } = body;

    const updatedBattle = await prisma.battle.update({
      where: { id: params.id },
      data: {
        ...(status && { status }),
        ...(redScore !== undefined && { redScore }),
        ...(blueScore !== undefined && { blueScore }),
        ...(duration !== undefined && { duration }),
        ...(nashEquilibrium !== undefined && { nashEquilibrium }),
      },
      include: {
        metrics: true,
      },
    });

    return NextResponse.json({
      message: 'Battle updated successfully',
      battle: updatedBattle,
    });
  } catch (error) {
    console.error('Update battle error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// DELETE /api/battles/:id - Delete a battle
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const battle = await prisma.battle.findUnique({
      where: { id: params.id },
    });

    if (!battle) {
      return NextResponse.json(
        { error: 'Battle not found' },
        { status: 404 }
      );
    }

    if (battle.userId !== session.userId) {
      return NextResponse.json(
        { error: 'Forbidden' },
        { status: 403 }
      );
    }

    await prisma.battle.delete({
      where: { id: params.id },
    });

    return NextResponse.json({
      message: 'Battle deleted successfully',
    });
  } catch (error) {
    console.error('Delete battle error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
