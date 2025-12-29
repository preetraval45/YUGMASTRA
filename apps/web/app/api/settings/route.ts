import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

// GET /api/settings - Get user settings
export async function GET() {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const settings = await prisma.settings.findUnique({
      where: { userId: session.userId },
    });

    if (!settings) {
      // Create default settings if they don't exist
      const newSettings = await prisma.settings.create({
        data: {
          userId: session.userId,
        },
      });
      return NextResponse.json({ settings: newSettings });
    }

    return NextResponse.json({ settings });
  } catch (error) {
    console.error('Get settings error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// PATCH /api/settings - Update user settings
export async function PATCH(request: NextRequest) {
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
      notifyAttacks,
      notifyDefenses,
      notifySystem,
      emailNotifications,
      learningRate,
      episodes,
      batchSize,
      theme,
    } = body;

    const settings = await prisma.settings.upsert({
      where: { userId: session.userId },
      create: {
        userId: session.userId,
        ...(notifyAttacks !== undefined && { notifyAttacks }),
        ...(notifyDefenses !== undefined && { notifyDefenses }),
        ...(notifySystem !== undefined && { notifySystem }),
        ...(emailNotifications !== undefined && { emailNotifications }),
        ...(learningRate !== undefined && { learningRate }),
        ...(episodes !== undefined && { episodes }),
        ...(batchSize !== undefined && { batchSize }),
        ...(theme !== undefined && { theme }),
      },
      update: {
        ...(notifyAttacks !== undefined && { notifyAttacks }),
        ...(notifyDefenses !== undefined && { notifyDefenses }),
        ...(notifySystem !== undefined && { notifySystem }),
        ...(emailNotifications !== undefined && { emailNotifications }),
        ...(learningRate !== undefined && { learningRate }),
        ...(episodes !== undefined && { episodes }),
        ...(batchSize !== undefined && { batchSize }),
        ...(theme !== undefined && { theme }),
      },
    });

    return NextResponse.json({
      message: 'Settings updated successfully',
      settings,
    });
  } catch (error) {
    console.error('Update settings error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
