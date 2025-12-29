import { NextRequest, NextResponse } from 'next/server';
import { getSession, removeAuthCookie } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

export async function POST(request: NextRequest) {
  try {
    const session = await getSession();

    if (session) {
      // Create activity log
      await prisma.activityLog.create({
        data: {
          userId: session.userId,
          action: 'logout',
          details: `User ${session.email} logged out`,
        },
      });
    }

    // Remove auth cookie
    await removeAuthCookie();

    return NextResponse.json({
      message: 'Logout successful',
    });
  } catch (error) {
    console.error('Logout error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
