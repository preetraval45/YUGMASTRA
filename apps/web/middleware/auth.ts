import { NextRequest, NextResponse } from 'next/server';
import { verify } from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

export interface AuthenticatedRequest extends NextRequest {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

export async function authenticateRequest(
  request: NextRequest
): Promise<{ authenticated: boolean; user?: any; error?: string }> {
  try {
    const authHeader = request.headers.get('authorization');

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return {
        authenticated: false,
        error: 'No authorization token provided',
      };
    }

    const token = authHeader.substring(7);

    try {
      const decoded = verify(token, JWT_SECRET) as {
        userId: string;
        email: string;
        role: string;
      };

      return {
        authenticated: true,
        user: {
          id: decoded.userId,
          email: decoded.email,
          role: decoded.role,
        },
      };
    } catch (verifyError) {
      return {
        authenticated: false,
        error: 'Invalid or expired token',
      };
    }
  } catch (error) {
    return {
      authenticated: false,
      error: 'Authentication failed',
    };
  }
}

export function withAuth(
  handler: (request: AuthenticatedRequest) => Promise<NextResponse>,
  options?: {
    requireAdmin?: boolean;
    requireRole?: string[];
  }
) {
  return async (request: NextRequest) => {
    const authResult = await authenticateRequest(request);

    if (!authResult.authenticated) {
      return NextResponse.json(
        { error: authResult.error || 'Unauthorized' },
        { status: 401 }
      );
    }

    if (options?.requireAdmin && authResult.user?.role !== 'admin') {
      return NextResponse.json(
        { error: 'Admin access required' },
        { status: 403 }
      );
    }

    if (
      options?.requireRole &&
      !options.requireRole.includes(authResult.user?.role || '')
    ) {
      return NextResponse.json(
        { error: 'Insufficient permissions' },
        { status: 403 }
      );
    }

    const authenticatedRequest = request as AuthenticatedRequest;
    authenticatedRequest.user = authResult.user;

    return handler(authenticatedRequest);
  };
}

export async function getUserFromRequest(
  request: NextRequest
): Promise<{ id: string; email: string; role: string } | null> {
  const authResult = await authenticateRequest(request);

  if (authResult.authenticated && authResult.user) {
    return authResult.user;
  }

  return null;
}
