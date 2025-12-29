import { NextRequest, NextResponse } from 'next/server';

interface RateLimitConfig {
  interval: number; // in milliseconds
  uniqueTokenPerInterval: number;
}

const rateLimitMap = new Map<string, { count: number; resetTime: number }>();

export function rateLimit(config: RateLimitConfig) {
  const { interval, uniqueTokenPerInterval } = config;

  return {
    check: (req: NextRequest, limit: number, token: string) => {
      const tokenKey = `${token}`;
      const now = Date.now();

      const tokenData = rateLimitMap.get(tokenKey);

      if (!tokenData || now > tokenData.resetTime) {
        rateLimitMap.set(tokenKey, {
          count: 1,
          resetTime: now + interval,
        });
        return { success: true };
      }

      if (tokenData.count >= limit) {
        return {
          success: false,
          retryAfter: Math.ceil((tokenData.resetTime - now) / 1000),
        };
      }

      tokenData.count += 1;
      rateLimitMap.set(tokenKey, tokenData);

      return { success: true };
    },
  };
}

// Clean up old entries every hour
setInterval(() => {
  const now = Date.now();
  for (const [key, value] of rateLimitMap.entries()) {
    if (now > value.resetTime) {
      rateLimitMap.delete(key);
    }
  }
}, 60 * 60 * 1000);

// Default rate limiter: 10 requests per 10 seconds
export const limiter = rateLimit({
  interval: 10 * 1000,
  uniqueTokenPerInterval: 500,
});

// Strict rate limiter for sensitive endpoints: 5 requests per minute
export const strictLimiter = rateLimit({
  interval: 60 * 1000,
  uniqueTokenPerInterval: 500,
});
