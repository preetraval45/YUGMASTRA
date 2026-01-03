import * as Sentry from '@sentry/nextjs';

export function initializeMonitoring() {
  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    Sentry.init({
      dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
      environment: process.env.NODE_ENV,

      // Performance monitoring
      tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

      // Profiling
      profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

      // Error filtering
      beforeSend(event, hint) {
        // Don't send errors in development
        if (process.env.NODE_ENV === 'development') {
          console.error('Sentry Error:', hint.originalException || hint.syntheticException);
          return null;
        }

        // Filter out known non-critical errors
        const error = hint.originalException;
        if (error && typeof error === 'object' && 'message' in error) {
          const message = String(error.message);

          // Skip network errors that are user-related
          if (message.includes('NetworkError') || message.includes('fetch failed')) {
            return null;
          }

          // Skip authentication errors (handled by UI)
          if (message.includes('Unauthorized') || message.includes('Invalid credentials')) {
            return null;
          }
        }

        return event;
      },

      // Additional context
      integrations: [
        Sentry.browserTracingIntegration(),
        Sentry.replayIntegration({
          maskAllText: true,
          blockAllMedia: true,
        }),
      ],

      // Session replay for production errors
      replaysSessionSampleRate: 0.0,
      replaysOnErrorSampleRate: process.env.NODE_ENV === 'production' ? 1.0 : 0.0,
    });
  }
}

// Custom error tracking
export function trackError(error: Error, context?: Record<string, any>) {
  console.error('Error:', error, context);

  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    Sentry.captureException(error, {
      extra: context,
    });
  }
}

// Custom event tracking
export function trackEvent(name: string, data?: Record<string, any>) {
  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    Sentry.captureMessage(name, {
      level: 'info',
      extra: data,
    });
  }
}

// Performance tracking
export function startTransaction(name: string, op: string) {
  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    return Sentry.startSpan({
      name,
      op,
    }, () => {
      // Span implementation
    });
  }
  return null;
}

// User context
export function setUser(user: { id: string; email?: string; username?: string }) {
  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    Sentry.setUser({
      id: user.id,
      email: user.email,
      username: user.username,
    });
  }
}

// Clear user context on logout
export function clearUser() {
  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    Sentry.setUser(null);
  }
}
