"""
Error monitoring and logging for AI Engine
Integrates with Sentry for production error tracking
"""

import os
import logging
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Sentry initialization
sentry_sdk = None
SENTRY_ENABLED = False

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration

    SENTRY_DSN = os.getenv('SENTRY_DSN')

    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=os.getenv('ENVIRONMENT', 'development'),
            traces_sample_rate=0.1 if os.getenv('ENVIRONMENT') == 'production' else 1.0,
            profiles_sample_rate=0.1 if os.getenv('ENVIRONMENT') == 'production' else 1.0,
            integrations=[
                FastApiIntegration(),
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR
                ),
            ],
            before_send=filter_errors,
        )
        SENTRY_ENABLED = True
        logger.info("Sentry monitoring initialized")
    else:
        logger.info("Sentry DSN not configured, monitoring disabled")

except ImportError:
    logger.warning("Sentry SDK not installed, monitoring disabled")


def filter_errors(event, hint):
    """Filter out non-critical errors"""
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']

        # Skip common expected errors
        if isinstance(exc_value, (ConnectionError, TimeoutError)):
            return None

        # Skip validation errors (handled by API)
        if 'ValidationError' in str(exc_type):
            return None

    return event


def track_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Track error with Sentry and logging"""
    logger.error(f"Error: {error}", exc_info=True, extra=context or {})

    if SENTRY_ENABLED and sentry_sdk:
        sentry_sdk.capture_exception(error, extras=context or {})


def track_event(name: str, data: Optional[Dict[str, Any]] = None):
    """Track custom event"""
    logger.info(f"Event: {name}", extra=data or {})

    if SENTRY_ENABLED and sentry_sdk:
        sentry_sdk.capture_message(name, level='info', extras=data or {})


def monitor_endpoint(func: Callable) -> Callable:
    """Decorator to monitor endpoint execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()

        try:
            result = await func(*args, **kwargs)

            # Log successful execution
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.2f}s")

            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            # Track error
            track_error(e, {
                'endpoint': func.__name__,
                'duration': duration,
                'args': str(args),
                'kwargs': str(kwargs),
            })

            # Re-raise for FastAPI to handle
            raise

    return wrapper


def set_user_context(user_id: str, email: Optional[str] = None):
    """Set user context for error tracking"""
    if SENTRY_ENABLED and sentry_sdk:
        sentry_sdk.set_user({
            'id': user_id,
            'email': email,
        })


def clear_user_context():
    """Clear user context"""
    if SENTRY_ENABLED and sentry_sdk:
        sentry_sdk.set_user(None)


def add_breadcrumb(message: str, category: str = 'default', level: str = 'info', data: Optional[Dict] = None):
    """Add breadcrumb for error context"""
    if SENTRY_ENABLED and sentry_sdk:
        sentry_sdk.add_breadcrumb({
            'message': message,
            'category': category,
            'level': level,
            'data': data or {},
        })


class PerformanceMonitor:
    """Context manager for performance monitoring"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.transaction = None

    def __enter__(self):
        self.start_time = datetime.now()

        if SENTRY_ENABLED and sentry_sdk:
            self.transaction = sentry_sdk.start_transaction(
                op='ai.operation',
                name=self.operation_name
            )
            self.transaction.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type:
            logger.error(f"{self.operation_name} failed after {duration:.2f}s: {exc_val}")
        else:
            logger.info(f"{self.operation_name} completed in {duration:.2f}s")

        if self.transaction:
            self.transaction.__exit__(exc_type, exc_val, exc_tb)


# Export monitoring functions
__all__ = [
    'track_error',
    'track_event',
    'monitor_endpoint',
    'set_user_context',
    'clear_user_context',
    'add_breadcrumb',
    'PerformanceMonitor',
    'SENTRY_ENABLED',
]
