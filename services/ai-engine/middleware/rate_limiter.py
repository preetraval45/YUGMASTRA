import time
import redis
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_prefix: str = "rate_limit"
    ):
        self.redis_client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    def _get_key(self, identifier: str) -> str:
        return f"{self.key_prefix}:{identifier}"

    def is_allowed(self, identifier: str) -> tuple[bool, dict]:
        key = self._get_key(identifier)
        current_time = time.time()
        window_start = current_time - self.window_seconds

        try:
            # Remove old entries outside the window
            self.redis_client.zremrangebyscore(key, 0, window_start)

            # Count requests in current window
            request_count = self.redis_client.zcard(key)

            if request_count < self.max_requests:
                # Add current request
                self.redis_client.zadd(key, {str(current_time): current_time})
                self.redis_client.expire(key, self.window_seconds)

                return True, {
                    "remaining": self.max_requests - request_count - 1,
                    "reset": int(current_time + self.window_seconds),
                    "limit": self.max_requests
                }
            else:
                # Rate limit exceeded
                oldest_request = self.redis_client.zrange(key, 0, 0, withscores=True)
                reset_time = int(oldest_request[0][1] + self.window_seconds) if oldest_request else int(current_time + self.window_seconds)

                return False, {
                    "remaining": 0,
                    "reset": reset_time,
                    "limit": self.max_requests
                }

        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {str(e)}")
            # Fail open - allow request if Redis is down
            return True, {
                "remaining": -1,
                "reset": -1,
                "limit": self.max_requests
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_func: Callable = None
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(redis_client, max_requests, window_seconds)
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, request: Request) -> str:
        # Use IP address as default identifier
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/"]:
            return await call_next(request)

        # Get identifier
        identifier = self.key_func(request)

        # Check rate limit
        is_allowed, metadata = self.rate_limiter.is_allowed(identifier)

        # Add rate limit headers
        response_headers = {
            "X-RateLimit-Limit": str(metadata["limit"]),
            "X-RateLimit-Remaining": str(metadata["remaining"]),
            "X-RateLimit-Reset": str(metadata["reset"])
        }

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again after {metadata['reset'] - int(time.time())} seconds",
                    "retry_after": metadata["reset"]
                },
                headers=response_headers
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in response_headers.items():
            response.headers[key] = value

        return response


class TokenBucketRateLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        rate: int = 10,
        capacity: int = 100,
        key_prefix: str = "token_bucket"
    ):
        self.redis_client = redis_client
        self.rate = rate  # Tokens added per second
        self.capacity = capacity  # Maximum tokens
        self.key_prefix = key_prefix

    def _get_keys(self, identifier: str) -> tuple[str, str]:
        return (
            f"{self.key_prefix}:tokens:{identifier}",
            f"{self.key_prefix}:timestamp:{identifier}"
        )

    def consume(self, identifier: str, tokens: int = 1) -> bool:
        token_key, timestamp_key = self._get_keys(identifier)
        current_time = time.time()

        try:
            # Get current state
            current_tokens = self.redis_client.get(token_key)
            last_update = self.redis_client.get(timestamp_key)

            if current_tokens is None:
                current_tokens = self.capacity
            else:
                current_tokens = float(current_tokens)

            if last_update is None:
                last_update = current_time
            else:
                last_update = float(last_update)

            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - last_update
            tokens_to_add = time_elapsed * self.rate
            current_tokens = min(self.capacity, current_tokens + tokens_to_add)

            # Check if enough tokens
            if current_tokens >= tokens:
                current_tokens -= tokens

                # Update Redis
                pipe = self.redis_client.pipeline()
                pipe.set(token_key, str(current_tokens), ex=3600)
                pipe.set(timestamp_key, str(current_time), ex=3600)
                pipe.execute()

                return True
            else:
                return False

        except redis.RedisError as e:
            logger.error(f"Redis error in token bucket: {str(e)}")
            # Fail open
            return True
