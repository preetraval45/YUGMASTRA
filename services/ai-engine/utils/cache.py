import json
import hashlib
import functools
from typing import Any, Optional, Callable
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client: Optional[aioredis.Redis] = None

    async def connect(self):
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Connected to Redis cache")

    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

    def _make_key(self, key: str, namespace: str = "") -> str:
        if namespace:
            return f"{namespace}:{key}"
        return key

    async def get(self, key: str, namespace: str = "") -> Optional[Any]:
        await self.connect()
        cache_key = self._make_key(key, namespace)

        try:
            value = await self.redis_client.get(cache_key)
            if value:
                logger.debug(f"Cache hit: {cache_key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache miss: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = ""
    ):
        await self.connect()
        cache_key = self._make_key(key, namespace)
        ttl = ttl or self.default_ttl

        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"Cache set: {cache_key} (ttl={ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

    async def delete(self, key: str, namespace: str = ""):
        await self.connect()
        cache_key = self._make_key(key, namespace)

        try:
            await self.redis_client.delete(cache_key)
            logger.debug(f"Cache deleted: {cache_key}")
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")

    async def exists(self, key: str, namespace: str = "") -> bool:
        await self.connect()
        cache_key = self._make_key(key, namespace)

        try:
            return await self.redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {str(e)}")
            return False

    async def clear_namespace(self, namespace: str):
        await self.connect()

        try:
            pattern = f"{namespace}:*"
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys from namespace {namespace}")
        except Exception as e:
            logger.error(f"Cache clear namespace error: {str(e)}")


def cache_key_from_args(*args, **kwargs) -> str:
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    ttl: int = 3600,
    namespace: str = "",
    key_func: Optional[Callable] = None
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance (assumes first arg is self with cache attribute)
            cache = None
            if args and hasattr(args[0], 'cache'):
                cache = args[0].cache

            if not cache:
                # No cache available, call function directly
                return await func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache_key_from_args(*args[1:], **kwargs)}"

            # Try to get from cache
            cached_value = await cache.get(cache_key, namespace)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await cache.set(cache_key, result, ttl, namespace)

            return result

        return wrapper
    return decorator
