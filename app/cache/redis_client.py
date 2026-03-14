import uuid
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import get_settings

settings = get_settings()

_redis: Redis | None = None


async def init_redis(url: str) -> None:
    """Initialize Redis connection pool."""
    global _redis
    # from_url handles password extraction automatically when format is redis://[:password@]host[:port][/db]
    _redis = redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )


async def get_redis() -> Redis:
    """Get Redis client."""
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis
    if _redis:
        await _redis.close()
        _redis = None


async def cache_set(key: str, value: str, ttl: int) -> None:
    """Set a value with TTL."""
    redis_client = await get_redis()
    await redis_client.setex(key, ttl, value)


async def cache_get(key: str) -> str | None:
    """Get a value."""
    redis_client = await get_redis()
    return await redis_client.get(key)


async def cache_delete(key: str) -> None:
    """Delete a key."""
    redis_client = await get_redis()
    await redis_client.delete(key)


async def check_rate_limit(user_id: uuid.UUID, limit: int) -> None:
    """Sliding window rate limit. Raises HTTPException(429) if exceeded."""
    redis_client = await get_redis()
    key = f"rate:{user_id}"

    now = await redis_client.time()
    current_time = int(now[0])

    window_start = current_time - 60

    await redis_client.zremrangebyscore(key, 0, window_start)

    request_count = await redis_client.zcard(key)

    if request_count >= limit:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    await redis_client.zadd(key, {str(current_time): current_time})
    await redis_client.expire(key, 60)