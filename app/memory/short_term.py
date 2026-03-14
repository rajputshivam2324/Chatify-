import json

from app.cache.redis_client import cache_get, cache_delete, cache_set
from app.core.config import get_settings

settings = get_settings()


async def get_messages(session_id: str) -> list[dict]:
    """Get messages for a session from Redis."""
    key = f"session:{session_id}:messages"
    data = await cache_get(key)
    if data:
        return json.loads(data)
    return []


async def append_messages(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append messages to session and refresh TTL."""
    key = f"session:{session_id}:messages"
    messages = await get_messages(session_id)

    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "content": assistant_msg})

    await cache_set(key, json.dumps(messages), settings.session_ttl_seconds)


async def clear_session(session_id: str) -> None:
    """Clear all messages for a session."""
    key = f"session:{session_id}:messages"
    await cache_delete(key)