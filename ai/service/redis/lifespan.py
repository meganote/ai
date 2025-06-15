from fastapi import FastAPI
from redis.asyncio import ConnectionPool

from ai.settings import settings


def init_reids(app: FastAPI) -> None:
    app.state.redis_pool = ConnectionPool.from_url(str(settings.redis_url))


async def shutdown_redis(app: FastAPI) -> None:
    await app.state.redis_pool.disconnect()
