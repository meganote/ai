from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ai.service.redis.lifespan import init_reids, shutdown_redis


@asynccontextmanager
async def lifespan_setup(app: FastAPI) -> AsyncGenerator[None, None]:
    # init
    # init_reids(app)

    yield

    # destroy
    # await shutdown_redis(app)
