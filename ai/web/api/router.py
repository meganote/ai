from fastapi.routing import APIRouter

from ai.web.api import bot, health

api_router = APIRouter()
api_router.include_router(router=health.router, prefix="/health", tags=["health"])
api_router.include_router(router=bot.router, prefix="/bot", tags=["bot"])
