import asyncio
import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette import EventSourceResponse

from ai.protocol.bot.bot import ChatChunk
from ai.protocol.bot.chat_context import ChatContext
from ai.service.openai.bot import Bot

router = APIRouter()


@router.post(
    "/stream",
    response_class=EventSourceResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "example": "data: {'content': 'Hello'}\n\ndata: {'content': 'World'}\n\n"
                }
            },
            "description": "Server-Sent Events stream",
        }
    },
)
async def stream():
    bot = Bot.with_deepseek()
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Tell me a joke")

    async def event_generator():
        async with bot.chat(chat_ctx=chat_ctx) as stream:
            try:
                async for chunk in stream:
                    yield {
                        "event": "message",
                        "data": json.dumps(chunk.model_dump()),
                    }
            except asyncio.CancelledError:
                pass
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
                }
            finally:
                if stream is not None and not stream._closed:
                    await stream.aclose()

    return EventSourceResponse(event_generator())
