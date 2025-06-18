import asyncio
import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette import EventSourceResponse

from ai.protocol.bot.chat_context import ChatContext
from ai.protocol.bot.inference import _ToolOutput, execute_tools_task
from ai.protocol.bot.model import ChatChunk
from ai.protocol.bot.tool_context import FunctionTool, RawFunctionTool, ToolContext
from ai.service.openai.model import Model
from ai.service.tools import get_weather

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
    bot = Model.with_deepseek()
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="北京今天天气怎么样？")

    tools: list[FunctionTool | RawFunctionTool] = [get_weather]
    tool_output = _ToolOutput(output=[], first_tool_fut=asyncio.Future())

    async def event_generator():
        async with bot.chat(chat_ctx=chat_ctx, tools=[get_weather]) as stream:
            try:
                async for chunk in stream:
                    yield {
                        "event": "message",
                        "data": json.dumps(chunk.model_dump(), ensure_ascii=False),
                    }
                    task = asyncio.create_task(
                        execute_tools_task(
                            tool_ctx=ToolContext(tools),
                            tool_choice="auto",
                            function_stream=chunk,
                            tool_output=tool_output,
                        ),
                        name="execute_tools_task",
                    )
                    # chat_ctx.insert(chunk.delta.tool_calls)
                    # logger.debug(f"Chat Context updated: {chat_ctx.model_dump()}")
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

            await task
            if len(tool_output.output) > 0:
                for py_out in tool_output.output:
                    print(py_out)

    return EventSourceResponse(event_generator())
