import asyncio
import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette import EventSourceResponse

from ai.protocol.bot.chat_context import ChatContext, FunctionCall, FunctionCallOutput
from ai.protocol.bot.inference import (
    _ModelGenerationData,
    _ToolOutput,
    execute_tools_task,
)
from ai.protocol.bot.model import ChatChunk
from ai.protocol.bot.tool_context import FunctionTool, RawFunctionTool, ToolContext
from ai.service.openai.agent import Agent
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
    chat_ctx.add_message(role="user", content="你好！")

    async def event_generator():
        async with bot.chat(chat_ctx=chat_ctx) as stream:
            try:
                async for chunk in stream:
                    yield {
                        "event": "message",
                        "data": json.dumps(chunk.model_dump(), ensure_ascii=False),
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


@router.post(
    "/weather",
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
async def weather():
    model = Model.with_deepseek()
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="北京今天天气怎么样？")

    tools: list[FunctionTool | RawFunctionTool] = [get_weather]
    agent = Agent(
        model=model,
        instructions="You're a helpful assistant",
        chat_ctx=chat_ctx,
        tools=tools,
    )

    return EventSourceResponse(agent.stream())

    # tool_output = _ToolOutput(output=[], first_tool_fut=asyncio.Future())

    # data = _ModelGenerationData()

    # async def event_generator():
    #     async with bot.chat(chat_ctx=chat_ctx, tools=[get_weather]) as stream:
    #         try:
    #             async for chunk in stream:
    #                 if isinstance(chunk, str):
    #                     data.generated_text += chunk
    #                     yield {
    #                         "event": "message",
    #                         "data": json.dumps(chunk.model_dump(), ensure_ascii=False),
    #                     }

    #                 elif isinstance(chunk, ChatChunk):
    #                     if not chunk.delta:
    #                         continue

    #                     if chunk.delta.tool_calls:
    #                         for tool in chunk.delta.tool_calls:
    #                             if tool.type != "function":
    #                                 continue

    #                             fnc_call = FunctionCall(
    #                                 id=f"{data.id}/func_{len(data.generated_functions)}",
    #                                 call_id=tool.call_id,
    #                                 name=tool.name,
    #                                 arguments=tool.arguments,
    #                             )
    #                             data.generated_functions.append(fnc_call)
    #                             # yield {
    #                             #     "event": "message",
    #                             #     "data": json.dumps(
    #                             #         data.generated_functions, ensure_ascii=False
    #                             #     ),
    #                             # }

    #                     if chunk.delta.content:
    #                         data.generated_text += chunk.delta.content

    #                 else:
    #                     logger.warning(
    #                         f"Model returned an unexpected type: {type(chunk)}"
    #                     )

    #             task = asyncio.create_task(
    #                 execute_tools_task(
    #                     tool_ctx=ToolContext(tools),
    #                     tool_choice="auto",
    #                     function_stream=data.generated_functions,
    #                     tool_output=tool_output,
    #                 ),
    #                 name="execute_tools_task",
    #             )

    #         except asyncio.CancelledError:
    #             pass
    #         except Exception as e:
    #             yield {
    #                 "event": "error",
    #                 "data": json.dumps({"error": str(e)}),
    #             }
    #         finally:
    #             if stream is not None and not stream._closed:
    #                 await stream.aclose()

    #         await task
    #         if len(tool_output.output) > 0:
    #             new_calls: list[FunctionCall] = []
    #             new_fnc_outputs: list[FunctionCallOutput] = []
    #             generate_tool_reply: bool = False
    #             for py_out in tool_output.output:
    #                 sanitized_out = py_out.sanitize()
    #                 print(sanitized_out)

    #                 if sanitized_out.fnc_call is not None:
    #                     new_calls.append(sanitized_out.fnc_call)
    #                     new_fnc_outputs.append(sanitized_out.fnc_call_out)
    #                     if sanitized_out.reply_required:
    #                         generate_tool_reply = True

    #                 tool_messages = new_calls + new_fnc_outputs
    #                 if generate_tool_reply:
    #                     chat_ctx.items.extend(tool_messages)
    #                     logger.debug(chat_ctx.to_dict())
    #                 elif len(new_fnc_outputs) > 0:
    #                     print(tool_messages)
    #
    # return EventSourceResponse(event_generator())
