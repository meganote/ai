from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from ai.protocol import bot
from ai.protocol.bot.chat_context import (
    ChatContext,
    FunctionCall,
    FunctionCallOutput,
    _ReadOnlyChatContext,
)
from ai.protocol.bot.inference import (
    _ModelGenerationData,
    _ToolOutput,
    execute_tools_task,
)
from ai.protocol.bot.tool_context import (
    FunctionTool,
    RawFunctionTool,
    ToolContext,
    find_function_tools,
)
from ai.protocol.types import NOT_GIVEN, NotGivenOr
from ai.utils.misc import shortuuid


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self) -> Any: ...


class Agent:
    def __init__(
        self,
        *,
        model: bot.Model,
        instructions: str,
        chat_ctx: NotGivenOr[bot.ChatContext | None] = NOT_GIVEN,
        tools: list[bot.FunctionTool | bot.RawFunctionTool] | None = None,
        # mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        self._model = model
        self._instructions = instructions
        # self._tools = tools.copy() + find_function_tools(self)
        self._tools = tools.copy()
        self._chat_ctx = (
            chat_ctx.copy(tools=self._tools) if chat_ctx else ChatContext.empty()
        )
        self._tool_ctx = ToolContext(self._tools)

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def tools(self) -> list[bot.FunctionTool | bot.RawFunctionTool]:
        return self._tools.copy()

    @property
    def chat_ctx(self) -> bot.ChatContext:
        return _ReadOnlyChatContext(self._chat_ctx.items)

    async def update_instructions(self, instructions: str) -> None:
        return

    async def update_tools(self, tools: list[FunctionTool | RawFunctionTool]) -> None:
        self._tools = list(set(tools))
        self._chat_ctx = self._chat_ctx.copy(tools=self._tools)
        return

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        self._chat_ctx = chat_ctx.copy(tools=self._tools)
        return

    async def on_enter(self) -> None:
        """Called when the task is entered"""
        pass

    async def on_exit(self) -> None:
        """Called when the task is exited"""
        pass

    async def run(self):
        max_rounds = 2
        round = 0
        tool_output = _ToolOutput(output=[], first_tool_fut=asyncio.Future())

        while round < max_rounds:
            round += 1

            logger.debug(f"round #{round}: {self._chat_ctx.to_dict()}")
            data = _ModelGenerationData()
            async with self._model.chat(
                chat_ctx=self._chat_ctx, tools=self._tools
            ) as stream:
                try:
                    async for chunk in stream:
                        if isinstance(chunk, str):
                            data.generated_text += chunk
                            yield self._create_event("message", chunk.model_dump())

                        elif isinstance(chunk, bot.ChatChunk):
                            if not chunk.delta:
                                continue

                            if chunk.delta.tool_calls:
                                for tool in chunk.delta.tool_calls:
                                    if tool.type != "function":
                                        continue

                                    fnc_call = FunctionCall(
                                        id=f"{data.id}/func_{len(data.generated_functions)}",
                                        call_id=tool.call_id,
                                        name=tool.name,
                                        arguments=tool.arguments,
                                    )
                                    data.generated_functions.append(fnc_call)
                                    yield self._create_event(
                                        "tool_call", fnc_call.model_dump()
                                    )

                            if chunk.delta.content:
                                data.generated_text += chunk.delta.content

                        else:
                            logger.warning(
                                f"Model returned an unexpected type: {type(chunk)}"
                            )

                    task = asyncio.create_task(
                        execute_tools_task(
                            tool_ctx=ToolContext(self.tools),
                            tool_choice="auto",
                            function_stream=data.generated_functions,
                            tool_output=tool_output,
                        ),
                        name="execute_tools_task",
                    )
                    await task

                    if len(tool_output.output) > 0:
                        new_calls: list[FunctionCall] = []
                        new_fnc_outputs: list[FunctionCallOutput] = []
                        generate_tool_reply: bool = False
                        for py_out in tool_output.output:
                            sanitized_out = py_out.sanitize()

                            if sanitized_out.fnc_call is not None:
                                new_calls.append(sanitized_out.fnc_call)
                                new_fnc_outputs.append(sanitized_out.fnc_call_out)

                                yield self._create_event(
                                    "tool_result",
                                    {
                                        "name": sanitized_out.fnc_call.name,
                                        "result": sanitized_out.fnc_call_out.output,
                                    },
                                )

                                if sanitized_out.reply_required:
                                    generate_tool_reply = True

                        tool_messages = new_calls + new_fnc_outputs
                        if generate_tool_reply:
                            self._chat_ctx.insert(tool_messages)

                        if generate_tool_reply:
                            continue

                    break

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    pass
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": str(e)}),
                    }
                finally:
                    if isinstance(stream, _ACloseable):
                        await stream.aclose()

            logger.debug(f"data: {data}")

    def _create_event(self, event: str, data: Any) -> dict:
        return {"event": event, "data": json.dumps(data, ensure_ascii=False)}
