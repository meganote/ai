from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from ai.protocol import bot
from ai.protocol.bot.chat_context import ChatContext, FunctionCall, _ReadOnlyChatContext
from ai.protocol.bot.inference import _ModelGenerationData
from ai.protocol.bot.tool_context import ToolContext, find_function_tools
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
        self._tools = tools.copy() + find_function_tools(self)
        self._tool_ctx = ToolContext(self._tools)
        self._chat_ctx = chat_ctx.copy(tools=self._tools) if chat_ctx else ChatContext.empty()

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def tools(self) -> list[bot.FunctionTool | bot.RawFunctionTool]:
        return self._tools.copy()

    @property
    def chat_ctx(self) -> bot.ChatContext:
        return _ReadOnlyChatContext(self._chat_ctx.items)

    async def run(self):

        data = _ModelGenerationData()
        async with self._model.chat(chat_ctx=self._chat_ctx, tools=self._tools) as stream:
            try:
                async for chunk in stream:
                    if isinstance(chunk, str):
                        data.generated_text += chunk

                    elif isinstance(chunk, bot.ChatChunk):
                        if not chunk.delta:
                            continue

                        if chunk.delta.tool_calls:
                            for tool in chunk.delta.tool_calls:
                                if tool.tpye != "function":
                                    continue

                                fnc_call = FunctionCall(
                                    id=f"{data.id}/func_{len(data.generated_functions)}",
                                    call_id=tool.call_id,
                                    name=tool.name,
                                    arguments=tool.arguments,
                                )
                                data.generated_functions.append(fnc_call)

                        if chunk.delta.content:
                            data.generated_text += chunk.delta.content

                    else:
                        logger.warning(f"Model returned an unexpected type: {type(chunk)}")
            finally:
                if isinstance(stream, _ACloseable):
                    await stream.aclose()

        logger.debug(f"data: {data}")
