from __future__ import annotations

import os

from openai.types.chat import ChatCompletionToolParam

from ai.protocol import bot


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("MEGANOTE_BASE_URL", "https://api.meganote.ai/v1")

    return base_url


def to_fnc_ctx(
    fnc_ctx: list[bot.FunctionTool | bot.RawFunctionTool],
) -> list[ChatCompletionToolParam]:
    tools: list[ChatCompletionToolParam] = []
    for fnc in fnc_ctx:
        if bot.is_raw_function_tool(fnc):
            info = bot.get_raw_function_info(fnc)
            tools.append(
                {
                    "type": "function",
                    "function": info.raw_schema,  # type: ignore
                }
            )
        elif bot.is_function_tool(fnc):
            tools.append(bot.utils.build_strict_openai_schema(fnc))

    return tools
