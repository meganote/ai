from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, cast

import httpx
import openai
from loguru import logger
from openai import AsyncClient, AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_chunk import Choice

from ai.protocol import bot
from ai.protocol.bot.chat_context import ChatContext
from ai.protocol.bot.tool_context import FunctionTool, RawFunctionTool, ToolChoice
from ai.protocol.exceptions import APIConnectionError, APIStatusError, APITimeoutError
from ai.protocol.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ai.utils import is_given

from .utils import to_fnc_ctx


@dataclass
class _BotOptions:
    model: str
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    store: NotGivenOr[bool]
    metadata: NotGivenOr[dict[str, str]]
    max_completion_tokens: NotGivenOr[int]


class Bot(bot.Bot):
    def __init__(
        self,
        *,
        model: str,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        _provider_fmt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Create OpenAI instance"""
        super().__init__()
        self._opts = _BotOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            max_completion_tokens=max_completion_tokens,
        )
        self._provider_fmt = _provider_fmt or "openai"
        self._client = client or AsyncClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=(
                    timeout
                    if timeout
                    else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
                ),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=500,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )

    @staticmethod
    def with_deepseek(
        *,
        model: str = "deepseek/deepseek-chat-v3-0324:free",  # deepseek-chat"
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        client: AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> Bot:
        """Create Meganote instance"""
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required")

        return Bot(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[Any]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> BotStream:
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.metadata):
            extra["metadata"] = self._opts.metadata

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        parallel_tool_calls = (
            parallel_tool_calls
            if is_given(parallel_tool_calls)
            else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice  # type: ignore
        if is_given(tool_choice):
            oai_tool_choice: ChatCompletionToolChoiceOptionParam
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice
                extra["tool_choice"] = oai_tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)  # type: ignore

        return BotStream(
            self,
            model=self._opts.model,
            provider_fmt=self._provider_fmt,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class BotStream(bot.BotStream):
    def __init__(
        self,
        bot: Bot,
        *,
        model: str,
        provider_fmt: str,
        client: AsyncClient,
        chat_ctx: bot.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[dict, Any],
    ) -> None:
        super().__init__(bot, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._bot = bot
        self._model = model
        self._provider_fmt = provider_fmt
        self._client = client
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        self._oai_stream: AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._func_raw_arguments: str | None = None
        self._tool_index: int | None = None
        retryable = True

        try:
            chat_ctx, _ = self._chat_ctx.to_provider_format(format=self._provider_fmt)
            fnc_ctx = to_fnc_ctx(self._tools) if self._tools else openai.NOT_GIVEN

            logger.debug(
                "chat.completions.create",
                extra={"fnc_ctx": fnc_ctx, "tool_choice": "auto", "chat_ctx": chat_ctx},
            )

            self._oai_stream = stream = await self._client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], chat_ctx),
                tools=fnc_ctx,
                model=self._model,
                stream_options={"include_usage": True},
                stream=True,
                timeout=httpx.Timeout(self._conn_options.timeout),
                **self._extra_kwargs,
            )

            thinking = asyncio.Event()
            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        chat_chunk = self._parse_choice(chunk.id, choice, thinking)
                        if chat_chunk is not None:
                            retryable = False
                            self._event_ch.send_nowait(chat_chunk)

                    if chunk.usage is not None:
                        retryable = False
                        chunk = bot.ChatChunk(
                            id=chunk.id,
                            usage=bot.CompletionUsage(
                                completions_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                        )
                        self._event_ch.send_nowait(chunk)

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e
        finally:
            self._event_ch.close()

    def _parse_choice(
        self, id: str, choice: Choice, thinking: asyncio.Event
    ) -> bot.ChatChunk | None:
        delta = choice.delta

        if delta is None:
            return None

        if delta.tool_calls:
            for tool in delta.tool_calls:
                if not tool.function:
                    continue

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = bot.ChatChunk(
                        id=id,
                        delta=bot.ChoiceDelta(
                            role="assistant",
                            content=delta.content,
                            tool_calls=[
                                bot.FunctionCall(
                                    arguments=self._func_raw_arguments or "",
                                    name=self._fnc_name or "",
                                    call_id=self._tool_call_id or "",
                                )
                            ],
                        ),
                    )
                    self._tool_call_id = self._fnc_name = self._func_raw_arguments = (
                        None
                    )

                if tool.function.name:
                    self._tool_index = tool.index
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._func_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._func_raw_arguments += tool.function.arguments

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason in ("tool_calls", "stop") and self._tool_call_id:
            call_chunk = bot.ChatChunk(
                id=id,
                delta=bot.ChoiceDelta(
                    role="assistant",
                    content=delta.content,
                    tool_calls=[
                        bot.FunctionToolCall(
                            arguments=self._func_raw_arguments or "",
                            name=self._fnc_name or "",
                            call_id=self._tool_call_id or "",
                        )
                    ],
                ),
            )
            self._tool_call_id = self._fnc_name = self._func_raw_arguments = None
            return call_chunk

        delta.content = bot.utils.strip_thinking_tokens(delta.content, thinking)

        if not delta.content:
            return None

        return bot.ChatChunk(
            id=id,
            delta=bot.ChoiceDelta(content=delta.content, role="assistant"),
        )
