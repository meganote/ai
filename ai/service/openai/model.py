from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, cast

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
from ai.protocol.bot import (
    ChatContext,
    FunctionTool,
    RawFunctionTool,
    ToolChoice,
    is_given,
)
from ai.protocol.exceptions import APIConnectionError, APIStatusError, APITimeoutError
from ai.protocol.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .utils import to_fnc_ctx


@dataclass
class _ModelOptions:
    model: str
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    store: NotGivenOr[bool]
    metadata: NotGivenOr[dict[str, str]]
    max_completion_tokens: NotGivenOr[int]


class Model(bot.Model):
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
        self._opts = _ModelOptions(
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

    @property
    def opts(self) -> str:
        """Get model name"""
        return self._opts

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
    ) -> Model:
        """Create Meganote instance"""
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required")

        return Model(
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
    ) -> ModelStream:
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
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
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
            extra["response_format"] = bot.to_openai_response_format(response_format)  # type: ignore

        return ModelStream(
            self,
            provider_fmt=self._provider_fmt,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class ModelStream(bot.ModelStream):
    def __init__(
        self,
        model: Model,
        *,
        provider_fmt: str,
        client: AsyncClient,
        chat_ctx: bot.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[dict, Any],
    ) -> None:
        super().__init__(model, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._provider_fmt = provider_fmt
        self._client = client
        self._extra_kwargs = extra_kwargs

    async def _create_stream(self) -> AsyncIterator[bot.ChatChunk]:
        tool_call_id = None
        fnc_name = None
        func_raw_arguments = None
        tool_index = None

        chat_ctx, _ = self._chat_ctx.to_provider_format(format=self._provider_fmt)
        fnc_ctx = to_fnc_ctx(self._tools) if self._tools else openai.NOT_GIVEN

        oai_stream = await self._client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], chat_ctx),
            tools=fnc_ctx,
            model=self._model.opts.model,
            stream_options={"include_usage": True},
            stream=True,
            timeout=httpx.Timeout(self._conn_options.timeout),
            **self._extra_kwargs,
        )

        try:
            async with oai_stream:
                async for chunk in oai_stream:
                    chunk: ChatCompletionChunk = cast(ChatCompletionChunk, chunk)
                    if chunk.usage:
                        yield bot.ChatChunk(
                            id=chunk.id,
                            usage=bot.CompletionUsage(
                                completions_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                        )

                    for choice in chunk.choices:
                        choice: Choice = cast(Choice, choice)
                        if not choice.delta:
                            continue

                        delta = choice.delta

                        logger.debug(
                            f"Processing OpenAI chat completion chunk: {chunk.id}, choice: {choice.index}, delta: {delta}"
                        )

                        if delta.tool_calls:
                            for tool in delta.tool_calls:
                                if not tool.function:
                                    continue

                                if tool.id and tool.index != tool_index:
                                    if tool_call_id:
                                        yield self._create_tool_chunk(
                                            chunk.id,
                                            tool_call_id,
                                            fnc_name,
                                            func_raw_arguments,
                                        )
                                        tool_call_id = fnc_name = func_raw_arguments = None

                                    tool_index = tool.index
                                    tool_call_id = tool.id
                                    fnc_name = tool.function.name
                                    func_raw_arguments = tool.function.arguments or ""

                                elif tool.function.arguments:
                                    func_raw_arguments += tool.function.arguments

                        if choice.finish_reason in ("tool_calls", "stop") and tool_call_id:
                            yield self._create_tool_chunk(
                                chunk.id, tool_call_id, fnc_name, func_raw_arguments
                            )
                            tool_call_id = fnc_name = func_raw_arguments = None

                        if delta.content:
                            cleaned_content = delta.content
                            if cleaned_content:
                                yield bot.ChatChunk(
                                    id=chunk.id,
                                    delta=bot.ChoiceDelta(
                                        content=cleaned_content, role="assistant"
                                    ),
                                )

        except Exception as e:
            raise self._convert_error(e) from e

    def _create_tool_chunk(
        self, chunk_id: str, call_id: str, name: str | None, arguments: str | None
    ) -> bot.ChatChunk:
        return bot.ChatChunk(
            id=chunk_id,
            delta=bot.ChoiceDelta(
                role="assistant",
                tool_calls=[
                    bot.FunctionToolCall(
                        name=name or "", arguments=arguments or "", call_id=call_id
                    )
                ],
            ),
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, asyncio.CancelledError):
            return False
        if isinstance(error, openai.APITimeoutError):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code in {429, 500, 502, 503, 504}
        if isinstance(error, APIConnectionError):
            return True
        return False

    def _convert_error(self, error: Exception) -> Exception:
        if isinstance(error, openai.APITimeoutError):
            return APITimeoutError()
        if isinstance(error, openai.APIStatusError):
            return APIStatusError(error.message, status_code=error.status_code)
        return APIConnectionError(f"Connection error: {str(error)}")
