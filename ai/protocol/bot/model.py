from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, AsyncIterable, AsyncIterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ai.log import logger

from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from .chat_context import ChatContext, ChatRole
from .tool_context import FunctionTool, RawFunctionTool


class CompletionUsage(BaseModel):
    completions_tokens: int
    prompt_tokens: int
    total_tokens: int


class FunctionToolCall(BaseModel):
    type: Literal["function"] = "function"
    name: str
    arguments: str
    call_id: str


class ChoiceDelta(BaseModel):
    role: ChatRole | None = None
    content: str | None = None
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)


class ChatChunk(BaseModel):
    id: str
    delta: ChoiceDelta | None = None
    usage: CompletionUsage | None = None


class ModelError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["error"] = "error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ModelStream: ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> Model:
        return self

    async def __aexit__(
        self,
        exec_type: type[BaseException] | None,
        exec: BaseException | None,
        exec_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class ModelStream(ABC, AsyncIterator[ChatChunk]):
    def __init__(
        self,
        model: Model,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._model = model
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options
        self._current_attempt = 0
        self._generator: Optional[AsyncIterator[ChatChunk]] = None
        self._closed = False

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[FunctionTool | RawFunctionTool]:
        return self._tools

    @abstractmethod
    async def _create_stream(self) -> AsyncIterator[ChatChunk]: ...

    async def __anext__(self) -> ChatChunk:
        if self._closed:
            raise RuntimeError("Stream is closed")

        if self._generator is None:
            self._generator = await self._create_retry_stream()

        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            self._generator = None
            raise
        except asyncio.CancelledError:
            self._generator = None
            raise
        except Exception as e:
            if self._is_retryable_error(e) and self._current_attempt < self._conn_options.max_retry:
                self._current_attempt += 1
                retry_interval = self._conn_options._interval_for_retry(self._current_attempt)
                logger.warning(
                    f"Stream error (attempt {self._current_attempt}), retrying in {retry_interval}s ...",
                    exc_info=e,
                )
                await asyncio.sleep(retry_interval)

                self._generator = await self._create_retry_stream()
                return await self._generator.__anext__()

            self._generator = None
            raise

    async def _create_retry_stream(self) -> AsyncIterator[ChatChunk]:
        self._current_attempt = 0

        for attempt in range(self._conn_options.max_retry + 1):
            try:
                self._current_attempt = 0
                return self._create_stream()
            except Exception as e:
                if attempt < self._conn_options.max_retry and self._is_retryable_error(e):
                    retry_interval = self._conn_options._interval_for_retry(attempt)
                    logger.warning(
                        f"Connection error (attempt {attempt+1}), retrying in {retry_interval}s ...",
                        exc_info=e,
                    )
                    await asyncio.sleep(retry_interval)
                else:
                    raise

    def _is_retryable_error(self, error: Exception) -> bool:
        return True

    async def aclose(self) -> None:
        self._closed = True
        if self._generator is not None:
            if hasattr(self._generator, "aclose"):
                await self._generator.aclose()
            self._generator = None

    async def __aenter__(self) -> ModelStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def to_str_iterable(self) -> AsyncIterable[str]:
        async def _iterable() -> AsyncIterable[str]:
            async with self:
                async for chunk in self:
                    if chunk.delta and chunk.delta.content:
                        yield chunk.delta.content

        return _iterable()
