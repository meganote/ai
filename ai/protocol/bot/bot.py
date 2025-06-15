from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, AsyncIterable, AsyncIterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai.log import logger
from ai.utils import aio

from ..exceptions import APIConnectionError, APIError
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


class BotError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["error"] = "error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


class Bot(ABC):
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
    ) -> BotStream: ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> Bot:
        return self

    async def __aexit__(
        self,
        exec_type: type[BaseException] | None,
        exec: BaseException | None,
        exec_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class BotStream(ABC):
    def __init__(
        self,
        bot: Bot,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._bot = bot
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options

        self._event_ch = aio.Chan[ChatChunk]()
        self._event_aiter, monitor_aiter = aio.iter_tools.tee(self._event_ch, 2)
        self._current_attempt_has_error = False
        # self._metrics_task = asyncio.create_task(
        #     self._metrics_monitor_task(monitor_aiter), name="MODEL._metrics_task"
        # )
        self._task = asyncio.create_task(self._main_task())

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            try:
                logger.debug("Starting _run attempt %d", i)
                return await self._run()
            except APIError as e:
                retry_interval = self._conn_options._interval_for_retry(i)

                if self._conn_options.max_retry == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to generate model completion after {self._conn_options.max_retry + 1} attempts",
                    ) from e

                else:
                    self._emit_error(e, retryable=True)
                    logger.warning(
                        f"failed to generate model completion, retring in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "model": self._model._label,
                            "attempt": i + 1,
                        },
                    )

                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)

                self._current_attempt_has_error = False

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._current_attempt_has_error = True
        self._bot.emit(
            "error",
            BotError(
                timestamp=time.time(),
                label=self._bot._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[FunctionTool | RawFunctionTool]:
        return self._tools

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)

    async def __anext__(self) -> ChatChunk:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __aenter__(self) -> BotStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def to_str_iterale(self) -> AsyncIterable[str]:
        async def _iterable() -> AsyncIterable[str]:
            async with self:
                async for chunk in self:
                    if chunk.delta and chunk.delta.content:
                        yield chunk.delta.content

        return _iterable()
