from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import Any, AsyncIterable

from loguru import logger
from pydantic import ValidationError

from ai.utils.misc import shortuuid

from ..types import NotGivenOr
from .chat_context import FunctionCall, FunctionCallOutput
from .tool_context import (
    StopResponse,
    ToolChoice,
    ToolContext,
    ToolError,
    is_function_tool,
    is_raw_function_tool,
)
from .utils import RunContext, cancel_and_wait, prepare_function_arguments


@dataclass
class _ModelGenerationData:
    generated_text: str = ""
    generated_functions: list[FunctionCall] = field(default_factory=list)
    id: str = field(default_factory=lambda: shortuuid("item_"))


@dataclass
class _ToolOutput:
    output: list[_PythonOutput]
    first_tool_fut: asyncio.Future[None]


async def execute_tools_task(
    *,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[ToolChoice],
    function_stream: AsyncIterable[FunctionCall],
    tool_output: _ToolOutput,
) -> None:

    tasks: list[asyncio.Task[Any]] = []
    try:
        async for fnc_call in function_stream:
            if tool_choice == "none":
                logger.error(
                    "A tool call with tool_choice set to 'none', ignoring",
                    extra={
                        "function": fnc_call.name,
                    },
                )
                continue

            # TODO: other tool_choice values

            if (function_tool := tool_ctx.function_tools.get(fnc_call.name)) is None:
                logger.warning(
                    f"unknown function: {fnc_call.name}",
                    extra={
                        "function": fnc_call.name,
                    },
                )
                continue

            if not is_function_tool(function_tool) and not is_raw_function_tool(function_tool):
                logger.error(
                    f"unknown tool type: {type(function_tool)}",
                    extra={
                        "function": fnc_call.name,
                    },
                )
                continue

            py_out = _PythonOutput(fnc_call=fnc_call, output=None, exception=None)
            try:
                json_args = fnc_call.arguments or "{}"
                fnc_args, fnc_kwargs = prepare_function_arguments(
                    fnc=function_tool,
                    json_arguments=json_args,
                    call_ctx=RunContext(function_call=fnc_call),
                )
            except (ValidationError, ValueError) as e:
                logger.exception(
                    f"tried to call function: {fnc_call.name} with invalid arguments",
                    extra={
                        "function": fnc_call.name,
                        "arguments": fnc_call.arguments,
                    },
                )
                py_out.exception = e
                tool_output.output.append(py_out)
                continue

            if not tool_output.first_tool_fut.done():
                tool_output.first_tool_fut.set_result(None)

            logger.debug(
                "executing tool",
                extra={
                    "function": fnc_call.name,
                    "arguments": fnc_call.arguments,
                },
            )

            try:
                task = asyncio.create_task(
                    function_tool(*fnc_args, **fnc_kwargs),
                    name=f"function_tool_{fnc_call.name}",
                )

                tasks.apend(task)
            except Exception as e:
                logger.exception(
                    "exception occurred while executing tool",
                    extra={
                        "function": fnc_call.name,
                    },
                )
                py_out.exception = e
                tool_output.output.append(py_out)
                continue

            def _log_exceptions(
                task: asyncio.Task[Any],
                *,
                py_out: _PythonOutput,
                fnc_call: FunctionCall,
            ) -> None:
                if task.exception() is not None:
                    logger.error(
                        "exception occurred while executing tool",
                        extra={
                            "function": fnc_call.name,
                        },
                        exc_info=task.exception(),
                    )
                    py_out.exception = task.exception()
                    tool_output.output.append(py_out)
                    return

                py_out.output = task.result()
                tool_output.output.append(py_out)
                tasks.remove(task)

            task.add_done_callback(partial(_log_exceptions, py_out=py_out, fnc_call=fnc_call))

        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    except asyncio.CancelledError:
        if len(tasks) > 0:
            names = [task.get_name() for task in tasks]
            logger.debug(
                "waiting for function call to finish before fully cancelling",
                extra={
                    "functions": names,
                },
            )
            await asyncio.gather(*tasks)
    finally:
        await cancel_and_wait(*tasks)

        if len(tool_output.output) > 0:
            logger.debug(
                "tools execution completed",
                extra={},
            )


@dataclass
class _SanitizedOutput:
    fnc_call: FunctionCall
    fnc_call_output: FunctionCallOutput | None
    reply_required: bool = field(default=True)


@dataclass
class _PythonOutput:
    fnc_call: FunctionCall
    output: Any
    exception: BaseException | None

    def sanitize(self) -> _SanitizedOutput:

        if isinstance(self.exception, ToolError):
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output=self.exception.message,
                    is_error=True,
                ),
            )

        if isinstance(self.exception, StopResponse):
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=None,
                # agent_task=None,
            )

        if self.exception is not None:
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output="An internal error occurred",  # Don't send the actual error message, as it may contain sensitive information  # noqa: E501
                    is_error=True,
                ),
                agent_task=None,
            )

        fnc_output: Any = self.output
        if (
            isinstance(self.output, list)
            or isinstance(self.output, frozenset)
            or isinstance(self.output, tuple)
        ):
            pass
