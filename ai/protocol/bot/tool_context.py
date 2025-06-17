from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from ai.protocol.types import ATTRIBUTE_RAW_TOOL_INFO, ATTRIBUTE_TOOL_INFO


class Function(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = Field(...)  # 必填字段
    function: Function


ToolChoice = Union[NamedToolChoice, Literal["auto", "required", "none"]]


class ToolError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._message = message

    @property
    def message(self) -> str:
        return self._message


class StopResponse(Exception):
    def __init__(self) -> None:
        super().__init__()


@dataclass
class _FunctionToolInfo:
    name: str
    description: str | None


@runtime_checkable
class FunctionTool(Protocol):
    __tool_info: _FunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class RawFunctionDescription(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    parameters: Dict[str, Any] = Field(...)


@dataclass
class _RawFunctionToolInfo:
    name: str
    raw_schema: dict[str, Any]


@runtime_checkable
class RawFunctionTool(Protocol):
    __raw_tool_info: _RawFunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
Raw_F = TypeVar("Raw_F", bound=Callable[..., Awaitable[Any]])


@overload
def function_tool(
    f: Raw_F, *, raw_schema: RawFunctionDescription | dict[str, Any]
) -> RawFunctionTool: ...


@overload
def function_tool(
    f: None = None, *, raw_schema: RawFunctionDescription | dict[str, Any]
) -> Callable[[Raw_F], RawFunctionTool]: ...


@overload
def function_tool(
    f: F, *, name: str | None = None, description: str | None = None
) -> FunctionTool: ...


@overload
def function_tool(
    f: None = None, *, name: str | None = None, description: str | None = None
) -> Callable[[F], FunctionTool]: ...


def function_tool(
    f: F | Raw_F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    raw_schema: RawFunctionDescription | dict[str, Any] | None = None,
) -> (
    FunctionTool
    | RawFunctionTool
    | Callable[[F], FunctionTool]
    | Callable[[Raw_F], RawFunctionTool]
):
    def deco_raw(func: Raw_F) -> RawFunctionTool:
        assert raw_schema is not None

        if not raw_schema.get("name"):
            raise ValueError("raw function name cannot be empty")

        if "parameters" not in raw_schema:
            raise ValueError("raw function description must contain a parameters key")

        info = _RawFunctionToolInfo(raw_schema={**raw_schema}, name=raw_schema["name"])
        setattr(func, ATTRIBUTE_RAW_TOOL_INFO, info)
        return cast(RawFunctionTool, func)

    def deco_func(func: F) -> FunctionTool:
        from docstring_parser import parse_from_object

        docstring = parse_from_object(func)
        info = _FunctionToolInfo(
            name=name or func.__name__,
            description=description or docstring.description,
        )
        setattr(func, ATTRIBUTE_TOOL_INFO, info)
        return cast(FunctionTool, func)

    if f is not None:
        return deco_raw(cast(Raw_F, f)) if raw_schema is not None else deco_func(cast(F, f))

    return deco_raw if raw_schema is not None else deco_func


def is_function_tool(f: Callable[..., Any]) -> TypeGuard[FunctionTool]:
    return hasattr(f, ATTRIBUTE_TOOL_INFO)


def get_function_info(f: FunctionTool) -> _FunctionToolInfo:
    return cast(_FunctionToolInfo, getattr(f, ATTRIBUTE_TOOL_INFO))


def is_raw_function_tool(f: Callable[..., Any]) -> TypeGuard[RawFunctionTool]:
    return hasattr(f, ATTRIBUTE_RAW_TOOL_INFO)


def get_raw_function_info(f: RawFunctionTool) -> _RawFunctionToolInfo:
    return cast(_RawFunctionToolInfo, getattr(f, ATTRIBUTE_RAW_TOOL_INFO))


def find_function_tools(cls_or_obj: Any) -> list[FunctionTool | RawFunctionTool]:
    methods: list[FunctionTool | RawFunctionTool] = []
    for _, member in inspect.getmembers(cls_or_obj):
        if is_function_tool(member) or is_raw_function_tool(member):
            methods.append(member)
    return methods


class ToolContext:
    def __init__(self, tools: list[FunctionTool | RawFunctionTool]) -> None:
        self.update_tools[tools]

    @classmethod
    def empty(cls) -> ToolContext:
        return cls([])

    @property
    def function_tools(self) -> dict[str, FunctionTool | RawFunctionTool]:
        return self._tools_map.copy()

    def update_tools(self, tools: list[FunctionTool | RawFunctionTool]) -> None:
        self._tools = tools.copy()

        for method in find_function_tools(self):
            tools.apend(method)

        self._tools_map: dict[str, FunctionTool | RawFunctionTool] = {}
        info: _FunctionToolInfo | _RawFunctionToolInfo
        for tool in tools:
            if is_raw_function_tool(tool):
                info = get_raw_function_info(tool)
            elif is_function_tool(tool):
                info = get_function_info(tool)
            else:
                ## MCP Servers
                raise ValueError(f"unknown tool type: {type(tool)}")

            if info.name in self._tools_map:
                raise ValueError(f"duplicate function name: {info.name}")

            self._tools_map[info.name] = tool

        def copy(self) -> ToolContext:
            return ToolContext(self._tools.copy())
