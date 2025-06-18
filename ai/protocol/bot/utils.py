from __future__ import annotations

import asyncio
import functools
import inspect
import sys
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.fields import Field, FieldInfo
from pydantic_core import PydanticUndefined, from_json
from typing_extensions import TypeGuard

from ..types import NotGiven, NotGivenOr
from . import _strict
from .chat_context import FunctionCall
from .tool_context import (
    FunctionTool,
    RawFunctionTool,
    get_function_info,
    is_function_tool,
    is_raw_function_tool,
)

THINK_TAG_START = "<think>"
THINK_TAG_END = "</think>"


_T = TypeVar("_T")
Userdata_T = TypeVar("Userdata_T")


class RunContext(Generic[Userdata_T]):
    # private ctor
    def __init__(
        self,
        *,
        function_call: FunctionCall,
    ) -> None:
        self._function_call = function_call

    @property
    def function_call(self) -> FunctionCall:
        return self._function_call


def is_given(obj: NotGivenOr[_T]) -> TypeGuard[_T]:
    return not isinstance(obj, NotGiven)


# def is_context_type(ty: type) -> bool:
#     from ..voice.events import RunContext

#     origin = get_origin(ty)
#     is_call_context = ty is RunContext or origin is RunContext

#     return is_call_context


def is_typed_dict(cls: type | Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, dict) and hasattr(cls, "__annotations__")


def build_strict_openai_schema(
    function_tool: FunctionTool,
) -> dict[str, Any]:
    """strict mode tool description"""
    model = function_arguments_to_pydantic_model(function_tool)
    info = get_function_info(function_tool)
    schema = _strict.to_strict_json_schema(model)

    return {
        "type": "function",
        "function": {
            "name": info.name,
            "strict": True,
            "description": info.description or "",
            "parameters": schema,
        },
    }


def function_arguments_to_pydantic_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function's signature. (excluding context types)"""

    from docstring_parser import parse_from_object

    fnc_names = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_names)
    model_name = fnc_name + "Args"

    docstring = parse_from_object(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # field_name -> (type, FieldInfo or default)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        type_hint = type_hints[param_name]

        # if is_context_type(type_hint):
        #     continue

        default_value = param.default if param.default is not param.empty else ...
        field_info = Field()

        # Annotated[str, Field(description="...")]
        if get_origin(type_hint) is Annotated:
            annotated_args = get_args(type_hint)
            type_hint = annotated_args[0]
            field_info = next(
                (x for x in annotated_args[1:] if isinstance(x, FieldInfo)), field_info
            )

        if default_value is not ... and field_info.default is PydanticUndefined:
            field_info.default = default_value

        if field_info.description is None:
            field_info.description = param_docs.get(param_name, None)

        fields[param_name] = (type_hint, field_info)

    return create_model(model_name, **fields)


def to_response_format_param(
    response_format: type | dict[str, Any],
) -> tuple[str, type[BaseModel] | TypeAdapter[Any]]:
    if isinstance(response_format, dict):
        # TODO(theomonnom): better type validation, copy TypedDict from OpenAI
        if response_format.get("type", "") not in (
            "text",
            "json_schema",
            "json_object",
        ):
            raise TypeError("Unsupported response_format type")

        # TODO(long): fix return value
        raise TypeError("Unsupported response_format type")
        return response_format

    # add support for TypedDict
    if is_typed_dict(response_format):
        response_format = create_model(
            response_format.__name__,
            **{k: (v, ...) for k, v in response_format.__annotations__.items()},  # type: ignore
        )
    json_schema_type: type[BaseModel] | TypeAdapter[Any] | None = None
    if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
        name = response_format.__name__
        json_schema_type = response_format
    elif inspect.isclass(response_format) and hasattr(
        response_format, "__pydantic_config__"
    ):  # @pydantic.dataclass
        name = response_format.__name__
        json_schema_type = TypeAdapter(response_format)
    else:
        raise TypeError(f"Unsupported response_format type - {response_format}")

    return name, json_schema_type


def to_openai_response_format(response_format: type | dict[str, Any]) -> dict[str, Any]:
    name, json_schema_type = to_response_format_param(response_format)

    schema = _strict.to_strict_json_schema(json_schema_type)
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": schema,
            "name": name,
            "strict": True,
        },
    }


def function_arguments_to_pydantic_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function's signature. (excluding context types)"""

    from docstring_parser import parse_from_object

    fnc_names = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_names)
    model_name = fnc_name + "Args"

    docstring = parse_from_object(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # field_name -> (type, FieldInfo or default)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        type_hint = type_hints[param_name]

        # if is_context_type(type_hint):
        #     continue

        default_value = param.default if param.default is not param.empty else ...
        field_info = Field()

        # Annotated[str, Field(description="...")]
        if get_origin(type_hint) is Annotated:
            annotated_args = get_args(type_hint)
            type_hint = annotated_args[0]
            field_info = next(
                (x for x in annotated_args[1:] if isinstance(x, FieldInfo)), field_info
            )

        if default_value is not ... and field_info.default is PydanticUndefined:
            field_info.default = default_value

        if field_info.description is None:
            field_info.description = param_docs.get(param_name, None)

        fields[param_name] = (type_hint, field_info)

    return create_model(model_name, **fields)


def prepare_function_arguments(
    *,
    fnc: FunctionTool | RawFunctionTool,
    json_arguments: str,  # raw function output from the LLM
    call_ctx: RunContext[Any] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:  # returns args, kwargs
    """
    Create the positional and keyword arguments to call a function tool from
    the raw function output from the LLM.
    """

    signature = inspect.signature(fnc)
    type_hints = get_type_hints(fnc, include_extras=True)
    args_dict = from_json(json_arguments)

    if is_function_tool(fnc):
        model_type = function_arguments_to_pydantic_model(fnc)

        # Function arguments with default values are treated as optional
        # when converted to strict LLM function descriptions. (e.g., we convert default
        # parameters to type: ["string", "null"]).
        # The following make sure to use the default value when we receive None.
        # (Only if the type can't be Optional)
        for param_name, param in signature.parameters.items():
            type_hint = type_hints[param_name]
            if param_name in args_dict and args_dict[param_name] is None:
                if not _is_optional_type(type_hint):
                    if param.default is not inspect.Parameter.empty:
                        args_dict[param_name] = param.default
                    else:
                        raise ValueError(
                            f"Received None for required parameter '{param_name} ;"
                            "this argument cannot be None and no default is available."
                        )

        model = model_type.model_validate(args_dict)  # can raise ValidationError
        raw_fields = _shallow_model_dump(model)
    elif is_raw_function_tool(fnc):
        # e.g async def open_gate(self, raw_arguments: dict[str, object]):
        # raw_arguments is required when using raw function tools
        raw_fields = {
            "raw_arguments": args_dict,
        }
    else:
        raise ValueError(f"Unsupported function tool type: {type(fnc)}")

    # inject RunContext if needed
    context_dict = {}
    for param_name, _ in signature.parameters.items():
        type_hint = type_hints[param_name]
        if is_context_type(type_hint) and call_ctx is not None:
            context_dict[param_name] = call_ctx

    bound = signature.bind(**{**raw_fields, **context_dict})
    bound.apply_defaults()
    return bound.args, bound.kwargs


def is_context_type(ty: type) -> bool:
    origin = get_origin(ty)
    is_call_context = ty is RunContext or origin is RunContext

    return is_call_context


def _is_optional_type(hint: Any) -> bool:
    if get_origin(hint) is Annotated:
        hint = get_args(hint)[0]

    origin = get_origin(hint)

    is_union = origin is Union
    if sys.version_info >= (3, 10):
        is_union = is_union or origin is UnionType

    return is_union and type(None) in get_args(hint)


def _shallow_model_dump(model: BaseModel, *, by_alias: bool = False) -> dict[str, Any]:
    result = {}
    for name, field in model.model_fields.items():
        key = field.alias if by_alias and field.alias else name
        result[key] = getattr(model, name)
    return result


def strip_thinking_tokens(content: str | None, thinking: asyncio.Event) -> str | None:
    if content is None:
        return None

    if thinking.is_set():
        idx = content.find(THINK_TAG_END)
        if idx >= 0:
            thinking.clear()
            content = content[idx + len(THINK_TAG_END) :]
        else:
            content = None
    else:
        idx = content.find(THINK_TAG_START)
        if idx >= 0:
            thinking.set()
            content = content[idx + len(THINK_TAG_START) :]

    return content


async def cancel_and_wait(*futures: asyncio.Future[Any]) -> None:
    loop = asyncio.get_running_loop()
    waiters = []

    for fut in futures:
        waiter = loop.create_future()
        cb = functools.partial(_release_waiter, waiter)
        waiters.append((waiter, cb))
        fut.add_done_callback(cb)
        fut.cancel()

    try:
        for waiter, _ in waiters:
            await waiter
    finally:
        for i, fut in enumerate(futures):
            _, cb = waiters[i]
            fut.remove_done_callback(cb)


def _release_waiter(waiter: asyncio.Future[Any], *_: Any) -> None:
    if not waiter.done():
        waiter.set_result(None)


gracefully_cancel = cancel_and_wait
