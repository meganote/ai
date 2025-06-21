from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Sequence, TypeAlias, Union, overload

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter

from ai import utils

from ..types import NOT_GIVEN, NotGivenOr, is_given
from . import _provider_format
from .tool_context import FunctionTool, RawFunctionTool

ChatRole: TypeAlias = Literal["system", "user", "assistant", "tool"]


class ImageContent(BaseModel):
    type: Literal["image_content"] = Field(default="image_content")
    image: str
    mime_type: str | None = None


class AudioContent(BaseModel):
    type: Literal["audio_content"] = Field(default="audio_content")
    frame: list[Any]
    transcript: str | None = None


class FileContent(BaseModel):
    type: Literal["file_content"] = Field(default="file_content")
    file: str
    mime_type: str | None = None


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["message"] = "message"
    role: ChatRole
    content: list[ChatContent]
    created: float = Field(default_factory=time.time)

    @property
    def text_content(self) -> str | None:
        text_parts = [c for c in self.content if isinstance[c, str]]
        if not text_parts:
            return None
        return "\n".join(text_parts)


ChatContent: TypeAlias = Union[ImageContent, FileContent, str]


class FunctionCall(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str
    created: float = Field(default_factory=time.time)


class FunctionCallOutput(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    name: str = Field(default="")
    output: str
    is_error: bool
    created: float = Field(default_factory=time.time)


ChatItem = Annotated[
    Union[ChatMessage, FunctionCall, FunctionCallOutput], Field(description="type")
]


class ChatContext:
    def __init__(self, items: list[ChatItem] | None = None):
        self._items: list[ChatItem] = items if items else []

    @classmethod
    def empty(cls) -> ChatContext:
        return cls([])

    @property
    def items(self) -> list[ChatItem]:
        return self._items

    @items.setter
    def items(self, items: list[ChatItem]) -> None:
        self._items = items

    def add_message(
        self,
        *,
        role: ChatRole,
        content: list[ChatContent] | str,
        id: str | None = None,
        created: float | None = None,
    ) -> ChatMessage:
        kwargs: dict[str, Any] = {}
        if id:
            kwargs["id"] = id
        if created:
            kwargs["created"] = created

        if isinstance(content, str):
            message = ChatMessage(role=role, content=[content], **kwargs)
        else:
            message = ChatMessage(role=role, content=content, **kwargs)

        self._items.append(message)
        return message

    def insert(self, item: ChatItem | Sequence[ChatItem]) -> None:
        """Insert an item or list of items into the chat context by creation time."""
        items = item if isinstance(item, list) else [item]

        for _item in items:
            print(f"!!! Inserting item into chat context: {_item}")
            idx = self.find_insertion_index(created_at=_item.created)
            self._items.insert(idx, _item)

    def get_by_id(self, item_id: str) -> ChatItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def index_by_id(self, item_id: str) -> int | None:
        return next(
            (i for i, item in enumerate(self.items) if item.id == item_id), None
        )

    def copy(
        self,
        *,
        exclude_function_call: bool = False,
        exclude_instructions: bool = False,
        exclude_empty_message: bool = False,
        tools: NotGivenOr[
            Sequence[FunctionTool | RawFunctionTool | str | Any]
        ] = NOT_GIVEN,
    ) -> ChatContext:
        items = []

        from .tool_context import (
            get_function_info,
            get_raw_function_info,
            is_function_tool,
            is_raw_function_tool,
        )

        valid_tools = set[str]()
        if is_given(tools):
            for tool in tools:
                if isinstance(tool, str):
                    valid_tools.add(tool)
                elif is_function_tool(tool):
                    valid_tools.add(get_function_info(tool).name)
                elif is_raw_function_tool(tool):
                    valid_tools.add(get_raw_function_info(tool).name)
                # TODO(theomonnom): other tools

        for item in self.items:
            if exclude_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                continue

            if (
                exclude_instructions
                and item.type == "message"
                and item.role in ["system", "developer"]
            ):
                continue

            if exclude_empty_message and item.type == "message" and not item.content:
                continue

            if (
                is_given(tools)
                and (
                    item.type == "function_call" or item.type == "function_call_output"
                )
                and item.name not in valid_tools
            ):
                continue

            items.append(item)

        return ChatContext(items)

    def truncate(self, *, max_items: int) -> ChatContext:
        """Truncate the chat context to the last N items in place.

        Removes leading function calls to avoid partial function outputs.
        Preserves the first system message by adding it back to the beginning.
        """
        instructions = next(
            (
                item
                for item in self._items
                if item.type == "message" and item.role == "system"
            ),
            None,
        )

        new_items = self._items[-max_items:]
        # chat ctx shouldn't start with function_call or function_call_output
        while new_items and new_items[0].type in [
            "function_call",
            "function_call_output",
        ]:
            new_items.pop(0)

        if instructions:
            new_items.insert(0, instructions)

        self._items[:] = new_items
        return self

    def to_dict(
        self,
        *,
        exclude_image: bool = True,
        exclude_audio: bool = True,
        exclude_timestamp: bool = True,
        exclude_function_call: bool = False,
    ) -> dict[str, Any]:
        items: list[ChatItem] = []
        for item in self.items:
            if exclude_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                continue

            if item.type == "message":
                item = item.model_copy()
                if exclude_image:
                    item.content = [
                        c for c in item.content if not isinstance(c, ImageContent)
                    ]
                if exclude_audio:
                    item.content = [
                        c for c in item.content if not isinstance(c, AudioContent)
                    ]

            items.append(item)

        exclude_fields = set()
        if exclude_timestamp:
            exclude_fields.add("created_at")

        return {
            "items": [
                item.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_defaults=False,
                    exclude=exclude_fields,
                )
                for item in items
            ],
        }

    @overload
    def to_provider_format(
        self, format: Literal["openai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], Literal[None]]: ...

    @overload
    def to_provider_format(
        self, format: str, **kwargs: Any
    ) -> tuple[list[dict], Any]: ...

    def to_provider_format(
        self,
        format: Literal["openai", "google", "aws", "anthropic"] | str,
        *,
        inject_dummy_user_message: bool = True,
        **kwargs: Any,
    ) -> tuple[list[dict], Any]:
        """Convert the chat context to a provider-specific format.

        If ``inject_dummy_user_message`` is ``True``, a dummy user message will be added
        to the beginning or end of the chat context depending on the provider.

        This is necessary because some providers expect a user message to be present for
        generating a response.
        """
        kwargs["inject_dummy_user_message"] = inject_dummy_user_message

        if format == "openai":
            return _provider_format.openai.to_chat_ctx(self, **kwargs)
        # elif format == "google":
        #     return _provider_format.google.to_chat_ctx(self, **kwargs)
        # elif format == "aws":
        #     return _provider_format.aws.to_chat_ctx(self, **kwargs)
        # elif format == "anthropic":
        #     return _provider_format.anthropic.to_chat_ctx(self, **kwargs)
        else:
            raise ValueError(f"Unsupported provider format: {format}")

    def find_insertion_index(self, *, created_at: float) -> int:
        """
        Returns the index to insert an item by creation time.

        Iterates in reverse, assuming items are sorted by `created_at`.
        Finds the position after the last item with `created_at <=` the given timestamp.
        """
        for i in reversed(range(len(self._items))):
            if self._items[i].created <= created_at:
                return i + 1

        return 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatContext:
        item_adapter = TypeAdapter(list[ChatItem])
        items = item_adapter.validate_python(data["items"])
        return cls(items)

    @property
    def readonly(self) -> bool:
        return False


class _ReadOnlyChatContext(ChatContext):
    """A read-only wrapper for ChatContext that prevents modifications."""

    error_msg = (
        "trying to modify a read-only chat context, "
        "please use .copy() and agent.update_chat_ctx() to modify the chat context"
    )

    class _ImmutableList(list[ChatItem]):
        def _raise_error(self, *args: Any, **kwargs: Any) -> None:
            logger.error(_ReadOnlyChatContext.error_msg)
            raise RuntimeError(_ReadOnlyChatContext.error_msg)

        # override all mutating methods to raise errors
        append = extend = pop = remove = clear = sort = reverse = _raise_error  # type: ignore
        __setitem__ = __delitem__ = __iadd__ = __imul__ = _raise_error  # type: ignore

        def copy(self) -> list[ChatItem]:
            return list(self)

    def __init__(self, items: list[ChatItem]):
        self._items = self._ImmutableList(items)

    @property
    def readonly(self) -> bool:
        return True
