from __future__ import annotations

import time
from typing import Annotated, Any, Literal, TypeAlias, Union, overload

from pydantic import BaseModel, Field

from ai import utils

from . import _provider_format

ChatRole: TypeAlias = Literal["system", "user", "assistant", "tool"]


class ImageContent(BaseModel):
    type: Literal["image_content"] = Field(default="image_content")
    image: str
    mime_type: str | None = None


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
        elif format == "google":
            return _provider_format.google.to_chat_ctx(self, **kwargs)
        elif format == "aws":
            return _provider_format.aws.to_chat_ctx(self, **kwargs)
        elif format == "anthropic":
            return _provider_format.anthropic.to_chat_ctx(self, **kwargs)
        else:
            raise ValueError(f"Unsupported provider format: {format}")
