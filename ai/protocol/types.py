from dataclasses import dataclass
from typing import Literal, TypeGuard, TypeVar, Union

from typing_extensions import TypeAlias

ATTRIBUTE_TOOL_INFO = "__tool_info"
ATTRIBUTE_RAW_TOOL_INFO = "__raw_tool_info"

_T = TypeVar("_T")


class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr: TypeAlias = Union[_T, NotGiven]
NOT_GIVEN = NotGiven()


def is_given(obj: NotGivenOr[_T]) -> TypeGuard[_T]:
    return not isinstance(obj, NotGiven)


@dataclass(frozen=True)
class APIConnectOptions:
    max_retry: int = 3
    retry_interval: float = 2.0
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if self.max_retry < 0:
            raise ValueError("max_retry must be greater than or equal to 0")

        if self.retry_interval < 0:
            raise ValueError("retry_interval must be greater than or equal to 0")

        if self.timeout < 0:
            raise ValueError("timeout must be greater than or equal to 0")

    def _interval_for_retry(self, num_retries: int) -> float:
        """
        Return the interval for the given number of retries.

        The first retry is immediate, and then uses specified retry_interval
        """
        if num_retries == 0:
            return 0.1
        return self.retry_interval


DEFAULT_API_CONNECT_OPTIONS = APIConnectOptions()
