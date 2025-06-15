from __future__ import annotations


class AssignmentTimeoutError(Exception):
    """Raised when accepting a job but not receiving an assignment within the specified timeout.
    The server may have chosen another worker to handle this job."""

    pass


class APIError(Exception):
    message: str
    body: object | None
    retryable: bool = False

    def __init__(self, message: str, *, body: object | None = None, retryable: bool = True) -> None:
        super().__init__(message)

        self.message = message
        self.body = body
        self.retryable = retryable

    def __str__(self) -> str:
        return f"{self.message} (body={self.body}, retryable={self.retryable})"


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    status_code: int
    request_id: str | None

    def __init__(
        self,
        message: str,
        *,
        status_code: int = -1,
        request_id: str | None = None,
        body: object | None = None,
        retryable: bool | None = None,
    ) -> None:
        if retryable is None:
            retryable = True
            # 4xx errors are not retryable
            if status_code >= 400 and status_code < 500:
                retryable = False

        super().__init__(message, body=body, retryable=retryable)

        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        return (
            f"{self.message} "
            f"(status_code={self.status_code}, "
            f"request_id={self.request_id}, "
            f"body={self.body}, "
            f"retryable={self.retryable})"
        )


class APIConnectionError(APIError):
    """Raised when an API request failed due to a connection error."""

    def __init__(self, message: str = "Connection error.", *, retryable: bool = True) -> None:
        super().__init__(message, body=None, retryable=retryable)


class APITimeoutError(APIConnectionError):
    """Raised when an API request timed out."""

    def __init__(self, message: str = "Request timed out.", *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)