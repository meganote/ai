import enum
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """Application settings."""

    host: str = "127.0.0.1"
    port: int = 8080
    workers_count: int = 1
    reload: bool = False

    profile: str = "dev"
    log_level: LogLevel = LogLevel.DEBUG

    # MongoDB
    mongo_host: str = "loalhost"
    mongo_port: int = 27017

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_user: Optional[str] = None
    redis_pass: Optional[str] = None

    @property
    def db_url(self) -> str:
        return f"mongodb://{self.mongo_host}:{self.mongo_port}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    model_config = SettingsConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
