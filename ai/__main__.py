import uvicorn

from ai.settings import settings


def main() -> None:
    uvicorn.run(
        "ai.web.application:get_app",
        workers=settings.workers_count,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.value.lower(),
        factory=True,
        # log_config=None,
    )


if __name__ == "__main__":
    main()
