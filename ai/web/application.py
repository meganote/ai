from importlib import metadata

import shortuuid
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.log import configure_logging
from ai.settings import settings
from ai.web.api.router import api_router
from ai.web.lifespan import lifespan_setup


def get_app() -> FastAPI:

    try:
        docs_disable = settings.profile != "dev"

        configure_logging()

        app = FastAPI(
            title="ai",
            lifespan=lifespan_setup,
            docs_url=None if docs_disable else "/docs",
            redoc_url=None if docs_disable else "/redoc",
            openapi_url=None if docs_disable else "/openapi.json",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_middleware(CorrelationIdMiddleware, generator=lambda: shortuuid.uuid())

        app.include_router(router=api_router, prefix="/api")

        return app
    except Exception as e:
        print(f"Application failed {str(e)}")
        raise
