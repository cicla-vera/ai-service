from fastapi import FastAPI

from app.api.routes.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cicla Vera AI Service",
        version="0.1.0",
        summary="Evidence analysis support service for the Vera safety layer.",
    )
    app.include_router(health_router)

    return app


app = create_app()
