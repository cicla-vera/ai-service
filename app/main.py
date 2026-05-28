from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router

load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cicla Vera AI Service",
        version="0.1.0",
        summary="Evidence analysis support service for the Vera safety layer.",
    )
    app.include_router(analyze_router)
    app.include_router(health_router)

    return app


app = create_app()
