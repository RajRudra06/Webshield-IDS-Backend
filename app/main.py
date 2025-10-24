from fastapi import FastAPI
from .routers import inference, feedback

def create_app() -> FastAPI:
    app = FastAPI(title="WebShield IDS Backend", version="0.1")
    app.include_router(inference.router, prefix="/inference", tags=["inference"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
    return app

app = create_app()
