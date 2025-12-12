from fastapi import FastAPI
from .routers import inference, feedback
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

def create_app() -> FastAPI:
    app = FastAPI(title="WebShield IDS Backend", version="0.1")

    # Routers
    app.include_router(inference.router, prefix="/inference", tags=["inference"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])

    # Middleware BEFORE handler
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

app = create_app()

# Handler AFTER app is fully configured
handler = Mangum(app)
