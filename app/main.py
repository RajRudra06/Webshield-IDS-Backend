from fastapi import FastAPI
from .routers import inference, feedback
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

def create_app() -> FastAPI:
    # Add redirect_slashes=False to stop the redirect loop
    app = FastAPI(title="WebShield IDS Backend", version="0.1", redirect_slashes=False)

    # Add root route
    @app.get("/")
    async def root():
        return {"status": "healthy", "service": "WebShield IDS"}

    # Routers
    app.include_router(inference.router, prefix="/inference", tags=["inference"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

app = create_app()

# Add lifespan="off" to Mangum
handler = Mangum(app, lifespan="off")