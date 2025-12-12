from fastapi import FastAPI
from .routers import inference, feedback
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="WebShield IDS Backend", version="0.1")
    app.include_router(inference.router, prefix="/inference", tags=["inference"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
    return app

app = create_app()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Chrome extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# from fastapi import FastAPI
# from .routers import inference, feedback

# def create_app() -> FastAPI:
#     app = FastAPI(
#         title="WebShield IDS UI",  # Changed
#         version="0.2-RL",  # Changed
#         description="URL threat detection with Reinforcement Learning adaptive ensemble"  # Added
#     )
#     app.include_router(inference.router, prefix="/inference", tags=["inference"])
#     app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
#     return app

# app = create_app()

# # Optional: Add a root endpoint
# @app.get("/")
# def root():
#     return {
#         "message": "WebShield IDS API with Reinforcement Learning",
#         "version": "0.2-RL",
#         "endpoints": {
#             "inference": "/inference/ - Predict URL threat with RL-based ensemble",
#             "feedback": "/feedback/submit - Submit ground truth for RL learning",
#             "rl_stats": "/feedback/stats - View RL agent learning statistics",
#             "docs": "/docs - Swagger UI documentation"
#         }
#     }