"""FastAPI application entrypoint for Neural Machine Translation & Localization Platform."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.api.v1.routes import router as api_v1_router
from backend.app.services.translation_service import translation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize model and tokenizers
    print(f"[*] Initializing {settings.PROJECT_NAME} v{settings.VERSION}...")
    translation_service.initialize()
    print("[+] Model service initialized and ready.")
    yield
    # Shutdown: clean up any resources
    print("[-] Shutting down translation service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-oriented Generative AI Translation Platform for English <-> Hindi using custom Transformer architecture.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Set up Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Neural Machine Translation & Localization API",
        "version": settings.VERSION,
        "docs": "/docs",
        "endpoints": {
            "translations": f"{settings.API_V1_STR}/translations",
            "health": f"{settings.API_V1_STR}/health",
            "ready": f"{settings.API_V1_STR}/ready",
            "models_info": f"{settings.API_V1_STR}/models/info",
            "history": f"{settings.API_V1_STR}/history",
        },
    }
