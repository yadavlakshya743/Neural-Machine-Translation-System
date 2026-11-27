"""FastAPI application entrypoint for the Neural Machine Translation & Localization Platform."""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app import db
from backend.app.api.v1.routes import router as api_v1_router
from backend.app.core.config import settings
from backend.app.services.translation_service import translation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("nmt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    start = time.time()
    translation_service.initialize()  # load models once per process
    logger.info("startup_complete loaded=%s missing=%s load_sec=%.1f",
                sorted(translation_service.translators), translation_service.load_errors,
                time.time() - start)
    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="English <-> Hindi translation with a custom Transformer trained on the IIT Bombay corpus.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Request id + access log (method, path, status, latency; never request bodies)."""
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("http method=%s path=%s status=%s ms=%.1f request_id=%s", request.method,
                request.url.path, response.status_code, (time.time() - start) * 1000, request_id)
    return response


def _envelope(request: Request, status_code: int, code: str, message: str, **extra) -> JSONResponse:
    body = {"error": {"code": code, "message": message,
                      "request_id": getattr(request.state, "request_id", None), **extra}}
    return JSONResponse(status_code=status_code, content=body)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    resp = _envelope(request, exc.status_code, detail.get("code", "HTTP_ERROR"), detail.get("message", ""))
    for k, v in (exc.headers or {}).items():
        resp.headers[k] = v
    return resp


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return _envelope(request, 422, "VALIDATION_ERROR", "Invalid request.", details=details)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("unhandled_error request_id=%s", getattr(request.state, "request_id", None))
    return _envelope(request, 500, "INTERNAL_ERROR", "An unexpected error occurred.")


app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# In deployment the built React app is served from the same origin as the API
# (one URL, no CORS). Locally the Vite dev server on :5173 is used instead.
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", "frontend/dist")
if os.path.isfile(os.path.join(FRONTEND_DIST, "index.html")):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", tags=["Root"])
    def root():
        return {
            "message": "Neural Machine Translation & Localization API",
            "version": settings.VERSION,
            "docs": "/docs",
            "endpoints": [f"{settings.API_V1_STR}/{p}" for p in (
                "auth/register", "auth/login", "translations", "models/active", "evaluations", "health", "ready")],
        }
