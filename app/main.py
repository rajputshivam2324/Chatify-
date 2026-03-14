import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, chat, deepseek, gemma, image, qwen
from app.cache.redis_client import close_redis, init_redis
from app.core.config import get_settings
from app.db.session import close_db, init_db
from app.memory import init_chroma

settings = get_settings()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logger.info("Starting Chatify AI", env=settings.app_env)

    await init_redis(settings.redis_url)
    await init_db()
    await init_chroma()

    logger.info("Application started successfully")

    yield

    logger.info("Shutting down Chatify AI")
    await close_redis()
    await close_db()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Chatify AI",
    description="Production-grade AI chatbot backend with multiple models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(SessionMiddleware, secret_key=settings.app_secret)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(qwen.router)
app.include_router(gemma.router)
app.include_router(deepseek.router)
app.include_router(image.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler."""
    logger.warning(
        "HTTP error",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global generic exception handler."""
    logger.error(
        "Unhandled error",
        error=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "env": settings.app_env}