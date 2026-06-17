"""Main FastAPI application for the tokenai REST API."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv()

from tokenai.api.config import settings
from tokenai.api.models import HealthResponse
from tokenai.api.routes.cache import router as cache_router
from tokenai.api.routes.compress import router as compress_router

app = FastAPI(title="tokenai API", version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cache_router)
app.include_router(compress_router)


@app.get("/")
def root() -> dict:
    """API index — links to docs, demo, and MCP endpoint."""
    return {
        "name": "tokenai",
        "version": settings.version,
        "docs": "/docs",
        "demo": "/demo",
        "mcp": "/mcp",
    }


@app.get("/demo", include_in_schema=False)
def demo_page() -> FileResponse:
    """Serve the interactive demo frontend."""
    return FileResponse("demo.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check — reports whether cache storage and MCP server are available."""
    return HealthResponse(
        status="ok",
        version=settings.version,
        cache_ready=os.path.exists(settings.cache_dir),
        mcp_available=os.path.exists("mcp/server.py"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tokenai.api.server:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
    )
