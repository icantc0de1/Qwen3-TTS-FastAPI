"""Routers package for API endpoints."""

from api.src.routers import openai_compatible
from api.src.routers.openai_compatible import router

__all__ = ["openai_compatible", "router"]
