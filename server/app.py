#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
FastAPI 应用工厂
================
组装：CORS → 路由 → 静态资源（web-ui/dist）。
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.formparsers import MultiPartParser

from server.routes import image, photo, system
import server.matting_extra  # noqa: F401  (注册 BEN2 等扩展抠图模型)

MultiPartParser.max_part_size = 100 * 1024 * 1024
MultiPartParser.max_file_size = 100 * 1024 * 1024

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_UI_DIR = os.path.join(ROOT_DIR, "web-ui", "dist")


def create_app() -> FastAPI:
    app = FastAPI(title="HivisionID-X", version="2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(photo.router)
    app.include_router(image.router)

    if os.path.exists(WEB_UI_DIR):
        app.mount("/static", StaticFiles(directory=WEB_UI_DIR), name="static")

    return app


app = create_app()
