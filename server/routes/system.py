#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""系统级路由：健康检查 / 图标 / 首页"""
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from server import super_res

router = APIRouter()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_UI_DIR = os.path.join(ROOT_DIR, "web-ui", "dist")


@router.get("/health")
async def health():
    return {
        "status": True,
        "service": "HivisionID-X",
        "version": "2.0",
        "super_res": super_res.available(),
        "endpoints": ["/idphoto", "/human_matting", "/add_background", "/generate_layout_photos", "/watermark", "/set_kb", "/idphoto_crop"],
    }


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = os.path.join(ROOT_DIR, "assets", "hivision_logo.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return HTMLResponse("", status_code=404)


@router.get("/", include_in_schema=False)
async def serve_index():
    index_html = os.path.join(WEB_UI_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return HTMLResponse("<h1>Web UI not found</h1><p>web-ui/dist/index.html is missing.</p>")
