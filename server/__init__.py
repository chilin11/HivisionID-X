#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
HivisionID-X 服务层
===================

模块结构：
- server/smart_crop.py       智能裁剪引擎（纯几何，无模型代码）
- server/encoding.py         图像编解码与背景合成
- server/creator_service.py  IDCreator 生命周期 / 抠图缓存 / 失败降级
- server/routes/             路由层（system / photo / image）
- server/app.py              FastAPI 应用工厂
"""
from server.app import app, create_app

__all__ = ["app", "create_app"]
