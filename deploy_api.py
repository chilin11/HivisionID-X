#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
HivisionID-X API 启动入口（向后兼容）

实现已模块化至 server/ 包：
    python deploy_api.py
等价于
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""
import os

import uvicorn

from server.app import app  # noqa: F401  (供 uvicorn server.app:app 引用)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
