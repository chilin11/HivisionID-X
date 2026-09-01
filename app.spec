# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置：python -m PyInstaller app.spec
from PyInstaller.utils.hooks import collect_data_files

datas = [
    ('hivision/creator/weights', 'hivision/creator/weights'),
    ('hivision/creator/retinaface/weights', 'hivision/creator/retinaface/weights'),
    ('hivision/plugin/font', 'hivision/plugin/font'),
    ('hivision/plugin/template', 'hivision/plugin/template'),
    ('hivision/plugin/beauty/lut', 'hivision/plugin/beauty/lut'),
    ('assets/hivision_logo.png', 'assets'),
    ('web-ui/dist', 'web-ui/dist'),
]
datas += collect_data_files('web-ui')

a = Analysis(
    ['deploy_api.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=['server', 'server.app', 'server.routes', 'uvicorn'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['gradio', 'gradio_client'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HivisionID-X',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
