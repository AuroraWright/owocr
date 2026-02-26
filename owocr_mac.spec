# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path().absolute()))
from owocr import __version__, __version_string__

datas = collect_data_files('rapidocr', include_py_files=False)
datas += collect_data_files('unidic_lite', include_py_files=False)
datas += collect_data_files('manga_ocr', include_py_files=True)
datas += [( 'owocr/data', 'owocr/data' )]
datas += [( 'ndlocr_lite/config', 'ndlocr_lite/config' )]
hiddenimports = collect_submodules('pynputfix')
hiddenimports += collect_submodules('pystrayfix')
hiddenimports += collect_submodules('desktop_notifier')

info_plist = {
    'CFBundleShortVersionString': __version_string__,
    'CFBundleVersion': __version_string__,
    'CFBundleSupportedPlatforms': ['MacOSX'],
    'NSHumanReadableCopyright': '©2026 Aurora Wright and contributors',
    'LSMinimumSystemVersion': '10.15',
    'LSApplicationCategoryType': 'public.app-category.utilities',
    'LSUIElement': 1,
    'LSMultipleInstancesProhibited': 1,
}

a = Analysis(
    ['owocr/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='owocr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name='owocr',
)
app = BUNDLE(
    coll,
    name='owocr.app',
    icon='owocr/data/icon.png',
    bundle_identifier='com.aury.owocr',
    info_plist=info_plist,
)
