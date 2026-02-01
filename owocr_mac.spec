# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from owocr import __version_string__

datas = collect_data_files('rapidocr', include_py_files=False)
datas += collect_data_files('unidic_lite', include_py_files=False)
datas += collect_data_files('manga_ocr', include_py_files=False)
datas += [( 'owocr/data', 'owocr/data' )]
hiddenimports = collect_submodules('pynputfix')
hiddenimports += collect_submodules('pystrayfix')
hiddenimports += collect_submodules('desktop_notifier')

info_plist = {
    "CFBundleShortVersionString": __version_string__,
    "NSHumanReadableCopyright": "Aurora Wright",
    "LSUIElement": 1,
    "LSMultipleInstancesProhibited": 1,
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='owocr',
)
app = BUNDLE(
    coll,
    name='owocr.app',
    icon='owocr.icns',
    bundle_identifier='com.aury.owocr',
    info_plist=info_plist,
)
