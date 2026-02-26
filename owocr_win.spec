# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    VSVersionInfo,
    FixedFileInfo,
    StringFileInfo,
    StringTable,
    StringStruct,
    VarFileInfo,
    VarStruct,
)

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

version_info = VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=__version__ + (0,),
        prodvers=__version__ + (0,),
        mask=0x3F,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0),
    ),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    '040904B0',
                    [
                        StringStruct('Comments', 'owocr'),
                        StringStruct('CompanyName', '©2026 Aurora Wright and contributors'),
                        StringStruct('FileDescription', 'owocr'),
                        StringStruct('FileVersion', __version_string__),
                        StringStruct('LegalCopyright', '©2026 Aurora Wright and contributors'),
                        StringStruct('ProductName', 'owocr'),
                        StringStruct('ProductVersion', __version_string__),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct('Translation', [1033, 1200])]),
    ],
)

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
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='owocr/data/icon.png',
    version=version_info,
    contents_directory='data',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='owocr',
)
