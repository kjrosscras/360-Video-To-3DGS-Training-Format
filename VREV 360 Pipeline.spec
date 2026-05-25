# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# Helper scripts / resources bundled beside the EXE in _internal.
# Make sure these filenames match the files in the folder where you run PyInstaller.
datas = [
    ('panorama_sfm.py', '.'),
    ('run_panorama_sfm.py', '.'),
    ('alpha_to_bw_mask.py', '.'),
    ('delete_pano0.py', '.'),
    ('rename_colmap_rig_images_and_update_images_txt.py', '.'),
    ('segment_images.py', '.'),
    ('settings_gui.json', '.'),
    ('colmap_pipeline_icon.ico', '.'),
]

binaries = []
hiddenimports = [
    'pycolmap',
    'pycolmap._core',
]

# Bundle pycolmap's package data, dynamic libraries, and hidden submodules.
# This works for the CPU-only conda-forge pycolmap build too.
datas += collect_data_files('pycolmap')
binaries += collect_dynamic_libs('pycolmap')
hiddenimports += collect_submodules('pycolmap')

_pycolmap_all = collect_all('pycolmap')
datas += _pycolmap_all[0]
binaries += _pycolmap_all[1]
hiddenimports += _pycolmap_all[2]


a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=binaries,
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
    name='VREV 360 Pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # Keep UPX off for pycolmap/COLMAP native DLL stability.
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['colmap_pipeline_icon.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    # Keep UPX off for pycolmap/COLMAP native DLL stability.
    upx=False,
    upx_exclude=[],
    name='VREV 360 Pipeline',
)
