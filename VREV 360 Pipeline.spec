# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('panorama_sfm.py', '.'), ('run_panorama_sfm.py', '.'), ('alpha_to_bw_mask.py', '.'), ('delete_pano0.py', '.'), ('rename_colmap_rig_images_and_update_images_txt.py', '.'), ('segment_images.py', '.'), ('settings_gui.json', '.'), ('colmap_pipeline_icon.ico', '.')]
binaries = []
hiddenimports = ['pycolmap', 'pycolmap._core']
datas += collect_data_files('pycolmap')
binaries += collect_dynamic_libs('pycolmap')
hiddenimports += collect_submodules('pycolmap')
tmp_ret = collect_all('pycolmap')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='VREV 360 Pipeline',
)
