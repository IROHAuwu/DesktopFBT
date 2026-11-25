# -*- mode: python ; coding: utf-8 -*-
# PyInstaller：pyinstaller --noconfirm DesktopFBT.spec  →  dist/DesktopFBT/DesktopFBT.exe（免安裝資料夾）
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

binaries = collect_dynamic_libs('onnxruntime')
datas = [('web', 'web'), ('unity', 'unity'), ('THIRD_PARTY_NOTICES', 'THIRD_PARTY_NOTICES'), ('models_bundled', 'models_bundled'),
         ('LICENSE', '.'), ('NOTICE', '.')]   # 第三方元件的授權聲明（原文）

hiddenimports = collect_submodules('rtmlib') + collect_submodules('onnxconverter_common') + ['pythonosc.udp_client']
for mod in ('dxcam', 'comtypes', 'winui3', 'winrt', 'aiohttp', 'segno', 'pygrabber'):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['tkinter', 'matplotlib', 'scipy', 'torch', 'IPython', 'pytest'],
    noarchive=False,
)
# Windows ML 文件：winrt-runtime 附的 msvcp140.dll 會和其他元件衝突，不要帶
a.binaries = [b for b in a.binaries
              if not (b[0].replace('\\', '/').lower().endswith('winrt/msvcp140.dll'))]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='DesktopFBT',
    console=False,
    icon='assets/icon.ico',
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='DesktopFBT')
