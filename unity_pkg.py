"""產生 .unitypackage：在 Unity 開著專案時雙擊就能匯入。
內容：Assets/DesktopFBT/Editor/DesktopFBTBvhImporter.cs（把 .bvh 自動轉成 Humanoid 動畫）、桌面模式工具＋動作檔。"""
import gzip
import hashlib
import io
import os
import tarfile
import time

from paths import resource_dir

SCRIPT_GUID = '1b4a3480f27141a78947752e23578cfc'      # 固定：重複匯入不會多出一份腳本
SCRIPT_PATH = 'Assets/DesktopFBT/Editor/DesktopFBTBvhImporter.cs'
MOTION_DIR = 'Assets/DesktopFBT/Motions/'

_CS_META = """fileFormatVersion: 2
guid: {guid}
MonoImporter:
  externalObjects: {{}}
  serializedVersion: 2
  defaultReferences: []
  executionOrder: 0
  icon: {{instanceID: 0}}
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""
_META = "fileFormatVersion: 2\nguid: {guid}\n"
_ANIM_META = ("fileFormatVersion: 2\nguid: {guid}\nNativeFormatImporter:\n  externalObjects: {{}}\n"
              "  mainObjectFileID: 7400000\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n")
_ASMDEF_META = "fileFormatVersion: 2\nguid: {guid}\nAssemblyDefinitionImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"

# 桌面模式工具（只有裝了 VRChat Avatar SDK 的專案才會編譯）
TOOLS = [
    ('Assets/DesktopFBT/Editor/Desktop/DesktopFBTDesktop.cs', ('Desktop', 'DesktopFBTDesktop.cs'),
     '2c24161406ac4ce882c04439402b8284', _CS_META),
    ('Assets/DesktopFBT/Editor/Desktop/DesktopFBT.Desktop.Editor.asmdef', ('Desktop', 'DesktopFBT.Desktop.Editor.asmdef'),
     '787aae7c424f46ef961faacdb3f630b8', _ASMDEF_META),
    # Param Compressor（把其他同步參數輪流送，空出 bit 給全身同步用）。元件要在 Editor 資料夾外面；
    # 上傳時執行的部分要有 NDMF（Modular Avatar）才會編譯。GUID 固定：重新安裝後 Avatar 上的元件不會變成 Missing Script
    ('Assets/DesktopFBT/ParamCompressor/Runtime/DesktopFBTParamCompressor.cs', ('ParamCompressor', 'Runtime', 'DesktopFBTParamCompressor.cs'),
     '03780ebbcd91441abca743676728938b', _CS_META),
    ('Assets/DesktopFBT/ParamCompressor/Runtime/DesktopFBT.ParamCompressor.Runtime.asmdef',
     ('ParamCompressor', 'Runtime', 'DesktopFBT.ParamCompressor.Runtime.asmdef'), '0e9041995e7b42ee8cbcb71e49d65fa9', _ASMDEF_META),
    ('Assets/DesktopFBT/ParamCompressor/Editor/DesktopFBTParamCompressorPass.cs', ('ParamCompressor', 'Editor', 'DesktopFBTParamCompressorPass.cs'),
     '8bf66b1eed95492587e78ac15c54061f', _CS_META),
    ('Assets/DesktopFBT/ParamCompressor/Editor/DesktopFBT.ParamCompressor.Editor.asmdef',
     ('ParamCompressor', 'Editor', 'DesktopFBT.ParamCompressor.Editor.asmdef'), '2df0a40ee3ee4f488bc1f7061fefdcf3', _ASMDEF_META),
    ('Assets/DesktopFBT/ParamCompressor/THIRD_PARTY_NOTICES.txt', ('ParamCompressor', 'THIRD_PARTY_NOTICES.txt'),
     'df6b10badb024ef8931e18589ac64c08', _META),
]
FOLDERS = [('Assets/DesktopFBT/Editor/Desktop', '4c670aacbef54f0b8739fde045ae78f7')]
_FOLDER_META = "fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"


def _read(*parts):
    with open(os.path.join(resource_dir(), 'unity', *parts), 'rb') as f:
        return f.read()


def script_source():
    return _read('DesktopFBTBvhImporter.cs')


def _add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o777
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def _dir(tar, guid):
    d = tarfile.TarInfo(guid)
    d.type = tarfile.DIRTYPE
    d.mode = 0o777
    d.mtime = int(time.time())
    tar.addfile(d)


def _folder_guid(path):
    return next((g for p, g in FOLDERS if p == path), None) or hashlib.md5(('DesktopFBT folder ' + path).encode('utf-8')).hexdigest()


def build(out_path, motions):
    """motions: [(檔名.anim 或 .bvh, 文字或 bytes)]。格式和 Unity 自己匯出的一樣：每個上層資料夾也要有項目，
    少了的話 Unity 會把檔案全部丟到 Assets 底下並換掉 GUID。"""
    items = [(SCRIPT_PATH, SCRIPT_GUID, script_source(), _CS_META)]
    for path, src, guid, meta in TOOLS:
        items.append((path, guid, _read(*src), meta))
    for name, data in motions:
        if isinstance(data, str):
            data = data.encode('utf-8')
        path = MOTION_DIR + name
        guid = hashlib.md5((path + hashlib.md5(data).hexdigest()).encode('utf-8')).hexdigest()
        items.append((path, guid, data, _ANIM_META if name.lower().endswith('.anim') else _META))
    folders = set()
    for path, *_ in items:
        parts = path.split('/')[:-1]
        for i in range(2, len(parts) + 1):
            folders.add('/'.join(parts[:i]))
    entries = [(_folder_guid(f), f, None, _FOLDER_META) for f in folders]
    entries += [(g, p, d, m) for p, g, d, m in items]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.GNU_FORMAT) as tar:
        for guid, path, data, meta in sorted(entries):
            _dir(tar, guid)
            if data is not None:
                _add(tar, f'{guid}/asset', data)
            _add(tar, f'{guid}/asset.meta', meta.format(guid=guid).encode('utf-8'))
            _add(tar, f'{guid}/pathname', path.encode('utf-8'))
    with open(out_path, 'wb') as f, gzip.GzipFile(filename='archtemp.tar', mode='wb', fileobj=f, mtime=0) as g:
        g.write(buf.getvalue())
    return out_path


# ---------------------------------------------------------------- 直接放進 Unity 專案
def is_project(path):
    """是不是 Unity 專案的根目錄（有 Assets 和 ProjectSettings）。"""
    return bool(path) and os.path.isdir(os.path.join(path, 'Assets')) and \
        os.path.isdir(os.path.join(path, 'ProjectSettings'))


def find_project(path):
    """使用者選到專案裡面的資料夾（例如 Assets）時，往上找專案根目錄。"""
    p = os.path.abspath(path or '')
    for _ in range(6):
        if is_project(p):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


# 改名前（v1.7 以前）的腳本位置。GUID 和現在的一樣，兩份同時存在的話 Unity 會有兩個 .bvh 匯入器、GUID 也會衝突。
_OLD = 'Youtube' + 'FBT'
_OLD_SCRIPTS = [(f'Assets/{_OLD}/Editor/{_OLD}BvhImporter.cs', SCRIPT_GUID)] + [
    (path.replace('DesktopFBT', _OLD), guid) for path, _, guid, _ in TOOLS]


def _remove_old_scripts(project):
    """只刪「檔名和 GUID 都是我們舊版裝的」腳本（和它的 .meta）；動作檔、產生的控制器都不動。刪不掉就留著。"""
    for path, guid in _OLD_SCRIPTS:
        f = os.path.join(project, *path.split('/'))
        try:
            with open(f + '.meta', encoding='utf-8') as m:
                if f'guid: {guid}' not in m.read():
                    continue
            os.remove(f)
            os.remove(f + '.meta')
        except OSError:
            continue
    for folder in (f'Assets/{_OLD}/Editor/Desktop', f'Assets/{_OLD}/Editor'):
        d = os.path.join(project, *folder.split('/'))
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                if os.path.exists(d + '.meta'):
                    os.remove(d + '.meta')
        except OSError:
            pass


def install(project, motions):
    """把匯入腳本和動作檔直接寫進專案的 Assets/DesktopFBT。Unity 切回前景時會自動匯入並轉成動畫。
    回傳寫入的動作檔路徑清單。"""
    if not is_project(project):
        raise RuntimeError('ERR_UNITY_PROJECT')
    _remove_old_scripts(project)
    ed = os.path.join(project, *SCRIPT_PATH.split('/')[:-1])
    os.makedirs(ed, exist_ok=True)
    cs = os.path.join(project, *SCRIPT_PATH.split('/'))
    src = script_source()
    old = None
    if os.path.exists(cs):
        with open(cs, 'rb') as f:
            old = f.read()
    if old != src:                                   # 新版腳本才覆寫（不會讓 Unity 白白重新編譯）
        with open(cs, 'wb') as f:
            f.write(src)
    if not os.path.exists(cs + '.meta'):
        with open(cs + '.meta', 'w', encoding='utf-8', newline='\n') as f:
            f.write(_CS_META.format(guid=SCRIPT_GUID))
    for path, src, guid, meta in TOOLS:
        dst = os.path.join(project, *path.split('/'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = _read(*src)
        old = None
        if os.path.exists(dst):
            with open(dst, 'rb') as f:
                old = f.read()
        if old != data:
            with open(dst, 'wb') as f:
                f.write(data)
        if not os.path.exists(dst + '.meta'):
            with open(dst + '.meta', 'w', encoding='utf-8', newline='\n') as f:
                f.write(meta.format(guid=guid))
    md = os.path.join(project, *MOTION_DIR.strip('/').split('/'))
    os.makedirs(md, exist_ok=True)
    out = []
    for name, data in motions:
        if isinstance(data, str):
            data = data.encode('utf-8')
        base, ext = os.path.splitext(name)
        dst = os.path.join(md, name)
        n = 2
        while os.path.exists(dst):
            dst = os.path.join(md, f'{base} ({n}){ext}')
            n += 1
        with open(dst, 'wb') as f:
            f.write(data)
        out.append(dst)
    return out
