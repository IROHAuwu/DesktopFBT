"""啟動前確認：程式資料夾（設定、校正、紀錄、憑證都存在 EXE 旁邊）能不能寫入。

只在「一開始就一定要寫入」的地方擋下：
- 資料夾本身不能建立檔案或資料夾；
- 已經存在的 config.json / calibration.json 不能寫入（之後存設定、存校正一定會失敗）。
不會擋下：models\\ 不能寫入但模型已經在裡面（唯讀的模型快取照常使用；之後需要下載或修復時由下載流程自己說明）、
certs\\ 不能寫入（用到 iPhone 直連時才需要）、磁碟滿（之後的操作才會遇到）。
擋下時在開啟介面之前就用系統對話框說清楚，然後以結束碼 2 結束（--selftest 不跳對話框）。
檢查只建立並刪除名稱唯一的暫存檔和暫存資料夾；既有檔案只用「不建立、不截斷、不附加」的方式開啟測試，不會修改，也不會搬移資料位置。
"""
import json
import os
import sys
import uuid

EXIT_NOT_WRITABLE = 2
_PATH_MAX_SHOWN = 240

_FOLDER = {
    'zh': ('デスクトップFBT 無法在這個資料夾裡儲存設定、校正和紀錄：\n\n{path}\n\n'
           '請把整個 DesktopFBT 資料夾移到可以寫入的位置（例如「文件」或桌面），不要放在 Program Files 或其他受保護的位置，'
           '然後再開啟一次。\n\n（不需要系統管理員權限。）'),
    'ja': ('デスクトップFBT はこのフォルダーに設定・キャリブレーション・ログを保存できません：\n\n{path}\n\n'
           'DesktopFBT フォルダーごと、書き込みできる場所（「ドキュメント」やデスクトップなど）に移動してから、もう一度開いてください。'
           'Program Files などの保護された場所には置かないでください。\n\n（管理者権限は不要です。）'),
    'en': ("デスクトップFBT can't save its settings, calibration and log in this folder:\n\n{path}\n\n"
           'Move the whole DesktopFBT folder to a place you can write to (for example Documents or the Desktop), '
           'not Program Files or another protected location, then open it again.\n\n(No administrator rights are needed.)'),
}
_FILE = {
    'zh': ('デスクトップFBT 無法寫入這個檔案：\n\n{path}\n\n'
           '請在檔案的「內容」取消勾選「唯讀」（或允許寫入這個檔案），或把整個 DesktopFBT 資料夾移到可以寫入的位置後，再開啟一次。\n\n'
           '（不需要系統管理員權限。）'),
    'ja': ('デスクトップFBT はこのファイルに書き込めません：\n\n{path}\n\n'
           'ファイルの「プロパティ」で「読み取り専用」を外す（またはこのファイルへの書き込みを許可する）か、'
           'DesktopFBT フォルダーごと書き込みできる場所に移動してから、もう一度開いてください。\n\n（管理者権限は不要です。）'),
    'en': ("デスクトップFBT can't write to this file:\n\n{path}\n\n"
           "Clear the file's Read-only setting in its Properties (or allow writing to it), or move the whole DesktopFBT folder "
           'to a place you can write to, then open it again.\n\n(No administrator rights are needed.)'),
}
MESSAGES = {'folder': _FOLDER, 'file': _FILE}


def _probe_dir(d):
    """在 d 建立名稱唯一的檔案與資料夾，再刪除。
    回傳 (錯誤, 警告)：建立失敗 → 錯誤；建立成功但刪除失敗 → 只有警告（留下的檔名一併回報），不重試。"""
    name = os.path.join(d, f'.desktopfbt-write-test-{os.getpid()}-{uuid.uuid4().hex}')
    made, warn = [], []
    try:
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)        # O_EXCL：絕不覆寫既有檔案
        made.append((name, os.remove))
        try:
            os.write(fd, b'ok')
        finally:
            os.close(fd)
        os.mkdir(name + '.d')
        made.append((name + '.d', os.rmdir))
        err = None
    except OSError as e:
        err = e
    finally:
        for p, rm in reversed(made):
            try:
                rm(p)
            except OSError as e:
                warn.append((p, e))
    return err, warn


def check(base):
    """回傳 (問題, 警告)。問題 = {'kind': 'folder'|'file', 'path', 'error'} 或 None；警告 = [(留下的路徑, 例外)]。"""
    err, warn = _probe_dir(base)
    if err is not None:
        return dict(kind='folder', path=base, error=err), warn
    for f in ('config.json', 'calibration.json'):
        p = os.path.join(base, f)
        if os.path.isfile(p):
            try:
                os.close(os.open(p, os.O_WRONLY))                        # 不建立、不截斷、不附加：內容不變
            except OSError as e:
                return dict(kind='file', path=p, error=e), warn
    return None, warn


def problem(base):
    """相容用：回傳 (路徑, 例外) 或 None。"""
    p, _ = check(base)
    return None if p is None else (p['path'], p['error'])


def language(base):
    """存過的語言（設定檔可讀、內容正確時）→ Windows 介面語言 → 英文。"""
    try:
        with open(os.path.join(base, 'config.json'), encoding='utf-8') as f:
            lang = json.load(f).get('language')
        if lang in _FOLDER:
            return lang
    except Exception:
        pass
    if sys.platform == 'win32':
        try:
            import ctypes
            primary = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF
            return {0x04: 'zh', 0x11: 'ja'}.get(primary, 'en')
        except Exception:
            pass
    return 'en'


def _short(path):
    path = str(path)
    if len(path) <= _PATH_MAX_SHOWN:
        return path
    keep = _PATH_MAX_SHOWN // 2 - 2
    return path[:keep] + ' … ' + path[-keep:]


def message(prob, base):
    return MESSAGES[prob['kind']][language(base)].format(path=_short(prob['path']))


def report(prob, base, interactive):
    """說明問題。interactive=False（--selftest）時不跳出任何視窗。回傳顯示的文字。"""
    text = message(prob, base)
    if interactive and sys.platform == 'win32':
        import ctypes
        # MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST；沒有桌面工作階段時會立即返回，不會卡住
        ctypes.windll.user32.MessageBoxW(None, text, 'デスクトップFBT', 0x10 | 0x10000 | 0x40000)
    elif sys.stderr is not None:
        try:
            print(text, file=sys.stderr)
        except Exception:
            pass
    return text
