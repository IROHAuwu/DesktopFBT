"""從網路下載的 zip 被 Windows 標成「來自網路」時，讓程式照樣開得起來。

用 Windows 內建的功能解壓縮下載來的 zip，裡面每個檔案都會帶著「來自網路」的標記（檔案上一條叫 Zone.Identifier 的附加資料流）。
.NET Framework 不肯載入帶這個標記的 DLL，介面用的 pywebview（pythonnet）就起不來，使用者只會看到
「Failed to resolve Python.Runtime.Loader.Initialize from …\\Python.Runtime.dll」。
手動的解法是解壓縮前在 zip 的「內容」勾「解除封鎖」；這裡在啟動時把自己資料夾裡的 DLL 的標記拿掉，使用者不用做這一步。
（使用者已經自己選擇執行這個 EXE 了；拿掉的只是隨它一起來的檔案上的標記。）"""
import os
import sys

EXIT_BLOCKED = 4
MARK = ':Zone.Identifier'
KINDS = ('.dll', '.exe', '.pyd')


def clear(base):
    """把 base 底下 DLL／EXE 的「來自網路」標記拿掉，回傳拿掉了幾個。不是 Windows、沒有標記、拿不掉（唯讀資料夾）都不算錯。"""
    if sys.platform != 'win32' or not base:
        return 0
    n = 0
    try:
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(KINDS):
                    try:
                        os.remove(os.path.join(root, f) + MARK)
                        n += 1
                    except OSError:
                        pass
    except OSError:
        pass
    return n


def is_blocked_error(exc):
    """這個例外是不是「DLL 被 Windows 擋住」造成的（pythonnet 載不進來）。"""
    text = f'{type(exc).__name__}: {exc}'
    return 'Python.Runtime' in text or 'clr_loader' in text or 'pythonnet' in text


def message():
    return ('アプリを起動できませんでした。ダウンロードした zip が Windows にブロックされている可能性があります。\n'
            '  1. ダウンロードした zip を右クリック →「プロパティ」\n'
            '  2. 下のほうの「許可する」（または「ブロックの解除」）にチェックを入れて「OK」\n'
            '  3. zip を展開し直して、DesktopFBT.exe を実行\n'
            '直らない場合は .NET Framework 4.8 と Microsoft Edge WebView2 ランタイムが入っているか確認してください。\n\n'
            'The app could not start. Windows may be blocking the downloaded zip.\n'
            '  1. Right-click the downloaded zip → Properties\n'
            '  2. Tick "Unblock" at the bottom → OK\n'
            '  3. Extract the zip again and run DesktopFBT.exe\n\n'
            '程式無法啟動，下載的 zip 可能被 Windows 封鎖了。\n'
            '  1. 對下載的 zip 按右鍵 →「內容」\n'
            '  2. 勾選下方的「解除封鎖」→「確定」\n'
            '  3. 重新解壓縮後執行 DesktopFBT.exe')


def report(interactive=True):
    text = message()
    if interactive and sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, 'デスクトップFBT', 0x10 | 0x10000 | 0x40000)
    elif sys.stderr is not None:
        try:
            print(text, file=sys.stderr)
        except Exception:
            pass
    return text
