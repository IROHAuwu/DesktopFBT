"""下載的 zip 解壓縮後 DLL 被 Windows 標成「來自網路」：啟動時把標記拿掉（不然 pywebview 起不來）。
標記是 NTFS 的附加資料流，只有 Windows 測得到；其他系統只測「不會出錯」。執行：python tests/test_unblock.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unblock  # noqa: E402


def test_clear():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, '_internal', 'pythonnet', 'runtime'))
    files = [os.path.join(d, 'DesktopFBT.exe'), os.path.join(d, '_internal', 'pythonnet', 'runtime', 'Python.Runtime.dll'),
             os.path.join(d, '_internal', 'a.pyd'), os.path.join(d, '_internal', 'readme.txt')]
    for f in files:
        with open(f, 'wb') as fh:
            fh.write(b'x')
    assert unblock.clear(None) == 0 and unblock.clear(os.path.join(d, 'nothing')) == 0
    if sys.platform != 'win32':
        assert unblock.clear(d) == 0
        print('  不是 Windows：什麼都不做')
        return
    for f in files:                                     # 和 Windows 解壓縮下載的 zip 時寫的一樣
        with open(f + unblock.MARK, 'w') as fh:
            fh.write('[ZoneTransfer]\nZoneId=3\n')
    assert all(os.path.exists(f + unblock.MARK) for f in files)
    assert unblock.clear(d) == 3
    assert not any(os.path.exists(f + unblock.MARK) for f in files[:3])
    assert os.path.exists(files[3] + unblock.MARK)      # 不是程式檔的不動
    assert all(open(f, 'rb').read() == b'x' for f in files)
    assert unblock.clear(d) == 0                        # 已經沒有標記：再跑一次不會出錯
    print('  3 個程式檔的標記拿掉了，內容沒變，其他檔案不動')


def test_error_and_message():
    e = RuntimeError(r'Failed to resolve Python.Runtime.Loader.Initialize from C:\Users\x\Downloads\DesktopFBT\_internal\pythonnet\runtime\Python.Runtime.dll')
    assert unblock.is_blocked_error(e) and not unblock.is_blocked_error(ValueError('ERR_VIDEO_OPEN'))
    text = unblock.report(interactive=False)
    assert '許可する' in text and 'Unblock' in text and '解除封鎖' in text and unblock.EXIT_BLOCKED not in (0, 2, 3)
    print('  使用者回報的那個錯誤認得出來，訊息三種語言都有')


if __name__ == '__main__':
    for f in (test_clear, test_error_and_message):
        print(f.__name__)
        f()
    print('OK')
