"""設定、校正檔的安全存檔：先寫到同一個資料夾裡名稱唯一的暫存檔，寫完、刷到磁碟後，再一次取代原本的檔案。

存檔中途失敗（磁碟滿、權限、被其他程式鎖住）時，原本的檔案完全不變，暫存檔會被刪掉。
取代時遇到「被其他程式暫時鎖住」（PermissionError），固定只等 0.2 秒再試一次。
保護範圍：程式內的寫入失敗或中斷；不保證斷電時的資料持久性。
"""
import json
import os
import time
import uuid

RETRY_WAIT_S = 0.2


class SaveError(OSError):
    """存檔失敗。file：檔名（不含路徑）；cause：原本的例外。"""

    def __init__(self, file, cause, cleanup_error=None):
        super().__init__(f'{file}: {type(cause).__name__}')
        self.file = file
        self.cause = cause
        self.cleanup_error = cleanup_error


def _write_text(fd, text):
    """寫入、刷到磁碟、關閉（測試時可以替換這個函式來模擬寫到一半失敗）。"""
    with os.fdopen(fd, 'w', encoding='utf-8') as f:            # 文字模式：換行和以前的存檔方式相同（Windows 為 CRLF）
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path, obj, **json_kw):
    """把 obj 以 JSON 存到 path。成功時一次取代；失敗時 path 不變並丟出 SaveError。"""
    name = os.path.basename(path)
    try:
        text = json.dumps(obj, **json_kw)                       # 先轉成文字：失敗時完全沒有碰到檔案
    except Exception as e:
        raise SaveError(name, e)
    folder = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(folder, f'.{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    created = False
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
        created = True
        _write_text(fd, text)
        try:
            os.replace(tmp, path)
        except PermissionError:                                 # 固定的重試規則：等 0.2 秒、只再試一次
            time.sleep(RETRY_WAIT_S)
            os.replace(tmp, path)
    except Exception as e:
        cleanup = None
        if created:
            try:
                os.remove(tmp)
            except OSError as ce:
                cleanup = ce
        raise SaveError(name, e, cleanup)
