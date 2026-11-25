"""DesktopFBT 桌面程式入口（pywebview：Windows 上使用內建的 Edge WebView2）。"""
import os
import sys
import traceback

from paths import data, resource

_unblocked = 0                               # 啟動時解除了幾個檔案的「來自網路」標記
EXIT_SELFTEST_FAILED = 3                     # --selftest 完成但回報有問題（報告照樣寫進 selftest.txt）


class Api:
    """給網頁介面呼叫的方法（JavaScript：window.pywebview.api.xxx()）。"""

    def __init__(self, engine):
        self._e = engine           # 底線開頭：不會被 pywebview 公開到 JS
        self._win = []

    def get_state(self):
        return self._e.get_state()

    def get_config(self):
        return self._e.get_config()

    def update_config(self, patch):
        return self._e.update_config(patch)

    def environment(self):
        return self._e.environment()

    def list_windows(self):
        return self._e.list_windows()

    def list_cameras(self):
        return self._e.list_cameras()

    def phone_info(self, start=True):
        return self._e.phone_info(bool(start))

    def start_live(self, src):
        self._e.start_live(src)

    def start_export(self, path):
        self._e.start_export(path)

    def start_image_export(self, paths):
        self._e.start_image_export(paths)

    def stop(self):
        self._e.stop()

    def open_folder(self):
        return self._e.open_folder()

    def pick_unity_project(self):
        """選擇 Unity 專案資料夾。回傳 {'path': ...} 或 {'error': 'ERR_UNITY_PROJECT'}；取消時回傳 None。"""
        import webview
        w = self._win[0] if self._win else None
        if w is None:
            return None
        r = w.create_folder_dialog() if hasattr(w, 'create_folder_dialog') else \
            w.create_file_dialog(webview.FOLDER_DIALOG)
        if not r:
            return None
        path = r[0] if isinstance(r, (list, tuple)) else r
        proj = self._e.set_unity_project(path)
        return {'path': proj} if proj else {'error': 'ERR_UNITY_PROJECT'}

    def install_unity_tools(self):
        return self._e.install_unity_tools()

    def clear_unity_project(self):
        self._e.set_unity_project('')

    def pick_images(self):
        """選擇一張或多張圖片，回傳路徑清單。"""
        import webview
        from video import IMAGE_EXT
        w = self._win[0] if self._win else None
        if w is None:
            return []
        pat = ';'.join('*' + e for e in IMAGE_EXT)
        r = w.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True,
                                 file_types=(f'Image ({pat})', 'All files (*.*)'))
        return list(r) if r else []

    # 動作庫
    def list_motions(self):
        return self._e.list_motions()

    def start_motion(self, path):
        self._e.start_motion(path)

    def remove_motion(self, path):
        return self._e.remove_motion(path)

    def restore_motions(self):
        return self._e.restore_motions()

    def add_motions(self):
        """選擇一個或多個 .anim 加進動作庫，回傳 {added, errors}。"""
        import webview
        w = self._win[0] if self._win else None
        if w is None:
            return {'added': [], 'errors': [], 'warnings': []}
        r = w.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True,
                                 file_types=('Unity animation (*.anim)', 'All files (*.*)'))
        return self._e.add_motions(list(r)) if r else {'added': [], 'errors': [], 'warnings': []}

    def open_exports(self, path=None):
        return self._e.open_exports(path)

    def import_to_unity(self, path):
        try:
            return dict(package=self._e.import_to_unity(path))
        except Exception as e:
            return dict(error=str(e) if str(e).startswith('ERR_') else 'ERR_UNITY_PROJECT')

    def pick_video(self):
        """開啟「選擇影片」對話框，回傳路徑或 None。"""
        import webview
        from video import VIDEO_EXT
        w = self._win[0] if self._win else None
        if w is None:
            return None
        pat = ';'.join('*' + e for e in VIDEO_EXT)
        r = w.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False,
                                 file_types=(f'Video ({pat})', 'All files (*.*)'))
        if not r:
            return None
        return r[0] if isinstance(r, (list, tuple)) else r

    def clear_error(self):
        self._e._set(error=None, message='')


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def selftest(startup_warnings=None):
    """打包檢查用：確認 EXE 裡的模組都載入得了，結果寫到 selftest.txt。"""
    import json

    import cv2
    import onnxruntime
    import rtmlib  # noqa: F401
    import webview  # noqa: F401
    from pythonosc.udp_client import SimpleUDPClient  # noqa: F401

    from engine import Engine
    eng = Engine(preload=False)
    try:
        info = dict(ok=True, env=eng.environment(), providers=onnxruntime.get_available_providers(),
                    opencv=cv2.__version__, web=os.path.exists(resource('web', 'index.html')),
                    windows=len(eng.list_windows()))
        import bvh  # noqa: F401
        import clips  # noqa: F401
        import sync  # noqa: F401
        import desktop  # noqa: F401
        import lift  # noqa: F401
        import precise  # noqa: F401
        import unity_pkg
        from rtmlib import RTMPose3d  # noqa: F401
        info['unity_script'] = len(unity_pkg.script_source())
        try:
            import onnx  # noqa: F401
            from onnxconverter_common import float16  # noqa: F401
            info['fp16'] = True
        except Exception as e:
            info['fp16'] = f'{type(e).__name__}: {e}'
        try:
            import dxcam  # noqa: F401
            info['dxcam'] = True
        except Exception as e:
            info['dxcam'] = f'{type(e).__name__}: {e}'
        try:
            import temporal
            found = {n: temporal.model_path(n) for n in temporal.MODELS}
            if not any(found.values()):
                info['temporal'] = 'missing'
            else:
                for n, path in found.items():
                    if path:
                        temporal.Model(path, 'cpu', temporal.MODELS[n][1])
                info['temporal'] = True
                info['temporal_models'] = sorted(n for n, v in found.items() if v)
        except Exception as e:
            info['temporal'] = f'{type(e).__name__}: {e}'
        try:
            import winui3.microsoft.windows.ai.machinelearning  # noqa: F401
            import winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap  # noqa: F401
            info['winml'] = True
        except Exception as e:
            info['winml'] = f'{type(e).__name__}: {e}'
        info['unblocked'] = _unblocked
        if sys.platform == 'win32':
            try:
                import clr  # noqa: F401            介面（pywebview）要靠它載入 .NET：DLL 被 Windows 擋住時就是在這裡失敗
                info['dotnet'] = True
            except Exception as e:
                info['dotnet'] = f'{type(e).__name__}: {e}'
        info['startup_warnings'] = [str(w[0]) for w in (startup_warnings or [])]
        info['ok'] = bool(info['web'] and info['unity_script'] > 0 and info['fp16'] is True
                          and info['temporal'] in (True, 'missing') and info.get('dotnet', True) is True)
    finally:
        eng.shutdown()
    with open(data('selftest.txt'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info['ok']


def log_startup_warnings(warnings):
    """啟動檢查刪不掉的測試檔（資料夾允許建立但不允許刪除時）：每次啟動寫一次到 log.txt，並印到 stderr（有的話）。
    只記錄，不重試、不刪除其他檔案；寫不進去也不出錯。回傳記錄的筆數。"""
    if not warnings:
        return 0
    import time
    lines = [time.strftime('%H:%M:%S ') + f'啟動檢查：測試用的檔案刪不掉，留在 {path}（{type(err).__name__}）'
             for path, err in warnings]
    try:
        with open(data('log.txt'), 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    except Exception:
        pass
    if sys.stderr is not None:
        try:
            print('\n'.join(lines), file=sys.stderr)
        except Exception:
            pass
    return len(lines)


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == '--tm-check':    # temporal.py 的子行程：在顯示卡上試跑時間序列模型
        import temporal
        temporal.check_main(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if len(sys.argv) >= 3 and sys.argv[1] == '--winml-eps':   # accel.py 的子行程：查 MIGraphX 的位置
        import accel
        accel.helper(sys.argv[2])
        return
    _utf8_console()
    import unblock
    global _unblocked                          # 下載的 zip 解壓縮後 DLL 被 Windows 擋住：先解除，不然介面起不來
    if getattr(sys, 'frozen', False) and not os.environ.get('DESKTOPFBT_KEEP_BLOCKED'):      # 這個環境變數只給建置時的對照測試用
        _unblocked = unblock.clear(os.path.dirname(os.path.abspath(sys.executable)))
    import startup_check                       # 一開始就必須能寫入的地方不能寫：開介面之前就說清楚並結束（不跳 traceback）
    bad, warnings = startup_check.check(data())
    if bad:
        startup_check.report(bad, data(), interactive='--selftest' not in sys.argv)
        sys.exit(startup_check.EXIT_NOT_WRITABLE)
    log_startup_warnings(warnings)             # 刪不掉的測試檔：記錄下來（不阻擋啟動）
    if '--selftest' in sys.argv:
        if not selftest(warnings):               # 結果寫在 selftest.txt；有問題時以 3 結束（不跳視窗），正常時照舊以 0 結束
            sys.exit(EXIT_SELFTEST_FAILED)
        return
    engine = None
    try:
        import webview

        from engine import Engine
        engine = Engine()
        api = Api(engine)
        window = webview.create_window(
            'デスクトップFBT', url=resource('web', 'index.html'), js_api=api,
            width=1180, height=760, min_size=(900, 620), background_color='#14161B')

        api._win.append(window)

        def on_closed():
            engine.shutdown()
        window.events.closed += on_closed
        webview.start(gui='edgechromium' if sys.platform == 'win32' else None, debug='--debug' in sys.argv)
    except Exception as e:
        if not unblock.is_blocked_error(e):
            raise
        write_crash(traceback.format_exc())        # 介面的元件載不進來（多半是 DLL 被 Windows 擋住）：說清楚怎麼解，不要只跳一串英文
        if engine is not None:
            engine.shutdown()
        unblock.report()
        sys.exit(unblock.EXIT_BLOCKED)
    engine.shutdown()


def write_crash(text):
    """把錯誤寫進 crash.txt；程式資料夾不能寫入時改寫到暫存資料夾。本身絕不再出錯。回傳寫入的路徑或 None。"""
    import tempfile
    for p in (data('crash.txt'), os.path.join(tempfile.gettempdir(), 'DesktopFBT_crash.txt')):
        try:
            with open(p, 'w', encoding='utf-8') as f:
                f.write(text)
            return p
        except Exception:
            continue
    return None


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write_crash(traceback.format_exc())
        raise
