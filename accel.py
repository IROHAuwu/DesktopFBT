"""AMD MIGraphX 加速（透過 Windows ML 取得）。

Windows ML 的執行模組由系統下載管理；取得路徑要用 WinRT，和 onnxruntime 放在同一個行程會互相干擾，
所以在子行程（DesktopFBT.exe --winml-eps <輸出檔>）裡查，主程式只拿路徑註冊。
需要 Windows 11 24H2 以上、Windows App SDK Runtime（沒有時系統會跳出安裝提示）、相容的 AMD 顯卡驅動。
"""
import json
import os
import subprocess
import sys
import tempfile
import threading

EP = 'MIGraphXExecutionProvider'
VENDORS = {'1002': 'amd', '1022': 'amd', '10DE': 'nvidia', '8086': 'intel', '5143': 'qualcomm'}


def gpus():
    """這台電腦的顯示卡：[(廠商, 名稱)]，廠商是 'amd' / 'nvidia' / 'intel' / 'qualcomm' / 'other'。查不到時回傳 []。"""
    if sys.platform != 'win32':
        return []
    try:
        import ctypes
        import re
        from ctypes import wintypes

        class DD(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('DeviceName', wintypes.WCHAR * 32), ('DeviceString', wintypes.WCHAR * 128),
                        ('StateFlags', wintypes.DWORD), ('DeviceID', wintypes.WCHAR * 128), ('DeviceKey', wintypes.WCHAR * 128)]
        out, i = [], 0
        while i < 32:
            d = DD()
            d.cb = ctypes.sizeof(DD)
            if not ctypes.windll.user32.EnumDisplayDevicesW(None, i, ctypes.byref(d), 0):
                break
            i += 1
            m = re.search(r'VEN_([0-9A-Fa-f]{4})', d.DeviceID or '')
            if not m:
                continue                                   # 遠端桌面、虛擬顯示器等
            item = (VENDORS.get(m.group(1).upper(), 'other'), d.DeviceString)
            if item not in out:
                out.append(item)
        return out
    except Exception:
        return []


def wanted(found=None):
    """要不要向系統查 MIGraphX：有 AMD 顯示卡，或查不出這台有什麼顯示卡時才查。
    其他廠牌（NVIDIA、Intel）用 DirectML；不查的話就不會跳出安裝 Windows App SDK 的提示，也不用等。"""
    found = gpus() if found is None else found
    return not found or any(v == 'amd' for v, _ in found)
_lock = threading.Lock()
_state = {'done': False, 'ok': False, 'error': None}


def helper(out_path):
    """子行程：找出（必要時下載）MIGraphX 執行模組，把 {名稱: 路徑} 寫到 out_path。"""
    res = {'providers': {}, 'error': None}
    try:
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions, initialize)
        import winui3.microsoft.windows.ai.machinelearning as winml
        with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
            catalog = winml.ExecutionProviderCatalog.get_default()
            for p in catalog.find_all_providers():
                if p.name != EP:
                    continue
                if p.ready_state != winml.ExecutionProviderReadyState.READY:
                    r = p.ensure_ready_async().get()
                    if r.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                        res['error'] = f'ensure_ready: {r.status} {getattr(r, "diagnostic_text", "")}'
                        continue
                res['providers'][p.name] = p.library_path
        if not res['providers'] and not res['error']:
            res['error'] = 'not offered for this PC'
    except Exception as e:
        res['error'] = f'{type(e).__name__}: {e}'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)


def _run_helper(timeout):
    fd, out = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, '--winml-eps', out]
        else:
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py'), '--winml-eps', out]
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        try:
            subprocess.run(cmd, timeout=timeout, creationflags=flags, check=False)
        except subprocess.TimeoutExpired:
            return {'providers': {}, 'error': 'timeout (Windows App SDK Runtime prompt not answered?)'}
        with open(out, encoding='utf-8') as f:
            return json.load(f)
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def ensure(timeout=240):
    """註冊 MIGraphX（整個程式只做一次）。回傳是否可以用。
    status()['done'] 在「做完」之後才是 True：正在查詢／下載時是 False，介面才知道要顯示「準備中」。"""
    with _lock:
        if _state['done']:
            return _state['ok']
        try:
            if sys.platform != 'win32':
                _state['error'] = 'windows only'
                return False
            if not wanted():
                _state['error'] = 'no AMD GPU (' + ', '.join(n for _, n in gpus()) + ')'
                return False
            import onnxruntime as ort
            if not hasattr(ort, 'register_execution_provider_library'):
                raise RuntimeError('onnxruntime too old')
            res = _run_helper(timeout)
            path = res['providers'].get(EP)
            if not path:
                raise RuntimeError(res.get('error') or 'not available')
            ort.register_execution_provider_library(EP, path)
            if not devices():
                raise RuntimeError('no device')
            _state['ok'] = True
        except Exception as e:
            _state['error'] = f'{type(e).__name__}: {e}' if not isinstance(e, RuntimeError) else str(e)
        finally:
            _state['done'] = True
        return _state['ok']


def status():
    return dict(_state)


def devices():
    import onnxruntime as ort
    try:
        return [d for d in ort.get_ep_devices() if d.ep_name == EP]
    except Exception:
        return []


def session_options(so, cache_dir=None):
    """把 MIGraphX 加進 SessionOptions。模型編譯結果存在 cache_dir（第二次開始就不用再編譯）。"""
    devs = devices()
    if not devs:
        raise RuntimeError('MIGraphX not registered')
    opts = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault('ORT_MIGRAPHX_MODEL_CACHE_PATH', cache_dir)
        os.environ.setdefault('ORT_MIGRAPHX_CACHE_PATH', cache_dir)
        opts['migraphx_model_cache_dir'] = cache_dir
    try:
        so.add_provider_for_devices(devs, opts)
    except Exception:
        so.add_provider_for_devices(devs, {})
    return so


def bench(model_path, cache_path, batch=4, tag=''):
    """在這台電腦上實際比較 MIGraphX 和 DirectML 跑姿勢模型的速度（毫秒／批），結果存起來只做一次。
    回傳 {'migraphx': ms 或 None, 'dml': ms 或 None}。"""
    key = f'{tag}|{os.path.getsize(model_path)}|{os.path.basename(model_path)}'
    try:
        with open(cache_path, encoding='utf-8') as f:
            c = json.load(f)
        if c.get('key') == key:
            return c['ms']
    except Exception:
        pass
    import time

    import numpy as np
    import onnxruntime as ort

    import pose as P
    P._patch_onnxruntime()
    x = np.zeros((batch, 3, 384, 288), np.float32)
    ms = {}
    for dev in ('migraphx', 'dml'):
        try:
            sess = ort.InferenceSession(model_path, providers=[P.PROVIDER[dev]])
            name = sess.get_inputs()[0].name
            for _ in range(2):
                P.run(sess, None, {name: x})
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                P.run(sess, None, {name: x})
                ts.append(time.perf_counter() - t)
            ms[dev] = 1000 * float(np.median(ts))
            del sess
        except Exception:
            ms[dev] = None
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'key': key, 'ms': ms}, f)
    except OSError:
        pass
    return ms
