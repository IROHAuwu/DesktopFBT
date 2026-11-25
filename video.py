"""影片來源：本機影片檔、擷取 Windows 視窗（例如正在播放 YouTube 的瀏覽器）。"""
import os
import shutil
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

VIDEO_EXT = ('.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.wmv', '.flv', '.mpg', '.mpeg')


WARNINGS = []          # 不致命的提醒（例如暫存複本刪不掉）；引擎取出後寫進 log.txt
_COPY_PREFIX = 'desktopfbt_'


def take_warnings():
    out = WARNINGS[:]
    del WARNINGS[:]
    return out


def _short_ascii(path):
    """Windows 的 ASCII 短檔名；沒有（或不是 ASCII）時回傳 None。"""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024) and buf.value.isascii():
            return buf.value
    except Exception:
        pass
    return None


def _owned_temp_dir(d):
    """只有「這個程式剛建立、位在暫存資料夾正下方、名稱是 desktopfbt_ 開頭」的資料夾才可以整個刪除。"""
    try:
        real = os.path.realpath(d)
        parent = os.path.normcase(os.path.dirname(real))
        return (os.path.isdir(real) and parent == os.path.normcase(os.path.realpath(tempfile.gettempdir()))
                and os.path.basename(real).startswith(_COPY_PREFIX))
    except Exception:
        return False


def _remove_owned(d):
    """刪除自己建立的暫存資料夾；不丟出例外，失敗只記一筆提醒。"""
    if not _owned_temp_dir(d):
        if os.path.exists(d):
            WARNINGS.append(f'影片暫存複本的資料夾不是預期的位置，沒有刪除：{os.path.basename(d)}')
        return
    try:
        shutil.rmtree(d)
    except Exception as e:
        WARNINGS.append(f'影片暫存複本刪不掉（不影響使用）：{os.path.basename(d)}（{type(e).__name__}）')


def _open_path(path):
    """Windows 上 OpenCV 打不開含中文、日文的路徑：改用短檔名，不行就複製到暫存資料夾。
    回傳 (要開啟的路徑, 這次建立的暫存資料夾或 None)；呼叫的人用完要刪掉那個資料夾。"""
    if sys.platform != 'win32' or path.isascii():
        return path, None
    short = _short_ascii(path)
    if short:
        return short, None
    d = tempfile.mkdtemp(prefix=_COPY_PREFIX)
    try:
        dst = os.path.join(d, 'video' + os.path.splitext(path)[1].lower())
        shutil.copyfile(path, dst)
    except BaseException:
        _remove_owned(d)                     # 複製到一半失敗：不留下半個檔案
        raise
    return dst, d


def read_image(path):
    """可以讀中文路徑的 imread。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('ERR_IMAGE_OPEN')
    return img


IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


class VideoFile:
    """依影片本身的時間播放（跟不上時跳幀）。read() → (BGR, 影片內時間秒, 幀號) 或 None（播完）。"""

    def __init__(self, path, loop=True):
        self.path = path
        self.cap = None
        self._stopped = False
        open_path, self._tmp_dir = _open_path(path)          # 暫存複本（如果有）由這個物件負責刪除
        try:
            self.cap = cv2.VideoCapture(open_path)
            if not self.cap.isOpened():
                raise RuntimeError('ERR_VIDEO_OPEN')
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
            if self.fps > 240:
                self.fps = 30.0
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except BaseException:
            self.stop()
            raise
        self.loop = loop
        self.idx = 0
        self.t0 = None

    @property
    def duration(self):
        return self.frames / self.fps if self.frames else 0.0

    def read_next(self):
        """不管時間，讀下一幀（匯出用）。"""
        ok, img = self.cap.read()
        if not ok:
            return None
        i = self.idx
        self.idx += 1
        return img, i / self.fps, i

    def read(self):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now - self.idx / self.fps
        target = int((now - self.t0) * self.fps)
        while self.idx < target - 1:                    # 處理比播放慢：直接跳過
            if not self.cap.grab():
                break
            self.idx += 1
        ahead = (self.idx / self.fps) - (now - self.t0)
        if ahead > 0:
            time.sleep(min(ahead, 0.1))
        r = self.read_next()
        if r is None and self.loop and self.idx > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.idx = 0
            self.t0 = time.perf_counter()
            r = self.read_next()
        return r

    def stop(self):
        """先放開影片（Windows 開著的檔案刪不掉），再刪這次建立的暫存複本。可以重複呼叫，不會丟出例外。"""
        if not self._stopped and self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                WARNINGS.append(f'影片關閉時發生錯誤：{type(e).__name__}')
        self._stopped = True
        d, self._tmp_dir = self._tmp_dir, None
        if d:
            _remove_owned(d)


# ------------------------------------------------------------------ Windows 視窗擷取
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    _u32 = ctypes.windll.user32
    _g32 = ctypes.windll.gdi32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # 視窗座標用實際像素
    except Exception:
        try:
            _u32.SetProcessDPIAware()
        except Exception:
            pass

    class _BIH(ctypes.Structure):
        _fields_ = [('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
                    ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD),
                    ('biCompression', wintypes.DWORD), ('biSizeImage', wintypes.DWORD),
                    ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
                    ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)]

    _ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _u32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _u32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _u32.IsWindowVisible.argtypes = [wintypes.HWND]
    _u32.IsIconic.argtypes = [wintypes.HWND]
    _u32.IsWindow.argtypes = [wintypes.HWND]
    _u32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _u32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _u32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _u32.GetWindowDC.argtypes = [wintypes.HWND]
    _u32.GetWindowDC.restype = wintypes.HDC
    _u32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _u32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    _g32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    _g32.CreateCompatibleDC.restype = wintypes.HDC
    _g32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    _g32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    _g32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    _g32.SelectObject.restype = wintypes.HGDIOBJ
    _g32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    _g32.DeleteDC.argtypes = [wintypes.HDC]
    _g32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                               ctypes.c_void_p, ctypes.POINTER(_BIH), wintypes.UINT]
    _g32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]

    def list_windows(exclude=('デスクトップFBT', 'DesktopFBT')):
        out = []

        def cb(hwnd, _):
            if not _u32.IsWindowVisible(hwnd) or _u32.IsIconic(hwnd):
                return True
            n = _u32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            _u32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.strip()
            r = wintypes.RECT()
            _u32.GetClientRect(hwnd, ctypes.byref(r))
            if not title or title in exclude or r.right < 200 or r.bottom < 150:
                return True
            if title in ('Program Manager', 'Windows Input Experience', 'Settings'):
                return True
            out.append({'id': int(hwnd), 'title': title, 'size': [r.right, r.bottom]})
            return True
        _u32.EnumWindows(_ENUM(cb), 0)
        return out

    def grab_window(hwnd):
        """擷取視窗的內容區（不含標題列）。被別的視窗擋住也可以，但不能縮到最小。"""
        hwnd = wintypes.HWND(hwnd)
        if not _u32.IsWindow(hwnd):
            raise RuntimeError('ERR_WINDOW_GONE')
        if _u32.IsIconic(hwnd):
            return None
        wr, cr = wintypes.RECT(), wintypes.RECT()
        _u32.GetWindowRect(hwnd, ctypes.byref(wr))
        _u32.GetClientRect(hwnd, ctypes.byref(cr))
        pt = wintypes.POINT(0, 0)
        _u32.ClientToScreen(hwnd, ctypes.byref(pt))
        W, H = wr.right - wr.left, wr.bottom - wr.top
        if W <= 0 or H <= 0:
            return None
        hdc = _u32.GetWindowDC(hwnd)
        mdc = _g32.CreateCompatibleDC(hdc)
        bmp = _g32.CreateCompatibleBitmap(hdc, W, H)
        old = _g32.SelectObject(mdc, bmp)
        try:
            # PW_RENDERFULLCONTENT (2)：Chrome / Edge 等硬體加速的視窗也拍得到
            if not _u32.PrintWindow(hwnd, mdc, 2):
                _g32.BitBlt(mdc, 0, 0, W, H, hdc, 0, 0, 0x00CC0020)
            bih = _BIH()
            bih.biSize = ctypes.sizeof(_BIH)
            bih.biWidth, bih.biHeight, bih.biPlanes, bih.biBitCount = W, -H, 1, 32
            buf = np.empty((H, W, 4), np.uint8)
            _g32.GetDIBits(mdc, bmp, 0, H, buf.ctypes.data, ctypes.byref(bih), 0)
        finally:
            _g32.SelectObject(mdc, old)
            _g32.DeleteObject(bmp)
            _g32.DeleteDC(mdc)
            _u32.ReleaseDC(hwnd, hdc)
        ox, oy = pt.x - wr.left, pt.y - wr.top
        img = buf[oy:oy + cr.bottom, ox:ox + cr.right, :3]
        return np.ascontiguousarray(img)

    def client_rect(hwnd):
        """視窗內容區在螢幕上的位置 (left, top, right, bottom)；縮到最小時 None。"""
        hwnd = wintypes.HWND(hwnd)
        if not _u32.IsWindow(hwnd) or _u32.IsIconic(hwnd):
            return None
        cr = wintypes.RECT()
        _u32.GetClientRect(hwnd, ctypes.byref(cr))
        pt = wintypes.POINT(0, 0)
        _u32.ClientToScreen(hwnd, ctypes.byref(pt))
        return pt.x, pt.y, pt.x + cr.right, pt.y + cr.bottom

    def primary_size():
        return _u32.GetSystemMetrics(0), _u32.GetSystemMetrics(1)
else:
    def list_windows(exclude=()):
        return []

    def grab_window(hwnd):
        raise RuntimeError('ERR_WINDOW_UNSUPPORTED')

    def client_rect(hwnd):
        return None

    def primary_size():
        return 0, 0


def crop_black(img, thr=12):
    """去掉影片周圍的黑邊（YouTube 播放器常見），回傳 (裁切後影像, (x0, y0))。"""
    g = img.max(axis=2)
    rows = np.where(g.max(axis=1) > thr)[0]
    cols = np.where(g.max(axis=0) > thr)[0]
    if len(rows) < 50 or len(cols) < 50:
        return img, (0, 0)
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return img[y0:y1, x0:x1], (int(x0), int(y0))


class _FastGrab:
    """DXGI 桌面複製（dxcam）：比 PrintWindow 快很多，60 FPS 沒問題。只能拍主螢幕上看得到的畫面。"""

    _cam = None                     # dxcam 每個螢幕只能建立一個：整個程式共用，不釋放

    def __init__(self):
        if _FastGrab._cam is None:
            import dxcam
            _FastGrab._cam = dxcam.create(output_color='BGR')
        self.cam = _FastGrab._cam

    def grab(self, hwnd):
        r = client_rect(hwnd)
        if r is None:
            return None
        W, H = primary_size()
        l, t, rr, b = r
        if l < 0 or t < 0 or rr > W or b > H or rr - l < 64 or b - t < 64:
            raise ValueError('outside')               # 不在主螢幕上：改用 PrintWindow
        img = self.cam.grab(region=(l, t, rr, b))
        if img is None:                                # 畫面沒有變化
            return None
        return np.ascontiguousarray(img)

    def close(self):
        pass


class WindowSource:
    """在背景執行緒持續擷取視窗（預設 60fps）。read() → (BGR, 時間, 幀號)。"""

    def __init__(self, hwnd, fps=60, fast=True):
        self.hwnd = int(hwnd)
        self.period = 1.0 / fps
        self.frame = None
        self.fid = 0
        self.err = None
        self.fps = 0.0
        self.fast = None
        self._stop = threading.Event()
        first = grab_window(self.hwnd)
        if first is None:
            raise RuntimeError('ERR_WINDOW_MINIMIZED')
        self.size = (first.shape[1], first.shape[0])
        self.frame = (first, time.perf_counter(), 0)
        if fast and sys.platform == 'win32':
            try:
                self.fast = _FastGrab()
            except Exception as e:
                WARNINGS.append(f'快速擷取無法使用，改用一般擷取：{type(e).__name__}')
                self.fast = None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _grab(self):
        if self.fast is not None:
            try:
                img = self.fast.grab(self.hwnd)
                if img is not None and (img.shape[1], img.shape[0]) != self.size:
                    self.size = (img.shape[1], img.shape[0])
                return img
            except ValueError:
                pass
            except Exception as e:
                WARNINGS.append(f'快速擷取失敗，改用一般擷取：{type(e).__name__}')
                self.fast.close()
                self.fast = None
        return grab_window(self.hwnd)

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                img = self._grab()
            except Exception as e:
                self.err = str(e) if str(e).startswith('ERR_') else 'ERR_WINDOW_GONE'
                return
            if img is not None:
                self.fid += 1
                self.frame = (img, t, self.fid)
                self.fps = 0.9 * self.fps + 0.1 / max(t - last, 1e-3)
                last = t
            time.sleep(max(0.0, self.period - (time.perf_counter() - t)))

    def read(self, last_fid=-1, timeout=1.0):
        if self.err:
            raise RuntimeError(self.err)
        t0 = time.perf_counter()
        while self.frame is None or self.frame[2] == last_fid:
            if self.err:
                raise RuntimeError(self.err)
            if time.perf_counter() - t0 > timeout:
                return None
            time.sleep(0.002)
        return self.frame

    def stop(self):
        self._stop.set()
        if self.fast is not None:
            self.th.join(timeout=1.0)
            self.fast.close()


def list_cameras(max_probe=6):
    """接在電腦上的相機（含 DroidCam、Iriun、OBS 虛擬相機等外部 App 的虛擬相機）。[{id, name}]"""
    names = []
    if sys.platform == 'win32':
        try:
            from pygrabber.dshow_graph import FilterGraph       # DirectShow 的裝置名稱，順序和 OpenCV 的編號一樣
            names = list(FilterGraph().get_input_devices())
        except Exception:
            names = []
    if names:
        return [dict(id=i, name=n) for i, n in enumerate(names)]
    out = []
    for i in range(max_probe):                                    # 拿不到名稱：一個一個試
        cap = _open_camera(i)
        try:
            if cap is not None and cap.isOpened():
                out.append(dict(id=i, name=f'Camera {i}'))
        finally:
            if cap is not None:
                cap.release()
    return out


def _open_camera(src):
    if isinstance(src, str):
        return cv2.VideoCapture(src)
    if sys.platform == 'win32':
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(int(src), api)
            if cap.isOpened():
                return cap
            cap.release()
        return None
    cap = cv2.VideoCapture(int(src))
    return cap if cap.isOpened() else None


class CameraSource:
    """相機（編號）或網路串流（http://…/video、rtsp://…，例如手機的 IP Webcam App）。
    背景執行緒一直讀最新的一幀。read() → (BGR, 時間, 幀號)，和 WindowSource 一樣。"""

    def __init__(self, src, width=1280, height=720, fps=30):
        self.src = src
        self.cap = _open_camera(src)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('ERR_CAMERA_OPEN')
        if not isinstance(src, str):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   # 高解析度時 USB 頻寬夠
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # 只要最新的一幀
        except Exception:
            pass
        ok, img = False, None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 8.0:
            ok, img = self.cap.read()
            if ok and img is not None:
                break
            time.sleep(0.05)
        if not ok or img is None:
            self.cap.release()
            raise RuntimeError('ERR_CAMERA_OPEN')
        self.size = (img.shape[1], img.shape[0])
        self.fid = 0
        self.frame = (img, time.perf_counter(), 0)
        self.err = None
        self.fps = 0.0
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        last, fails = time.perf_counter(), 0
        while not self._stop.is_set():
            ok, img = self.cap.read()
            t = time.perf_counter()
            if not ok or img is None:
                fails += 1
                if fails > 60:
                    self.err = 'ERR_CAMERA_LOST'
                    return
                time.sleep(0.03)
                continue
            fails = 0
            self.fid += 1
            self.frame = (img, t, self.fid)
            self.size = (img.shape[1], img.shape[0])
            self.fps = 0.9 * self.fps + 0.1 / max(t - last, 1e-3)
            last = t

    def read(self, last_fid=-1, timeout=1.0):
        if self.err:
            raise RuntimeError(self.err)
        t0 = time.perf_counter()
        while self.frame is None or self.frame[2] == last_fid:
            if self.err:
                raise RuntimeError(self.err)
            if time.perf_counter() - t0 > timeout:
                return None
            time.sleep(0.002)
        return self.frame

    def stop(self):
        self._stop.set()
        self.th.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass


def export_dir():
    from paths import data
    d = data('exports')
    os.makedirs(d, exist_ok=True)
    return d
