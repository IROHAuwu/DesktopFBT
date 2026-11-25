"""手機直連：PC 開一個區網內的 HTTPS 伺服器，手機掃 QR code 用瀏覽器開網頁，就能把相機畫面傳過來。

- 不需要在手機安裝 App（iPhone Safari、Android Chrome 都可以）
- 手機瀏覽器只在 HTTPS 下允許網頁使用相機，所以用自簽憑證（第一次要按「繼續前往／瀏覽此網站」）
- 畫面以 JPEG 用 HTTP POST 傳送（Safari 對自簽憑證的 WebSocket 支援不穩，POST 最可靠）
"""
import asyncio
import datetime
import ipaddress
import os
import socket
import ssl
import threading
import time
import uuid

import cv2
import numpy as np

from paths import data, resource

# 鏡頭 → 水平視角（度，橫放 16:9 的大略值），給 3D 估計當初始值
LENS_HFOV = {'wide': 66.0, 'ultra': 104.0, 'front': 66.0}


# ---------------------------------------------------------------- 網路與憑證
def lan_ips():
    """這台電腦在區網的 IPv4 位址（最可能的排第一個）。"""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))          # 不會真的送出封包，只是讓系統選網卡
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    good = [ip for ip in ips if not ip.startswith(('127.', '169.254.', '0.'))]
    return good or ['127.0.0.1']


CERT_ERROR = 'ERR_PHONE_CERT'


class CertError(Exception):
    """憑證讀寫或檢查失敗。訊息只有例外類型與檔名（不含路徑或憑證內容），寫進 log.txt；介面只顯示 ERR_PHONE_CERT。"""


def _pair_loads(cert_p, key_p):
    """這組憑證與金鑰能不能一起用（金鑰和憑證相符）。檔案讀不到時丟出 OSError；內容不對時回傳 False。"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        ctx.load_cert_chain(cert_p, key_p)
        return True
    except ssl.SSLError:                      # SSLError 也是 OSError：先處理「內容不對」
        return False


def _stage(folder, data_bytes):
    """在 certs 資料夾建立唯一的暫存檔並寫入。"""
    p = os.path.join(folder, f'.desktopfbt-cert-{uuid.uuid4().hex}.tmp')
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    try:
        os.write(fd, data_bytes)
    finally:
        os.close(fd)
    return p


def ensure_cert(ips):
    """建立（或在 IP 改變、檔案損壞、憑證和金鑰不相符時重建）自簽憑證。回傳 (cert_path, key_path)。
    tools/dev/cert_pair_contract.md：只重用「能一起載入」的一組；重建時先寫暫存檔、檢查後才換上；
    讀不到既有檔案（權限、被鎖住）時不覆蓋，丟出 CertError。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    d = data('certs')
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise CertError(f'certs 資料夾無法建立（{type(e).__name__}）') from None
    cert_p, key_p = os.path.join(d, 'cert.pem'), os.path.join(d, 'key.pem')
    if os.path.exists(cert_p) and os.path.exists(key_p):
        try:
            with open(cert_p, 'rb') as f:
                raw = f.read()
        except OSError as e:
            raise CertError(f'cert.pem 讀不到（{type(e).__name__}）') from None
        try:
            cert = x509.load_pem_x509_certificate(raw)
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            have = {str(v) for v in san.get_values_for_type(x509.IPAddress)}
            usable = set(ips) <= have and cert.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc)
        except Exception:
            usable = False                    # 損壞：重建
        if usable:
            try:
                if _pair_loads(cert_p, key_p):
                    return cert_p, key_p
            except OSError as e:
                raise CertError(f'key.pem 讀不到（{type(e).__name__}）') from None
            # 金鑰和憑證不相符（例如上次重建寫到一半）：重建
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'DesktopFBT (local)')])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.DNSName('localhost')] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=800))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .sign(key, hashes.SHA256()))
    key_b = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    cert_b = cert.public_bytes(serialization.Encoding.PEM)
    temps = []
    try:
        try:
            temps.append(_stage(d, cert_b))
            temps.append(_stage(d, key_b))
        except OSError as e:
            raise CertError(f'certs 資料夾無法寫入（{type(e).__name__}）') from None
        tmp_c, tmp_k = temps
        if not _pair_loads(tmp_c, tmp_k):
            raise CertError('新產生的憑證檢查失敗')
        old_cert = None
        try:
            with open(cert_p, 'rb') as f:
                old_cert = f.read()           # 第二步失敗時用來還原（最多一次）
        except OSError:
            pass
        try:
            os.replace(tmp_c, cert_p)         # 第一步失敗：原本的一組完全沒動
        except OSError as e:
            raise CertError(f'cert.pem 無法更新（{type(e).__name__}）') from None
        try:
            os.replace(tmp_k, key_p)
        except OSError as e:
            if old_cert is not None:          # 盡量還原舊的 cert.pem；還原不了時，下次啟動會發現不相符並重建
                try:
                    back = _stage(d, old_cert)
                    temps.append(back)
                    os.replace(back, cert_p)
                except OSError:
                    pass
            raise CertError(f'key.pem 無法更新（{type(e).__name__}）') from None
        return cert_p, key_p
    finally:
        for t in temps:
            try:
                if os.path.exists(t):
                    os.remove(t)
            except OSError:
                pass


def qr_svg(text):
    import segno
    return segno.make(text, error='m').svg_data_uri(scale=6, border=2, dark='#15161f', light='#ffffff')


# ---------------------------------------------------------------- 伺服器
class _Slot:
    def __init__(self):
        self.jpeg = None
        self.ts = 0.0
        self.id = 0
        self.fps = 0.0
        self.name = ''
        self.lens = ''
        self.size = None
        self.last = 0.0
        self.client = None          # 目前在傳的手機（避免兩支手機搶同一個位置）
        self.offsets = []           # (到達時間 - 手機拍攝時間)，取最小值當作時鐘差
        self._decoded_id = -1
        self._decoded = None


class PhoneHub:
    """一支手機的畫面。同時只收一支：另一支連進來時回 busy。"""

    def __init__(self, port=8443):
        self.port = port
        self.fmt = dict(width=1280, height=720, fps=30, quality=0.75)
        self._lock = threading.Lock()
        self._thread = None
        self._loop = None
        self._runner = None
        self.error = None
        self.error_detail = ''
        self.ips = []
        self.slot = _Slot()

    # -------------------------------------------------------- 啟動
    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive() and self.error is None

    def start(self):
        if self.running:
            return True
        self.error = None
        ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, args=(ready,), daemon=True)
        self._thread.start()
        ready.wait(10)
        return self.running

    def _serve(self, ready):
        from aiohttp import web
        try:
            self.ips = lan_ips()
            cert, key = ensure_cert(self.ips)
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ctx.load_cert_chain(cert, key)
            app = web.Application(client_max_size=8 * 1024 * 1024)
            app.router.add_get('/', self._page)
            app.router.add_post('/frame', self._frame)
            app.router.add_post('/bye', self._bye)
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._runner = web.AppRunner(app, access_log=None)
            self._loop.run_until_complete(self._runner.setup())
            last_err = None
            for port in range(self.port, self.port + 8):          # 被占用時往後找
                try:
                    site = web.TCPSite(self._runner, '0.0.0.0', port, ssl_context=ctx)
                    self._loop.run_until_complete(site.start())
                    self.port = port
                    last_err = None
                    break
                except OSError as e:
                    last_err = e
            if last_err:
                raise last_err
        except CertError as e:
            self.error, self.error_detail = CERT_ERROR, str(e)
            ready.set()
            return
        except Exception as e:
            self.error = 'ERR_PHONE_SERVER'
            self.error_detail = f'{type(e).__name__}: {e}'
            ready.set()
            return
        ready.set()
        self._loop.run_forever()

    def stop(self):
        if self._loop and self._loop.is_running():
            async def _shutdown():
                await self._runner.cleanup()
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            try:
                fut.result(5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    # -------------------------------------------------------- HTTP
    async def _page(self, request):
        from aiohttp import web
        with open(resource('web', 'phone.html'), encoding='utf-8') as f:
            html = f.read()
        return web.Response(text=html, content_type='text/html', headers={'Cache-Control': 'no-store'})

    async def _frame(self, request):
        from aiohttp import web
        client = request.query.get('client', '')
        body = await request.read()
        now = time.perf_counter()
        s = self.slot
        with self._lock:
            if s.client and s.client != client and now - s.last < 3.0:
                return web.json_response({'busy': True, **self.fmt})   # 另一支手機正在用這個位置
            if s.client != client:
                s.offsets = []
            s.client = client
            # 拍攝時間：手機時鐘 → PC 時鐘。時鐘差 = 最近 (到達 - 拍攝) 的最小值（網路最快的那一次）
            cap = request.headers.get('X-Capture')
            ts = now
            if cap:
                try:
                    c = float(cap) / 1000.0
                    s.offsets.append(now - c)
                    if len(s.offsets) > 300:
                        s.offsets = s.offsets[-300:]
                    ts = min(now, c + min(s.offsets))
                except ValueError:
                    pass
            if s.last:
                dt = now - s.last
                if dt > 0:
                    s.fps = 0.85 * s.fps + 0.15 * (1.0 / dt) if s.fps else 1.0 / dt
            s.jpeg = body
            s.ts = ts
            s.id += 1
            s.last = now
            s.name = request.headers.get('X-Device', s.name or 'Phone')
            s.lens = request.headers.get('X-Lens', s.lens)
            w, h = request.headers.get('X-Width'), request.headers.get('X-Height')
            if w and h:
                s.size = (int(w), int(h))
        return web.json_response({'ok': True, **self.fmt})

    async def _bye(self, request):
        from aiohttp import web
        client = request.query.get('client', '')
        s = self.slot
        with self._lock:
            if not client or client == s.client:
                s.last = 0.0
                s.client = None
                s.fps = 0.0
        return web.json_response({'ok': True})

    # -------------------------------------------------------- 給程式使用
    def set_format(self, width, height, fps):
        self.fmt.update(width=int(width), height=int(height), fps=int(fps))

    def connected(self):
        s = self.slot
        return s.last > 0 and time.perf_counter() - s.last < 2.0

    def latest(self):
        """(BGR, 時間, 幀號)；JPEG 在需要時才解碼。"""
        s = self.slot
        with self._lock:
            jpeg, ts, fid = s.jpeg, s.ts, s.id
            if fid == s._decoded_id:
                return s._decoded, ts, fid
        if jpeg is None:
            return None, 0.0, 0
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        with self._lock:
            s._decoded_id, s._decoded = fid, img
        return img, ts, fid

    def url(self):
        ip = self.ips[0] if self.ips else '127.0.0.1'
        return f'https://{ip}:{self.port}/'

    def info(self, with_qr=False):
        s = self.slot
        on = self.connected()
        out = dict(running=self.running, error=self.error, port=self.port, ips=self.ips, connected=on,
                   name=s.name, lens=s.lens, fps=round(s.fps, 1) if on else 0.0,
                   size=list(s.size) if s.size else None)
        if self.running:
            out['url'] = self.url()
            if with_qr:
                out['qr'] = qr_svg(out['url'])
        return out


def fit_size(img, size):
    """縮放成固定大小；長寬比不同（手機轉向）就等比縮放後置中補黑邊。"""
    W, H = size
    h, w = img.shape[:2]
    if (w, h) == (W, H):
        return img
    import cv2
    if abs(w / h - W / H) < 0.02:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    f = min(W / w, H / h)
    nw, nh = max(1, round(w * f)), max(1, round(h * f))
    out = np.zeros((H, W, 3), img.dtype)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


class PhoneSource:
    """手機畫面當即時來源（和 WindowSource 一樣的介面）。伺服器由 Engine 管理，這裡不關。"""

    def __init__(self, hub, timeout=None, should_stop=None):
        self.hub = hub
        t0 = time.perf_counter()
        while self.hub.latest()[0] is None or not self.hub.connected():      # 等手機開始傳
            if should_stop is not None and should_stop():
                raise InterruptedError('stopped')
            if timeout is not None and time.perf_counter() - t0 > timeout:
                raise RuntimeError('ERR_PHONE_TIMEOUT')
            time.sleep(0.05)
        f = self.hub.latest()[0]
        # 手機會因為 Wi‑Fi 慢自動降解析度：對外固定一個大小（長邊 = 設定寬度），不然相機內參會對不上
        h, w = f.shape[:2]
        L = max(int(self.hub.fmt.get('width', 1280)), w, h)
        self.size = (L, round(L * h / w)) if w >= h else (round(L * w / h), L)

    @property
    def fps(self):
        return self.hub.slot.fps if self.hub.connected() else 0.0

    @property
    def hfov(self):
        return LENS_HFOV.get(self.hub.slot.lens)

    @property
    def name(self):
        return self.hub.slot.name or 'Phone'

    def read(self, last_fid=-1, timeout=1.0):
        t0 = time.perf_counter()
        while True:
            img, ts, fid = self.hub.latest()
            if img is not None and fid != last_fid:
                return fit_size(img, self.size), ts, fid
            if time.perf_counter() - t0 > timeout:
                if not self.hub.connected():
                    raise RuntimeError('ERR_PHONE_LOST')
                return None
            time.sleep(0.003)

    def stop(self):
        pass
