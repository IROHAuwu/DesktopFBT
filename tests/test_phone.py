"""手機直連（掃 QR code）：模擬手機用 HTTPS POST 傳 JPEG，確認收得到畫面、另一支手機搶不到、網頁打得開，
並且當成即時來源讀得到。也測相機來源（用影片檔代替相機）。執行：python tests/test_phone.py
"""
import json
import os
import ssl
import sys
import tempfile
import threading
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = tempfile.mkdtemp()
import paths  # noqa: E402
paths.base_dir = lambda: TMP
import phone_server as PS  # noqa: E402
import video as V  # noqa: E402

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=body, method='POST', headers=headers or {})
    with urllib.request.urlopen(req, context=CTX, timeout=5) as r:
        return json.loads(r.read())


def jpeg(color):
    img = np.full((720, 1280, 3), color, np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


def test_phone():
    hub = PS.PhoneHub(18443)
    assert hub.start(), (hub.error, hub.error_detail)
    info = hub.info(with_qr=True)
    assert info['url'].startswith('https://') and info['qr'].startswith('data:image/svg')
    base = f'https://127.0.0.1:{hub.port}'
    with urllib.request.urlopen(base + '/', context=CTX, timeout=5) as r:
        page = r.read().decode()
    assert 'デスクトップFBT' in page and 'getUserMedia' in page

    got = []

    def consumer():                      # 手機還沒連上就開始等（和按下「開始」一樣）
        src = PS.PhoneSource(hub, timeout=10)
        last = -1
        for _ in range(20):
            r = src.read(last, timeout=2)
            if r is not None:
                got.append(r)
                last = r[2]
        got.append(('size', src.size, src.hfov))
    th = threading.Thread(target=consumer)
    th.start()
    time.sleep(0.3)
    a = jpeg((200, 120, 40))
    t_end = time.time() + 1.5
    while time.time() < t_end:
        r = post(f'{base}/frame?client=phoneA', a, {'X-Width': '1280', 'X-Height': '720', 'X-Device': 'iPhone', 'X-Lens': 'wide'})
        time.sleep(1 / 30)
    intruder = post(f'{base}/frame?client=other', a)
    th.join(5)
    assert r['ok'] and r['width'] == 1280 and r['fps'] == 30
    assert intruder.get('busy') is True
    frames = [g for g in got if not isinstance(g[0], str)]
    assert len(frames) >= 15 and frames[0][0].shape == (720, 1280, 3) and abs(int(frames[0][0][5, 5, 0]) - 200) < 10
    assert got[-1] == ('size', (1280, 720), PS.LENS_HFOV['wide'])
    info = hub.info()
    print(f"  手機：收到 {len(frames)} 幀，{info['fps']} FPS，{info['name']}；另一支手機被擋下")
    inf = hub.info(with_qr=True)
    assert inf['connected'] and inf['url'].endswith(f':{hub.port}/') and inf['qr']
    post(f'{base}/bye?client=phoneA', b'')
    assert not hub.connected()
    hub.stop()


def test_camera_source():
    path = os.path.join(TMP, 'cam.avi')
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))
    for i in range(60):
        w.write(np.full((240, 320, 3), i * 4, np.uint8))
    w.release()
    src = V.CameraSource(path)                         # 網址也是走這條（cv2.VideoCapture）
    last, n = -1, 0
    for _ in range(10):
        r = src.read(last, timeout=1)
        if r is None:
            break
        last, n = r[2], n + 1
    src.stop()
    assert src.size == (320, 240) and n >= 3
    print(f'  相機來源：{src.size}，讀到 {n} 幀')


if __name__ == '__main__':
    test_phone()
    test_camera_source()
    print('全部通過')
