"""模型下載與損壞恢復的測試（不需要網路、不需要真正的模型、不碰使用者的 models 資料夾）。
用一個手工編碼的小 ONNX 模型（Identity）和本機 loopback HTTP 伺服器（作業系統分配的臨時埠）模擬各種下載失敗；
檔案好不好用真正的 onnxruntime（CPU）開啟判斷。執行：python tests/test_model_download.py
"""
import http.server
import os
import shutil
import socket
import sys
import tempfile
import threading
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
BASE = tempfile.mkdtemp(prefix='desktopfbt_modeltest_')
import paths  # noqa: E402
paths.base_dir = lambda: BASE
import onnxruntime as ort  # noqa: E402
import models  # noqa: E402
import engine as E  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('ok  ' + msg)


# ------------------------------------------------------------ 小 ONNX 模型（Identity，手工編碼 protobuf）
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _f(num, payload):          # length-delimited field
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _v(num, value):            # varint field
    return _varint(num << 3) + _varint(value)


def tiny_onnx(seed=0, op=b'Identity'):
    tensor = _v(1, 1) + _f(2, _f(1, _v(1, 1 + seed)))            # float, shape [1+seed]
    vtype = _f(1, tensor)
    node = _f(1, b'x') + _f(2, b'y') + _f(4, op)
    graph = _f(1, node) + _f(2, b'g') + _f(11, _f(1, b'x') + _f(2, vtype)) + _f(12, _f(1, b'y') + _f(2, vtype))
    return _v(1, 7) + _f(7, graph) + _f(8, _v(2, 13))


GOOD_DET, GOOD_3D = tiny_onnx(0), tiny_onnx(2)
GARBAGE = os.urandom(700)


def zbytes(name, data):
    p = os.path.join(BASE, '_z.zip')
    with zipfile.ZipFile(p, 'w') as zf:
        zf.writestr(name, data)
    return open(p, 'rb').read()


# ------------------------------------------------------------ loopback 伺服器
REQ = []
ROUTES = {}          # path -> (mode, bytes)
HOOK = {'on': None}  # 收到請求時呼叫（測試用：模擬使用者在下載途中按停止）


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def do_GET(self):
        REQ.append(self.path)
        if HOOK['on']:
            HOOK['on'](self.path)
        mode, data = ROUTES.get(self.path, ('404', b''))
        if mode == '404':
            self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
        if mode == 'ok':
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data); return
        cut = data[:max(1, len(data) * 2 // 5)]
        if mode == 'tcl':
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(cut)
        elif mode == 'close':
            self.protocol_version = 'HTTP/1.0'
            self.send_response(200); self.send_header('Connection', 'close'); self.end_headers(); self.wfile.write(cut)
        elif mode == 'chunk':
            self.send_response(200); self.send_header('Transfer-Encoding', 'chunked'); self.end_headers()
            self.wfile.write(b'%x\r\n' % len(cut) + cut + b'\r\n')
        self.wfile.flush()
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


SRV = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
PORT = SRV.server_address[1]
assert PORT not in (8443, 9000) and not (39500 <= PORT <= 39599)
threading.Thread(target=SRV.serve_forever, daemon=True).start()
URL = f'http://127.0.0.1:{PORT}'

DET_ZIP = os.path.basename(models.DET_URL)
DET_ONNX = DET_ZIP[:-4] + '.onnx'
R3D = os.path.basename(models.POSE3D_URL)
models.DET_URL = f'{URL}/p/{DET_ZIP}'
models.MIRROR = f'{URL}/m/'
models.POSE3D_URL = f'{URL}/h/{R3D}'
MD = os.path.join(BASE, 'models')


def reset(det='ok', mirror='ok', r3d='ok', r3d_data=None):
    shutil.rmtree(MD, ignore_errors=True)
    os.makedirs(MD)
    REQ.clear()
    ROUTES.clear()
    good = zbytes(DET_ONNX, GOOD_DET)
    ROUTES[f'/p/{DET_ZIP}'] = ('ok', b'<html>login</html>') if det == 'html' else (det, good)
    ROUTES[f'/m/{DET_ZIP}'] = ('ok', zbytes(DET_ONNX, GARBAGE)) if mirror == 'bad' else (mirror, good)
    ROUTES[f'/h/{R3D}'] = ('ok', b'<html>login</html>') if r3d == 'html' else (r3d, GOOD_3D if r3d_data is None else r3d_data)


def _write(p, data):
    with open(p, 'wb') as f:
        f.write(data)
    return p


def loads(p):
    try:
        ort.InferenceSession(p, providers=['CPUExecutionProvider'])
        return True
    except Exception:
        return False


def leftovers():
    return [f for f in os.listdir(MD) if f.endswith('.part')] if os.path.isdir(MD) else []


def code_of(fn):
    try:
        fn()
        return None
    except Exception as e:
        return str(e)


def download_cases():
    reset()
    out = models.ensure_pose3d()
    check(loads(out['det']) and loads(out['pose3d']) and models.pose3d_ready() and not leftovers(), '正常下載：兩個模型都能開啟')
    n = len(REQ)
    models.ensure_pose3d()
    check(len(REQ) == n, '已經下載過：不再連線')

    for mode in ('404', 'tcl', 'close', 'chunk', 'html'):
        reset(det=mode)
        out = models.ensure_pose3d()
        check(loads(out['det']) and any(r.startswith('/m/') for r in REQ) and not leftovers(), f'主站 {mode}：改用鏡像')

    reset(det='404', mirror='bad')
    err = code_of(models.ensure_pose3d)
    check(err == f'ERR_MODEL_DOWNLOAD:{DET_ONNX}|invalid' and not os.path.exists(os.path.join(MD, DET_ONNX)) and not leftovers(),
          '兩個來源都不是模型：invalid，沒有留下檔案')

    for mode, why in (('404', 'network'), ('tcl', 'incomplete'), ('html', 'invalid')):
        reset(r3d=mode)
        err = code_of(models.ensure_pose3d)
        check(err == f'ERR_MODEL_DOWNLOAD:{R3D}|{why}' and not os.path.exists(os.path.join(MD, R3D)) and not leftovers(),
              f'RTMW3D {mode} → {why}')

    reset()
    REQ.clear()
    stop = {'n': 0}

    def prog(part, frac):
        stop['n'] += 1
        raise E.Stopped()
    try:
        models.ensure_pose3d(prog)
        raise AssertionError
    except E.Stopped:
        pass
    check(not leftovers(), '下載途中停止：不留下 .part')


def engine_cases():
    """模型檔損壞時：刪掉、重新下載一次。GPU 失敗時改用 CPU。"""
    E.P.resolve_device = lambda pref: pref
    opened = []

    class Est:
        def __init__(self, paths_, dev, f, prec='auto'):
            for k in ('det', 'pose3d'):
                ort.InferenceSession(paths_[k], providers=['CPUExecutionProvider'])
            if dev == 'dml':
                raise RuntimeError('no dml')
            opened.append(dev)
            self.precision, self.device, self.tta, self.freq = 'fp32', dev, False, f
    E.Pose3DEstimator = Est

    reset()
    models.ensure_pose3d()
    _write(os.path.join(MD, R3D), GARBAGE)
    eng = E.Engine()
    eng.cfg['pose']['device'] = 'cpu'
    est = eng._estimator()
    check(isinstance(est, Est) and loads(os.path.join(MD, R3D)) and any(r.startswith('/h/') for r in REQ),
          '損壞的 RTMW3D：刪掉後重新下載，載入成功')
    check(eng.get_state(False)['model']['status'] == 'ready', '狀態：ready')

    reset()
    models.ensure_pose3d()
    eng = E.Engine()
    eng.cfg['pose']['device'] = 'dml'
    eng._estimator()
    check(opened[-1] == 'cpu' and eng.get_state(False)['device'] == 'cpu', 'DirectML 失敗：改用 CPU')

    reset(r3d='404')
    eng = E.Engine()
    eng.cfg['pose']['device'] = 'cpu'
    err = code_of(eng._estimator)
    check(err.startswith('ERR_MODEL_DOWNLOAD') and eng.get_state(False)['model']['status'] == 'error', '下載失敗：回報錯誤')


def conversion_cases():
    """可變批次、半精度的轉換：結果和原模型相同（半精度在誤差內），轉不了時回傳 None。"""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        print('skip 轉換（沒有 onnx 套件）')
        return
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.3, (4, 3, 3, 3)).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node('Conv', ['x', 'w'], ['c'], pads=[1, 1, 1, 1]),
         helper.make_node('Shape', ['c'], ['sh']),
         helper.make_node('Reshape', ['c', 'sh'], ['y'])],
        'g', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 8, 6])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 8, 6])],
        [helper.make_tensor('w', TensorProto.FLOAT, w.shape, w.flatten())])
    src = os.path.join(BASE, 'conv.onnx')
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    m.ir_version = 8
    onnx.save(m, src)
    nb, f16 = models.batch_model(src), models.fp16_model(src)
    check(nb and f16 and os.path.exists(nb) and os.path.exists(f16), '轉出可變批次與半精度模型')
    x = rng.normal(0, 1, (3, 3, 8, 6)).astype(np.float32)
    a = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    ref = np.concatenate([a.run(None, {'x': x[i:i + 1]})[0] for i in range(3)])
    y = ort.InferenceSession(nb, providers=['CPUExecutionProvider']).run(None, {'x': x})[0]
    y16 = ort.InferenceSession(f16, providers=['CPUExecutionProvider']).run(None, {'x': x})[0]
    check(np.abs(y - ref).max() < 1e-6 and y16.dtype == np.float32 and np.abs(y16 - ref).max() < 0.02,
          '一次三幀：和一幀一幀完全相同；半精度輸入輸出仍是 float32、誤差小')
    bad = _write(os.path.join(BASE, 'bad.onnx'), GARBAGE)
    check(models.batch_model(bad) is None and models.take_warnings(), '壞的檔案：不轉換、記一筆提醒')


def main():
    check(loads(_write(os.path.join(BASE, 'fx.onnx'), GOOD_DET)) and not loads(_write(os.path.join(BASE, 'gx.onnx'), GARBAGE)),
          '測試用小模型：onnxruntime CPU 能開；同樣大小的亂碼不能開')
    download_cases()
    engine_cases()
    conversion_cases()
    print('全部通過')


if __name__ == '__main__':
    main()
