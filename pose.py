"""2D 姿勢估計（rtmlib：YOLOX 偵測 + RTMPose，Halpe26 含腳尖、腳跟）。

推論裝置：
- migraphx : AMD MIGraphX（Windows ML 提供；AMD 顯卡預設）
- dml  : DirectML（舊版；Windows 上任何 DX12 顯卡：NVIDIA / AMD / Intel）
- cuda : 需要另外安裝 onnxruntime-gpu 和 CUDA
- cpu  : 最慢，但一定能跑
"""

import numpy as np

# Halpe26 關節編號
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
HEAD, NECK, HIP = 17, 18, 19
L_BIGTOE, R_BIGTOE, L_SMALLTOE, R_SMALLTOE, L_HEEL, R_HEEL = 20, 21, 22, 23, 24, 25
NUM_JOINTS = 26

SKELETON = [(L_ANK, L_KNEE), (L_KNEE, L_HIP), (R_ANK, R_KNEE), (R_KNEE, R_HIP),
            (L_HIP, R_HIP), (L_SHO, L_HIP), (R_SHO, R_HIP), (L_SHO, R_SHO),
            (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
            (HEAD, NECK), (NECK, HIP), (L_ANK, L_HEEL), (L_ANK, L_BIGTOE),
            (R_ANK, R_HEEL), (R_ANK, R_BIGTOE), (L_BIGTOE, L_SMALLTOE), (R_BIGTOE, R_SMALLTOE)]
LEFT = {L_SHO, L_ELB, L_WRI, L_HIP, L_KNEE, L_ANK, L_BIGTOE, L_SMALLTOE, L_HEEL}
RIGHT = {R_SHO, R_ELB, R_WRI, R_HIP, R_KNEE, R_ANK, R_BIGTOE, R_SMALLTOE, R_HEEL}

_PATCHED = False
GPU_LOCK = __import__('threading').RLock()    # 同一張顯示卡上的推論一次一個（兩條執行緒同時跑 DirectML 會讓顯示卡重置）


class GPUError(RuntimeError):
    def __init__(self, msg='ERR_GPU', detail=''):
        super().__init__(msg)
        self.detail = detail


def _system_text(e):
    """UnicodeDecodeError 裡原本的訊息（系統語言，例如 cp932）。"""
    raw = getattr(e, 'object', b'') or b''
    for enc in ('cp932', 'cp950', 'gbk', 'cp1252'):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode('utf-8', 'replace')


def run(sess, outs, feed):
    """session.run 的包裝：排隊使用顯示卡；顯示卡出錯時的訊息常是系統語言（例如 cp932），無法用 UTF-8 解讀。"""
    with GPU_LOCK:
        try:
            return sess.run(outs, feed)
        except UnicodeDecodeError as e:
            raise GPUError('ERR_GPU', _system_text(e).strip()[-600:]) from e
        except Exception as e:
            if 'DML' in str(e) or 'DXGI' in str(e) or '887A' in str(e).upper() or 'MIGraphX' in str(e):
                raise GPUError('ERR_GPU', str(e)[-600:]) from e
            raise


def _patch_onnxruntime():
    """讓 rtmlib 支援 device='dml'，並套用 DirectML 需要的 SessionOptions。"""
    global _PATCHED
    if _PATCHED:
        return
    import onnxruntime as ort
    from rtmlib.tools import base
    base.RTMLIB_SETTINGS['onnxruntime']['dml'] = 'DmlExecutionProvider'
    base.RTMLIB_SETTINGS['onnxruntime']['migraphx'] = 'MIGraphXExecutionProvider'
    orig = ort.InferenceSession

    class Session(orig):
        def __init__(self, path_or_bytes=None, sess_options=None, providers=None, **kw):
            if providers and 'MIGraphXExecutionProvider' in str(providers):
                import accel
                from paths import data
                so = accel.session_options(sess_options or ort.SessionOptions(), data('models', 'migraphx_cache'))
                super().__init__(path_or_bytes, sess_options=so, **kw)
                return
            if providers and 'DmlExecutionProvider' in str(providers):
                so = sess_options or ort.SessionOptions()
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options = so
                providers = list(providers) + ['CPUExecutionProvider']
            super().__init__(path_or_bytes, sess_options=sess_options, providers=providers, **kw)

    ort.InferenceSession = Session
    _PATCHED = True


PROVIDER = {'migraphx': 'MIGraphXExecutionProvider', 'dml': 'DmlExecutionProvider',
            'cuda': 'CUDAExecutionProvider', 'cpu': 'CPUExecutionProvider'}


def fallbacks(device):
    """這個裝置失敗時依序改用的裝置。"""
    order = {'migraphx': ['migraphx', 'dml', 'cpu'], 'dml': ['dml', 'cpu'], 'cuda': ['cuda', 'cpu']}
    av = set(available_devices())
    return [d for d in order.get(device, [device]) if d in av or d == 'cpu']


def available_devices():
    try:
        import onnxruntime as ort
        av = ort.get_available_providers()
    except Exception:
        return ['cpu']
    out = []
    try:
        import accel
        if accel.status()['ok']:
            out.append('migraphx')
    except Exception:
        pass
    if 'DmlExecutionProvider' in av:
        out.append('dml')
    if 'CUDAExecutionProvider' in av:
        out.append('cuda')
    out.append('cpu')
    return out


def resolve_device(pref):
    av = available_devices()
    if pref in av:
        return pref
    return av[0]     # auto：migraphx > dml > cuda > cpu


def draw(img, kps, scores, thr, marks=None):
    """畫骨架。marks：{關節: +1 朝鏡頭 / -1 背向鏡頭}——正對鏡頭的手腳在畫面上縮成一個點、看不到線，
    在末端畫記號：圓圈加中心點 = 朝鏡頭伸過來，圓圈加叉 = 往後伸。"""
    import cv2
    if kps is None:
        return img
    s = max(1, int(round(img.shape[0] / 360)))
    for a, b in SKELETON:
        if scores[a] > thr and scores[b] > thr:
            col = (255, 170, 60) if a in LEFT else (90, 120, 255) if a in RIGHT else (230, 230, 230)
            cv2.line(img, tuple(int(v) for v in kps[a]), tuple(int(v) for v in kps[b]), col, 2 * s,
                     cv2.LINE_AA)
    for j in range(NUM_JOINTS):
        if scores[j] > thr:
            cv2.circle(img, tuple(int(v) for v in kps[j]), 3 * s, (255, 255, 255), -1, cv2.LINE_AA)
    for j, toward in (marks or {}).items():
        if not np.all(np.isfinite(kps[j])):
            continue
        c = tuple(int(v) for v in kps[j])
        col = (255, 170, 60) if j in LEFT else (90, 120, 255) if j in RIGHT else (230, 230, 230)
        r = 9 * s
        cv2.circle(img, c, r, (20, 20, 20), 4 * s, cv2.LINE_AA)
        cv2.circle(img, c, r, col, 2 * s, cv2.LINE_AA)
        if toward > 0:
            cv2.circle(img, c, 3 * s, col, -1, cv2.LINE_AA)
        else:
            k = int(0.6 * r)
            cv2.line(img, (c[0] - k, c[1] - k), (c[0] + k, c[1] + k), col, 2 * s, cv2.LINE_AA)
            cv2.line(img, (c[0] - k, c[1] + k), (c[0] + k, c[1] - k), col, 2 * s, cv2.LINE_AA)
    return img
