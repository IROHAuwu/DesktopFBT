"""GEM-X（NVIDIA，影片 → 全身動作；程式 Apache-2.0、權重 NVIDIA Open Model License）：影片轉動作的高精度模式。

流程（和官方相同；人物框由我們的骨架算出，相機當成固定）：
  每幀：人物框 → 裁切 → ① Body Pose 影像特徵 (1024) ② 77 點 2D 關鍵點（熱圖，左右翻轉平均）
  整段：2D 關鍵點 ＋ 人物框 ＋ 相機參數 ＋ 影像特徵 → 時序模型 → 655 維（76 個關節旋轉、體型、根部旋轉與速度）
  SOMA 骨架（平均體型的零姿勢關節位置，匯出時算好）→ 正向運動學 → 77 個關節 → Halpe26 → 原本的匯出流程
三個 ONNX 由 tools/export_gemx.py 匯出，並在 CI 用真模型和官方 PyTorch 實作逐項比對（這個檔的 numpy 部分就是比對的對象）。
"""
import os
import time

import cv2
import numpy as np

import pose as P

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_LEN = 120
CHUNK = 480                     # 長影片分段推論（每段前後各多看 MAX_LEN 幀）
NJ = 77
FILES = ('gemx_body_fp16.onnx', 'gemx_body_fp16.onnx.data', 'gemx_kp2d_fp16.onnx', 'gemx_kp2d_fp16.onnx.data',
         'gemx_motion.onnx', 'gemx_skeleton.npz')
RELEASE_URL = 'https://github.com/IROHAuwu/DesktopFBT/releases/tag/gemx-models'

# SOMA 的關節編號（公開骨架去掉虛擬的 Root）
HIPS, NECK1, HEAD_J, HEAD_END, L_EYE_J, R_EYE_J = 0, 4, 6, 7, 9, 10
L_ARM, L_FOREARM, L_HAND, R_ARM, R_FOREARM, R_HAND = 12, 13, 14, 40, 41, 42
L_LEG, L_SHIN, L_FOOT, L_TOE, L_TOE_END = 67, 68, 69, 70, 71
R_LEG, R_SHIN, R_FOOT, R_TOE, R_TOE_END = 72, 73, 74, 75, 76


def folder():
    from paths import data
    return data('models', 'gemx')


def paths():
    d = folder()
    return {f: os.path.join(d, f) for f in FILES}


def missing():
    """缺少的檔案（檔名）。"""
    return [f for f, p in paths().items() if not os.path.exists(p)]


# ---------------------------------------------------------------- 自動下載
BASE_URL = 'https://github.com/IROHAuwu/DesktopFBT/releases/download/gemx-models/'
SIZES = {'gemx_skeleton.npz': 7372, 'gemx_motion.onnx': 177190316, 'gemx_body_fp16.onnx': 1300000,
         'gemx_kp2d_fp16.onnx': 950000, 'gemx_body_fp16.onnx.data': 1761000000, 'gemx_kp2d_fp16.onnx.data': 1693000000}


RETRIES, RETRY_WAIT = 3, 2.0     # 連線中斷時再試幾次、第一次等幾秒（之後每次加長）


class Unavailable(Exception):
    """下載來源不存在或需要登入（放模型的 repo 還沒公開）：請使用者自己下載。"""


class DownloadError(RuntimeError):
    """下載失敗。why：network／incomplete／invalid／storage（介面的 MODEL_WHY_*）。"""

    def __init__(self, file, why, detail=''):
        super().__init__(f'ERR_MODEL_DOWNLOAD:{file}|{why}')
        self.file, self.why, self.detail = file, why, detail


def _open(url, start=0, timeout=30):
    import urllib.request
    headers = {'User-Agent': 'DesktopFBT'}
    if start:
        headers['Range'] = f'bytes={start}-'
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def broken():
    """模型載入失敗時用：和 Release 的 SHA256SUMS 核對，內容不對的檔案刪掉（之後會重新下載）。回傳刪掉的檔名；連不上網路時回傳 []。"""
    try:
        with _open(BASE_URL + 'SHA256SUMS', timeout=20) as r:
            sums = dict(reversed(line.split()) for line in r.read().decode('utf-8', 'replace').splitlines() if len(line.split()) == 2)
    except Exception:
        return []
    bad = []
    for f, p in paths().items():
        try:
            if f in sums and os.path.exists(p) and _sha256(p) != sums[f]:
                os.remove(p)
                bad.append(f)
        except OSError:
            pass
    return bad


def download(progress=lambda done, total: None):
    """缺的檔案從 Release 下載到 models\\gemx：先小檔、再大檔；中斷後從 .part 接著下載；每個檔核對 SHA256SUMS。
    progress(已下載位元組, 全部位元組) 丟出的例外（例如按停止）原樣往外傳，下載到一半的 .part 會留著。"""
    import shutil
    import urllib.error
    need = sorted(missing(), key=lambda f: SIZES.get(f, 0))
    if not need:
        return
    try:
        with _open(BASE_URL + 'SHA256SUMS', timeout=20) as r:
            sums = dict(reversed(line.split()) for line in r.read().decode('utf-8', 'replace').splitlines() if len(line.split()) == 2)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            raise Unavailable(str(e)) from None
        raise DownloadError('SHA256SUMS', 'network', repr(e)) from None
    except Exception as e:
        raise DownloadError('SHA256SUMS', 'network', repr(e)) from None
    if any(f not in sums for f in need):
        raise DownloadError('SHA256SUMS', 'invalid', f'no checksum for {[f for f in need if f not in sums]}')
    d = folder()
    try:
        os.makedirs(d, exist_ok=True)
        part = {f: os.path.join(d, f + '.part') for f in need}
        have = {f: os.path.getsize(part[f]) if os.path.exists(part[f]) else 0 for f in need}
        left = sum(max(SIZES.get(f, 0) - have[f], 0) for f in need)
        if shutil.disk_usage(d).free < left + (200 << 20):
            raise DownloadError(need[-1], 'storage', f'need {left >> 20} MB free')
    except OSError as e:
        raise DownloadError(need[0], 'storage', repr(e)) from None
    total = sum(SIZES.get(f, 0) for f in need)
    done = {f: 0 for f in need}

    def report():
        progress(sum(done.values()), max(total, sum(done.values()), 1))

    def fetch(f):
        """下載一個檔到 .part（有 .part 就接著下載）。"""
        start = os.path.getsize(part[f]) if os.path.exists(part[f]) else 0
        try:
            r = _open(BASE_URL + f, start)
        except urllib.error.HTTPError as e:
            if e.code == 416 and start:                     # .part 已經是完整的（或比遠端大）：直接核對
                return
            if e.code in (401, 403, 404):
                raise Unavailable(str(e)) from None
            raise DownloadError(f, 'network', repr(e)) from None
        except Exception as e:
            raise DownloadError(f, 'network', repr(e)) from None
        with r:
            resumed = start and r.status == 206
            size = int(r.headers.get('Content-Length') or 0) + (start if resumed else 0)
            try:
                out = open(part[f], 'ab' if resumed else 'wb')
            except OSError as e:
                raise DownloadError(f, 'storage', repr(e)) from None
            done[f] = start if resumed else 0
            with out:
                while True:
                    try:
                        chunk = r.read(1 << 20)
                    except Exception as e:
                        raise DownloadError(f, 'incomplete', repr(e)) from None
                    if not chunk:
                        break
                    try:
                        out.write(chunk)
                    except OSError as e:
                        raise DownloadError(f, 'storage', repr(e)) from None
                    done[f] += len(chunk)
                    report()
            if size and done[f] != size:
                raise DownloadError(f, 'incomplete', f'{done[f]} of {size} bytes')
    for f in need:
        dst = os.path.join(d, f)
        for attempt in (0, 1):
            for retry in range(RETRIES + 1):                # 幾 GB 的下載：連線中途斷掉時自己接著下載
                try:
                    fetch(f)
                    break
                except DownloadError as e:
                    if e.why not in ('network', 'incomplete') or retry == RETRIES:
                        raise
                    time.sleep(RETRY_WAIT * (retry + 1))
                    progress(sum(done.values()), max(total, sum(done.values()), 1))     # 等的時候按停止要有反應
            if _sha256(part[f]) == sums[f]:
                break
            os.remove(part[f])                              # 內容不對（接續下載接錯、檔案換過）：整個重來一次
            done[f] = 0
            if attempt:
                raise DownloadError(f, 'invalid', 'checksum mismatch')
        try:
            os.replace(part[f], dst)
        except OSError as e:
            raise DownloadError(f, 'storage', repr(e)) from None
        done[f] = max(done[f], os.path.getsize(dst))
        report()

# 左右對調時的關節對應（官方 SOMA77_FLIP）
FLIP77 = list(range(NJ))
for _l, _r in [(9, 10), *[(i, i + 28) for i in range(11, 39)], *[(i, i + 5) for i in range(67, 72)]]:
    FLIP77[_l], FLIP77[_r] = _r, _l
FLIP77 = np.array(FLIP77)

# 655 維的欄位
S_POSE, S_ID, S_SCALE, S_RC, S_RGV, S_VEL = (slice(0, 456), slice(456, 584), slice(584, 640),
                                             slice(640, 646), slice(646, 652), slice(652, 655))


# ---------------------------------------------------------------- 前處理
def intrinsics(w, h):
    """官方 motion_intrinsics：焦距 = 畫面對角線。"""
    f = float(np.hypot(w, h))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float32)


def cliff(boxes, K):
    """人物框相對主點的位置與大小，除以焦距。boxes (L,3) = 中心 x、y、邊長。"""
    boxes = np.asarray(boxes, np.float32)
    return np.stack([boxes[:, 0] - K[0, 2], boxes[:, 1] - K[1, 2], boxes[:, 2]], -1) / K[0, 0]


def crop_body(rgb, box, size=256):
    """Body Pose 的裁切（官方 crop_convention='motion_training'）。rgb：HWC uint8。回傳 (3,256,256)。"""
    cx, cy, s = (float(v) for v in box)
    src = np.asarray(((cx - s / 2, cy - s / 2), (cx + s / 2, cy - s / 2), (cx, cy + s / 2)), np.float32)
    dst = np.asarray(((0, 0), (size, 0), (size / 2, size)), np.float32)
    A = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(rgb, A, (size, size), flags=cv2.INTER_LINEAR)
    return ((out.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy()


def crop_kp(rgb, box):
    """2D 關鍵點模型的裁切：256×256（邊緣對到 255）取中間 192 欄。回傳 (3,256,192)。"""
    cx, cy, s = (float(v) for v in box)
    h = s / 2
    src = np.array([[cx - h, cy - h], [cx + h, cy - h], [cx, cy]], np.float32)
    dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], np.float32)
    A = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(rgb, A, (256, 256), flags=cv2.INTER_LINEAR)
    out = (out.astype(np.float32) / 255.0 - MEAN) / STD
    return out[:, 32:224].transpose(2, 0, 1).copy()


def flip_heatmaps(maps):
    """左右翻轉的那一份熱圖轉回來（關節左右對調＋水平翻轉）。"""
    return maps[:, FLIP77][..., ::-1]


def decode_heatmaps(maps, boxes):
    """熱圖 (N,77,H,W) → 全畫面座標＋信心 (N,77,3)。官方 decode_native77_heatmaps（UDP、四分之一像素修正）。"""
    maps = np.asarray(maps, np.float32)
    boxes = np.asarray(boxes, np.float32)
    N, J, H, W = maps.shape
    flat = maps.reshape(N, J, -1)
    idx = flat.argmax(-1)
    x, y = (idx % W).astype(np.float32), (idx // W).astype(np.float32)
    ix, iy = idx % W, idx // W
    n, k = np.meshgrid(np.arange(N), np.arange(J), indexing='ij')
    okx = (ix > 1) & (ix < W - 1)
    oky = (iy > 1) & (iy < H - 1)
    dx = np.sign(maps[n, k, iy, np.clip(ix + 1, 0, W - 1)] - maps[n, k, iy, np.clip(ix - 1, 0, W - 1)])
    dy = np.sign(maps[n, k, np.clip(iy + 1, 0, H - 1), ix] - maps[n, k, np.clip(iy - 1, 0, H - 1), ix])
    x += np.where(okx, dx, 0) * 0.25
    y += np.where(oky, dy, 0) * 0.25
    s = boxes[:, 2:3]
    px = x / (W - 1) * (s * 0.75) + boxes[:, 0:1] - s * 0.375
    py = y / (H - 1) * s + boxes[:, 1:2] - s * 0.5
    return np.stack([px, py, flat.max(-1)], -1).astype(np.float32)


def observations(kp, boxes):
    """關鍵點 (L,77,3)（畫素＋信心）→ 模型輸入：以人物框正規化到 [-1,1]，框外的點信心設 0。"""
    kp = np.asarray(kp, np.float32)
    boxes = np.asarray(boxes, np.float32)
    xy, conf = kp[..., :2], kp[..., 2].copy()
    c, s = boxes[:, None, :2], boxes[:, None, 2:3]
    lo, hi = c - s / 2, c + s / 2
    conf[((xy < lo) | (xy > hi)).any(-1)] = 0
    return np.concatenate([2 * (xy - c) / s, conf[..., None]], -1).astype(np.float32)


def attn_mask(L):
    """超過 120 幀時每一幀只看前後各 60 幀（True = 不看）。"""
    if L <= MAX_LEN:
        return np.zeros((L, L), bool)
    i = np.arange(L)
    lo = np.clip(np.clip(i - MAX_LEN // 2, 0, None), None, L - MAX_LEN)
    hi = np.clip(np.clip(i + MAX_LEN // 2, None, L), MAX_LEN, None)
    return ~((i[None] >= lo[:, None]) & (i[None] < hi[:, None]))


# ---------------------------------------------------------------- 後處理
def rot6d(d6):
    """(...,6) → (...,3,3)（列向量 b1, b2, b3）。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def decode(x):
    """時序模型輸出 (L,655)（已還原正規化）→ 各個量。旋轉都是矩陣。"""
    x = np.asarray(x, np.float64)
    L = len(x)
    return dict(R_body=rot6d(x[:, S_POSE].reshape(L, 76, 6)), identity=x[:, S_ID], scale=np.exp(x[:, S_SCALE]),
                R_c=rot6d(x[:, S_RC]), R_gv=rot6d(x[:, S_RGV]), vel=x[:, S_VEL])


def camera_transl(cam, boxes, K):
    """弱透視相機 (s, tx, ty) → 相機座標的位移。"""
    cam, boxes = np.asarray(cam, np.float64), np.asarray(boxes, np.float64)
    sb = cam[:, 0] * boxes[:, 2] + 1e-9
    return np.stack([cam[:, 1] + 2 * (boxes[:, 0] - K[0, 2]) / sb, cam[:, 2] + 2 * (boxes[:, 1] - K[1, 2]) / sb,
                     2 * K[0, 0] / sb], -1)


FLIP_Z = np.diag([-1.0, -1.0, 1.0])            # 繞 z 轉 180°：模型的重力座標 → y 朝上


def world_static(R_gv, vel, step=None):
    """固定相機的世界座標：根部旋轉 (L,3,3) 與位置 (L,3)（官方 rollout_world 在相機不動時的結果）。
    step：每一幀相對 30fps 的時間倍率（不是 30fps 的影片用）。"""
    v = np.asarray(vel, np.float64)
    if step is not None:
        v = v * np.asarray(step, np.float64)[:, None]
    move = np.einsum('lij,lj->li', R_gv, v)
    transl = np.concatenate([np.zeros((1, 3)), np.cumsum(move[:-1], 0)])
    return FLIP_Z @ R_gv, transl @ FLIP_Z.T


def fk(rest, parents, R_root, R_body, transl=None, rotations=False):
    """正向運動學。rest (77,3)：零姿勢的關節位置；parents (77,)：父關節（根 = -1）；
    R_root (L,3,3)、R_body (L,76,3,3)：相對零姿勢的局部旋轉。回傳 (L,77,3)，根在 transl。
    rotations=True 時另外回傳每個關節相對零姿勢的世界旋轉 (L,77,3,3)。"""
    L = len(R_root)
    G = np.zeros((L, NJ, 3, 3))
    pos = np.zeros((L, NJ, 3))
    for j in range(NJ):
        p = int(parents[j])
        if p < 0:
            G[:, j] = R_root
            if transl is not None:
                pos[:, j] = transl
        else:
            G[:, j] = G[:, p] @ R_body[:, j - 1]
            pos[:, j] = pos[:, p] + np.einsum('lij,j->li', G[:, p], rest[j] - rest[p])
    return (pos, G) if rotations else pos


# ---------------------------------------------------------------- 人物框
def person_box(k, s, thr=0.3, pad=1.08):
    """我們的骨架（26 點）→ 人物框 (x0,y0,x1,y1)。骨架比偵測框小一點（頭頂、手腳的邊緣），所以放大一點。"""
    pts = k[s > thr]
    if len(pts) < 6:
        return None
    lo, hi = pts.min(0), pts.max(0)
    c, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * pad
    return np.concatenate([c - half, c + half])


def square_boxes(xyxy, enlarge=1.2):
    """官方的做法：框做 5 幀移動平均，再變成正方形（邊長 = max(高, 寬/0.75) × 1.2）。回傳 (L,3)。"""
    b = np.asarray(xyxy, np.float64)
    if len(b) > 5:
        k5 = np.ones(5) / 5
        b = np.stack([np.convolve(np.pad(b[:, c], 2, mode='edge'), k5, mode='valid') for c in range(4)], -1)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, np.maximum(h, w / 0.75) * enlarge], -1).astype(np.float32)


# ---------------------------------------------------------------- 模型
def _session(path, device):
    """半精度的大模型：長度／批次會變，不用 MIGraphX（每種大小都要重新編譯）。"""
    import onnxruntime as ort
    P._patch_onnxruntime()
    for dev in ['dml', 'cpu'] if device == 'migraphx' else P.fallbacks(device):
        try:
            return ort.InferenceSession(path, providers=[P.PROVIDER[dev]]), dev
        except Exception:
            continue
    raise RuntimeError('ERR_GEMX_MODEL')


class Features:
    """Body Pose 影像特徵（DINOv3 ViT-H+，半精度）。"""

    def __init__(self, path, device):
        self.sess, self.device = _session(path, device)

    def __call__(self, crops, cliffs):
        return P.run(self.sess, None, {'img': np.stack(crops).astype(np.float32),
                                       'cliff': np.asarray(cliffs, np.float32)})[0]


class Keypoints:
    """77 點 2D 關鍵點（DINOv3 ViT-H+ 熱圖，半精度）。官方做左右翻轉平均：每幀跑兩次。"""

    def __init__(self, path, device, flip=True):
        self.sess, self.device = _session(path, device)
        self.flip = flip

    def __call__(self, crops, boxes):
        x = np.stack(crops).astype(np.float32)
        if self.flip:
            both = P.run(self.sess, None, {'img': np.concatenate([x, x[..., ::-1]])})[0]
            heat = 0.5 * (both[:len(x)] + flip_heatmaps(both[len(x):]))
        else:
            heat = P.run(self.sess, None, {'img': x})[0]
        return decode_heatmaps(heat, boxes)


class Net:
    """時序模型。"""

    def __init__(self, path, device):
        self.sess, self.device = _session(path, device)

    def run(self, obs, cliffs, feats, pos):
        L = len(obs)
        out_x = np.zeros((L, 655), np.float32)
        out_cam = np.zeros((L, 3), np.float32)
        angvel = np.tile(np.float32([1, 0, 0, 0, 1, 0]), (L, 1))          # 固定相機
        for s0 in list(range(0, L, CHUNK)) or [0]:
            a, b = max(0, s0 - MAX_LEN), min(L, s0 + CHUNK + MAX_LEN)
            feed = {'obs': obs[None, a:b].astype(np.float32), 'cliff': cliffs[None, a:b].astype(np.float32),
                    'angvel': angvel[None, a:b], 'img': feats[None, a:b].astype(np.float32),
                    'mask': attn_mask(b - a), 'pos': (pos[None, a:b] - pos[a]).astype(np.float32)}
            x, cam, _ = P.run(self.sess, None, feed)
            e = min(L, s0 + CHUNK)
            out_x[s0:e] = x[0, s0 - a:e - a]
            out_cam[s0:e] = cam[0, s0 - a:e - a]
        out_x[:, S_ID] = out_x[:, S_ID].mean(0)                           # 體型、比例：整段一個
        out_x[:, S_SCALE] = out_x[:, S_SCALE].mean(0)
        return out_x, out_cam


# ---------------------------------------------------------------- SOMA 骨架 → Halpe26
class Body:
    """SOMA 的 77 個關節 → 我們的 26 點。SOMA 沒有鼻子、耳朵、腳跟和大小腳趾：
    用零姿勢的身體方向算出固定的偏移，跟著頭／腳的關節轉。"""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.rest = z['rest'].astype(np.float64)
        self.parents = z['parents'].astype(np.int64)
        r = self.rest
        left = r[L_ARM] - r[R_ARM]
        left /= np.linalg.norm(left)
        up = 0.5 * (r[L_LEG] + r[R_LEG]) - 0.5 * (r[L_FOOT] + r[R_FOOT])       # 腿的方向：零姿勢是站直的
        up -= (up @ left) * left
        up /= np.linalg.norm(up)
        fwd = np.cross(left, up)
        self.left, self.up, self.fwd = left, up, fwd
        k = float((r[HEAD_END] - 0.5 * (r[L_TOE_END] + r[R_TOE_END])) @ up) / 1.70      # 零姿勢的身高相對 170cm
        self.height = 1.70 * k
        eyes = 0.5 * (r[L_EYE_J] + r[R_EYE_J]) - r[HEAD_J]
        self.extra = {                                   # 26 點的編號 → (跟著哪個關節轉, 相對那個關節的偏移)
            P.NOSE: (HEAD_J, eyes + (0.03 * fwd - 0.035 * up) * k),
            P.L_EAR: (HEAD_J, eyes + (0.075 * left - 0.085 * fwd - 0.005 * up) * k),
            P.R_EAR: (HEAD_J, eyes + (-0.075 * left - 0.085 * fwd - 0.005 * up) * k),
        }
        for sign, foot, toe, end, big, small, heel in ((1, L_FOOT, L_TOE, L_TOE_END, P.L_BIGTOE, P.L_SMALLTOE, P.L_HEEL),
                                                       (-1, R_FOOT, R_TOE, R_TOE_END, P.R_BIGTOE, P.R_SMALLTOE, P.R_HEEL)):
            out = sign * left                              # 腳的外側
            tip = r[end] - r[toe]
            self.extra[big] = (toe, tip - 0.02 * k * out)
            self.extra[small] = (toe, 0.7 * tip + 0.04 * k * out)
            sole = float((r[end] - r[foot]) @ up)          # 腳尖比腳踝低多少：腳跟放在同一個高度
            self.extra[heel] = (foot, -0.06 * k * fwd + sole * up)
        self.joints = {P.L_EYE: L_EYE_J, P.R_EYE: R_EYE_J, P.L_SHO: L_ARM, P.R_SHO: R_ARM, P.L_ELB: L_FOREARM,
                       P.R_ELB: R_FOREARM, P.L_WRI: L_HAND, P.R_WRI: R_HAND, P.L_HIP: L_LEG, P.R_HIP: R_LEG,
                       P.L_KNEE: L_SHIN, P.R_KNEE: R_SHIN, P.L_ANK: L_FOOT, P.R_ANK: R_FOOT, P.HEAD: HEAD_END,
                       P.NECK: NECK1}

    def points(self, R_root, R_body, transl, scale=1.0):
        """R_root (L,3,3)、R_body (L,76,3,3)、transl (L,3) → Halpe26 (L,26,3)。scale：這個人相對平均體型的大小。"""
        J, G = fk(self.rest * scale, self.parents, R_root, R_body, transl, rotations=True)
        X = np.zeros((len(R_root), P.NUM_JOINTS, 3))
        for name, j in self.joints.items():
            X[:, name] = J[:, j]
        X[:, P.HIP] = 0.5 * (J[:, L_LEG] + J[:, R_LEG])
        for name, (j, off) in self.extra.items():
            X[:, name] = J[:, j] + np.einsum('lij,j->li', G[:, j], off * scale)
        return X


BONES_2D = ((L_LEG, L_SHIN), (L_SHIN, L_FOOT), (R_LEG, R_SHIN), (R_SHIN, R_FOOT), (L_ARM, L_FOREARM),
            (L_FOREARM, L_HAND), (R_ARM, R_FOREARM), (R_FOREARM, R_HAND), (L_ARM, R_ARM), (L_LEG, R_LEG),
            (L_ARM, L_LEG), (R_ARM, R_LEG))


def body_scale(body, d, cam, kp, boxes, K, thr=0.5):
    """這個人比平均體型大多少。模型算的相機距離是配合它自己預測的體型，App 的骨架是平均體型：
    把平均體型的骨架投影回畫面，和偵測到的 2D 關鍵點比骨頭長度。資料不夠時回傳 1。"""
    J = fk(body.rest, body.parents, d['R_c'], d['R_body'], camera_transl(cam, boxes, K))
    z = np.maximum(J[..., 2], 0.05)
    uv = np.stack([K[0, 0] * J[..., 0] / z + K[0, 2], K[1, 1] * J[..., 1] / z + K[1, 2]], -1)
    ratios = []
    for a, b in BONES_2D:
        ok = (kp[:, a, 2] > thr) & (kp[:, b, 2] > thr)
        proj = np.linalg.norm(uv[ok, a] - uv[ok, b], axis=-1)
        seen = np.linalg.norm(kp[ok, a, :2] - kp[ok, b, :2], axis=-1)
        good = proj > 0.05 * boxes[ok, 2]                 # 正對鏡頭、縮成一點的骨頭不算
        ratios.append(seen[good] / proj[good])
    ratios = np.concatenate(ratios) if ratios else np.zeros(0)
    if len(ratios) < 30:
        return 1.0
    return float(np.clip(np.median(ratios), 0.75, 1.3))


def yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def solve(net, body, kp, boxes, feats, size, fps=30.0, view=False):
    """kp (L,77,3) 畫素＋信心、boxes (L,3)、feats (L,1024)、fps：這些幀的間隔 → Halpe26 世界座標 (L,26,3)（y 上、公尺）。
    view=True 時多回傳每幀「從人指向相機」的方向 (L,3)：相機俯視／仰視時它不是水平的。"""
    L = len(kp)
    K = intrinsics(*size)
    step = 30.0 / float(fps)                               # 模型的時間單位是 1/30 秒
    x, cam = net.run(observations(kp, boxes), cliff(boxes, K), feats, np.arange(L) * step)
    d = decode(x)
    R_w, transl = world_static(d['R_gv'], d['vel'], np.full(L, step))
    # 方向對齊相機：讓前一秒身體的朝向和「相機看到的朝向」一樣（我們的座標：x 右、y 上、z 朝相機）。
    # 用整個旋轉找最接近的水平角（不是只看「身體前方」那一個方向：坐著、躺著時前方幾乎朝上，水平角會亂掉）。
    D = np.diag([1.0, -1.0, -1.0])
    n = max(1, min(L, int(round(fps))))
    M = np.einsum('lij,lkj->ik', D @ d['R_c'][:n], R_w[:n])          # Σ 相機看到的旋轉 × 世界旋轉的轉置
    Y = yaw(np.arctan2(M[0, 2] - M[2, 0], M[0, 0] + M[2, 2]))
    scale = body_scale(body, d, cam, kp, boxes, K)
    X = body.points(Y @ R_w, d['R_body'], transl @ Y.T, scale)
    if not view:
        return X, scale
    to_cam = np.einsum('lij,lkj->lik', Y @ R_w, D @ d['R_c'])[:, :, 2]      # 世界←相機 的第三軸（朝相機）
    return X, scale, to_cam
