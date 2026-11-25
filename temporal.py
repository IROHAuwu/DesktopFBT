"""時間序列 3D（MotionAGFormer / MotionBERT-Lite）：用前後一段時間的 2D 骨架一起推算深度。

單張畫面的模型（RTMW3D）每幀各猜各的深度，前後不連貫、身體前後常判錯；
MotionBERT 看一整段 2D 動作，深度連貫很多。這裡只取它的「各關節相對骨盆的深度」，
其餘（距離、地板、尺度）照舊由 Lifter 算。沒有模型時整個略過。
"""
import collections
import os
import threading
import time

import numpy as np

import pose as P

T = 243                 # MotionBERT、MotionAGFormer-B 一次看的幀數（-S 是 81）
FPS = 50.0              # 訓練資料（Human3.6M）的幀率：輸入先重新取樣到 50fps

# Halpe26 → H36M 17 點（和 MotionBERT 的 halpe2h36m 相同）
H36M = [P.HIP, P.R_HIP, P.R_KNEE, P.R_ANK, P.L_HIP, P.L_KNEE, P.L_ANK, None, P.NECK, P.NOSE, P.HEAD,
        P.L_SHO, P.L_ELB, P.L_WRI, P.R_SHO, P.R_ELB, P.R_WRI]
_FLIP_L, _FLIP_R = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
# 沒有對應的點：跟著最近的有對應的點（保留單張模型給的相對深度）
FOLLOW = {P.L_EYE: P.NOSE, P.R_EYE: P.NOSE, P.L_EAR: P.NOSE, P.R_EAR: P.NOSE,
          P.L_BIGTOE: P.L_ANK, P.L_SMALLTOE: P.L_ANK, P.L_HEEL: P.L_ANK,
          P.R_BIGTOE: P.R_ANK, P.R_SMALLTOE: P.R_ANK, P.R_HEEL: P.R_ANK}
# 尺度：H36M 骨頭（a, b）的長度 / 身高
_BONES = [((4, 5), 0.245), ((1, 2), 0.245), ((5, 6), 0.246), ((2, 3), 0.246),
          ((11, 12), 0.186), ((14, 15), 0.186), ((12, 13), 0.146), ((15, 16), 0.146)]


MODELS = {                                  # 設定值 → (檔名, 輸入格式)
    'agformer_b': ('agformer_base.onnx', 'screen'),
    'agformer_s': ('agformer_small.onnx', 'screen'),
    'motionbert': ('motionbert_lite.onnx', 'crop'),
}


def model_path(name='agformer_b'):
    """打包在程式裡的模型（沒有時 None）。"""
    from paths import resource
    p = resource('models_bundled', MODELS.get(name, MODELS['agformer_b'])[0])
    return p if os.path.exists(p) else None


DML_TAG = '_dml1'


def dml_safe(m):
    """DirectML 的矩陣乘法最多 4 維：注意力裡 5 維的 MatMul（和 Softmax）先攤平成 3 維（2 維）算完再變回原形狀。
    數值完全相同；回傳 (模型, 改了幾個)。"""
    import onnx
    from onnx import helper, numpy_helper, shape_inference
    g = m.graph
    inf = shape_inference.infer_shapes(m)
    rank = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        t = vi.type.tensor_type
        if t.HasField('shape'):
            rank[vi.name] = len(t.shape.dim)
    for init in g.initializer:
        rank[init.name] = len(init.dims)
    consts = {}

    def const(name, vals):
        if name not in consts:
            g.initializer.append(numpy_helper.from_array(np.array(vals, np.int64), name))
            consts[name] = name
        return name

    big = 2 ** 62
    out_nodes, n_fix = [], 0
    for i, nd in enumerate(g.node):
        r = [rank.get(x, 0) for x in nd.input]
        p = f'dmlfix{i}_'
        if nd.op_type == 'MatMul' and len(r) == 2 and r[0] >= 5 and r[0] == r[1]:
            a, b = nd.input
            out_nodes += [
                helper.make_node('Shape', [a], [p + 'sa']), helper.make_node('Shape', [b], [p + 'sb']),
                helper.make_node('Slice', [p + 'sa', const('dml_m2', [-2]), const('dml_big', [big])], [p + 'la']),
                helper.make_node('Slice', [p + 'sb', const('dml_m2', [-2]), const('dml_big', [big])], [p + 'lb']),
                helper.make_node('Concat', [const('dml_m1', [-1]), p + 'la'], [p + 'ra'], axis=0),
                helper.make_node('Concat', [const('dml_m1', [-1]), p + 'lb'], [p + 'rb'], axis=0),
                helper.make_node('Reshape', [a, p + 'ra'], [p + 'a3']),
                helper.make_node('Reshape', [b, p + 'rb'], [p + 'b3']),
                helper.make_node('MatMul', [p + 'a3', p + 'b3'], [p + 'c3']),
                helper.make_node('Slice', [p + 'sa', const('dml_0', [0]), const('dml_m1', [-1])], [p + 'lead']),
                helper.make_node('Slice', [p + 'sb', const('dml_m1', [-1]), const('dml_big', [big])], [p + 'n']),
                helper.make_node('Concat', [p + 'lead', p + 'n'], [p + 'so'], axis=0),
                helper.make_node('Reshape', [p + 'c3', p + 'so'], [nd.output[0]]),
            ]
            n_fix += 1
            continue
        if nd.op_type == 'Softmax' and r and r[0] >= 5:
            ax = next((a.i for a in nd.attribute if a.name == 'axis'), -1)
            if ax in (-1, r[0] - 1):
                x = nd.input[0]
                out_nodes += [
                    helper.make_node('Shape', [x], [p + 's']),
                    helper.make_node('Slice', [p + 's', const('dml_m1', [-1]), const('dml_big', [big])], [p + 'l']),
                    helper.make_node('Concat', [const('dml_m1', [-1]), p + 'l'], [p + 'r'], axis=0),
                    helper.make_node('Reshape', [x, p + 'r'], [p + 'x2']),
                    helper.make_node('Softmax', [p + 'x2'], [p + 'y2'], axis=-1),
                    helper.make_node('Reshape', [p + 'y2', p + 's'], [nd.output[0]]),
                ]
                n_fix += 1
                continue
        out_nodes.append(nd)
    del g.node[:]
    g.node.extend(out_nodes)
    del g.value_info[:]
    onnx.checker.check_model(m)
    return m, n_fix


def dml_model(src):
    """DirectML 用的版本（轉一次存在使用者資料的 models 資料夾）。失敗時 None。"""
    from paths import data
    dst = data('models', os.path.basename(src)[:-5] + DML_TAG + '.onnx')
    if os.path.exists(dst):
        return dst
    tmp = dst + '.part'
    try:
        import onnx
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        m, _ = dml_safe(onnx.load(src))
        onnx.save(m, tmp)
        os.replace(tmp, dst)
        return dst
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None


_LEVELS = ('ORT_ENABLE_BASIC', 'ORT_DISABLE_ALL')


def check_main(model, dev, out):
    """子行程（DesktopFBT.exe --tm-check 模型 裝置 結果.json）：在顯示卡上試跑時間序列模型。"""
    import json
    import onnxruntime as ort
    P._patch_onnxruntime()
    if dev == 'migraphx':
        import accel
        accel.ensure()
    res = {'ok': False, 'error': ''}
    for lv in _LEVELS:
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, lv)
            sess = ort.InferenceSession(model, sess_options=so, providers=[P.PROVIDER[dev]])
            T = int(sess.get_inputs()[0].shape[1])
            x = np.random.default_rng(0).uniform(-1, 1, (1, T, 17, 3)).astype(np.float32)
            y = P.run(sess, None, {'x': x})[0]
            if not np.all(np.isfinite(y)):
                raise RuntimeError('non-finite output')
            res = {'ok': True, 'level': lv}
            break
        except Exception as e:
            res['error'] = f'{lv}: {type(e).__name__}: {e} {getattr(e, "detail", "")}'.strip()[-800:]
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)


_checked = {}


def gpu_check(model, dev, timeout=180):
    """在另一個行程試跑（結果存在 models/temporal_gpu.json，同一個模型、裝置、版本只試一次）。"""
    import json
    import subprocess
    import sys
    import tempfile
    import onnxruntime as ort
    from paths import data
    key = f'{os.path.basename(model)}|{os.path.getsize(model)}|{dev}|{ort.__version__}'
    if key in _checked:
        return _checked[key]
    cache = data('models', 'temporal_gpu.json')
    try:
        with open(cache, encoding='utf-8') as f:
            db = json.load(f)
    except (OSError, ValueError):
        db = {}
    if key in db:
        _checked[key] = db[key]
        return db[key]
    fd, out = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.remove(out)
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable, '--tm-check', model, dev, out]
    else:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py'), '--tm-check', model, dev, out]
    try:
        p = subprocess.run(cmd, timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                           check=False, capture_output=True)
        try:
            with open(out, encoding='utf-8') as f:
                r = json.load(f)
        except (OSError, ValueError):
            r = {'ok': False, 'error': f'試跑的行程當掉了（結束碼 {p.returncode}）'}
    except subprocess.TimeoutExpired:
        r = {'ok': False, 'error': 'timeout', 'retry': True}
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    _checked[key] = r
    if not r.get('retry'):
        db[key] = r
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, 'w', encoding='utf-8') as f:
                json.dump(db, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
    return r


def to_h36m_wild(k, s):
    """MotionAGFormer 的示範程式用的轉換（模仿訓練資料 Stacked Hourglass 的頭、胸、脊椎位置）。"""
    out = to_h36m(k, s)
    c = k[:17]
    head = np.array([c[1:5, 0].mean(), c[1, 1] + c[2, 1] - c[0, 1]])
    sho = c[5:7].mean(0)
    thorax = sho + (c[0] - sho) / 3
    pelvis = c[11:13].mean(0)
    spine = c[[5, 6, 11, 12]].mean(0)
    out[10, :2], out[8, :2], out[0, :2], out[7, :2] = head, thorax, pelvis, spine
    out[9, :2] = c[0] - (c[0] - sho) / 4
    out[7, 0] += 2 * (out[7, 0] - 0.5 * (out[0, 0] + out[8, 0]))
    out[8, 1] -= (c[1:3, 1].mean() - c[0, 1]) * 2 / 3
    sc = s[:17]
    out[0, 2] = sc[[11, 12]].mean()
    out[8, 2] = sc[[5, 6]].mean()
    out[7, 2] = 0.5 * (out[0, 2] + out[8, 2])
    out[10, 2] = sc[1:5].mean()
    return out


def to_h36m(k, s):
    """Halpe26 (26,2)、分數 (26,) → (17,3)：x, y, 分數。"""
    out = np.zeros((17, 3))
    for i, j in enumerate(H36M):
        if j is not None:
            out[i, :2], out[i, 2] = k[j], s[j]
    out[7, :2] = 0.5 * (k[P.NECK] + k[P.HIP])
    out[7, 2] = min(s[P.NECK], s[P.HIP])
    return out


def normalize_screen(seq, size):
    """畫面座標正規化：x / 寬 * 2 - 1、y / 寬 * 2 - 高/寬（MotionAGFormer）。"""
    w, h = size
    out = seq.copy()
    out[..., 0] = seq[..., 0] / w * 2 - 1
    out[..., 1] = seq[..., 1] / w * 2 - h / w
    out[seq[..., 2] <= 0, :2] = 0.0
    return out.astype(np.float32)


def normalize(seq):
    """整段一起縮放到 [-1, 1]（MotionBERT 的 crop_scale）。seq (T,17,3)；分數 0 的點不算範圍。"""
    out = seq.copy()
    v = seq[seq[..., 2] > 0][:, :2]
    if len(v) < 4:
        return None
    lo, hi = v.min(0), v.max(0)
    scale = float(max(hi - lo))
    if scale <= 0:
        return None
    c = 0.5 * (lo + hi) - 0.5 * scale
    out[..., :2] = np.clip(((seq[..., :2] - c) / scale - 0.5) * 2, -1, 1)
    out[seq[..., 2] <= 0, :2] = 0.0
    return out.astype(np.float32)


def flip(x):
    y = x.copy()
    y[..., 0] *= -1
    y[..., _FLIP_L + _FLIP_R, :] = y[..., _FLIP_R + _FLIP_L, :]
    return y


def resample(ts, seq, t_out, max_gap=0.35):
    """把不規則時間的 (N,17,3) 線性內插到 t_out；離最近的觀測太遠的時間點分數 = 0。
    比第一筆觀測還早的時間點（剛開始、資料還不到模型的視窗長度）：當成人一直維持第一筆的姿勢。
    以前這一段全部填 0，模型看到的大半是空白，剛開始的一兩秒深度會亂掉（真實畫面上軀幹前傾到四十幾度）。"""
    ts = np.asarray(ts, float)
    i = np.clip(np.searchsorted(ts, t_out), 1, len(ts) - 1)
    t0, t1 = ts[i - 1], ts[i]
    u = np.clip((t_out - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)[:, None, None]
    a, b = seq[i - 1], seq[i]
    out = a + u * (b - a)
    out[..., 2] = np.minimum(a[..., 2], b[..., 2])
    near = np.minimum(np.abs(t_out - t0), np.abs(t_out - t1))
    far = ((near > max_gap) & (t_out >= ts[0])) | (t_out > ts[-1] + max_gap)
    out[far] = 0.0
    out[t_out < ts[0]] = seq[0]
    return out


class Model:
    """ONNX 推論。run(seq (T,17,3) 畫素) → 相對骨盆的 3D (T,17,3)（模型單位）或 None。"""

    def __init__(self, path, device='cpu', kind='screen'):
        import onnxruntime as ort
        self.error = None                   # 顯示卡上試過但失敗的原因（最後一個）
        P._patch_onnxruntime()
        self.sess, self.device = None, 'cpu'
        self.T = 0
        for dev in P.fallbacks(device):
            if dev == 'cpu':
                try:
                    self.sess, self.device = ort.InferenceSession(path, providers=[P.PROVIDER[dev]]), dev
                except Exception as e:
                    self.error = f'cpu: {e}'
                break
            # 顯示卡：5 維的 MatMul 先攤平（DirectML 不支援），而且先在另一個行程試跑——
            # 有的驅動出錯時會讓整個程式當掉，試跑的結果記下來，之後不再重試
            gp = dml_model(path) or path
            r = gpu_check(gp, dev)
            if not r.get('ok'):
                self.error = f'{dev}: {r.get("error", "?")}'
                continue
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, r.get('level', 'ORT_ENABLE_BASIC'))
                self.sess = ort.InferenceSession(gp, sess_options=so, providers=[P.PROVIDER[dev]])
                self.device = dev
                break
            except Exception as e:
                self.error = f'{dev}: {e}'
        if self.sess is not None:
            self.T = int(self.sess.get_inputs()[0].shape[1])
        if self.sess is None:
            raise RuntimeError('no provider')
        self.kind = kind
        self.size = (1920, 1080)          # 畫面大小（screen 格式用）
        self.ms = 0.0

    def h36m(self, k, s):
        return to_h36m_wild(k, s) if self.kind == 'screen' else to_h36m(k, s)

    def run(self, seqs, tta=False):
        """seqs：[(T,17,3)]。回傳 [(T,17,3) 或 None]。"""
        xs, idx = [], []
        for i, s in enumerate(seqs):
            if (s[..., 2] > 0).sum() < 4:
                continue
            n = normalize_screen(s, self.size) if self.kind == 'screen' else normalize(s)
            if n is not None:
                xs.append(n)
                idx.append(i)
        out = [None] * len(seqs)
        if not xs:
            return out
        x = np.stack(xs)
        if tta:
            x = np.concatenate([x, flip(x)])
        t = time.perf_counter()
        y = P.run(self.sess, None, {'x': x})[0]
        self.ms = 1000 * (time.perf_counter() - t)
        if tta:
            n = len(xs)
            y = 0.5 * (y[:n] + flip(y[n:]))
        for i, yy in zip(idx, y):
            out[i] = yy - yy[:, :1]
        return out


def metric_depth(Y, height):
    """模型單位 → 公尺：讓手腳骨頭長度接近身高比例（整段一個尺度）。回傳每點相對骨盆的深度 (T,17)。"""
    den = []
    for (a, b), r in _BONES:
        L = np.linalg.norm(Y[:, a] - Y[:, b], axis=1)
        den.append(np.median(L) / r)
    s = height / max(float(np.median(den)), 1e-6)
    return Y[..., 2] * s


def apply_depth(z, zt):
    """單張模型的相對深度 z (26,) 換成時間序列模型的 zt (17,)：對應的點直接換，其他點跟著最近的點平移。"""
    out = z.copy()
    for i, j in enumerate(H36M):
        if j is not None:
            out[j] = zt[i]
    for j, base in FOLLOW.items():
        out[j] = zt[H36M.index(base)] + (z[j] - z[base])
    return out


def wrist_shift(z_old, z_new):
    """手指跟著手腕：回傳 (左手位移, 右手位移)。"""
    return z_new[P.L_WRI] - z_old[P.L_WRI], z_new[P.R_WRI] - z_old[P.R_WRI]


# ---------------------------------------------------------------- 整段影片（匯出）
def sequence(model, times, frames, height, tta=True, stride=None):
    """frames：[(k, s) 或 None]（和 times 對齊）。回傳每幀的 zt (17,) 或 None。"""
    have = [i for i, f in enumerate(frames) if f is not None]
    if len(have) < 10:
        return [None] * len(frames)
    T = model.T
    stride = stride or max(T // 2, 1)
    ts = np.array([times[i] for i in have])
    seq = np.stack([model.h36m(*frames[i]) for i in have])
    grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / FPS)
    X = resample(ts, seq, grid)
    n = len(grid)
    starts = list(range(0, max(n - T, 0) + 1, stride))
    if starts[-1] + T < n:
        starts.append(n - T)
    acc = np.zeros((n, 17))
    w = np.zeros(n)
    taper = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(float)
    taper = np.minimum(taper / (40.0 * T / 243), 1.0)                  # 視窗兩端（前後資訊少）權重低
    for i in range(0, len(starts), 4):
        chunk = []
        for st in starts[i:i + 4]:
            c = X[st:st + T]
            if len(c) < T:                                  # 影片比視窗短：重複最後一幀補滿
                c = np.concatenate([c, np.repeat(c[-1:], T - len(c), 0)])
            chunk.append(c)
        for st, Y in zip(starts[i:i + 4], model.run(chunk, tta)):
            if Y is None:
                continue
            m = min(T, n - st)
            acc[st:st + m] += metric_depth(Y, height)[:m] * taper[:m, None]
            w[st:st + m] += taper[:m]
    ok = w > 0
    Z = np.where(ok[:, None], acc / np.maximum(w, 1e-9)[:, None], 0.0)
    out = [None] * len(frames)
    for i in have:
        g = (times[i] - grid[0]) * FPS
        a = int(np.clip(np.floor(g), 0, n - 1))
        b = min(a + 1, n - 1)
        if ok[a] and ok[b]:
            u = g - a
            out[i] = (1 - u) * Z[a] + u * Z[b]
    return out


# ---------------------------------------------------------------- 即時
class Live:
    """push() 收逐幀 2D；背景一直用最近一段（模型的視窗長度）推論；pull() 交出「已經有後面 lag 秒資料」的幀（已換好深度）。

    模型太慢或沒有結果時，等太久的幀照原樣交出（不會卡住）。"""

    def __init__(self, model, height, lag=0.4, min_interval=0.08, give_up=1.5, fallback=None):
        self.model = model
        self.fallback = fallback                   # fallback(錯誤訊息) → 另一個模型（同輸入格式）或 None：一直失敗時換掉
        self.ok = 0
        self.last_error = None
        self.dead = False                          # 放棄了：之後的幀直接照原樣交出
        self.height = height
        self.lag = lag
        self.min_interval = self.base_interval = min_interval
        self.give_up = give_up
        self.buf = collections.deque()             # (t, h36m (17,3))：給模型的 2D
        self.pending = collections.deque()         # (t, item)：等著換深度的幀
        self.res = None                            # (grid 時間, Z (T,17))
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.stop_ev = threading.Event()
        self.last_t = None
        self.run_s = 0.25                          # 推論一次的時間（秒，平均）
        self.used = 0
        self.passed = 0
        self.errors = 0
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    @property
    def delay_s(self):
        return self.lag + self.min_interval + self.run_s

    def push(self, t, k, s, item):
        with self.cv:
            self.buf.append((t, self.model.h36m(k, s)))
            while self.buf and t - self.buf[0][0] > (self.model.T + 10) / FPS:
                self.buf.popleft()
            self.pending.append((t, item))
            self.last_t = t
            self.cv.notify()

    def skip(self, t, item):
        """沒有骨架的幀：照順序交出，不給模型。"""
        with self.cv:
            self.pending.append((t, item))

    def _loop(self):
        last_run = 0.0
        while not self.stop_ev.is_set():
            with self.cv:
                self.cv.wait(0.2)
                if len(self.buf) < 10 or time.perf_counter() - last_run < self.min_interval:
                    continue
                ts = np.array([b[0] for b in self.buf])
                seq = np.stack([b[1] for b in self.buf])
            last_run = time.perf_counter()
            grid = ts[-1] - np.arange(self.model.T - 1, -1, -1) / FPS
            X = resample(ts, seq, grid)
            try:
                Y = self.model.run([X])[0]
                self.ok += 1
            except Exception as e:
                Y = None
                self.errors += 1
                self.last_error = f'{type(e).__name__}: {e} {getattr(e, "detail", "")}'.strip()
                if self.ok == 0 and self.errors >= 2 and self.fallback is not None:
                    fb, self.fallback = self.fallback, None
                    try:
                        m = fb(self.last_error)
                    except Exception:
                        m = None
                    if m is not None:
                        self.model = m
                        self.errors = 0
                        continue
                if self.ok == 0 and self.errors >= 2 and self.fallback is None:
                    self.dead = True
                    return
            dt = time.perf_counter() - last_run
            self.run_s = 0.8 * self.run_s + 0.2 * dt
            # 顯示卡很忙（例如同時開著 VRChat）時這個模型一次要跑很久，而它和姿勢模型共用顯示卡：
            # 不要佔超過約四分之一的時間，不然姿勢模型跟不上、一直丟幀。結果會晚一點，輸出的延遲會自己跟著拉長
            self.min_interval = float(np.clip(3.0 * min(self.run_s, dt), self.base_interval, 0.4))
            if Y is not None:
                Z = metric_depth(Y, self.height)
                with self.lock:
                    self.res = (grid, Z)

    def pull(self, now_t):
        """回傳 [(t, item, zt 或 None)]，照時間順序。"""
        out = []
        with self.lock:
            if self.dead:
                while self.pending:
                    t, item = self.pending.popleft()
                    self.passed += 1
                    out.append((t, item, None))
                return out
            res = self.res
            while self.pending:
                t, item = self.pending[0]
                zt = None
                if res is not None:
                    grid, Z = res
                    if grid[0] <= t <= grid[-1] - self.lag:
                        g = (t - grid[0]) * FPS
                        a = int(np.floor(g))
                        b = min(a + 1, len(grid) - 1)
                        u = g - a
                        zt = (1 - u) * Z[a] + u * Z[b]
                    elif t < grid[0]:
                        zt = None                       # 太舊（剛開始或中斷過）：照原樣
                    elif now_t - t < self.give_up:
                        break                           # 還沒有足夠的後續資料：再等
                elif now_t - t < self.give_up:
                    break
                self.pending.popleft()
                if zt is not None:
                    self.used += 1
                else:
                    self.passed += 1
                out.append((t, item, zt))
        return out

    def stop(self):
        self.stop_ev.set()
        with self.cv:
            self.cv.notify_all()
        self.th.join(timeout=3.0)
