"""動作庫：把 Unity 的 Humanoid 動畫檔（.anim）直接播成 Avatar 參數（OSC）。

來源有兩種：這個程式自己轉出來的（影片 → 動作、圖片 → 姿勢，在匯出資料夾裡），和使用者自己加進來的。
.anim 的肌肉曲線和桌面模式送出的參數是同一套定義（anim.py 寫的就是它），所以讀回來幾乎不用換算：
肌肉值直接對到 m0…m54，RootQ（身體旋轉）拆成偏航和傾斜，RootT.y 是身體高度。
"""
import json
import os
import re
import shutil

import numpy as np

import anim as ANIM
import desktop as DK
import video as V
from paths import data

RATE = 60.0                     # 讀進來之後取樣的頻率（和送出的頻率一樣）
MAX_BYTES = 300 << 20           # 再大的檔案不讀（一小時的動作約 100 MB）
MAX_SECONDS = 3600.0
BLEND_S = 0.3                   # 循環播放：最後這麼久漸漸接回第一幀
_MUSCLE = {n: i for i, n in enumerate(ANIM.MUSCLE_NAMES)}
_FOLD = {f'UpperChest {a}': f'Chest {a}' for a in ('Front-Back', 'Left-Right', 'Twist Left-Right')}
_KEY_FIELDS = ('time', 'value', 'inSlope', 'outSlope')
_ROOT = ('RootT.x', 'RootT.y', 'RootT.z', 'RootQ.x', 'RootQ.y', 'RootQ.z', 'RootQ.w')


# ------------------------------------------------------------------ 讀 .anim
def parse_anim(path):
    """Unity 文字格式的 .anim → {屬性名: (時間, 值, 進入斜率, 離開斜率)}（只讀 m_FloatCurves）。
    不是文字格式 → ValueError('ERR_ANIM_BINARY')；讀不了 → ValueError('ERR_ANIM_READ')。
    一行一行讀、讀完 m_FloatCurves 就停：後面的 m_EditorCurves 是同樣內容的另一份，大檔案不用整個放進記憶體。"""
    try:
        if os.path.getsize(path) > MAX_BYTES:
            raise ValueError('ERR_ANIM_READ')
        fh = open(path, 'rb')
    except OSError:
        raise ValueError('ERR_ANIM_READ') from None
    out, inside, keys, cur, in_keys = {}, False, None, None, False
    with fh:
        try:
            if not fh.read(8).lstrip(b'\xef\xbb\xbf').startswith(b'%YAML'):
                raise ValueError('ERR_ANIM_BINARY')
            fh.readline()
            for raw in fh:
                line = raw.decode('utf-8', 'replace').rstrip('\r\n')
                s = line.strip()
                if not inside:
                    inside = s == 'm_FloatCurves:'
                    continue
                if line.startswith('  ') and not line.startswith('   ') and s.startswith('m_'):      # 下一個區段
                    break
                if s in ('- curve:', 'curve:'):                 # 舊版 Unity 寫「- curve:」，新版是「- serializedVersion: 2」換行「curve:」
                    keys, cur, in_keys = [], None, False
                elif keys is None:
                    continue
                elif s.startswith('m_Curve:'):
                    in_keys = True
                elif in_keys and s.startswith('- '):            # 一個關鍵影格（舊版直接從「- time:」開始）
                    cur = {}
                    keys.append(cur)
                    k, _, v = s[2:].partition(':')
                    if k in _KEY_FIELDS:
                        cur[k] = v
                elif in_keys and s.startswith('m_'):            # m_PreInfinity…：關鍵影格結束
                    in_keys = False
                elif in_keys and cur is not None:
                    k, _, v = s.partition(':')
                    if k in _KEY_FIELDS:
                        cur[k] = v
                elif s.startswith('attribute:'):
                    name = s[10:].strip().strip('\'"')
                    try:
                        pts = sorted((float(k['time']), float(k['value']), _num(k.get('inSlope')), _num(k.get('outSlope')))
                                     for k in keys if 'time' in k and 'value' in k)
                    except ValueError:
                        pts = []
                    if pts and name not in out:
                        out[name] = tuple(np.array(c, float) for c in zip(*pts))
                    keys, cur = None, None
        except OSError:
            raise ValueError('ERR_ANIM_READ') from None
    return out


def transform_curves(path):
    """這個動畫除了肌肉曲線以外，有沒有「直接移動骨頭」的曲線（旋轉／位置）。有的話那部分不會送出，播出來會和原本不同。"""
    try:
        with open(path, 'rb') as fh:
            head = fh.read(4 << 20)                # 這幾個區段都在 m_FloatCurves 前面
    except OSError:
        return False
    return re.search(rb'^  m_(?:Rotation|CompressedRotation|Euler|Position)Curves:[ \t]*\r?$', head, re.M) is not None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _sample(curve, grid):
    """一條曲線在 grid 的時間上的值（Unity 的三次 Hermite；斜率不是有限值的那一段維持前一個關鍵影格的值）。"""
    t, v, sin, sout = curve
    if len(t) == 1:
        return np.full(len(grid), v[0])
    i = np.clip(np.searchsorted(t, grid, side='right') - 1, 0, len(t) - 2)
    h = np.maximum(t[i + 1] - t[i], 1e-9)
    u = np.clip((grid - t[i]) / h, 0.0, 1.0)
    m0, m1 = sout[i], sin[i + 1]
    step = ~(np.isfinite(m0) & np.isfinite(m1))
    m0, m1 = np.where(step, 0.0, m0), np.where(step, 0.0, m1)
    u2, u3 = u * u, u * u * u
    val = (2 * u3 - 3 * u2 + 1) * v[i] + (u3 - 2 * u2 + u) * h * m0 + (-2 * u3 + 3 * u2) * v[i + 1] + (u3 - u2) * h * m1
    return np.where(step, np.where(u >= 1.0, v[i + 1], v[i]), val)


# ------------------------------------------------------------------ 肌肉 → 骨架（預覽、沒有 RootT 時算高度）
def local_rotation(name, mus):
    """desktop.muscles_of 的反運算：{肌肉編號: 值} → 這根骨頭的區域旋轉（Unity 四元數）。"""
    pre, post, sign, dofs = DK.MODEL[name]
    t = [0.0, 0.0, 0.0]
    for d in range(3):
        if dofs[d] is None:
            continue
        idx, mn, mx = dofs[d]
        v = float(np.clip(mus.get(idx, 0.0), -1.5, 1.5))
        ang = (v * mx if v >= 0 else -v * mn) * sign[d]
        t[d] = float(np.tan(np.radians(ang) / 2.0))
    tx = t[0]
    q = np.array([tx, t[1] + tx * t[2], t[2] - tx * t[1], 1.0])
    q /= np.linalg.norm(q)
    return DK.qmul(DK.qmul(np.asarray(pre, float), q), DK.qinv(np.asarray(post, float)))


def _local_v(name, M, n):
    """local_rotation 的整段版本：M = {肌肉編號: (n,) 陣列}，值可以超過 ±1（Unity 照比例外推）→ (4, n) 四元數。"""
    pre, post, sign, dofs = DK.MODEL[name]
    t = []
    for d in range(3):
        v = M.get(dofs[d][0]) if dofs[d] is not None else None
        if v is None:
            t.append(np.zeros(n))
            continue
        _, mn, mx = dofs[d]
        ang = np.clip(np.where(v >= 0, v * mx, -v * mn) * sign[d], -170.0, 170.0)
        t.append(np.tan(np.radians(ang) / 2.0))
    q = np.stack([t[0], t[1] + t[0] * t[2], t[2] - t[0] * t[1], np.ones(n)])
    q /= np.linalg.norm(q, axis=0)
    return DK.qmul(DK.qmul(np.asarray(pre, float), q), DK.qinv(np.asarray(post, float)))


def _muscles_v(name, L):
    """desktop.muscles_of 的整段版本：(4, n) 區域旋轉 → {肌肉編號: (n,) 陣列}，夾在 ±1。"""
    pre, post, sign, dofs = DK.MODEL[name]
    q = DK.qmul(DK.qmul(DK.qinv(np.asarray(pre, float)), L), np.asarray(post, float))
    q = np.where(q[3] < 0, -q, q)
    w = np.maximum(q[3], 1e-6)
    tx = q[0] / w if dofs[0] else np.zeros_like(w)
    a, b = q[1] / w, q[2] / w
    k = 1.0 + tx * tx
    t = (tx, (a - tx * b) / k, (b + tx * a) / k)
    out = {}
    for d in range(3):
        if dofs[d] is None:
            continue
        idx, mn, mx = dofs[d]
        ang = np.degrees(2.0 * np.arctan(t[d])) * sign[d]
        out[idx] = np.clip(np.where(ang >= 0, ang / mx, -ang / mn), -1.0, 1.0)
    return out


def fit_range(M, n):
    """肌肉值超過 ±1 的動畫（Unity 會照比例轉得更多，但 Avatar 參數只到 ±1）：從骨盆往外一根一根算，
    被夾住的骨頭少轉的部分由下一根補回來（例如肩膀抬不夠高 → 上臂多抬一點），手腳在空間中的方向盡量和原本一樣。
    M = {肌肉編號: (n,) 陣列} → 同樣格式、全部在 ±1 以內。本來就在範圍內的幀結果不變。"""
    ident = np.tile(np.array([0.0, 0.0, 0.0, 1.0])[:, None], (1, n))
    want, got, out = {'Hips': ident}, {'Hips': ident}, {}
    for name, (parent, _) in DK.RIG.items():
        if parent is None:
            continue
        if name not in DK.MODEL:
            want[name], got[name] = want[parent], got[parent]
            continue
        want[name] = DK.qmul(want[parent], _local_v(name, M, n))            # 原本的動畫要它在空間中的方向
        m = _muscles_v(name, DK.qmul(DK.qinv(got[parent]), want[name]))     # 接在（可能被夾過的）上一根後面要轉多少
        out.update(m)
        got[name] = DK.qmul(got[parent], _local_v(name, m, n))
    return out


def _fk(params):
    """一幀的參數 → (各骨頭的世界旋轉, 相對骨盆的位置, 重心)。座標：Unity（+X 角色右邊、+Y 上、+Z 前方）。"""
    mus = {int(k[1:]): v for k, v in params.items() if k[0] == 'm' and k[1:].isdigit()}
    W = {'Hips': np.array([0.0, 0.0, 0.0, 1.0])}
    for n, (p, _) in DK.RIG.items():
        if p is not None:
            W[n] = DK.qmul(W[p], local_rotation(n, mus)) if n in DK.MODEL else W[p]
    pos = DK._positions(W)
    yaw = np.degrees(np.arctan2(params.get('YawS', 0.0), params.get('YawC', 1.0)))
    want = DK.tilt_quat(yaw, np.array([params.get('TiltX', 0.0), params.get('TiltZ', 0.0)]))
    qh = DK.qmul(want, DK.qinv(DK.body_rotation(pos)))                    # 骨盆的旋轉：讓「身體旋轉」等於參數給的
    W = {n: DK.qmul(qh, q) for n, q in W.items()}
    pos = DK._positions(W)
    return W, pos, sum(DK.COM_W[n] * pos[n] for n in DK.RIG)


def grounded_y(params):
    """讓全身最低的點剛好貼地的身體高度參數（動畫沒有記高度時用）。"""
    W, pos, com = _fk(params)
    by = (DK.ground_height(W, pos) + com[1]) / DK.HUMAN_SCALE
    return float(np.clip(by / 0.75 - 1.0, -1.0, 1.0))


def skeleton(params):
    """一幀的參數 → 標準骨架每個關節的位置（腳踩在參數給的高度上）。"""
    _, pos, com = _fk(params)
    by = (float(params.get('Y', 0.3)) + 1.0) * 0.75
    hip = np.array([0.0, by * DK.HUMAN_SCALE - com[1], 0.0])
    return {n: p + hip for n, p in pos.items()}


_BONES = [(a, b) for b, (a, _) in DK.RIG.items() if a is not None]


def draw(params, size=560):
    """預覽圖：從正面看的骨架（BGR）。左邊橘色、右邊藍色，和即時畫面的顏色一樣。"""
    import cv2
    img = np.full((size, size, 3), (26, 22, 20), np.uint8)
    floor = int(size * 0.88)
    img[floor:] = (35, 30, 27)
    k = size * 0.42
    try:
        pos = skeleton(params)
    except Exception:
        return img
    pt = {n: (int(round(size / 2 - p[0] * k)), int(round(floor - p[1] * k))) for n, p in pos.items()}
    for a, b in _BONES:
        col = (60, 170, 255) if b.startswith('Left') else (255, 120, 90) if b.startswith('Right') else (230, 230, 230)
        cv2.line(img, pt[a], pt[b], col, 4, cv2.LINE_AA)
    up = pos['Head'] - pos['Neck']
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    c = pos['Head'] + 0.10 * up
    cv2.circle(img, (int(round(size / 2 - c[0] * k)), int(round(floor - c[1] * k))), int(0.10 * k), (255, 255, 255), -1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ 動作
class Clip:
    """讀好的動作：每個參數一條 RATE Hz 的陣列。sample(t) 取某一刻的參數。"""

    def __init__(self, path):
        cv = parse_anim(path)
        mus = {n: c for n, c in cv.items() if n in _MUSCLE and _MUSCLE[n] in DK.MUSCLES}
        if len(mus) < 5:
            raise ValueError('ERR_ANIM_NOT_HUMANOID')
        used = list(mus.values()) + [cv[n] for n in _ROOT if n in cv]
        t0 = min(float(c[0][0]) for c in used)
        dur = max(float(c[0][-1]) for c in used) - t0
        if not np.isfinite(dur) or dur > MAX_SECONDS:
            raise ValueError('ERR_ANIM_READ')
        self.path, self.name = path, os.path.splitext(os.path.basename(path))[0]
        self.seconds = max(dur, 0.0)
        self.pose = self.seconds < 0.2                              # 一個姿勢（圖片轉出來的只有兩個一樣的關鍵影格）
        n = 1 if self.pose else int(np.ceil(self.seconds * RATE - 1e-3)) + 1      # 檔案裡的時間只有幾位數：298.0002 格還是 298 格
        grid = t0 + np.linspace(0.0, self.seconds, n)
        P = {f'm{i}': np.zeros(n) for i in DK.MUSCLES}
        for name, c in mus.items():
            P[f'm{_MUSCLE[name]}'] = _sample(c, grid)
        for name, to in _FOLD.items():                              # 上胸不在送出的參數裡：角度加到胸（上胸 ±20°、胸 ±40°）
            if name in cv:
                P[f'm{_MUSCLE[to]}'] = P[f'm{_MUSCLE[to]}'] + 0.5 * _sample(cv[name], grid)
        for k in P:                                                 # 檔案裡壞掉的數字（NaN／無限大）：不能讓它傳到整條手臂
            P[k] = np.nan_to_num(P[k], nan=0.0, posinf=4.0, neginf=-4.0)
        over = np.zeros(n, bool)
        for k in P:
            over |= np.abs(P[k]) > 1.0 + 1e-9
        self.over = float(over.mean())                              # 有多少比例的幀超過 Avatar 參數的範圍（±1）
        if over.any():
            idx = np.nonzero(over)[0]
            fit = fit_range({i: P[f'm{i}'][idx] for i in DK.MUSCLES}, len(idx))
            for i, v in fit.items():
                P[f'm{i}'][idx] = v
        for k in P:
            P[k] = np.clip(P[k], -1.0, 1.0)
        if all(f'RootQ.{a}' in cv for a in 'xyzw'):
            Q = np.stack([_sample(cv[f'RootQ.{a}'], grid) for a in 'xyzw'], axis=1)
            Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        else:
            Q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        yt = [DK.body_tilt(q) for q in Q]
        yaw = np.radians([y for y, _ in yt])
        P['YawS'], P['YawC'] = np.sin(yaw), np.cos(yaw)
        P['TiltX'], P['TiltZ'] = np.array([t[0] for _, t in yt]), np.array([t[1] for _, t in yt])
        if 'RootT.y' in cv:
            P['Y'] = np.clip(_sample(cv['RootT.y'], grid) / 0.75 - 1.0, -1.0, 1.0)
        else:                                                       # 沒有記高度：讓最低的點貼地
            P['Y'] = np.array([grounded_y({k: float(v[i]) for k, v in P.items()}) for i in range(n)])
        self.P, self.n = P, n
        self.transforms = transform_curves(path)                    # 另外有直接移動骨頭的曲線（不會送出）

    def frame(self, i):
        return {k: float(v[i]) for k, v in self.P.items()}

    def sample(self, t, loop=True):
        """t 秒時的參數。loop：超過長度時從頭再來，而且最後 BLEND_S 秒漸漸接回第一幀（頭尾姿勢不同時不會跳一下）。"""
        if self.n == 1:
            return self.frame(0)
        t = t % self.seconds if loop else min(max(t, 0.0), self.seconds)
        x = t * (self.n - 1) / self.seconds
        i = min(int(x), self.n - 2)
        u = x - i
        out = {k: float(v[i] + u * (v[i + 1] - v[i])) for k, v in self.P.items()}
        blend = min(BLEND_S, 0.25 * self.seconds)
        if loop and blend > 0 and self.seconds - t < blend:
            w = 1.0 - (self.seconds - t) / blend
            w = w * w * (3.0 - 2.0 * w)
            first = self.frame(0)
            out = {k: out[k] + w * (first[k] - out[k]) for k in out}
        r = float(np.hypot(out['YawS'], out['YawC']))
        if r > 1e-6:
            out['YawS'], out['YawC'] = out['YawS'] / r, out['YawC'] / r
        return out


def mix(a, b, w):
    """兩組參數之間的過渡（w = 0 → a、1 → b）；朝向走圓弧上的短邊。"""
    w = float(np.clip(w, 0.0, 1.0))
    w = w * w * (3.0 - 2.0 * w)
    out = {k: a[k] + w * (b[k] - a[k]) if k in a else b[k] for k in b}
    if 'YawS' in a and 'YawS' in b:
        ya, yb = np.arctan2(a['YawS'], a['YawC']), np.arctan2(b['YawS'], b['YawC'])
        y = ya + w * np.arctan2(np.sin(yb - ya), np.cos(yb - ya))
        out['YawS'], out['YawC'] = float(np.sin(y)), float(np.cos(y))
    return out


# ------------------------------------------------------------------ 清單
def folder():
    """使用者自己加進來的動作放的地方。"""
    d = data('motions')
    os.makedirs(d, exist_ok=True)
    return d


_TIME = re.compile(rb'\btime: ([-+0-9.eE]+)')
_meta = {}
_SCAN_ALL = 8 << 20


def _info(path, kind):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _meta:
        sec = None
        if st.st_size <= MAX_BYTES:
            try:
                with open(path, 'rb') as fh:                # 大檔案只看頭尾：開頭的時間在最前面，最後一條曲線的結尾在最後面
                    blob = fh.read() if st.st_size <= _SCAN_ALL else fh.read(1 << 16) + (fh.seek(-(1 << 20), 2), fh.read())[1]
                ts = [float(m) for m in _TIME.findall(blob)]
                sec = (max(ts) - min(ts)) if ts else None
            except (OSError, ValueError):
                sec = None
        _meta[key] = sec
    sec = _meta[key]
    return dict(path=path, name=os.path.splitext(os.path.basename(path))[0], kind=kind, mtime=st.st_mtime,
                seconds=sec, pose=sec is not None and sec < 0.2)


def _hidden_file():
    return os.path.join(folder(), 'hidden.json')


def _key(path):
    return os.path.normcase(os.path.abspath(path))


def _hidden():
    """從清單隱藏的轉換結果：{路徑: 當時的修改時間}。同一個檔名之後重新轉一次（時間變了）會再出現。"""
    try:
        with open(_hidden_file(), encoding='utf-8') as fh:
            d = json.load(fh)
        return {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_hidden(d):
    try:
        if d:
            with open(_hidden_file(), 'w', encoding='utf-8') as fh:
                json.dump(d, fh, ensure_ascii=False, indent=1)
        elif os.path.exists(_hidden_file()):
            os.remove(_hidden_file())
        return True
    except OSError:
        return False


def _is_hidden(item, hidden):
    return abs(hidden.get(_key(item['path']), -1.0) - item['mtime']) < 1e-3


def _scan():
    out = []
    for d, kind in ((folder(), 'added'), (V.export_dir(), 'export')):
        try:
            names = [f for f in os.listdir(d) if f.lower().endswith('.anim')]
        except OSError:
            names = []
        items = [i for i in (_info(os.path.join(d, f), kind) for f in names) if i]
        out += sorted(items, key=lambda i: -i['mtime'])
    return out


def list_all():
    """可以播放的動作：自己加的在前面，然後是轉出來的（新的在前）。從清單隱藏的不列。"""
    hidden = _hidden()
    return [i for i in _scan() if not (i['kind'] == 'export' and _is_hidden(i, hidden))]


def hidden_count():
    hidden = _hidden()
    return sum(1 for i in _scan() if i['kind'] == 'export' and _is_hidden(i, hidden)) if hidden else 0


def restore_hidden():
    """隱藏的轉換結果全部放回清單。"""
    return _save_hidden({})


def add(paths):
    """把使用者選的 .anim 複製進動作資料夾（先確認讀得了）。回傳 {'added': [路徑], 'errors': [{'name', 'error'}], 'warnings': [{'name', 'warning'}]}。"""
    added, errors, warnings = [], [], []
    for p in paths or []:
        name = os.path.basename(p)
        try:
            if os.path.normcase(os.path.abspath(os.path.dirname(p))) == os.path.normcase(os.path.abspath(folder())):
                continue                                    # 選到的是已經在動作庫裡的檔案：不用再複製一份
            Clip(p)
        except ValueError as e:
            errors.append(dict(name=name, error=str(e)))
            continue
        except Exception:
            errors.append(dict(name=name, error='ERR_ANIM_READ'))
            continue
        stem, dst, i = os.path.splitext(name)[0], None, 1
        while dst is None or os.path.exists(dst):
            dst = os.path.join(folder(), f'{stem}{"" if i == 1 else f" ({i})"}.anim')
            i += 1
        try:
            shutil.copyfile(p, dst)
            added.append(dst)
        except OSError:
            errors.append(dict(name=name, error='ERR_ANIM_READ'))
            continue
        if transform_curves(dst):
            warnings.append(dict(name=name, warning='WARN_ANIM_TRANSFORMS'))
    return dict(added=added, errors=errors, warnings=warnings)


def remove(path):
    """從清單拿掉：自己加進來的 → 刪掉動作庫裡那一份（原本的檔案不動）；轉換結果 → 只是不列出來，匯出資料夾裡的檔案留著。"""
    p = _key(path or '')
    if not p.endswith('.anim') or not os.path.isfile(p):
        return False
    if os.path.dirname(p) == _key(folder()):
        try:
            os.remove(p)
        except OSError:
            return False
        return True
    if os.path.dirname(p) == _key(V.export_dir()):
        here = {_key(i['path']): i['mtime'] for i in _scan() if i['kind'] == 'export'}
        hidden = {k: v for k, v in _hidden().items() if abs(here.get(k, -1.0) - v) < 1e-3}      # 檔案已經不在或重新轉過的順便清掉
        hidden[p] = here.get(p, os.path.getmtime(p))
        return _save_hidden(hidden)
    return False
