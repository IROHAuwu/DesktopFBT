"""單一影片的 3D 姿勢：RTMW3D（2D 位置＋相對骨盆的深度）→ 公尺單位的 3D 關節點。

沒有第二台相機、也沒有頭顯可以對照，所以：
- 每個關節在畫面上的位置決定「它在哪條視線上」，模型給的相對深度決定「前後關係」。
- 整個人離相機多遠（尺度）用骨頭長度推算：選一個距離，讓大腿、小腿、手臂的長度最接近身高比例。
- 哪邊是「上」用「雙腳 → 身體重心」方向的長時間中位數推算（站著的人重心在腳的正上方，彎腰、前傾也一樣），
  再和「腳踩在地板上」擬合出的俯仰角合併；兩者差很多時只用前者。
輸出的座標系：x = 相機右方、y = 上、z = 朝向相機（右手系，公尺）。地板在 y = 0。
"""
import collections
import threading

import numpy as np

import limbs as LB
import steady
import pose as P
from filters import OneEuro

# COCO-WholeBody（133 點）→ Halpe26
_WB_FEET = {P.L_BIGTOE: 17, P.L_SMALLTOE: 18, P.L_HEEL: 19,
            P.R_BIGTOE: 20, P.R_SMALLTOE: 21, P.R_HEEL: 22}

# 骨頭長度 / 身高（人體測量學的平均比例）
BONES = [((P.L_HIP, P.L_KNEE), 0.245), ((P.R_HIP, P.R_KNEE), 0.245),
         ((P.L_KNEE, P.L_ANK), 0.246), ((P.R_KNEE, P.R_ANK), 0.246),
         ((P.L_SHO, P.L_ELB), 0.186), ((P.R_SHO, P.R_ELB), 0.186),
         ((P.L_ELB, P.L_WRI), 0.146), ((P.R_ELB, P.R_WRI), 0.146),
         ((P.L_SHO, P.L_HIP), 0.288), ((P.R_SHO, P.R_HIP), 0.288)]
SOLE = (P.L_HEEL, P.R_HEEL, P.L_BIGTOE, P.R_BIGTOE, P.L_SMALLTOE, P.R_SMALLTOE)   # 鞋底附近的點


class Hands(tuple):
    """手的點 (k (42,2), z (42,), s (42,))；另外帶著同一幀的 head_cue（.cue），一起傳到後面的流程。"""
    cue = None


def wholebody_to_halpe(k2d, z, sc):
    """(133,2), (133,), (133,) → (26,2), (26,), (26,)"""
    k = np.zeros((P.NUM_JOINTS, 2))
    d = np.zeros(P.NUM_JOINTS)
    s = np.zeros(P.NUM_JOINTS)
    k[:17], d[:17], s[:17] = k2d[:17], z[:17], sc[:17]
    for h, w in _WB_FEET.items():
        k[h], d[h], s[h] = k2d[w], z[w], sc[w]

    def mid(a, b):
        return 0.5 * (k[a] + k[b]), 0.5 * (d[a] + d[b]), min(s[a], s[b])
    k[P.NECK], d[P.NECK], s[P.NECK] = mid(P.L_SHO, P.R_SHO)
    k[P.HIP], d[P.HIP], s[P.HIP] = mid(P.L_HIP, P.R_HIP)
    ek, ed, es = mid(P.L_EAR, P.R_EAR)
    k[P.HEAD] = ek + 0.55 * (ek - k[P.NECK])          # 頭頂：耳朵再往上一點
    d[P.HEAD] = ed
    s[P.HEAD] = min(es, s[P.NECK])
    return k, d, s


ZR_WINDOW_S = 0.5             # 深度：取最近這麼久的中位數
ZR_CUTOFF, ZR_BETA = 0.25, 1.0   # 之後的 One Euro：站著不動時很慢（0.25 Hz），走動時放快


def rays(k, K):
    """畫素 → 相機座標中 z = 1 的視線方向。"""
    return np.column_stack([(k[:, 0] - K[0, 2]) / K[0, 0], (k[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(k))])


def solve_depth(d, z, ok, height):
    """找骨盆深度 Zr，讓骨頭長度最接近身高比例。d: 視線 (26,3)，z: 相對深度 (26,)。"""
    pairs = [(a, b, r * height) for (a, b), r in BONES if ok[a] and ok[b]]
    if len(pairs) < 3:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    L = np.array([p[2] for p in pairs])
    zr = np.geomspace(0.4, 60.0, 240)[:, None, None]
    Xa = d[a][None] * (zr + z[a][None, :, None])
    Xb = d[b][None] * (zr + z[b][None, :, None])
    err = np.linalg.norm(Xa - Xb, axis=2) / L[None] - 1.0
    cost = np.sum(np.minimum(err ** 2, 0.25), axis=1)     # 單根骨頭錯很大時不要拖累整體
    i = int(np.argmin(cost))
    lo, hi = zr[max(i - 1, 0), 0, 0], zr[min(i + 1, len(zr) - 1), 0, 0]
    for _ in range(30):                                     # 黃金分割再精修
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        c = []
        for m in (m1, m2):
            e = np.linalg.norm(d[a] * (m + z[a][:, None]) - d[b] * (m + z[b][:, None]), axis=1) / L - 1
            c.append(np.sum(np.minimum(e ** 2, 0.25)))
        if c[0] < c[1]:
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def _bone_arrays(ok):
    pairs = [(a, b, r) for (a, b), r in BONES if ok[a] and ok[b]]
    if len(pairs) < 3:
        return None
    return (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]),
            np.array([p[2] for p in pairs]))


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def fit_pitch(frames, up0, height, span_deg=20.0):
    """在 up0 附近找相機俯仰角的修正量：讓「腳踩地板」算出的深度，骨頭長度最一致。
    frames: [(d (26,3), z (26,), ok (26,), zr_bone)]。回傳 (up, h) 或 None。"""
    data = []
    for d, z, ok, zrb in frames:
        sole = np.array([j for j in SOLE if ok[j]])
        ba = _bone_arrays(ok)
        if len(sole) < 2 or ba is None:
            continue
        data.append((d, z, sole, ba, zrb))
    if len(data) < 30:
        return None

    def evaluate(delta):
        up = _rot_x(np.radians(delta)) @ up0
        hs, items = [], []
        for d, z, sole, ba, zrb in data:
            down = d[sole] @ up
            hh = np.sort(-down * (zrb + z[sole]))
            hs.append(0.5 * (hh[-1] + hh[-2]))
            items.append((down, z[sole]))
        h = float(np.median(hs))
        if h < 0.15:
            return 1e9, up, h
        cost = 0.0
        for (down, zs), (d, z, sole, (a, b, r), _) in zip(items, data):
            m = down < -0.03
            if m.sum() < 2:
                cost += 2.5                    # 腳的視線沒有往下：這個俯仰角不合理
                continue
            zr = float(np.sort(-h / down[m] - zs[m])[:2].mean())
            e = np.linalg.norm(d[a] * (zr + z[a])[:, None] - d[b] * (zr + z[b])[:, None], axis=1) \
                / (r * height) - 1
            cost += float(np.sum(np.minimum(e ** 2, 0.25)))
        return cost / len(data) + 1e-4 * delta ** 2, up, h      # 沒有資訊時偏好不修正

    best = min((evaluate(dl) for dl in np.arange(-span_deg, span_deg + 0.1, 1.0)), key=lambda r: r[0])
    d0 = float(np.degrees(np.arctan2(np.cross(up0, best[1])[0], up0 @ best[1])))
    best = min((evaluate(dl) for dl in np.arange(d0 - 1.0, d0 + 1.01, 0.2)), key=lambda r: r[0])
    return best[1], best[2]


def camera_points(k, z, s, K, height, thr, zr_hint=None):
    """回傳 (X_cam (26,3), ok, Zr)。X_cam 是 OpenCV 相機座標（x 右、y 下、z 前）。"""
    ok = s > thr
    d = rays(k, K)
    zr = solve_depth(d, z, ok, height)
    if zr is None:
        if zr_hint is None:
            return None, ok, None
        zr = zr_hint
    depth = np.maximum(zr + z, 0.2)
    return d * depth[:, None], ok, zr


class Grounder:
    """影片模式：最低的腳底永遠貼在地板（y = 0）。單鏡頭估的高度常常偏高，整個人看起來浮在空中。"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.off = None

    def __call__(self, X, ok):
        # 全身最低的點（站著時是腳底；倒立、躺下時是手、頭）。腳尖、腳跟的點深度不準，最多只算到腳踝下方一點
        o = steady.floor_level(X, ok)
        if o is None:
            return X
        self.off = o if self.off is None else self.off + self.alpha * (o - self.off)
        X = X.copy()
        X[:, 1] -= self.off
        return X


class Lifter:
    """逐幀把模型輸出轉成「直立、地板在 0」的 3D 關節點。即時與匯出共用。"""

    def __init__(self, size, hfov_deg=60.0, height=1.65, thr=0.3, smooth=True):
        self.hfov = float(hfov_deg)
        self.view = (0, 0, int(size[0]), int(size[1]))
        self.K = self._K(self.view)
        self.height = float(height)
        self.thr = thr
        self.pitches = collections.deque(maxlen=1500)       # 每幀從身體估的相機俯仰角（弧度）
        self.up_n = 0
        self.up_torso = np.array([0.0, -1.0, 0.0])           # 只看身體估出來的「上」
        self.pitch = 0.0                                    # 目前採用的俯仰角（弧度）
        self.pitch_t = None
        self.locked = False                                 # 站直校正／匯出第二輪：俯仰角固定
        self.zr = None
        # 最近的腳底點（相機座標，(6,3)，沒有的是 NaN）；匯出時整段都留著
        self.soles = collections.deque(maxlen=600 if smooth else None)
        self._h = None                                      # 相機離地高度（和目前的俯仰角一致）
        self._h_key = None
        self.h_lock = None                                  # 匯出第二輪：高度也固定
        self.floor_used = 0
        self.buf = collections.deque(maxlen=240)           # 用來修正俯仰角的取樣
        self.n = 0
        self.fit = None                                     # (up, h)：俯仰修正後的地板
        self.fit_at = 0
        # 離相機多遠（骨盆的深度）：單一幀的估計很吵（真實畫面量到：人站著不動時標準差 11–18 cm），
        # 而它一變整個人就跟著放大縮小、上下晃。先取最近半秒的中位數，再用慢的濾波；人真的走近走遠時濾波會自己放快。
        self.zr_f = OneEuro(ZR_CUTOFF, ZR_BETA) if smooth else None
        self.zr_hist = collections.deque()                  # (時間, 這一幀估的深度)
        self.R = np.diag([1.0, -1.0, -1.0])                # 相機 → 世界（先假設相機是水平的）

    def _K(self, view):
        """視角以「長邊」計算，直式影片也適用。view = (x0, y0, w, h)：影片在畫面中的範圍。"""
        x0, y0, w, h = view
        f = 0.5 * max(w, h) / np.tan(np.radians(self.hfov) / 2)
        return np.array([[f, 0, x0 + w / 2.0], [0, f, y0 + h / 2.0], [0, 0, 1.0]])

    def set_view(self, view):
        """影片在擷取畫面中的位置改變時（例如切換全螢幕）重新開始估計地板。"""
        view = tuple(int(v) for v in view)
        old = self.view
        if all(abs(a - b) <= 8 for a, b in zip(view, old)):
            return
        self.view = view
        self.K = self._K(view)
        self.soles.clear()
        self._h = self._h_key = self.h_lock = None
        self.buf.clear()
        self.fit = None
        self.zr = None
        if self.zr_f is not None:
            self.zr_f.reset()
            self.zr_hist.clear()

    def freeze(self):
        """匯出用：地板和俯仰角已估好，第二輪逐幀計算時不再變動。"""
        self.refit()
        self.pitch = self._target_pitch()
        self._set_up(self._up_of(self.pitch))
        self.locked = True
        self.h_lock = self.camera_height()                  # 用整段影片的腳底點算一次，第二輪不再變
        self.fit_at = 10 ** 12
        self.zr = None
        if self.zr_f is not None:
            self.zr_f.reset()
            self.zr_hist.clear()

    # ---- 站直校正（相機模式）：倒數最後一秒多請人站直，用整個身體的方向定出相機的俯仰角並固定
    def stand_sample(self, k, z, s):
        Xc, ok, _ = camera_points(k, z, s, self.K, self.height, self.thr, self.zr)
        if Xc is None or not (ok[P.NECK] and ok[P.HIP]):
            return
        if ok[P.L_ANK] and ok[P.R_ANK]:
            v = Xc[P.NECK] - 0.5 * (Xc[P.L_ANK] + Xc[P.R_ANK])          # 腳踝到脖子：比只看軀幹穩
        else:
            v = Xc[P.NECK] - Xc[P.HIP]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return
        v = v / n
        if abs(v[0]) < 0.3 and v[1] < -0.6:
            self._stand = getattr(self, '_stand', [])
            self._stand.append(float(np.arctan2(v[2], -v[1])))

    def stand_lock(self, min_n=6):
        """站直的取樣夠了：固定相機的「上」方向（之後不再用地板擬合或身體去改）。回傳俯仰角（度）或 None。"""
        st = getattr(self, '_stand', [])
        self._stand = []
        if len(st) < min_n:
            return None
        p = float(np.clip(np.median(st), -0.6, 0.6))
        self.up_torso = self._up_of(p)
        self.pitches.clear()
        self.fit = None
        self.fit_at = 10 ** 12
        self.pitch, self.locked = p, True
        self._set_up(self.up_torso)
        return float(np.degrees(p))

    @staticmethod
    def _up_of(p):
        return np.array([0.0, -np.cos(p), np.sin(p)])

    @staticmethod
    def _body_axis(Xc, ok):
        """「上」的取樣：雙腳中間 → 身體重心（站著時重心在腳的正上方，彎腰、前傾都成立）；看不到腳踝時用骨盆 → 脖子。"""
        if not (ok[P.NECK] and ok[P.HIP]):
            return None
        if ok[P.L_ANK] and ok[P.R_ANK]:
            v = 0.6 * Xc[P.HIP] + 0.4 * Xc[P.NECK] - 0.5 * (Xc[P.L_ANK] + Xc[P.R_ANK])
        else:
            v = Xc[P.NECK] - Xc[P.HIP]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    def _target_pitch(self):
        """身體估的俯仰角和地板擬合的合併：差不多時各半；差越多越不信地板擬合（腳在畫面邊緣、被裁掉時它會錯得離譜）。"""
        p = float(np.arctan2(self.up_torso[2], -self.up_torso[1]))
        if self.fit is None:
            return p
        pf = float(np.arctan2(self.fit[0][2], -self.fit[0][1]))
        d = abs(np.degrees(pf - p))
        w = 0.5 * float(np.clip((14.0 - d) / 6.0, 0.0, 1.0))
        return p + w * (pf - p)

    def _update_up(self, Xc, ok, t=None):
        v = self._body_axis(Xc, ok)
        if v is None:
            return
        if self.locked:
            return
        # 只用「在畫面上沒有往旁邊歪」的幀估相機的俯仰角，取中位數：一開始蹲著、彎腰、側傾都不會把整段帶歪
        if abs(v[0]) < 0.35 and v[1] < -0.5:
            self.pitches.append(float(np.arctan2(v[2], -v[1])))
            self.up_n += 1
        if self.pitches:
            p = float(np.median(self.pitches)) if len(self.pitches) >= 15 else 0.0     # 樣本太少：先當相機是水平的
            self.up_torso = self._up_of(float(np.clip(p, -0.6, 0.6)))
        target = self._target_pitch()
        # 開始的幾秒直接採用；之後慢慢移過去（每秒最多 2°），畫面上的人不會突然整個傾斜
        if self.zr_f is not None and self.pitch_t is not None and t is not None and self.up_n > 90:
            lim = np.radians(2.0) * float(np.clip(t - self.pitch_t, 0.0, 0.5))
            target = self.pitch + float(np.clip(target - self.pitch, -lim, lim))
        self.pitch, self.pitch_t = target, t
        self._set_up(self._up_of(target))

    def _set_up(self, up):
        # 影片、視窗畫面的相機幾乎不會左右傾斜：不採用側傾（不然一開始身體歪一邊，整段都會歪）
        up = np.array([0.0, up[1], up[2]], float)
        n = np.linalg.norm(up)
        if n < 1e-6:
            return
        up = up / n
        # 相機頂多會往上或下傾 50°，超過的多半是人在躺、彎腰，不要採用
        if up[1] > -0.64:
            return
        fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
        fwd /= np.linalg.norm(fwd)
        x = np.cross(up, -fwd)
        self.R = np.stack([x, up, -fwd])                  # 列 = 世界 x, y, z 在相機座標中的方向

    def camera_height(self):
        """相機離地多高：最近每一幀「最低的兩個腳底點」的中位數，用目前採用的「上」算（俯仰角改了就重算，兩者永遠一致）。"""
        if self.h_lock is not None:
            return self.h_lock
        n = len(self.soles)
        if n < 20:
            return None
        up = self.R[1]
        full = self.soles.maxlen is not None and n >= self.soles.maxlen
        key = (round(float(up[2]), 4), self.n // 15 if full else (n if n < 90 else n // 5))
        if key != self._h_key:
            hs = -(np.stack(self.soles) @ up)                 # (N,6)，沒有的點是 NaN
            hs = np.sort(np.where(np.isnan(hs), -np.inf, hs), axis=1)[:, -2:]
            self._h = float(np.median(hs.mean(axis=1)))
            self._h_key = key
        return self._h

    def refit(self):
        """用累積的畫面修正俯仰角與相機高度（匯出時跑完一輪後呼叫；即時模式改在背景跑，見 _refit_async）。"""
        if self.up_n < 30:
            return
        r = fit_pitch(list(self.buf), self.up_torso, self.height)
        if r is not None:
            self.fit = r

    def _refit_async(self):
        """即時模式：在背景執行緒算（以前在主迴圈算，每隔約 10 秒會卡住快 1 秒）。"""
        if self.up_n < 30 or getattr(self, '_job', None) is not None:
            return
        buf = list(self.buf)
        step = max(1, len(buf) // 80)                   # 取樣 80 幀就夠準
        args = (buf[::step], self.up_torso.copy(), self.height)
        box = {}

        def run():
            box['r'] = fit_pitch(*args)
        self._job = (threading.Thread(target=run, daemon=True), box)
        self._job[0].start()

    def _refit_poll(self):
        job = getattr(self, '_job', None)
        if job is None or job[0].is_alive():
            return
        self._job = None
        r = job[1].get('r')
        if r is not None and not self.locked:
            self.fit = r

    def __call__(self, k, z, s, t=None):
        """k (26,2), z (26,), s (26,) → (X (26,3) 世界座標、ok)；失敗時 (None, ok)。t：秒。"""
        Xc, ok, zr = camera_points(k, z, s, self.K, self.height, self.thr, self.zr)
        if Xc is None:
            return None, ok
        self._update_up(Xc, ok, t)
        self.n += 1
        if self.fit is None or self.n % 3 == 0:          # 一開始每幀取樣，盡快估好地板
            self.buf.append((rays(k, self.K), z.copy(), ok.copy(), zr))
        if self.fit_at < 10 ** 12 and self.n - self.fit_at >= (60 if self.fit is None else 300) \
                and len(self.buf) >= 40:
            self.fit_at = self.n
            self._refit_async()
        self._refit_poll()
        up = self.R[1]
        sole = [j for j in SOLE if ok[j]]
        if len(sole) >= 2:
            pts = np.full((len(SOLE), 3), np.nan)
            for i, j in enumerate(SOLE):
                if ok[j]:
                    pts[i] = Xc[j]
            self.soles.append(pts)                          # 每幀最低的兩點 ≈ 地板
            h = self.camera_height()
            d = rays(k[sole], self.K)
            down = d @ up
            m = down < -0.03
            if h is not None and h > 0.2 and m.sum() >= 2:
                # 地板限制：踩地的腳點在地板上 → 視線和地板的交點決定深度（比骨頭長度準很多）
                # 離地的點會算出比較遠的深度，所以取最小的兩個
                c = np.sort(-h / down[m] - z[np.array(sole)[m]])
                zr_f = float(c[:2].mean())
                if abs(zr_f / zr - 1.0) < 0.2:              # 差太多時多半是跳起來，維持骨頭長度的結果
                    zr = zr_f
                    self.floor_used += 1
        # 深度跳動是單鏡頭最大的雜訊來源：限制每幀的變化
        if self.zr is not None:
            zr = self.zr + float(np.clip(zr - self.zr, -0.25, 0.25))
        self.zr = zr
        if self.zr_f is not None:
            tt = self.n / 30.0 if t is None else t
            self.zr_hist.append((tt, zr))
            while self.zr_hist and (tt - self.zr_hist[0][0] > ZR_WINDOW_S or tt < self.zr_hist[0][0]):
                self.zr_hist.popleft()
            zr = float(self.zr_f(float(np.median([v for _, v in self.zr_hist])), tt))
        Xc = rays(k, self.K) * np.maximum(zr + z, 0.2)[:, None]
        X = Xc @ self.R.T
        h = self.camera_height()
        if h is None:
            return None, ok                                 # 還不知道地板在哪：先不輸出（約前 1 秒）
        X[:, 1] += h                                        # 地板 = 0
        self.last_zr, self.last_h = zr, h
        return X, ok

    def extra(self, k, z):
        """同一幀的其他點（例如手指）用剛才的深度與地板轉成世界座標。要先呼叫 __call__ 成功過。"""
        X = (rays(k, self.K) * np.maximum(self.last_zr + z, 0.2)[:, None]) @ self.R.T
        X[:, 1] += self.last_h
        return X

class Pose3DEstimator:
    """YOLOX 找人 + RTMW3D 估 3D。回傳 Halpe26 格式 (k2d, z, scores) 或 (None, None, None)。

    - batch()：連續幾幀一起送進模型（顯卡一次算多張，總吞吐量高很多）。
    - precision='auto'：顯卡上先試半精度模型，和全精度比對結果一致、而且比較快才換過去。"""

    # 左右翻轉時要對調的 Halpe26 關節
    FLIP = [(P.L_EYE, P.R_EYE), (P.L_EAR, P.R_EAR), (P.L_SHO, P.R_SHO), (P.L_ELB, P.R_ELB),
            (P.L_WRI, P.R_WRI), (P.L_HIP, P.R_HIP), (P.L_KNEE, P.R_KNEE), (P.L_ANK, P.R_ANK),
            (P.L_BIGTOE, P.R_BIGTOE), (P.L_SMALLTOE, P.R_SMALLTOE), (P.L_HEEL, P.R_HEEL)]
    TRIALS = 8

    def __init__(self, paths, device='dml', det_frequency=8, precision='auto'):
        import models
        P._patch_onnxruntime()
        from rtmlib import YOLOX, RTMPose3d
        self.device = device
        self.det = YOLOX(paths['det'], model_input_size=tuple(paths['det_size']),
                         backend='onnxruntime', device=device)
        src = paths['pose3d']
        nb = models.batch_model(src) if paths.get('convert', True) else None
        self.pose = RTMPose3d(nb or src, model_input_size=(288, 384), backend='onnxruntime', device=device)
        self.freq = max(1, int(det_frequency))
        self.bbox = None
        self.n = 0
        self.tta = False          # 匯出用：原圖和左右翻轉一起算再平均（比較準）
        self.live_tta = False     # 即時也這樣算（batch()）：模型時間加倍
        self.raw = None           # 最近一次的 133 點 (k2d, z, scores)
        self.hands = None
        self.cue = self.cue_detail = None
        self.precision = 'fp32'
        self.pad_batch = 0
        self.ms = 0.0             # 每幀的模型時間（毫秒，平均）
        self._mean = np.array(self.pose.mean, float)
        self._std = np.array(self.pose.std, float)
        self._use(self.pose.session)
        self._cand = None
        self._trials = []
        self._force16 = precision == 'fp16'
        if precision in ('auto', 'fp16') and device != 'cpu' and paths.get('convert', True):
            f16 = models.fp16_model(src)
            if f16:
                try:
                    import onnxruntime as ort
                    self._cand = ort.InferenceSession(f16, providers=[P.PROVIDER.get(device, device)])
                except Exception as e:
                    models.WARNINGS.append(f'半精度模型無法載入：{type(e).__name__}: {e}')

    def _use(self, sess):
        self.sess = sess
        self._in = sess.get_inputs()[0].name
        self._outs = [o.name for o in sess.get_outputs()]
        dim0 = sess.get_inputs()[0].shape[0]
        self.can_batch = not (isinstance(dim0, int) and dim0 == 1)

    def _detect(self, img):
        with P.GPU_LOCK:
            try:
                boxes = self.det(img)
            except UnicodeDecodeError as e:
                raise P.GPUError('ERR_GPU') from e
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return None
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if self.bbox is not None:                           # 盡量跟著上一個人
            b = self.bbox
            ix = np.clip(np.minimum(boxes[:, 2], b[2]) - np.maximum(boxes[:, 0], b[0]), 0, None)
            iy = np.clip(np.minimum(boxes[:, 3], b[3]) - np.maximum(boxes[:, 1], b[1]), 0, None)
            inter = ix * iy
            score = inter / (area + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
            if score.max() > 0.2:
                return boxes[int(np.argmax(score))]
        return boxes[int(np.argmax(area))]

    def _crop(self, img, bbox):
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
        center, scale = bbox_xyxy2cs(np.array(bbox, float), padding=1.25)
        x, scale = top_down_affine(self.pose.model_input_size, scale, center, img)
        x = (x - self._mean) / self._std
        return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32), center, scale

    def _infer(self, sess, x):
        n = len(x)
        if self.pad_batch and n < self.pad_batch and self.can_batch:   # MIGraphX：形狀固定，免得每次重新編譯
            x = np.concatenate([x, np.repeat(x[-1:], self.pad_batch - n, 0)])
            return [o[:n] for o in P.run(sess, self._outs, {self._in: x})]
        if len(x) > 1 and not self.can_batch:
            outs = [P.run(sess, self._outs, {self._in: x[i:i + 1]}) for i in range(len(x))]
            return [np.concatenate([o[k] for o in outs]) for k in range(len(outs[0]))]
        return P.run(sess, self._outs, {self._in: x})

    def _decode(self, outs, crops):
        res = []
        for i, (_, center, scale) in enumerate(crops):
            kpts, scores, _, k2d = self.pose.postprocess([o[i:i + 1] for o in outs], center, scale)
            res.append((np.asarray(k2d[0], float), np.asarray(kpts[0][:, 2], float), np.asarray(scores[0], float)))
        return res

    def _trial(self, x, crops, ref, t32):
        """半精度候選：同一張圖兩邊都算，比對位置和時間；看夠了就決定要不要換過去。"""
        import time
        k, _, s = ref
        good = s[:23] > 0.5
        if good.sum() < 8:                                          # 畫面裡沒有人：不能比對
            return
        t = time.perf_counter()
        try:
            o16 = self._decode(self._infer(self._cand, x), crops)[0]          # 同樣的批次大小才公平
        except Exception:
            self._cand = None
            return
        t16 = (time.perf_counter() - t) / len(x)
        span = float(np.ptp(k[:23][good], axis=0).max()) + 1e-6
        err = float(np.median(np.linalg.norm(k[:23][good] - o16[0][:23][good], axis=1))) / span
        self._trials.append((err, t32 / len(x), t16))
        if len(self._trials) >= self.TRIALS:
            tr = self._trials[2:]                                   # 前兩次含暖機
            err = float(np.median([a for a, _, _ in tr]))
            t32 = float(np.median([b for _, b, _ in tr]))
            t16 = float(np.median([c for _, _, c in tr]))
            if err < 0.012 and (self._force16 or t16 < 0.9 * t32):
                self._use(self._cand)
                self.pose.session = self._cand                  # 放掉全精度的那份（顯示卡記憶體）
                self.precision = 'fp16'
            self.trial_result = dict(err=err, ms32=1000 * t32, ms16=1000 * t16, used=self.precision)
            self._cand = None

    def _run(self, crops, frames=None):
        """crops：[(input (3,H,W), center, scale)] → [(k2d, z, scores)]（133 點）。frames：這批是幾幀畫面（算每幀時間用）。"""
        import time
        x = np.stack([c[0] for c in crops])
        t = time.perf_counter()
        res = self._decode(self._infer(self.sess, x), crops)
        dt = time.perf_counter() - t
        n = frames or len(crops)
        self.ms = 0.9 * self.ms + 0.1 * 1000 * dt / n if self.ms else 1000 * dt / n
        if self._cand is not None:
            self._trial(x, crops, res[0], dt)
        return res

    def _update_bbox(self, img, k, s):
        body = s[:17] > 0.3
        if body.sum() < 6:
            self.bbox = None
            return False
        pts = k[s > 0.3]
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.1 * (hi - lo) + 10
        h, w = img.shape[:2]
        self.bbox = np.array([max(lo[0] - pad[0], 0), max(lo[1] - pad[1], 0),
                              min(hi[0] + pad[0], w - 1), min(hi[1] + pad[1], h - 1)])
        return True

    def _halpe(self, raw):
        k2d, z, sc = raw
        self.raw = raw if len(k2d) >= 133 else None
        self.hands = self.cue = self.cue_detail = None
        if self.raw is not None:
            self.cue_detail = {}
            self.cue = LB.head_cue(k2d, sc, self.cue_detail)      # 手和頭重疊時在前面還是後面（看臉部的點）
            self.hands = Hands(a[91:133] for a in raw)
            self.hands.cue = self.cue
        return wholebody_to_halpe(k2d, z, sc)

    def batch(self, imgs):
        """連續的幾幀（同一個人物框）。回傳 [(k, z, s, hands)]；沒有人的幀是 (None, None, None, None)。"""
        if self.bbox is None or self.n % self.freq < len(imgs):
            b = self._detect(imgs[-1])
            if b is not None:
                self.bbox = b
        self.n += len(imgs)
        if self.bbox is None:
            return [(None, None, None, None)] * len(imgs)
        crops = [self._crop(im, self.bbox) for im in imgs]
        if self.live_tta:                                   # 每幀再算一次左右翻轉的，放在同一批
            fb = self._flip_box(imgs[0].shape[1], self.bbox)
            crops += [self._crop(np.ascontiguousarray(im[:, ::-1]), fb) for im in imgs]
        res = self._run(crops, len(imgs))
        out = []
        for i, im in enumerate(imgs):
            if self.live_tta:
                k, d, s = self._unflip_merge(res[i], res[len(imgs) + i], im.shape[1])
            else:
                k, d, s = self._halpe(res[i])
            out.append((k, d, s, self.hands) if (s[:17] > 0.3).sum() >= 6 else (None, None, None, None))
        self._update_bbox(imgs[-1], k, s)
        return out

    def __call__(self, img):
        if self.bbox is None or self.n % self.freq == 0:
            b = self._detect(img)
            if b is not None:
                self.bbox = b
        self.n += 1
        if self.bbox is None:
            return None, None, None
        if self.tta:
            k, d, s = self._pose_tta(img, self.bbox)
        else:
            k, d, s = self._halpe(self._run([self._crop(img, self.bbox)])[0])
        if not self._update_bbox(img, k, s):
            return None, None, None
        return k, d, s

    @staticmethod
    def _flip_box(w, bbox):
        return [w - 1 - bbox[2], bbox[1], w - 1 - bbox[0], bbox[3]]

    def _pose_tta(self, img, bbox):
        """原圖和左右翻轉放在同一批一起算；手部保留原圖的結果。"""
        w = img.shape[1]
        r1, r2 = self._run([self._crop(img, bbox), self._crop(np.ascontiguousarray(img[:, ::-1]), self._flip_box(w, bbox))], 1)
        return self._unflip_merge(r1, r2, w)

    def _unflip_merge(self, r1, r2, w):
        """r1：原圖的結果，r2：左右翻轉後的結果（133 點）→ 合併後的 Halpe26 (k, z, s)。self.hands 留原圖的。"""
        k2, d2, s2 = self._halpe(r2)
        k1, d1, s1 = self._halpe(r1)
        k2 = k2.copy()
        k2[:, 0] = w - 1 - k2[:, 0]
        for a, b in self.FLIP:
            k2[[a, b]], d2[[a, b]], s2[[a, b]] = k2[[b, a]], d2[[b, a]], s2[[b, a]]
        return self._merge(k1, d1, s1, k2, d2, s2)

    @staticmethod
    def _merge(k1, d1, s1, k2, d2, s2, max_px=None):
        """兩次結果依分數加權平均；位置差太多（多半是其中一次認錯）時取分數高的那個。"""
        span = float(np.ptp(k1[s1 > 0.3], axis=0).max()) if (s1 > 0.3).sum() > 2 else 200.0
        lim = max_px or max(12.0, 0.08 * span)
        w1, w2 = np.maximum(s1, 1e-3), np.maximum(s2, 1e-3)
        far = np.linalg.norm(k1 - k2, axis=1) > lim
        k = (w1[:, None] * k1 + w2[:, None] * k2) / (w1 + w2)[:, None]
        d = (w1 * d1 + w2 * d2) / (w1 + w2)
        s = 0.5 * (s1 + s2)
        pick1 = far & (s1 >= s2)
        pick2 = far & (s2 > s1)
        k[pick1], d[pick1], s[pick1] = k1[pick1], d1[pick1], s1[pick1]
        k[pick2], d[pick2], s[pick2] = k2[pick2], d2[pick2], s2[pick2]
        return k, d, s


# ---------------------------------------------------------------- 單張圖片
def lift_image(k, z, s, size, hfov_deg=60.0, height=1.65, thr=0.3):
    """一張圖：假設相機大致水平，深度用骨頭長度推算。回傳 (X (26,3), ok) 或 (None, ok)。
    座標同 Lifter：x = 相機右方、y = 上、z = 朝向相機；地板 = 0，骨盆水平位置在原點。"""
    w, h = size
    f = 0.5 * max(w, h) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    Xc, ok, _ = camera_points(k, z, s, K, height, thr)
    if Xc is None:
        return None, ok
    X = Xc @ np.diag([1.0, -1.0, -1.0]).T
    X = fill_missing(X, ok, height)
    low = [X[j, 1] for j in SOLE if ok[j]]
    if not low:
        low = [X[j, 1] for j in (P.L_ANK, P.R_ANK) if ok[j]]
        low = [v - 0.07 for v in low]
    floor = min(low) if low else float(X[:, 1].min())
    X[:, 1] -= floor
    X[:, [0, 2]] -= X[P.HIP, [0, 2]]
    return X, ok


def fill_missing(X, ok, height):
    """照片裡看不到的關節（例如只拍到上半身）用自然站姿補上，讓骨架完整。"""
    X = X.copy()
    have = ok.copy()
    up = np.array([0.0, 1.0, 0.0])

    def need(*js):
        return all(have[j] for j in js)
    if not need(P.L_HIP, P.R_HIP):
        if need(P.L_SHO, P.R_SHO):
            mid = 0.5 * (X[P.L_SHO] + X[P.R_SHO]) - up * 0.29 * height
            side = (X[P.L_SHO] - X[P.R_SHO]) * 0.5
            X[P.L_HIP], X[P.R_HIP] = mid + side, mid - side
            have[P.L_HIP] = have[P.R_HIP] = True
        else:
            return X
    X[P.HIP] = 0.5 * (X[P.L_HIP] + X[P.R_HIP])
    left = X[P.L_HIP] - X[P.R_HIP]
    left[1] = 0
    left /= max(np.linalg.norm(left), 1e-6)
    fwd = np.cross(left, up)                     # 角色的前方
    if not need(P.L_SHO, P.R_SHO):
        X[P.L_SHO] = X[P.L_HIP] + up * 0.29 * height + left * 0.08
        X[P.R_SHO] = X[P.R_HIP] + up * 0.29 * height - left * 0.08
    X[P.NECK] = 0.5 * (X[P.L_SHO] + X[P.R_SHO])
    for a, b, c, L1, L2 in ((P.L_SHO, P.L_ELB, P.L_WRI, 0.186, 0.146), (P.R_SHO, P.R_ELB, P.R_WRI, 0.186, 0.146)):
        if not have[b]:
            X[b] = X[a] - up * L1 * height
        if not have[c]:
            X[c] = X[b] - up * L2 * height
    for a, b, c, toe, small, heel in ((P.L_HIP, P.L_KNEE, P.L_ANK, P.L_BIGTOE, P.L_SMALLTOE, P.L_HEEL),
                                      (P.R_HIP, P.R_KNEE, P.R_ANK, P.R_BIGTOE, P.R_SMALLTOE, P.R_HEEL)):
        if not have[b]:
            X[b] = X[a] - up * 0.245 * height
        if not have[c]:
            X[c] = X[b] - up * 0.246 * height
        if not have[toe]:
            X[toe] = X[c] + fwd * 0.15 - up * 0.06
        if not have[small]:
            X[small] = X[c] + fwd * 0.12 - up * 0.06
        if not have[heel]:
            X[heel] = X[c] - fwd * 0.05 - up * 0.06
    if not (have[P.L_EAR] and have[P.R_EAR]):
        c = X[P.NECK] + up * 0.17
        X[P.L_EAR], X[P.R_EAR] = c + left * 0.075, c - left * 0.075
    if not have[P.NOSE]:
        X[P.NOSE] = 0.5 * (X[P.L_EAR] + X[P.R_EAR]) + fwd * 0.1
    return X
