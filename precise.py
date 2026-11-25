"""精度優先（桌面模式用）：延遲一小段時間，用前後的畫面一起決定。

1. 轉身／前後：整個人左右標籤對調是單鏡頭最常見的錯（模型把「面向鏡頭」看成「背對」）。
   每幀兩種假設（照原樣、左右對調），用 Viterbi 在一段時間內找最連貫的一條：
   - 證據：臉的分數（看得到臉 → 面向鏡頭）、鼻子在耳朵前面、腳尖在腳跟前面、膝蓋往前彎
   - 連貫：身體朝向每幀能轉的角度有限（真的轉身是連續的，標籤對調是瞬間 180°）
2. 上半身：扭腰時肩膀哪邊在前、彎腰時上半身往鏡頭還是往後倒，單鏡頭也分不出來。
   每幀四種假設（照原樣、肩膀手臂以脖子為中心前後鏡像、整個上半身以骨盆為中心前後鏡像、兩者），
   證據是身體構造（扭腰不超過約 60°、上半身不會往後倒超過約 30°、臉朝胸口前方），連貫是關節的速度不會突然反向。
3. 腿、手臂各自左右對調：同樣用 Viterbi，證據是左右位置，連貫是關節位置不會瞬移。
4. 去抖動：以每個時間點為中心、前後各一段的加權二次曲線擬合（分數低、離群的點權重小）。
"""
import collections

import numpy as np

import pose as P

_PAIRS_BODY = [(P.L_EYE, P.R_EYE), (P.L_EAR, P.R_EAR), (P.L_SHO, P.R_SHO), (P.L_ELB, P.R_ELB),
               (P.L_WRI, P.R_WRI), (P.L_HIP, P.R_HIP), (P.L_KNEE, P.R_KNEE), (P.L_ANK, P.R_ANK),
               (P.L_BIGTOE, P.R_BIGTOE), (P.L_SMALLTOE, P.R_SMALLTOE), (P.L_HEEL, P.R_HEEL)]
_PAIRS_LEGS = [(P.L_KNEE, P.R_KNEE), (P.L_ANK, P.R_ANK), (P.L_BIGTOE, P.R_BIGTOE),
               (P.L_SMALLTOE, P.R_SMALLTOE), (P.L_HEEL, P.R_HEEL)]
_PAIRS_ARMS = [(P.L_ELB, P.R_ELB), (P.L_WRI, P.R_WRI)]


def _perm(pairs):
    idx = np.arange(P.NUM_JOINTS)
    for a, b in pairs:
        idx[a], idx[b] = b, a
    return idx


PERM_BODY = _perm(_PAIRS_BODY)
PERM_LEGS = _perm(_PAIRS_LEGS)
PERM_ARMS = _perm(_PAIRS_ARMS)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


_Y = np.array([0.0, 1.0, 0.0])
UPPER = [P.NOSE, P.L_EYE, P.R_EYE, P.L_EAR, P.R_EAR, P.HEAD, P.NECK,
         P.L_SHO, P.R_SHO, P.L_ELB, P.R_ELB, P.L_WRI, P.R_WRI]
ARMS = [P.L_SHO, P.R_SHO, P.L_ELB, P.R_ELB, P.L_WRI, P.R_WRI]


def facing(X):
    """骨盆前方（水平、單位向量）；算不出來時 None。只用左右方向（彎腰時「上」是水平的，不能拿來算）；
    肩膀只佔一點（扭腰時肩膀不代表整個人的朝向，上半身另外判斷）。"""
    side = 0.3 * (X[P.R_SHO] - X[P.L_SHO]) + (X[P.R_HIP] - X[P.L_HIP])
    f = np.cross(_Y, side)
    f[1] = 0.0
    return _unit(f)


def head_facing(X, s, thr=0.3):
    """頭的前方（水平）：耳朵連線；看不到耳朵時 None。"""
    if s[P.L_EAR] < thr or s[P.R_EAR] < thr:
        return None
    return _unit(np.cross(_Y, X[P.R_EAR] - X[P.L_EAR]) * np.array([1.0, 0.0, 1.0]))


def upright(X):
    """上半身有多直（1 = 站直，0 = 彎到水平以下）：彎腰時臉、鼻子的線索不可靠。"""
    u = _unit(X[P.NECK] - X[P.HIP])
    return 0.0 if u is None else float(np.clip((u[1] - 0.3) / 0.5, 0.0, 1.0))


def anatomy(X, s, thr=0.3):
    """照目前的標籤，「前方」是不是和身體構造一致：> 0 一致，< 0 多半左右標反了。"""
    f = facing(X)
    if f is None:
        return 0.0
    c = 0.0                                                  # 只看腿：上半身（臉、肩膀）另外判斷
    for toe, heel in ((P.L_BIGTOE, P.L_HEEL), (P.R_BIGTOE, P.R_HEEL)):
        if s[toe] > thr and s[heel] > thr:
            c += 0.6 * float(np.clip(((X[toe] - X[heel]) @ f) / 0.06, -2, 2))
    for hip, knee, ank in ((P.L_HIP, P.L_KNEE, P.L_ANK), (P.R_HIP, P.R_KNEE, P.R_ANK)):
        if s[hip] > thr and s[knee] > thr and s[ank] > thr:
            m = 0.5 * (X[hip] + X[ank])
            d = (X[knee] - m) @ f
            if abs(d) > 0.04:                                # 腿打直時沒有資訊
                c += 0.5 * float(np.clip(d / 0.06, -2, 2))
    return c


def face_score(s):
    """看得到臉的程度（-1.5 背對 … +1.5 面向鏡頭）。"""
    v = float(np.mean([s[P.NOSE], s[P.L_EYE], s[P.R_EYE]]))
    return float(np.clip((v - 0.5) / 0.2, -1.5, 1.5))


class _Frame:
    __slots__ = ('t', 'X', 's', 'ok', 'hands', 'cam', 'cache')

    def __init__(self, t, X, s, ok, hands, cam):
        self.t, self.X, self.s, self.ok, self.hands, self.cam = t, X, s, ok, hands, cam
        self.cache = {}


def _swap_hands(h):
    if not h:
        return h
    return {'Left': h.get('Right'), 'Right': h.get('Left')}


def _apply(fr, perm, swap_hands):
    fr.X = fr.X[perm]
    fr.s = fr.s[perm]
    fr.ok = fr.ok[perm]
    if swap_hands:
        fr.hands = _swap_hands(fr.hands)


def _rate(frames):
    """每幀的權重 = 這幀代表的時間（以 30fps 為 1）：證據和連貫的成本都以「每秒」計，不同幀率結果一致。"""
    t = np.array([f.t for f in frames])
    d = np.diff(t, prepend=t[0] - 1 / 30.0) if len(t) > 1 else np.array([1 / 30.0])
    return np.clip(d * 30.0, 0.25, 2.0)


def viterbi(unary, pair, start=None, w=None):
    """unary (T,N) 成本、pair(t) → (N,N) 從 t-1 到 t 的成本；w (T,)：每幀的權重。回傳狀態序列。"""
    T, N = unary.shape
    if w is not None:
        unary = unary * w[:, None]
        p0 = pair
        pair = lambda t: p0(t) * w[t]  # noqa: E731
    cost = unary[0].copy()
    if start is not None:                       # 第一幀已經定案
        cost = np.where(np.arange(N) == start, 0.0, 1e9)
    back = np.zeros((T, N), int)
    for t in range(1, T):
        m = cost[:, None] + pair(t)
        back[t] = np.argmin(m, axis=0)
        cost = m[back[t], np.arange(N)] + unary[t]
    path = np.zeros(T, int)
    path[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def mirror(X, cam):
    """沿著相機視線、以骨盆為中心前後鏡像（單鏡頭分不出的另一種解）。"""
    d = cam - X[P.HIP]
    d[1] = 0.0
    d = _unit(d)
    if d is None:
        return X
    return X - 2.0 * np.outer((X - X[P.HIP]) @ d, d)


def body_variant(X, s, k, cam):
    """k：bit0 = 左右標籤對調，bit1 = 前後鏡像。"""
    if k & 1:
        X, s = X[PERM_BODY], s[PERM_BODY]
    if k & 2 and cam is not None:
        X = mirror(X, cam)
    return X, s


def _reflect(X, idx, center, cam):
    d = cam - center
    d[1] = 0.0
    d = _unit(d)
    if d is None:
        return X
    X = X.copy()
    X[idx] = X[idx] - 2.0 * np.outer((X[idx] - center) @ d, d)
    return X


def upper_variant(X, k, cam):
    """k：bit0 = 肩膀、手臂以脖子為中心前後鏡像（扭腰方向），bit1 = 整個上半身以骨盆為中心前後鏡像（彎腰方向）。"""
    if cam is None or k == 0:
        return X
    if k & 1:
        X = _reflect(X, ARMS, X[P.NECK].copy(), cam)
    if k & 2:
        X = _reflect(X, UPPER, X[P.HIP].copy(), cam)
    return X


def _signed(a, b, axis):
    return float(np.degrees(np.arctan2(np.cross(a, b) @ axis, a @ b)))


def torso_angles(X):
    """(扭腰角, 前彎角)（度）：肩線相對骨盆線繞脊椎的角度；上半身相對骨盆前方往前倒的角度（往後為負）。"""
    r_h = X[P.R_HIP] - X[P.L_HIP]
    r_s = X[P.R_SHO] - X[P.L_SHO]
    u = _unit(X[P.NECK] - X[P.HIP])
    f_p = _unit(np.cross(_Y, r_h))
    if u is None or f_p is None:
        return None
    a = r_h - (r_h @ u) * u
    b = r_s - (r_s @ u) * u
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return None
    tw = _signed(a, b, u)
    lean = float(np.degrees(np.arctan2(u @ f_p, u[1])))
    return tw, lean


def upper_anatomy(X, s, thr=0.3):
    """上半身姿勢不合人體構造的程度（越大越不可能）。"""
    ang = torso_angles(X)
    if ang is None:
        return 0.0
    tw, lean = ang
    c = min((max(0.0, abs(tw) - 55.0) / 15.0) ** 2, 9.0)          # 扭腰（含肩胛）大約到 60°
    c += min((max(0.0, -lean - 30.0) / 15.0) ** 2, 9.0)           # 往後仰大約到 30°，往前可以彎很多
    u = _unit(X[P.NECK] - X[P.HIP])
    f_c = _unit(np.cross(u, X[P.R_SHO] - X[P.L_SHO]))             # 胸口前方
    if f_c is not None and s[P.NOSE] > thr and s[P.L_EAR] > thr and s[P.R_EAR] > thr:
        d = (X[P.NOSE] - 0.5 * (X[P.L_EAR] + X[P.R_EAR])) @ f_c
        c -= 0.7 * float(np.clip(d / 0.04, -2, 2))                 # 臉在胸口前方
    if f_c is not None and s[P.L_EAR] > thr and s[P.R_EAR] > thr:  # 頭相對肩膀轉不到 80° 以上
        e = X[P.R_EAR] - X[P.L_EAR]
        b = X[P.R_SHO] - X[P.L_SHO]
        e, b = e - (e @ u) * u, b - (b @ u) * u
        if np.linalg.norm(e) > 1e-6 and np.linalg.norm(b) > 1e-6:
            c += min((max(0.0, abs(_signed(b, e, u)) - 80.0) / 20.0) ** 2, 4.0)
    return c


# ---------------------------------------------------------------- 去抖動
def smooth_window(frames, c, smooth_s):
    """快的動作用短的範圍（保留細節），靜止或慢的時候用長的範圍（壓掉抖動）。"""
    t0 = frames[c].t
    ts = np.array([f.t - t0 for f in frames])
    Xs = np.stack([f.X for f in frames])                   # (T,26,3)
    S = np.stack([np.where(f.ok, np.maximum(f.s, 0.05), 0.0) for f in frames])   # (T,26)
    A = np.column_stack([np.ones_like(ts), ts, ts * ts])
    out = frames[c].X.copy()

    T, J = S.shape
    AA = (A[:, :, None] * A[:, None, :]).reshape(T, 9)
    cols = np.arange(J)

    def fit(sig):
        """所有關節一起做加權二次擬合（正規方程式，3 次重新加權）。回傳 (係數 (J,3,3), 可用 (J,))。"""
        g = np.exp(-0.5 * (ts / sig) ** 2)
        w = g[:, None] * S
        used = w > 1e-3
        n = used.sum(0)
        valid = (w.sum(0) > 1e-6) & (n >= 4)
        eye = 1e-9 * np.eye(3)
        for _ in range(3):                                  # 離群的點（認錯、跳動）權重降低
            M = (w.T @ AA).reshape(J, 3, 3) + eye
            rhs = (A.T @ (w[:, :, None] * Xs).reshape(T, J * 3)).reshape(3, J, 3).transpose(1, 0, 2)
            coef = np.linalg.solve(M, rhs)
            pred = (A @ coef.transpose(1, 0, 2).reshape(3, J * 3)).reshape(T, J, 3)
            r = np.linalg.norm(Xs - pred, axis=2)
            srt = np.sort(np.where(used, r, np.inf), axis=0)
            m = np.maximum(n, 1)
            med = 0.5 * (srt[(m - 1) // 2, cols] + srt[m // 2, cols])
            scale = 1.4826 * np.where(valid, med, 0.0) + 0.01
            w = g[:, None] * S / np.maximum(1.0, r / (2.5 * scale))
        return coef, valid

    long, lv = fit(0.6 * smooth_s)
    short, sv = fit(0.25 * smooth_s)
    v = np.linalg.norm(short[:, 1, :], axis=1)              # 速度（公尺／秒）
    a = np.clip((v - 0.3) / 1.2, 0.0, 1.0)                  # 0.3 m/s 以下全用長的，1.5 m/s 以上全用短的
    a = np.where(sv, a, 0.0)[:, None]
    val = (1 - a) * long[:, 0, :] + a * short[:, 0, :]
    out[lv] = val[lv]
    hands = frames[c].hands
    if hands:
        hands = dict(hands)
        g = np.exp(-0.5 * (ts / (0.3 * smooth_s)) ** 2)
        for side in ('Left', 'Right'):
            seq = [f.hands.get(side) if f.hands else None for f in frames]
            if all(h is not None for h in seq):
                H = np.stack(seq)
                W = (g / g.sum())[:, None, None]
                hands[side] = (H * W).sum(0)
    return out, hands


class Smoother:
    """只做去抖動（不判斷左右、前後）：以每一幀為中心、前後各 smooth_s 秒一起擬合，所以沒有濾波造成的落後；
    代價是多 smooth_s 秒的延遲。push(t, X, s, ok, hands) → 延遲後的 (t, X, ok, hands) 或 None。"""

    def __init__(self, smooth_s=0.25):
        self.smooth_s = float(smooth_s)
        self.done = collections.deque()
        self.out_i = 0

    @property
    def delay_s(self):
        return self.smooth_s

    def push(self, t, X, s, ok, hands=None, cam=None):
        if X is None:
            return None
        if self.smooth_s <= 0.0:
            return float(t), X, ok, hands
        if self.done and (t - self.done[-1].t > 3.0 or t <= self.done[-1].t):      # 中斷太久或時間倒退：重新開始
            self.done.clear()
            self.out_i = 0
        self.done.append(_Frame(float(t), np.asarray(X, float).copy(), np.asarray(s, float).copy(),
                                np.asarray(ok, bool).copy(), dict(hands) if hands else None, None))
        return _emit(self)


def _emit(o):
    """o.done 裡已經有後面 smooth_s 秒資料的幀：平滑後交出（一次 push 最多交出最新的一個）。"""
    result = None
    while o.out_i < len(o.done) and o.done[-1].t - o.done[o.out_i].t >= o.smooth_s:
        frames = list(o.done)
        c = o.out_i
        tc = frames[c].t
        lo = c
        while lo > 0 and tc - frames[lo - 1].t <= o.smooth_s:
            lo -= 1
        hi = c
        while hi + 1 < len(frames) and frames[hi + 1].t - tc <= o.smooth_s:
            hi += 1
        win, ci = frames[lo:hi + 1], c - lo
        if len(win) > 40:                                    # 幀率很高時隔幾幀取一幀（中間那幀一定留著），計算量不會跟著幀率漲
            step = -(-len(win) // 32)
            win = win[ci % step::step]
            ci //= step
        X2, h2 = smooth_window(win, ci, o.smooth_s)
        result = (tc, X2, frames[c].ok, h2)
        o.out_i += 1
        while o.done and tc - o.done[0].t > o.smooth_s:     # 不再需要的舊幀
            o.done.popleft()
            o.out_i -= 1
    return result


class Precise:
    """push(t, X, s, ok, hands, cam) → 延遲後的 (t, X, ok, hands) 或 None。

    X：影片模式的世界座標 (26,3)；s：分數；cam：相機在世界座標的位置。"""

    def __init__(self, fps=30.0, decide_s=0.8, smooth_s=0.3):
        # 時間為準（幀率會變）：fps 只用來估計延遲
        self.decide_s = float(decide_s)
        self.smooth_s = float(smooth_s)
        self.fps = float(fps)
        self.out_i = 0
        self.win = collections.deque()            # 還沒決定標籤的
        self.done = collections.deque()           # 已決定、等著平滑的
        self.state = {'body': None, 'upper': None, 'legs': None, 'arms': None}
        self.anchor = None                        # 上一個定案的幀（原始標籤）
        self.omega = np.radians(400.0)            # 身體轉動的合理上限（度／秒）
        self.step = 0.05

    def reset(self):
        self.win.clear()
        self.done.clear()
        self.out_i = 0
        self.state = {'body': None, 'upper': None, 'legs': None, 'arms': None}
        self.anchor = None

    # ------------------------------------------------------------ 標籤
    def _decide_body(self, frames):
        T = len(frames)
        N = 4 if all(f.cam is not None for f in frames) else 2
        unary = np.zeros((T, N))
        F = np.full((T, N, 3), np.nan)
        for i, fr in enumerate(frames):
            c = fr.cache.get(('body', N))
            if c is not None:
                unary[i], F[i] = c
                continue
            fs = face_score(fr.s) * upright(fr.X)          # 彎腰時臉本來就看不到，不能當成背對
            for k in range(N):
                X, s = body_variant(fr.X, fr.s, k, fr.cam)
                f = facing(X)
                hf = head_facing(X, s)                     # 看得到臉 ↔ 頭朝向鏡頭（扭腰時頭不一定和骨盆同方向）
                toward = 0.0
                if hf is not None and fr.cam is not None:
                    d = fr.cam - X[P.HIP]
                    d[1] = 0.0
                    d = _unit(d)
                    toward = float(hf @ d) if d is not None else 0.0
                elif f is not None and fr.cam is not None:
                    d = fr.cam - X[P.HIP]
                    d[1] = 0.0
                    d = _unit(d)
                    toward = float(f @ d) if d is not None else 0.0
                unary[i, k] = -(0.7 * anatomy(X, s) + 1.0 * fs * toward) + (0.0, 0.2, 0.35, 0.35)[k]   # 沒有證據時相信模型（鏡像那兩種分不太出來，要多一點證據）
                if f is not None:
                    F[i, k] = f
            fr.cache[('body', N)] = (unary[i].copy(), F[i].copy())

        def pair(t):
            dt = max(frames[t].t - frames[t - 1].t, 1e-3)
            g0, g1 = F[t - 1], F[t]
            c = np.clip(g0 @ g1.T, -1, 1)
            ang = np.arccos(c)
            out = 4.0 * (ang / (self.omega * dt)) ** 2
            return np.nan_to_num(out, nan=0.0)
        return viterbi(unary, pair, self.state['body'], _rate(frames))

    def _decide_upper(self, frames):
        T = len(frames)
        N = 4 if all(f.cam is not None for f in frames) else 1
        if N == 1:
            return np.zeros(T, int)
        unary = np.zeros((T, N))
        V = []
        for i, fr in enumerate(frames):
            c = fr.cache.get('upper')
            if c is None:
                vs = np.stack([upper_variant(fr.X, k, fr.cam)[UPPER] for k in range(N)])
                u = np.array([upper_anatomy(upper_variant(fr.X, k, fr.cam), fr.s) for k in range(N)])
                c = fr.cache['upper'] = (vs, u + np.array((0.0, 0.25, 0.25, 0.45)))
            V.append(c[0])
            unary[i] = c[1]

        # Viterbi，連貫成本用等速預測（真的動作速度不會突然反向）；前一幀的速度沿著各自最好的路徑算
        w = _rate(frames)
        unary = unary * w[:, None]
        start = self.state['upper']
        cost = unary[0].copy() if start is None else np.where(np.arange(N) == start, 0.0, 1e9)
        back = np.zeros((T, N), int)
        for t in range(1, T):
            dt = max(frames[t].t - frames[t - 1].t, 1e-3)
            A, B = V[t - 1], V[t]                                  # (N,M,3)
            if t >= 2:
                dp = max(frames[t - 1].t - frames[t - 2].t, 1e-3)
                pred = A + 0.8 * (A - V[t - 2][back[t - 1]]) * min(dt / dp, 2.0)
            else:
                pred = A
            sig = 0.05 + 12.0 * dt * dt + 0.3 * dt
            d = np.linalg.norm(B[None] - pred[:, None], axis=3)     # (N,N,M)
            pc = np.mean(np.minimum((d / sig) ** 2, 25.0), axis=2) * 3.0 * w[t]
            m = cost[:, None] + pc
            back[t] = np.argmin(m, axis=0)
            cost = m[back[t], np.arange(N)] + unary[t]
        path = np.zeros(T, int)
        path[-1] = int(np.argmin(cost))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def _decide_pair(self, frames, key, perm, joints, lateral_w):
        T = len(frames)
        unary = np.zeros((T, 2))
        root = (P.L_HIP, P.R_HIP) if key == 'legs' else (P.L_SHO, P.R_SHO)
        for i, fr in enumerate(frames):
            c = fr.cache.get(('pair', key))
            if c is not None:
                unary[i] = c
                continue
            X, sc = fr.X, fr.s
            a, b = joints
            e = 0.0
            if min(sc[a], sc[b], sc[root[0]], sc[root[1]]) > 0.3:
                # 骨頭長度：膝蓋（手肘）離自己那一側的骨盆（肩膀）比較近
                own = np.linalg.norm(X[a] - X[root[0]]) + np.linalg.norm(X[b] - X[root[1]])
                other = np.linalg.norm(X[a] - X[root[1]]) + np.linalg.norm(X[b] - X[root[0]])
                e += float(np.clip((other - own) / 0.08, -2, 2))
                r = _unit(X[P.R_HIP] - X[P.L_HIP] + X[P.R_SHO] - X[P.L_SHO])
                if r is not None:
                    e += lateral_w * float(np.clip(((X[b] - X[a]) @ r) / 0.12, -2, 2))
            unary[i] = (-e, e)
            fr.cache[('pair', key)] = unary[i].copy()

        moving = [j for pr in (_PAIRS_LEGS if key == 'legs' else _PAIRS_ARMS) for j in pr]

        def pair(t):
            prev, cur = frames[t - 1], frames[t]
            c = cur.cache.get(('pp', key))
            if c is not None and c[0] is prev:
                return c[1]
            A, B = prev.X, cur.X
            dt = max(cur.t - prev.t, 1e-3)
            sig = 0.04 + 1.5 * dt                     # 這段時間能動多遠（公尺）
            a2 = np.stack([A[moving], A[perm][moving]])           # (2,M,3)
            b2 = np.stack([B[moving], B[perm][moving]])
            d = np.linalg.norm(a2[:, None] - b2[None], axis=3)    # (2,2,M)
            out = np.sum((d / sig) ** 2, axis=2) / len(moving) * 3.0
            cur.cache[('pp', key)] = (prev, out)
            return out
        return viterbi(unary, pair, self.state[key], _rate(frames))

    def _label(self):
        frames = ([self.anchor] if self.anchor is not None else []) + list(self.win)
        k = 1 if self.anchor is not None else 0
        body = self._decide_body(frames)
        # 先套用整個人的結果，再看腿、手臂（只看暫時的複本，最舊那幀才真正定案）
        tmp = []
        for i, f in enumerate(frames):
            key = ('var', int(body[i]))
            if key not in f.cache:
                f.cache[key] = _Frame(f.t, *body_variant(f.X, f.s, body[i], f.cam), f.ok, None, f.cam)
            tmp.append(f.cache[key])
        upper = self._decide_upper(tmp)
        tmp2 = []
        for i, f in enumerate(tmp):
            key = ('upv', int(upper[i]))
            if key not in f.cache:
                f.cache[key] = _Frame(f.t, upper_variant(f.X, int(upper[i]), f.cam), f.s, f.ok, None, f.cam)
            tmp2.append(f.cache[key])
        legs = self._decide_pair(tmp2, 'legs', PERM_LEGS, (P.L_KNEE, P.R_KNEE), 0.6)
        arms = self._decide_pair(tmp2, 'arms', PERM_ARMS, (P.L_ELB, P.R_ELB), 0.2)
        return body, upper, legs, arms, k

    # ------------------------------------------------------------
    def push(self, t, X, s, ok, hands=None, cam=None):
        if X is None:
            return None
        last = self.win[-1].t if self.win else (self.done[-1].t if self.done else None)
        if last is not None and t - last > 3.0:          # 中斷太久（人不見了）：重新開始
            self.reset()
        self.win.append(_Frame(float(t), np.asarray(X, float).copy(), np.asarray(s, float).copy(),
                               np.asarray(ok, bool).copy(), dict(hands) if hands else None,
                               None if cam is None else np.asarray(cam, float)))
        if len(self.win) > 2 and self.win[-1].t - self.win[0].t >= self.decide_s + self.step:
            # 一次決定一小段（step 秒）：每幀都重算整段太花時間，結果幾乎一樣
            body, upper, legs, arms, k = self._label()
            i = 0
            while len(self.win) > 2 and self.win[-1].t - self.win[0].t >= self.decide_s:
                b, u, l, a = int(body[k + i]), int(upper[k + i]), int(legs[k + i]), int(arms[k + i])
                i += 1
                self.state = {'body': b, 'upper': u, 'legs': l, 'arms': a}
                fr = self.win.popleft()
                self.anchor = _Frame(fr.t, fr.X.copy(), fr.s.copy(), fr.ok.copy(), None, fr.cam)
                if b & 1:
                    _apply(fr, PERM_BODY, True)
                if b & 2 and fr.cam is not None:
                    fr.X = mirror(fr.X, fr.cam)
                if u:
                    fr.X = upper_variant(fr.X, u, fr.cam)
                if l:
                    _apply(fr, PERM_LEGS, False)
                if a:
                    _apply(fr, PERM_ARMS, True)
                self.done.append(fr)
        return _emit(self)

    @property
    def delay_s(self):
        return self.decide_s + self.step + self.smooth_s
