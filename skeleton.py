"""骨架約束：把每幀算出來的關節點「拉回合理的人體」。

- 骨頭長度固定：先用身高比例當起點，追蹤時學出這個人的骨長：3D 長度的中位數，和「畫面上的長度」的第 85 百分位（扣掉畫面誤差）取大的。
  畫面上的長度不會超過真正的骨長，而且不受深度影響；只看 3D 長度的話，模型把深度縮回去（手常常朝鏡頭伸）時骨長會越學越短。
  同樣的理由，學到的骨長最短只到身高比例的 0.9 倍（人和人的差異大約就這麼多）。
- 沿視線修深度：單鏡頭在畫面上的位置準、前後（深度）不準。知道相機位置時，先讓每個關節只沿著自己的視線移動
  （畫面上的位置不變）去符合骨長；模型給的深度當先驗。兩種誤差各有多大不是寫死的，從骨長的殘差自己量出來：
  骨頭和視線垂直時殘差只來自畫面誤差，骨頭朝前後時多出來的部分就是深度誤差。
- 正對鏡頭的手腳（例如手往鏡頭伸直：肩膀、手肘、手腕在畫面上疊在一起）：畫面上看起來很短的骨頭一定是朝前或朝後，
  伸出去多少由骨長決定（這時幾何算得很準），模型常把這種深度縮回去甚至判反；模型和幾何差很多時才改用幾何。
  朝前還是朝後：模型有把握就聽模型的，沒把握時用人體構造（手在身體前方／外側、大腿往前抬），上臂、大腿不可能水平往背後伸。
- 再做一般的「位置約束」反覆投影收尾（修畫面上的誤差），膝蓋只能往前彎。

座標：y 上、公尺。
"""
import collections

import numpy as np

import pose as P

# (關節 a, 關節 b, 長度 / 身高)
BONES = [
    (P.L_HIP, P.L_KNEE, 0.245), (P.R_HIP, P.R_KNEE, 0.245),
    (P.L_KNEE, P.L_ANK, 0.246), (P.R_KNEE, P.R_ANK, 0.246),
    (P.L_HIP, P.R_HIP, 0.11),
    (P.L_ANK, P.L_HEEL, 0.045), (P.R_ANK, P.R_HEEL, 0.045),
    (P.L_ANK, P.L_BIGTOE, 0.10), (P.R_ANK, P.R_BIGTOE, 0.10),
    (P.L_HEEL, P.L_BIGTOE, 0.13), (P.R_HEEL, P.R_BIGTOE, 0.13),
    (P.L_SHO, P.L_ELB, 0.186), (P.R_SHO, P.R_ELB, 0.186),
    (P.L_ELB, P.L_WRI, 0.146), (P.R_ELB, P.R_WRI, 0.146),
    (P.L_SHO, P.L_HIP, 0.288), (P.R_SHO, P.R_HIP, 0.288),
    (P.L_SHO, P.R_SHO, 0.21),
]
# 骨盆是最重要的輸出，而且通常比膝蓋準：約束時讓它少動一點
MOBILITY = np.ones(P.NUM_JOINTS)
MOBILITY[[P.L_HIP, P.R_HIP]] = 0.35
MOBILITY[[P.L_SHO, P.R_SHO]] = 0.6
FOOT_POINTS = [P.L_BIGTOE, P.R_BIGTOE, P.L_SMALLTOE, P.R_SMALLTOE, P.L_HEEL, P.R_HEEL]
LEGS = ((P.L_HIP, P.L_KNEE, P.L_ANK, P.L_BIGTOE, P.L_HEEL),
        (P.R_HIP, P.R_KNEE, P.R_ANK, P.R_BIGTOE, P.R_HEEL))
# 沿視線修深度時用的骨頭：身體的（不含腳尖、腳跟），和腳掌的（腳踝固定，只移動腳尖、腳跟）
RAY_BONES = np.array([i for i, (a, b, _) in enumerate(BONES) if a not in FOOT_POINTS and b not in FOOT_POINTS])
_RA = np.array([BONES[i][0] for i in RAY_BONES])
_RB = np.array([BONES[i][1] for i in RAY_BONES])
FOOT_BONES = np.array([i for i, (a, b, _) in enumerate(BONES) if a in FOOT_POINTS or b in FOOT_POINTS])
_FA = np.array([BONES[i][0] for i in FOOT_BONES])
_FB = np.array([BONES[i][1] for i in FOOT_BONES])
# 模型的深度誤差：離軀幹越遠越大（相對值）
Z_GAIN = np.ones(P.NUM_JOINTS)
Z_GAIN[[P.L_HIP, P.R_HIP]] = 0.4
Z_GAIN[[P.L_SHO, P.R_SHO]] = 0.7
Z_GAIN[[P.L_WRI, P.R_WRI, P.L_ANK, P.R_ANK]] = 1.3
# 不在骨長約束裡的點：跟著這些關節一起移動（深度修正後頭、腳掌的形狀不變）
FOLLOW = {P.NECK: (P.L_SHO, P.R_SHO), P.HIP: (P.L_HIP, P.R_HIP),
          P.NOSE: (P.L_SHO, P.R_SHO), P.L_EYE: (P.L_SHO, P.R_SHO), P.R_EYE: (P.L_SHO, P.R_SHO),
          P.L_EAR: (P.L_SHO, P.R_SHO), P.R_EAR: (P.L_SHO, P.R_SHO), P.HEAD: (P.L_SHO, P.R_SHO),
          P.L_BIGTOE: (P.L_ANK,), P.L_SMALLTOE: (P.L_ANK,), P.L_HEEL: (P.L_ANK,),
          P.R_BIGTOE: (P.R_ANK,), P.R_SMALLTOE: (P.R_ANK,), P.R_HEEL: (P.R_ANK,)}
Z_GAIN[FOOT_POINTS] = 1.3
# 兩節的肢體：(根, 中, 末端, 另一側的根, 種類)
CHAINS = ((P.L_SHO, P.L_ELB, P.L_WRI, P.R_SHO, 'arm'), (P.R_SHO, P.R_ELB, P.R_WRI, P.L_SHO, 'arm'),
          (P.L_HIP, P.L_KNEE, P.L_ANK, P.R_HIP, 'leg'), (P.R_HIP, P.R_KNEE, P.R_ANK, P.L_HIP, 'leg'))
FS_START, FS_FULL = 0.75, 0.40      # 畫面上的長度 ÷ 骨長：小於 0.75 開始用幾何決定深度，0.40 以下完全用幾何
# 深度的幾何上限。軀幹：(骨盆側, 肩膀, 跟著肩膀一起移動的點)；手腳：(根, 中間, 末端, 跟著末端一起移動的點)
LIMIT_TORSO = ((P.L_HIP, P.L_SHO, (P.L_ELB, P.L_WRI)), (P.R_HIP, P.R_SHO, (P.R_ELB, P.R_WRI)))
LIMIT_LIMBS = ((P.L_SHO, P.L_ELB, P.L_WRI, ()), (P.R_SHO, P.R_ELB, P.R_WRI, ()),
               (P.L_HIP, P.L_KNEE, P.L_ANK, (P.L_HEEL, P.L_BIGTOE, P.L_SMALLTOE)),
               (P.R_HIP, P.R_KNEE, P.R_ANK, (P.R_HEEL, P.R_BIGTOE, P.R_SMALLTOE)))
LIMIT_TOL = 0.08                    # 骨長、遠近的誤差容許（8%：只收明顯不可能的；模擬資料上再緊會傷到本來正確的深度）
LIMIT_STAY = 0.05                   # 中間關節「留在模型給的深度附近」的權重（兩個解差不多好時選近的）


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


class Skeleton:
    def __init__(self, height=1.65, learn=True, stiffness=0.8, limit=True):
        self.height = float(height)
        self.limit = limit
        self.L = np.array([r * self.height for _, _, r in BONES])
        self.learn = learn
        self.hist = [collections.deque(maxlen=600) for _ in BONES]
        self.hist_lat = [collections.deque(maxlen=600) for _ in BONES]     # 畫面上的長度（公尺）
        self.k = stiffness
        self.n = 0
        # 誤差大小（公尺）：開始時保守（不太動深度），資料夠了改用量出來的
        self.sig_L, self.sig_z = 0.025, 0.03
        self._acc = np.zeros(5)                   # 骨長殘差的累計：垂直視線的（Σr², n）、朝前後的（Σr², Σc², n）
        self._fs = {}                             # 正對鏡頭的骨頭上一幀判斷的前／後（關節 → (±1, 第幾幀)）
        self.rho = 1.0                            # 這個人在畫面上的大小 ÷ 照目前的遠近和骨長該有的大小（慢慢更新）
        self.fs_now = {}                          # 這一幀正對鏡頭的骨頭：末端關節 → +1 朝鏡頭、-1 背向鏡頭（預覽畫面用）

    def length(self, a, b):
        for i, (x, y, _) in enumerate(BONES):
            if (x, y) == (a, b) or (x, y) == (b, a):
                return float(self.L[i])
        if P.L_SMALLTOE in (a, b) or P.R_SMALLTOE in (a, b):
            return 0.085 * self.height
        return 0.1 * self.height

    def set_lengths_from(self, frames):
        """用一批資料一次學好骨長（整段影片）。frames：[(X, ok) 或 (X, ok, 相機位置)]；
        有相機位置時順便量出畫面誤差、深度誤差的大小（沿視線修深度用）。"""
        for f in frames:
            self._observe(f[0], f[1], f[2] if len(f) > 2 else None)
        self._update()
        for f in frames:
            if len(f) > 2 and f[2] is not None:
                U = f[0] - np.asarray(f[2], float)
                self._calibrate(f[0], f[1], U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9))

    def _observe(self, X, ok, cam=None):
        cam = None if cam is None else np.asarray(cam, float)
        for i, (a, b, _) in enumerate(BONES):
            if ok[a] and ok[b]:
                d = X[b] - X[a]
                self.hist[i].append(float(np.linalg.norm(d)))
                if cam is not None:
                    u = _unit(0.5 * (X[a] + X[b]) - cam)                  # 這根骨頭的視線方向
                    if u is not None:
                        self.hist_lat[i].append(float(np.linalg.norm(d - (d @ u) * u)))

    def _update(self):
        for i, (a, b, r) in enumerate(BONES):
            h = self.hist[i]
            if len(h) >= 30:
                ref = r * self.height
                w = min(1.0, len(h) / 240.0)          # 樣本少時還是靠身高比例（開頭幾秒的姿勢不一定有代表性）
                m = float(np.median(h))
                if len(self.hist_lat[i]) >= 30:
                    # 第 85 百分位會被畫面誤差往上推大約一個標準差，扣回來
                    m = max(m, float(np.percentile(self.hist_lat[i], 85)) - self.sig_L)
                m = w * m + (1.0 - w) * ref
                self.L[i] = float(np.clip(m, 0.9 * ref, 1.3 * ref))   # 學到的骨長不能離譜

    def _calibrate(self, X, ok, U):
        """從骨長殘差量出「畫面誤差」和「深度誤差」各有多大（指數衰減的累計，約最近 20 秒）。"""
        m = ok[_RA] & ok[_RB]
        if not m.any():
            return
        a, b, L = _RA[m], _RB[m], self.L[RAY_BONES[m]]
        d = X[b] - X[a]
        ln = np.linalg.norm(d, axis=1)
        good = ln > 1e-6
        if not good.any():
            return
        a, b, L, d, ln = a[good], b[good], L[good], d[good], ln[good]
        u = d / ln[:, None]
        r2 = (ln - L) ** 2
        ca, cb = np.sum(u * U[a], axis=1), np.sum(u * U[b], axis=1)
        c = np.abs(0.5 * (ca + cb))                     # 骨頭有多朝前後（0 = 和視線垂直）
        c2 = (Z_GAIN[a] * ca) ** 2 + (Z_GAIN[b] * cb) ** 2
        flat, deep = c < 0.25, c > 0.45
        self._acc *= 1.0 - 1.0 / 600.0
        self._acc += (r2[flat].sum(), flat.sum(), r2[deep].sum(), c2[deep].sum(), deep.sum())
        s_in, n_in, s_t, c_t, n_t = self._acc
        if n_in > 150 and n_t > 150:
            v_l = s_in / n_in
            self.sig_L = float(np.clip(np.sqrt(v_l), 0.008, 0.05))
            v_z = (s_t / n_t - v_l) / max(c_t / n_t, 1e-3)
            self.sig_z = float(np.clip(np.sqrt(max(v_z, 0.0)), 0.005, 0.12))

    def _ray_fit(self, X, U, a, b, L, free, iters=4):
        """骨頭 (a, b, 長度 L)；free (26,) bool：可以沿視線移動的關節。回傳每個關節沿視線的移動量 (26,)。"""
        js = np.flatnonzero(free)
        s = np.zeros(P.NUM_JOINTS)
        if len(js) == 0 or len(a) == 0:
            return s
        col = np.full(P.NUM_JOINTS, -1)
        col[js] = np.arange(len(js))
        rows = np.arange(len(a))
        fa, fb = free[a], free[b]
        wp = 1.0 / (self.sig_z * Z_GAIN[js]) ** 2
        for _ in range(iters):
            Xc = X + s[:, None] * U
            d = Xc[b] - Xc[a]
            ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            u = d / ln[:, None]
            J = np.zeros((len(a), len(js)))
            J[rows[fb], col[b[fb]]] = np.sum(u[fb] * U[b[fb]], axis=1) / self.sig_L
            J[rows[fa], col[a[fa]]] = -np.sum(u[fa] * U[a[fa]], axis=1) / self.sig_L
            r = (ln - L) / self.sig_L
            step = np.linalg.solve(J.T @ J + np.diag(wp), -(J.T @ r + wp * s[js]))
            s[js] += np.clip(step, -0.08, 0.08)
        return s

    def _pick(self, key, cam, u, pa, L, t0, prior, veto=None):
        """骨頭從 pa 出發、末端在視線 u 上、長度 L：回傳 (末端到相機的距離, 採用幾何的比例 0~1, 朝向)；不夠短時 (t0, 0, 0)。
        t0：模型給的距離；prior：人體構造偏好的方向；veto(d) → True 表示這個方向不可能。"""
        w = pa - cam
        m = float(u @ w)
        lat = np.sqrt(max(float(w @ w) - m * m, 0.0))            # 骨頭在畫面上的長度（公尺）
        fs = float(np.clip((FS_START - lat / L) / (FS_START - FS_FULL), 0.0, 1.0))
        if fs <= 0.0:
            self._fs.pop(key, None)
            return t0, 0.0, 0
        g = np.sqrt(max(L * L - lat * lat, 0.0))                 # 前後要伸出去多少
        d = {sg: (cam + (m + sg * g) * u - pa) / L for sg in (1, -1)}
        v = 2.0 * float(np.clip((t0 - m) / max(g, 1e-6), -1.0, 1.0))          # 模型的意見（縮回去的時候很小）
        if prior is not None:
            v += 0.4 * float((d[1] - d[-1]) @ prior)                           # 人體構造（模型沒把握時才由它決定）
        last = self._fs.get(key)
        if last is not None and self.n - last[1] <= 15:
            v += 0.4 * last[0]                                                 # 和上一幀一樣（不要前後跳）
        sg = 1 if v >= 0 else -1
        if veto is not None and veto(d[sg]) and not veto(d[-sg]):
            sg = -sg
        self._fs[key] = (sg, self.n)
        # 模型的深度和幾何差不多時不插手（交給後面依誤差大小加權的修正）；差很多（縮回去、判反）才改用幾何。
        # 幾何本身的誤差 ≈ 畫面誤差 × 骨長 ÷ 前後的長度：骨頭只是稍微斜向鏡頭時幾何並不準，差距要更大才算
        noise = 2.0 * self.sig_L * L / max(g, 0.3 * L)
        fs *= float(np.clip((abs(t0 - (m + sg * g)) - noise - 0.15 * g) / max(0.2 * g, 1e-6), 0.0, 1.0))
        return t0 + fs * (m + sg * g - t0), fs, sg

    def _foreshorten(self, X, U, cam, ok):
        """正對鏡頭的手腳：見檔頭。回傳修過的 X（只沿視線移動手肘／手腕／膝蓋／腳踝）。"""
        self.fs_now = {}
        need = (P.L_SHO, P.R_SHO, P.L_HIP, P.R_HIP, P.NECK, P.HIP)
        front = upv = None
        if all(ok[j] for j in need):
            upv = _unit(X[P.NECK] - X[P.HIP])
            right = (X[P.R_SHO] - X[P.L_SHO]) + (X[P.R_HIP] - X[P.L_HIP])
            front = None if upv is None else _unit(np.cross(upv, right))      # 胸口的前方
        X = X.copy()

        def no_back(d):                         # 上臂、大腿：不會水平往背後伸（往後擺到離垂下約 70° 以內都算正常）
            return front is not None and d @ front < -0.5 and d @ upv > -0.35

        def note(j, fs, sg):
            if fs > 0.5:
                self.fs_now[j] = -sg            # 離相機近（距離變小）= 朝鏡頭

        for a, mid, end, other, kind in CHAINS:
            if not (ok[a] and ok[end]):
                continue
            L1, L2 = self.L[self._bone(a, mid)], self.L[self._bone(mid, end)]
            out = _unit(X[a] - X[other]) if ok[other] else None
            p1 = front
            if kind == 'arm' and front is not None and out is not None:
                p1 = front + 0.7 * out                                         # 手臂：往前、往外
            t_end = float(np.linalg.norm(X[end] - cam))
            if not ok[mid]:
                # 中間的關節看不到（被手擋住），而且末端在畫面上貼著根部：整隻朝鏡頭或背向鏡頭伸直
                if kind != 'arm':
                    continue
                t, fs, sg = self._pick((a, end), cam, U[end], X[a], 0.97 * (L1 + L2), t_end, p1, no_back)
                if fs > 0.5:
                    X[end] = cam + t * U[end]
                    X[mid] = X[a] + (X[end] - X[a]) * (L1 / (L1 + L2))
                    note(end, fs, sg)
                continue
            t_mid = float(np.linalg.norm(X[mid] - cam))
            t, fs, sg = self._pick((a, mid), cam, U[mid], X[a], L1, t_mid, p1, no_back)
            if fs <= 0.0:
                continue
            X[mid] = cam + t * U[mid]
            note(mid, fs, sg)
            # 末端先跟著中間的關節一起前後移動（模型給的是兩者的相對深度），再看下一節是不是也正對鏡頭
            t_end += t - t_mid
            if kind == 'arm':
                d1 = _unit(X[mid] - X[a])
                p2 = front if d1 is None or front is None else front + 0.5 * d1       # 前臂：手在身體前方、順著上臂
                t_end, fs, sg = self._pick((mid, end), cam, U[end], X[mid], L2, t_end, p2)
                note(end, fs, sg)
            X[end] = cam + t_end * U[end]
        return X

    def _limit(self, X, ok, cam):
        """深度的幾何上限。骨頭在畫面上的長度（垂直視線的部分）已經接近骨長時，前後不可能再差很多：
        模型的相對深度在真實畫面上會錯到 20–30 cm（站直的腿被算成彎的、軀幹忽前忽後），超出骨長容許的部分沿視線收回來。
        遠近（整個人的大小）不準時每根骨頭都會一起偏長或偏短，所以先用這一幀「最伸展的那些骨頭」量出大小的比例 rho。"""
        cam = np.asarray(cam, float)
        U = X - cam
        dist = np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        U = U / dist

        def lateral(a, b):
            d = X[b] - X[a]
            u = _unit(0.5 * (X[a] + X[b]) - cam)
            return float(np.linalg.norm(d - (d @ u) * u))
        m = ok[_RA] & ok[_RB]
        if m.sum() >= 6:
            ratio = np.array([lateral(a, b) for a, b in zip(_RA[m], _RB[m])]) / self.L[RAY_BONES[m]]
            self.rho += 0.03 * (float(np.clip(np.percentile(ratio, 75), 0.85, 1.2)) - self.rho)
        X = X.copy()

        def cap(a, b, L):
            """b 沿自己的視線收回到離 a 不超過 L 的範圍；回傳移動量（沒動是 0）。"""
            u = U[b]
            w = X[a] - cam
            mid = float(u @ w)                                   # a 沿「b 的視線」的位置
            lat2 = max(float(w @ w) - mid * mid, 0.0)
            g = float(np.linalg.norm(X[b] - cam)) - mid          # b 比 a 遠多少（沿視線）
            gmax = np.sqrt(max(L * L - lat2, 0.0))
            if abs(g) <= gmax:
                return 0.0
            shift = float(np.sign(g) * gmax - g)
            X[b] = X[b] + shift * u
            return shift
        k = self.rho * (1.0 + LIMIT_TOL)
        for a, b, tail in LIMIT_TORSO:
            if ok[a] and ok[b]:
                sh = cap(a, b, self.L[self._bone(a, b)] * k)
                for j in tail:                                   # 手臂跟著肩膀移（相對深度不變）
                    if sh and ok[j]:
                        X[j] = X[j] + sh * U[j]
        ts = np.linspace(-0.6, 0.6, 121)
        for a, m, e, tail in LIMIT_LIMBS:
            if not ok[a]:
                continue
            L1, L2 = self.L[self._bone(a, m)] * self.rho, self.L[self._bone(m, e)] * self.rho
            if ok[e]:
                # 末端（手腕、腳踝）離根部最遠就是整隻伸直：站直的腿、伸直的手剛好在這個上限上，
                # 模型的深度超過時收回來 → 伸直，而且這個位置只由畫面上的長度決定，不會跟著模型的深度晃
                sh = cap(a, e, (L1 + L2) * (1.0 + 0.5 * LIMIT_TOL))
                if sh and ok[m]:                                 # 收回來了 = 幾乎伸直：中間的關節（手肘、膝蓋）沿視線放到兩節長度都對得上的位置
                    C = X[m][None] + ts[:, None] * U[m][None]
                    cost = (np.linalg.norm(C - X[a], axis=1) - L1) ** 2 + (np.linalg.norm(C - X[e], axis=1) - L2) ** 2
                    X[m] = C[int(np.argmin(cost + LIMIT_STAY * ts ** 2))]
                elif ok[m]:                                      # 沒有超過：兩節各自檢查（末端跟著中間的關節移）
                    s1 = cap(a, m, L1 * (1.0 + LIMIT_TOL))
                    if s1:
                        X[e] = X[e] + s1 * U[e]
                        sh += s1
                    sh += cap(m, e, L2 * (1.0 + LIMIT_TOL))
                for j in tail:
                    if sh and ok[j]:
                        X[j] = X[j] + sh * U[j]
            elif ok[m]:
                cap(a, m, L1 * (1.0 + LIMIT_TOL))
        return X

    @staticmethod
    def _bone(a, b):
        for i, (x, y, _) in enumerate(BONES):
            if (x, y) == (a, b) or (x, y) == (b, a):
                return i
        raise KeyError((a, b))

    def _rays(self, X, ok, cam):
        """每個關節只沿著自己的視線移動（畫面上的位置不變），讓骨長對上；移動量有先驗（模型的深度）拉著。
        骨頭和視線垂直時長度對深度幾乎沒有資訊，這時自然就不會動。"""
        U = X - np.asarray(cam, float)
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        if self.learn:
            self._calibrate(X, ok, U)
        m = ok[_RA] & ok[_RB]
        if m.sum() < 3:
            return X
        if self.limit:
            X = self._limit(X, ok, cam)
            U = X - np.asarray(cam, float)
            U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        X1 = self._foreshorten(X, U, cam, ok)
        free = np.zeros(P.NUM_JOINTS, bool)
        free[_RA[m]] = free[_RB[m]] = True
        out = X1 + self._ray_fit(X1, U, _RA[m], _RB[m], self.L[RAY_BONES[m]], free)[:, None] * U
        move = out - X
        for j, base in FOLLOW.items():                       # 頭、腳掌先整個跟著移動
            if ok[j] and all(ok[k] for k in base):
                out[j] = X[j] + np.mean(move[list(base)], axis=0)
        m = ok[_FA] & ok[_FB]                                 # 腳掌：腳踝不動，腳尖、腳跟沿視線移到長度對得上的地方
        if m.any():
            free = np.zeros(P.NUM_JOINTS, bool)
            free[[P.L_BIGTOE, P.R_BIGTOE, P.L_HEEL, P.R_HEEL]] = True
            free &= ok
            sf = self._ray_fit(out, U, _FA[m], _FB[m], self.L[FOOT_BONES[m]], free)
            for toe, small in ((P.L_BIGTOE, P.L_SMALLTOE), (P.R_BIGTOE, P.R_SMALLTOE)):
                sf[small] = sf[toe] if ok[small] else 0.0     # 小趾跟著大趾
            out = out + sf[:, None] * U
        return out

    def apply(self, X, ok, conf=None, iters=6, use_toes=True, cam=None):
        """X (26,3)、ok (26,) bool、conf (26,) 0~1（越大越可信）。回傳修正後的 X（不改原本的）。
        cam：相機在同一個座標系的位置；有給時先沿視線修深度（見檔頭）。
        use_toes=False（「腳尖方向」關閉）：腳尖、腳跟完全不參與約束與骨長學習（骨長約束、膝蓋前後判斷都不用它們），
        所以它們不會移動腳踝、膝蓋、骨盆。呼叫者的 ok 不會被修改。"""
        X = np.array(X, dtype=np.float64, copy=True)
        ok_all = np.array(ok, dtype=bool, copy=True)
        ok = ok_all.copy()
        if not use_toes:
            ok[FOOT_POINTS] = False                           # 只在這裡把腳尖、腳跟當作沒有資料
        if self.learn:
            self._observe(X, ok, cam)
            self.n += 1
            if self.n % 30 == 0:
                self._update()
        if cam is not None:
            X = self._rays(X, ok_all, cam)
        if conf is None:
            conf = np.ones(P.NUM_JOINTS)
        mob = MOBILITY * np.clip(1.25 - np.asarray(conf, float), 0.25, 1.0)
        for _ in range(iters):
            for i, (a, b, _) in enumerate(BONES):
                if not (ok[a] and ok[b]):
                    continue
                d = X[b] - X[a]
                n = np.linalg.norm(d)
                if n < 1e-6:
                    continue
                wa, wb = mob[a], mob[b]
                corr = self.k * (n - self.L[i]) / n / (wa + wb) * d
                X[a] += wa * corr
                X[b] -= wb * corr
            self._knees(X, ok, use_toes)
        return X

    def _knees(self, X, ok, use_toes=True):
        """膝蓋只能往前彎：前方用腳尖方向（沒有、或 use_toes=False 時用骨盆方向）。"""
        up = np.array([0.0, 1.0, 0.0])
        body_f = None
        if ok[P.L_HIP] and ok[P.R_HIP]:
            body_f = np.cross(up, X[P.R_HIP] - X[P.L_HIP])
        for hip, knee, ank, toe, heel in LEGS:
            if not (ok[hip] and ok[knee] and ok[ank]):
                continue
            f = None
            if use_toes and ok[toe] and ok[heel]:
                f = X[toe] - X[heel]
            elif body_f is not None:
                f = body_f
            if f is None:
                continue
            line = X[ank] - X[hip]
            L = np.linalg.norm(line)
            if L < 1e-3:
                continue
            u = line / L
            f = f - np.dot(f, u) * u
            fn = np.linalg.norm(f)
            if fn < 1e-6:
                continue
            f /= fn
            v = X[knee] - (X[hip] + np.dot(X[knee] - X[hip], u) * u)
            back = float(np.dot(v, f))
            if back < -0.015:                     # 膝蓋在後面：翻到前面
                X[knee] -= 2.0 * back * f


def confidence(scores, thr):
    """把模型的分數換成 0~1 的可信度。"""
    s = np.asarray(scores, float)
    return np.clip((s - thr) / max(1.0 - thr, 1e-3), 0.0, 1.0)
