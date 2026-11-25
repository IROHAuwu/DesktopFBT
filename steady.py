"""即時追蹤的穩定化：模型給的「相對深度」和頭的方向在真實畫面上很吵，而且會跟著手腳的動作一起飄
（人站著不動、只舉手或抬腳時，頭和軀幹的深度會差到 10–20 cm：頭亂轉、身體前後晃、膝蓋忽彎忽直）。

- DepthSteady：每個關節的相對深度（相對骨盆）用「畫面上有沒有在動」決定濾波快慢——
  畫面上相對骨盆不動的關節，深度也不該動（慢濾波）；畫面上動得快的關節才跟著模型的深度走。
- head_from_face：頭的方向改用臉上五個點（鼻子、兩眼、兩耳）在畫面上的相對位置算（不用深度）。
"""
import numpy as np

import pose as P

# 軀幹和頭：更慢（它們的深度變化 = 前傾後仰，比手腳慢，而且錯了整個人都在晃）
TRUNK = (P.NECK, P.HEAD, P.NOSE, P.L_EYE, P.R_EYE, P.L_EAR, P.R_EAR, P.L_SHO, P.R_SHO, P.L_HIP, P.R_HIP, P.HIP)
C_MIN_TRUNK, C_MIN_LIMB = 0.25, 0.6       # 畫面上不動時的截止頻率（Hz）
C_MAX = 8.0
V0, GAIN = 0.06, 9.0                      # 畫面上的速度（身高／秒）超過 V0 才開始放快；每多 1 身高／秒多 GAIN Hz
V_SPAN_S = 0.17                           # 量速度用的時間差（太短會把抖動當成移動）


class DepthSteady:
    """z (26,) 相對深度 → 穩定後的 z。k：畫面位置 (26,2)，s：分數，t：秒。"""

    def __init__(self, thr=0.3):
        self.thr = thr
        self.z = None                       # 濾波後的相對深度（相對骨盆）
        self.t = None
        self.hist = []                      # [(t, 相對骨盆的畫面位置 (26,2), 人的大小 px)]
        self.size = None
        self.c_min = np.full(P.NUM_JOINTS, C_MIN_LIMB)
        self.c_min[list(TRUNK)] = C_MIN_TRUNK

    def reset(self):
        self.z = self.t = self.size = None
        self.hist = []

    def __call__(self, k, z, s, t):
        k, z = np.asarray(k, float), np.asarray(z, float)
        ok = np.asarray(s) > self.thr
        if not (ok[P.L_HIP] and ok[P.R_HIP] and np.isfinite(z).all()):
            return z
        hip2 = 0.5 * (k[P.L_HIP] + k[P.R_HIP])
        zh = 0.5 * (z[P.L_HIP] + z[P.R_HIP])
        rel2, relz = k - hip2, z - zh
        ext = np.ptp(k[ok], axis=0)
        size = float(max(ext[1], 0.6 * ext[0], 1.0))              # 人在畫面上的大小（站著時 ≈ 身高）
        self.size = size if self.size is None else max(0.98 * self.size, size)     # 蹲下、彎腰時不要跟著變小
        if self.z is None or self.t is None or not 0.0 < t - self.t < 0.5:         # 第一幀，或中斷過
            self.z, self.t, self.hist = relz.copy(), t, [(t, rel2, self.size)]
            return z
        dt = t - self.t
        self.hist.append((t, rel2, self.size))
        while len(self.hist) > 2 and t - self.hist[1][0] >= V_SPAN_S:
            self.hist.pop(0)
        t0, r0, _ = self.hist[0]
        v = np.linalg.norm(rel2 - r0, axis=1) / max(t - t0, 1e-3) / self.size       # 身高／秒
        c = np.clip(self.c_min + GAIN * np.maximum(v - V0, 0.0), self.c_min, C_MAX)
        a = 1.0 - np.exp(-2.0 * np.pi * c * dt)
        a = np.where(ok, a, 0.0)                                   # 看不到的關節：維持原本的
        self.z = self.z + a * (relz - self.z)
        self.t = t
        out = z.copy()
        out[ok] = self.z[ok] + zh
        return out


FEET = ((P.L_ANK, ((P.L_BIGTOE, 0.100), (P.L_SMALLTOE, 0.085), (P.L_HEEL, 0.045))),
        (P.R_ANK, ((P.R_BIGTOE, 0.100), (P.R_SMALLTOE, 0.085), (P.R_HEEL, 0.045))))
ANKLE_H = 0.05                # 腳踝比腳尖、腳跟的點高多少（相對身高）


def flat_feet(X, ok, cam, height=1.65):
    """腳尖、腳跟的點：畫面上的位置準，深度很吵（真實畫面上腳尖會被算到腳踝正下方 20 cm，像踮到極限；
    而且哪個點看得到每幀都在變）。腳掌的姿勢會決定「最低的點」，它一跳整個人就上下晃。
    做法：只從畫面取「腳尖朝哪個方向」（腳尖、腳跟的視線和腳底那個高度的水平面的交點，兩點的連線），
    腳掌的俯仰不用量的：踩在地上的腳放平；抬起來的腳，腳踝放在不勾不踮的位置（腳掌垂直小腿）。回傳修過的 X（腳尖、腳跟、小趾）。"""
    cam = np.asarray(cam, float)
    X = np.array(X, float)
    s = height / 1.65
    ys = [X[j][1] for j in (P.L_ANK, P.R_ANK) if ok[j]]
    low = min(ys) if ys else 0.0
    for side, (ank, pts) in zip((1.0, -1.0), FEET):
        toe, small, heel = pts[0][0], pts[1][0], pts[2][0]
        knee = P.L_KNEE if ank == P.L_ANK else P.R_KNEE
        if not (ok[ank] and ok[knee]):
            continue
        A = X[ank]
        shin = A - X[knee]
        ns = float(np.linalg.norm(shin))
        if ns < 1e-6:
            continue
        shin /= ns
        plane_y = A[1] - ANKLE_H * height

        def on_floor(j):
            u = X[j] - cam
            n = float(np.linalg.norm(u))
            if n < 1e-6 or u[1] / n > -0.12:                    # 視線幾乎是水平的（相機和腳差不多高）：交點不可靠
                return None
            return cam + ((plane_y - cam[1]) / u[1]) * u
        gt = on_floor(toe) if ok[toe] else None
        gh = on_floor(heel) if ok[heel] else None
        if gt is not None and gh is not None:
            head = gt - gh
        elif ok[toe] and ok[heel]:
            head = X[toe] - X[heel]
        elif gt is not None:
            head = gt - A
        elif ok[toe]:
            head = X[toe] - A
        else:
            head = np.zeros(3)
        # 踩在地上的腳（腳踝是兩隻腳裡比較低的）：腳掌放平，「下」就是正下方；抬起來的腳：腳踝不勾不踮，「下」是小腿的方向
        w = float(np.clip((0.20 * s - (A[1] - low)) / (0.12 * s), 0.0, 1.0))
        down = w * np.array([0.0, -1.0, 0.0]) + (1.0 - w) * shin
        down /= max(float(np.linalg.norm(down)), 1e-9)
        head = head - (head @ down) * down                      # 垂直「下」的部分 = 腳掌的方向
        nh = float(np.linalg.norm(head))
        if nh < 0.03 * s:                                       # 看不出來（腳尖正對著鏡頭往下、沒有腳尖的點）：朝身體的前方
            if not (ok[P.L_HIP] and ok[P.R_HIP]):
                continue
            fwd = np.cross([0.0, 1.0, 0.0], X[P.R_HIP] - X[P.L_HIP])
            fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
            head = fwd - (fwd @ down) * down
            nh = float(np.linalg.norm(head))
            if nh < 0.2:                                        # 小腿剛好朝前或朝後（往後勾腿）：腳尖朝下／朝上
                head = np.array([0.0, -1.0, 0.0]) * (1.0 if fwd @ down < 0 else -1.0)
                head = head - (head @ down) * down
                nh = float(np.linalg.norm(head))
                if nh < 1e-6:
                    continue
        head /= nh
        out = np.cross(head, down) * side                       # 腳的外側
        new = {toe: A + (0.150 * head + 0.068 * down) * s, heel: A + (-0.045 * head + 0.068 * down) * s,
               small: A + (0.115 * head + 0.068 * down + 0.045 * out) * s}
        # 抬起來的腳：量到的腳尖如果可信就照量到的（深度錯的時候「腳踝到腳尖的距離」會不對，因為畫面上的位置是固定的）。
        # 踩在地上的腳一律放平：它決定整個人的高度，量到的俯仰每幀差幾度，人就跟著上下晃（真實畫面上骨盆高度的抖動差一倍）
        m = 0.0
        if ok[toe] and w < 1.0:
            r = float(np.linalg.norm(X[toe] - A)) / (0.165 * s)
            m = (1.0 - w) * float(np.clip(min((r - 0.70) / 0.15, (1.35 - r) / 0.20), 0.0, 1.0))
        for j, p in new.items():
            X[j] = m * X[j] + (1.0 - m) * p if ok[j] else p
    return X


def floor_level(X, ok, height=1.65):
    """地板在哪裡（最低的點）。腳尖、腳跟最多只算到腳踝下方 12 cm：深度算錯時它們會掉到很下面，把整個人往上頂。"""
    hs = []
    for ank, pts in FEET:
        if ok[ank]:
            low = [X[j][1] for j, _ in pts if ok[j]]
            hs.append(max(min(low), X[ank][1] - 0.073 * height) if low else X[ank][1] - ANKLE_H * height)
    feet = {j for _, pts in FEET for j, _ in pts} | {P.L_ANK, P.R_ANK}
    hs += [X[j][1] for j in range(P.NUM_JOINTS) if ok[j] and j not in feet]          # 倒立、躺下時是手、頭
    if ok[P.HEAD]:
        hs.append(X[P.HEAD][1] - 0.04)
    return float(min(hs)) if hs else None


# 頭的尺寸（公尺，身高 1.70 的人；原點 = 兩耳中點，x 左、y 上、z 臉的前方）
EAR_W, NOSE_F, NOSE_U = 0.150, 0.115, -0.030
EYE = (0.032, 0.005, 0.085)               # 眼睛：左右、上、前
NECK_UP, TOP_UP = 0.110, 0.125            # 脖子根部 → 兩耳中點、兩耳中點 → 頭頂
FACE_MIN = 0.5                            # 鼻子和兩耳的分數都要超過這個才用


def head_rotation(k, sc, thr=0.3):
    """臉上的點在畫面上的相對位置 → 頭的方向（相對「看過去的方向」）。不用深度：
    轉頭 = 鼻子偏向哪隻耳朵（和兩耳距離的比例），點頭 = 鼻子比兩耳連線高或低多少，歪頭 = 兩耳連線的傾斜。
    回傳 3×3（欄 = 頭的左、上、前；座標 = 畫面右、畫面上、朝相機）；看不到臉（背對、被擋住）時 None。"""
    if min(sc[P.NOSE], sc[P.L_EAR], sc[P.R_EAR]) <= max(thr, FACE_MIN):
        return None
    flip = np.array([1.0, -1.0])                               # 畫面的 y 朝下
    e = (k[P.L_EAR] - k[P.R_EAR]) * flip
    n = (k[P.NOSE] - 0.5 * (k[P.L_EAR] + k[P.R_EAR])) * flip
    # 背對鏡頭時模型還是會「猜」出臉上的點，而且分數不低。面向鏡頭時自己的左耳在畫面右邊：
    # 相對身體的上方（脖子 → 頭頂；沒有時用畫面的上方），左右反過來就是背對，不用這個方法
    upb = (k[P.HEAD] - k[P.NECK]) * flip if min(sc[P.HEAD], sc[P.NECK]) > thr else np.array([0.0, 1.0])
    nu = float(np.linalg.norm(upb))
    upb = upb / nu if nu > 1e-6 else np.array([0.0, 1.0])
    span = max(float(np.linalg.norm(e)), abs(float(n @ np.array([upb[1], -upb[0]]))) / NOSE_F * EAR_W, 1e-6)
    if (e[0] * upb[1] - e[1] * upb[0]) < -0.15 * span:
        return None
    le = float(np.linalg.norm(e))
    if le < 1e-6 and float(np.linalg.norm(n)) < 1e-6:
        return None
    if sc[P.L_EYE] > thr and sc[P.R_EYE] > thr:                # 歪頭的方向：兩耳和兩眼的連線一起看（側臉時兩耳幾乎重疊）
        e_dir = e + 2.3 * (k[P.L_EYE] - k[P.R_EYE]) * flip
    else:
        e_dir = e
    ne = float(np.linalg.norm(e_dir))
    if ne < 1e-6:
        return None
    eh = e_dir / ne                                            # 畫面上「頭的左邊」的方向
    nx = float(n @ eh)                                         # 鼻子往頭的左邊偏多少（畫素）
    ny = float(n @ np.array([-eh[1], eh[0]]))                  # 鼻子比兩耳連線高多少
    a, b = nx / NOSE_F, max(float(e @ eh), 0.0) / EAR_W
    scale = float(np.hypot(a, b))                              # 畫素／公尺
    if scale < 1e-6:
        return None
    yaw = np.arctan2(a, b)                                     # 正的 = 臉轉向自己的左邊（畫面右邊）
    rho = float(np.hypot(NOSE_F, NOSE_U))
    side = np.array([eh[0], eh[1], 0.0])
    up2 = np.array([-eh[1], eh[0], 0.0])
    toward = np.array([0.0, 0.0, 1.0])
    left = np.cos(yaw) * side - np.sin(yaw) * toward           # 臉轉向左邊：左耳離相機比較遠
    # 鼻子（相對兩耳中點）的方向：畫面上的偏移 ＋ 朝相機的部分（看得到臉 → 鼻子在靠相機的這一側）
    v = np.array([nx, ny]) / (scale * rho)
    v = v / max(1.0, float(np.linalg.norm(v)))
    nose = v[0] * side + v[1] * up2 + np.sqrt(max(1.0 - float(v @ v), 0.0)) * toward
    nose = nose - (nose @ left) * left
    nn = float(np.linalg.norm(nose))
    if nn < 1e-6:
        return None
    nose /= nn
    t0 = np.arctan2(NOSE_U, NOSE_F)                            # 鼻尖比「正前方」低的角度
    cr = np.cross(nose, left)
    fwd = np.cos(t0) * nose - np.sin(t0) * cr
    up = np.sin(t0) * nose + np.cos(t0) * cr
    return _refine(np.stack([left, up, fwd], axis=1), k, sc, thr)


_FACE = {P.L_EAR: (0.5 * EAR_W, 0.0, 0.0), P.R_EAR: (-0.5 * EAR_W, 0.0, 0.0), P.NOSE: (0.0, NOSE_U, NOSE_F),
         P.L_EYE: EYE, P.R_EYE: (-EYE[0], EYE[1], EYE[2])}


def _rodrigues(w):
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    x, y, z = w / a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def _refine(H, k, sc, thr, iters=3):
    """上面的算式在大角度時是近似的：用五個點一起微調（大小和位置自由，只調方向），讓投影和畫面最接近。"""
    js = [j for j in _FACE if sc[j] > thr]
    if len(js) < 4:
        return H
    M = np.array([_FACE[j] for j in js])
    obs = np.array([k[j] for j in js], float) * np.array([1.0, -1.0])
    w = np.array([min(float(sc[j]), 1.0) for j in js])
    obs = obs - (w[:, None] * obs).sum(0) / w.sum()

    def resid(R):
        p = (M @ R.T)[:, :2]
        p = p - (w[:, None] * p).sum(0) / w.sum()
        den = float((w[:, None] * p * p).sum())
        sc_ = float((w[:, None] * p * obs).sum()) / den if den > 1e-12 else 0.0
        return ((sc_ * p - obs) * np.sqrt(w)[:, None]).ravel()
    r = resid(H)
    for _ in range(iters):
        J = np.stack([(resid(_rodrigues(d) @ H) - r) / 1e-4 for d in 1e-4 * np.eye(3)], axis=1)
        step = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3) * float(np.trace(J.T @ J) + 1e-9), -J.T @ r)
        n = float(np.linalg.norm(step))
        if n > 0.3:
            step *= 0.3 / n
        H2 = _rodrigues(step) @ H
        r2 = resid(H2)
        if r2 @ r2 >= r @ r:
            break
        H, r = H2, r2
    return H


def _logrot(R):
    """旋轉矩陣 → 旋轉向量（軸 × 角度）。"""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    a = float(np.arccos(c))
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(w))
    return w * (a / n) if n > 1e-9 else np.zeros(3)


def _frame(up, across):
    """（上、左右）兩個方向 → 3×3（欄 = 左、上、前）。"""
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    left = across - (across @ up) * up
    n = float(np.linalg.norm(left))
    if n < 1e-9:
        return None
    left /= n
    return np.stack([left, up, np.cross(left, up)], axis=1)


class HeadSteady:
    """頭的點（耳、眼、鼻、頭頂）照 head_rotation 算出的方向，重新接在脖子上。
    模型給的頭部深度很吵（耳朵和脖子只差十公分，深度錯五公分頭就轉二十幾度），而且會跟著手腳的動作一起飄。
    畫面算出來的方向很穩、反應快，但每個人的臉型、模型標點的習慣會讓它有固定的偏差（例如一直稍微抬頭）；
    模型深度算出來的方向很吵，但長時間平均是對的。所以兩者互補：快的變化聽畫面的，長時間的平均慢慢對齊模型的（約 TAU 秒）。"""
    TAU = 6.0
    MAX_BIAS = np.radians(30.0)

    def __init__(self, thr=0.3, height=1.65):
        self.thr, self.height = thr, height
        self.D = None                      # 「畫面的方向 → 模型的方向」的長時間平均
        self.n = 0
        self.t = None

    def __call__(self, X, ok, k, sc, cam, R, t=None):
        """cam：相機位置（世界座標）；R：lift.Lifter.R（列 = 世界的 x、y、z 軸在相機座標裡的方向）。看不到臉時原樣回傳。"""
        if not ok[P.NECK]:
            return X
        H = head_rotation(k, sc, self.thr)
        if H is None:
            return X
        X = np.array(X, float)
        d = np.asarray(cam, float) - X[P.NECK]
        nd = float(np.linalg.norm(d))
        if nd < 1e-6:
            return X
        d /= nd                                                    # 朝相機
        right = np.asarray(R, float)[:, 0]                         # 畫面右邊（世界座標）
        right = right - (right @ d) * d
        nr = float(np.linalg.norm(right))
        if nr < 1e-6:
            return X
        right /= nr
        F = np.stack([right, np.cross(d, right), d], axis=1) @ H   # 畫面算出來的方向（世界座標；欄 = 左、上、前）
        turn = abs(float(np.degrees(np.arctan2(H[0, 2], H[2, 2]))))    # 臉相對鏡頭轉了幾度（0 = 正對）
        Z = None
        if ok[P.L_EAR] and ok[P.R_EAR]:                            # 模型的點算出來的方向
            Z = _frame(0.5 * (X[P.L_EAR] + X[P.R_EAR]) - X[P.NECK], X[P.L_EAR] - X[P.R_EAR])
        if Z is not None and turn < 50.0:                          # 只在臉大致正對鏡頭時更新（側臉時畫面的算法不準）
            D = _logrot(Z @ F.T)
            n = float(np.linalg.norm(D))
            if n < np.radians(60.0):                               # 差太多的多半是模型把前後認反：不採用
                self.n += 1
                dt = 1.0 / 30.0 if t is None or self.t is None else float(np.clip(t - self.t, 1e-3, 0.5))
                a = max(1.0 / self.n, 1.0 - np.exp(-dt / self.TAU))
                D = _rodrigues(D)
                self.D = D if self.D is None else _rodrigues(a * _logrot(D @ self.D.T)) @ self.D
                v = _logrot(self.D)
                nv = float(np.linalg.norm(v))
                if nv > self.MAX_BIAS:                             # 固定的偏差不會很大
                    self.D = _rodrigues(v * (self.MAX_BIAS / nv))
        self.t = t
        if self.D is not None:
            F = self.D @ F
        if Z is not None and turn > 60.0:                          # 側臉：漸漸改用模型的點（畫面上兩耳疊在一起，鼻子在哪一側也分不清）
            w = float(np.clip((turn - 60.0) / 25.0, 0.0, 1.0))
            F = _rodrigues(w * _logrot(Z @ F.T)) @ F
        left, up, fwd = F.T
        s = self.height / 1.70
        mid = X[P.NECK] + NECK_UP * s * up
        X[P.L_EAR], X[P.R_EAR] = mid + 0.5 * EAR_W * s * left, mid - 0.5 * EAR_W * s * left
        X[P.NOSE] = mid + (NOSE_F * fwd + NOSE_U * up) * s
        X[P.L_EYE] = mid + (EYE[0] * left + EYE[1] * up + EYE[2] * fwd) * s
        X[P.R_EYE] = mid + (-EYE[0] * left + EYE[1] * up + EYE[2] * fwd) * s
        X[P.HEAD] = mid + TOP_UP * s * up
        return X
