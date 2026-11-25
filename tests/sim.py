"""模擬的姿勢模型：已知的 3D 動作 → 相機投影 → 加上單鏡頭模型常見的誤差（畫面上小、深度大而且前後連續）。
精度測試共用。座標：世界 y 上、公尺；相機是 OpenCV 慣例（x 右、y 下、z 前）。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import motion as M  # noqa: E402
import pose as P  # noqa: E402

LIMB_BONES = [(P.L_SHO, P.L_ELB), (P.L_ELB, P.L_WRI), (P.R_SHO, P.R_ELB), (P.R_ELB, P.R_WRI),
              (P.L_HIP, P.L_KNEE), (P.L_KNEE, P.L_ANK), (P.R_HIP, P.R_KNEE), (P.R_KNEE, P.R_ANK),
              (P.HIP, P.NECK)]
BODY = [P.L_SHO, P.R_SHO, P.L_ELB, P.R_ELB, P.L_WRI, P.R_WRI, P.L_HIP, P.R_HIP, P.L_KNEE, P.R_KNEE,
        P.L_ANK, P.R_ANK, P.NECK, P.HIP]
# 深度誤差：離軀幹越遠的關節越大
Z_GAIN = np.ones(P.NUM_JOINTS)
Z_GAIN[[P.L_HIP, P.R_HIP, P.HIP]] = 0.3
Z_GAIN[[P.L_SHO, P.R_SHO, P.NECK]] = 0.6
Z_GAIN[[P.L_ELB, P.R_ELB, P.L_KNEE, P.R_KNEE]] = 1.0
Z_GAIN[[P.L_WRI, P.R_WRI, P.L_ANK, P.R_ANK, P.L_BIGTOE, P.R_BIGTOE, P.L_SMALLTOE, P.R_SMALLTOE,
        P.L_HEEL, P.R_HEEL]] = 1.3


def look_at(pos, target):
    f = target - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f])


class Camera:
    def __init__(self, pos=(0.4, 1.15, -3.6), target=(0, 0.9, 0), size=(1280, 720), hfov=60.0):
        self.pos = np.asarray(pos, float)
        self.R = look_at(self.pos, np.asarray(target, float))
        self.size = size
        self.hfov = hfov
        f = 0.5 * max(size) / np.tan(np.radians(hfov) / 2)
        self.K = np.array([[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1.0]])

    def cam(self, J):
        return (J - self.pos) @ self.R.T

    def project(self, J):
        Xc = self.cam(J)
        uv = Xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3], Xc

    def world(self, J):
        """真實的關節點換到 Lifter 輸出的座標系（x = 相機右、y = 上、z = 朝向相機、地板 = 0、相機在原點上方）。"""
        up = self.R @ np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, up[1], up[2]])
        up /= np.linalg.norm(up)
        fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
        fwd /= np.linalg.norm(fwd)
        Rw = np.stack([np.cross(up, -fwd), up, -fwd])
        X = self.cam(J) @ Rw.T
        X[:, 1] += self.pos[1]
        return X


def spin(J, yaw):
    h = J[P.HIP].copy()
    return (J - h) @ M.rot_y(yaw).T + h


def motion_walk(t):
    """走動、踏步、揮手、偶爾蹲低，慢慢轉身（±70°）。"""
    return spin(np.array(M.person(t, jump=False), float), np.pi + 1.2 * np.sin(0.23 * t))


def motion_turn(t):
    """一直轉圈（每 12 秒一圈）。"""
    return spin(np.array(M.person(t, jump=False), float), np.pi + 2 * np.pi * t / 12.0)


def motion_bend(t):
    """扭腰、彎腰。"""
    return spin(M.bend_person(t)[0], np.pi + 0.4 * np.sin(0.2 * t))


class _AR1:
    """前後連續的誤差（模型的偏差跟著姿勢慢慢變，不是每幀獨立）。"""

    def __init__(self, rng, shape, tau):
        self.rng, self.tau = rng, tau
        self.x = rng.normal(0, 1, shape)

    def __call__(self, dt):
        a = np.exp(-dt / self.tau)
        self.x = a * self.x + np.sqrt(1 - a * a) * self.rng.normal(0, 1, self.x.shape)
        return self.x


class FakeModel:
    """照著真實動作產生 (k, z, s)：k 畫素、z 相對骨盆的深度（公尺）、s 分數。

    px：畫面誤差（佔人在畫面上高度的比例）；zs：深度誤差（公尺，慢慢變的部分）；zw：每幀獨立的深度抖動；
    comp：深度被壓扁的比例（單鏡頭模型常把前後的幅度估小）。"""

    def __init__(self, cam, seed=0, px=0.008, px_w=0.004, zs=0.05, zw=0.015, comp=0.88):
        self.cam = cam
        self.rng = np.random.default_rng(seed)
        self.px, self.px_w, self.zs, self.zw, self.comp = px, px_w, zs, zw, comp
        self.e2 = _AR1(self.rng, (P.NUM_JOINTS, 2), 0.5)
        self.ez = _AR1(self.rng, (P.NUM_JOINTS,), 0.7)

    def __call__(self, J, dt):
        uv, Xc = self.cam.project(J)
        hpx = float(np.ptp(uv[:, 1])) + 1e-6
        k = uv + self.px * hpx * self.e2(dt) + self.px_w * hpx * self.rng.normal(0, 1, uv.shape)
        z0 = Xc[:, 2] - 0.5 * (Xc[P.L_HIP, 2] + Xc[P.R_HIP, 2])
        z = self.comp * z0 + Z_GAIN * (self.zs * self.ez(dt) + self.zw * self.rng.normal(0, 1, P.NUM_JOINTS))
        return k, z, np.full(P.NUM_JOINTS, 0.85)


def sequence(motion, seconds=40.0, fps=30.0, cam=None, **noise):
    """回傳 (cam, [(t, 真實 J, (k, z, s))])。"""
    cam = cam or Camera()
    model = FakeModel(cam, **noise)
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        J = motion(t)
        out.append((t, J, model(J, 1.0 / fps)))
    return cam, out


ARMS = [(P.L_SHO, P.L_ELB, P.L_WRI), (P.R_SHO, P.R_ELB, P.R_WRI)]


def arms_forward(t, yaw=0.0, aim=0.0):
    """站著（yaw = 0 面向鏡頭、π 背對），兩手從垂下舉到正前方再放下（6 秒一次），手肘幾乎打直。
    手指向鏡頭時肩膀、手肘、手腕在畫面上疊成一點。aim：手臂偏離正前方的角度（弧度，往外）。回傳 (J, 舉起的角度)。"""
    J = spin(np.array(M.person(0.0, jump=False), float), yaw)
    right = J[P.R_SHO] - J[P.L_SHO]
    right /= np.linalg.norm(right)
    fwd = np.cross([0, 1.0, 0], right)                                       # 身體前方
    th = np.radians(95) * np.clip(1.6 * np.sin(np.pi * (t % 6.0) / 6.0), 0, 1)
    for (sh, el, wr), sx in zip(ARMS, (-1, 1)):
        d = -np.cos(th) * np.array([0, 1.0, 0]) + np.sin(th) * (np.cos(aim) * fwd + np.sin(aim) * right * sx)
        d2 = d + 0.08 * np.array([0, 1.0, 0])
        J[el] = J[sh] + d * M.UPPER
        J[wr] = J[el] + d2 / np.linalg.norm(d2) * M.FORE
    return J, th


class OverlapModel(FakeModel):
    """手臂在畫面上疊在一起時模型常見的反應：分不出前後，深度縮回肩膀附近（arm_comp：剩下幾成，負的 = 判反），
    手肘被手擋住、分數掉到 elbow_score。"""

    def __init__(self, cam, arm_comp=1.0, elbow_score=0.85, **kw):
        super().__init__(cam, **kw)
        self.arm_comp, self.elbow_score = arm_comp, elbow_score

    def __call__(self, J, dt):
        k, z, s = super().__call__(J, dt)
        uv, _ = self.cam.project(J)
        hpx = float(np.ptp(uv[:, 1]))
        for sh, el, wr in ARMS:
            if np.linalg.norm(uv[wr] - uv[sh]) / hpx < 0.12:
                for j in (el, wr):
                    z[j] = z[sh] + self.arm_comp * (z[j] - z[sh])
                s[el] = self.elbow_score
        return k, z, s


# ------------------------------------------------------------------ 評分
def bone_angles(X, T):
    """主要骨頭的方向誤差（度）。"""
    out = []
    for a, b in LIMB_BONES:
        u, v = X[b] - X[a], T[b] - T[a]
        c = u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
        out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.array(out)


def rel_error(X, T):
    """相對骨盆的位置誤差（公尺，每個身體關節）。"""
    return np.linalg.norm((X[BODY] - X[P.HIP]) - (T[BODY] - T[P.HIP]), axis=1)


def jitter(seq, fps):
    """抖動：位置的二次差分（公尺／秒²，身體關節的平均）。"""
    A = np.stack([x[BODY] - x[P.HIP] for x in seq])
    return float(np.linalg.norm(np.diff(A, 2, axis=0), axis=2).mean() * fps * fps)
