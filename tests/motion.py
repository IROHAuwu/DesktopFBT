"""測試用的人體動作：骨頭長度固定（像真人），有踏步、抬腳、小跳、轉身、蹲低、揮手。
回傳 Halpe26 關節點（世界座標，y 上，公尺）。"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose as P  # noqa: E402

HEIGHT = 1.70
THIGH, SHIN = 0.245 * HEIGHT, 0.246 * HEIGHT
UPPER, FORE = 0.186 * HEIGHT, 0.146 * HEIGHT
ANKLE_H, HIP_W, SHO_W = 0.075, 0.09, 0.18


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _ik(hip, ank, fwd, L1, L2):
    """膝蓋位置：兩節骨頭，膝蓋往前彎。"""
    d = ank - hip
    n = np.linalg.norm(d)
    n_c = min(max(n, 1e-6), L1 + L2 - 1e-4)
    a = (L1 ** 2 - L2 ** 2 + n_c ** 2) / (2 * n_c)
    h = math.sqrt(max(L1 ** 2 - a ** 2, 0.0))
    u = d / max(n, 1e-6)
    f = fwd - np.dot(fwd, u) * u
    f /= max(np.linalg.norm(f), 1e-6)
    return hip + u * a + f * h


def person(t, jump=True):
    """t 秒的姿勢。人的前方 = -z（OpenVR 慣例），右方 = +x（yaw=0 時）。"""
    yaw = 0.8 * math.sin(0.25 * t) + 0.3 * math.sin(0.9 * t)
    Rb = rot_y(yaw)
    fwd = Rb @ np.array([0.0, 0.0, -1.0])
    right = Rb @ np.array([1.0, 0.0, 0.0])
    base = np.array([0.6 * math.sin(0.21 * t), 0.0, 0.4 * math.sin(0.33 * t + 1.0)])
    squat = 0.18 * max(0.0, math.sin(0.35 * t)) ** 6           # 偶爾蹲低
    hop = 0.0
    if jump:
        ph = (t % 9.0)
        if 6.0 < ph < 6.45:                                       # 每 9 秒一次小跳
            hop = 4.0 * 0.16 * (ph - 6.0) / 0.45 * (1 - (ph - 6.0) / 0.45)
    pelvis = base + np.array([0, 0.85 - squat + hop + 0.015 * math.sin(4 * t), 0])
    J = np.zeros((P.NUM_JOINTS, 3))
    for side, sx in ((0, -1), (1, 1)):
        hip = pelvis + right * HIP_W * sx
        ph = 2.0 * t + side * math.pi
        step = 0.18 * math.sin(ph)
        lift = 0.14 * max(0.0, math.sin(ph + math.pi / 2)) ** 2   # 抬腳
        ank = base + right * 0.12 * sx + fwd * step + np.array([0, ANKLE_H + lift + hop, 0])
        knee = _ik(hip, ank, fwd, THIGH, SHIN)
        # 確保腳踝和骨盆距離不超過腿長
        L = np.linalg.norm(ank - hip)
        if L > THIGH + SHIN - 1e-3:
            ank = hip + (ank - hip) * (THIGH + SHIN - 1e-3) / L
            knee = _ik(hip, ank, fwd, THIGH, SHIN)
        foot_f = rot_y(yaw + 0.15 * sx) @ np.array([0, 0, -1.0])
        pitch = 0.5 * lift / 0.14
        toe_dir = foot_f * math.cos(pitch) + np.array([0, -math.sin(pitch) * 0.3, 0])
        J[[P.L_HIP, P.R_HIP][side]] = hip
        J[[P.L_KNEE, P.R_KNEE][side]] = knee
        J[[P.L_ANK, P.R_ANK][side]] = ank
        J[[P.L_BIGTOE, P.R_BIGTOE][side]] = ank + toe_dir * 0.17 + np.array([0, -ANKLE_H + 0.02, 0]) \
            - right * 0.01 * sx
        J[[P.L_SMALLTOE, P.R_SMALLTOE][side]] = ank + toe_dir * 0.13 + np.array([0, -ANKLE_H + 0.02, 0]) \
            + right * 0.04 * sx
        J[[P.L_HEEL, P.R_HEEL][side]] = ank - foot_f * 0.06 + np.array([0, -ANKLE_H + 0.03 + lift * 0.3, 0])
    lean = 0.12 * math.sin(0.5 * t) + 0.25 * squat / 0.18
    up = np.array([0, 1.0, 0]) * math.cos(lean) + fwd * math.sin(lean)
    hipc = pelvis
    neck = hipc + up * 0.50
    J[P.HIP] = hipc
    J[P.NECK] = neck
    for side, sx in ((0, -1), (1, 1)):
        sho = neck - up * 0.03 + right * SHO_W * sx
        sw = 0.9 * math.sin(2.0 * t + (1 - side) * math.pi) + (1.3 if side == 1 and (t % 7) > 4 else 0)
        ua = -up * math.cos(sw) + fwd * math.sin(sw) * 0.9 + right * sx * 0.25
        ua /= np.linalg.norm(ua)
        elb = sho + ua * UPPER
        bend = 0.5 + 0.5 * abs(math.sin(1.3 * t + side))
        fa = ua * math.cos(bend) + fwd * math.sin(bend)
        fa -= np.dot(fa, ua) * ua * 0.0
        fa /= np.linalg.norm(fa)
        J[[P.L_SHO, P.R_SHO][side]] = sho
        J[[P.L_ELB, P.R_ELB][side]] = elb
        J[[P.L_WRI, P.R_WRI][side]] = elb + fa * FORE
    head_c = neck + up * 0.17
    hy = rot_y(yaw + 0.3 * math.sin(0.8 * t))
    hf, hr = hy @ np.array([0, 0, -1.0]), hy @ np.array([1.0, 0, 0])
    J[P.L_EAR] = head_c - hr * 0.075
    J[P.R_EAR] = head_c + hr * 0.075
    J[P.L_EYE] = head_c + hf * 0.08 - hr * 0.032 + np.array([0, 0.02, 0])
    J[P.R_EYE] = head_c + hf * 0.08 + hr * 0.032 + np.array([0, 0.02, 0])
    J[P.NOSE] = head_c + hf * 0.10 - np.array([0, 0.02, 0])
    J[P.HEAD] = head_c + np.array([0, 0.12, 0])
    return J


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


UPPER_BODY = [P.NOSE, P.L_EYE, P.R_EYE, P.L_EAR, P.R_EAR, P.HEAD, P.NECK,
         P.L_SHO, P.R_SHO, P.L_ELB, P.R_ELB, P.L_WRI, P.R_WRI]


def bend_person(t):
    """扭腰（±45°）、彎腰摸腳尖（到 ~110°）的動作。回傳 (J, 扭腰角, 前彎角)（弧度）。"""
    J = person(t, jump=False)
    hip = J[P.HIP].copy()
    right = J[P.R_HIP] - J[P.L_HIP]
    right[1] = 0
    right /= np.linalg.norm(right)
    tw = 0.8 * math.sin(0.7 * t)
    bend = 1.9 * max(0.0, math.sin(0.4 * t)) ** 2
    U = J[UPPER_BODY] - hip
    U = U @ _rot((0, 1, 0), tw).T
    fwd = np.cross((0, 1.0, 0), right)
    ax = np.cross((0, 1.0, 0), fwd)
    U = U @ _rot(ax, bend).T
    J[UPPER_BODY] = U + hip
    return J, tw, bend
