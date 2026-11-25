"""一開始蹲著、側傾、彎腰：相機的「上」方向不能被帶歪（之後整段身體都會斜一邊）。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pose as P  # noqa: E402
from lift import Lifter  # noqa: E402


def feed(lifter, v):
    Xc = np.zeros((P.NUM_JOINTS, 3))
    Xc[P.NECK] = v * 0.5
    ok = np.ones(P.NUM_JOINTS, bool)
    lifter._update_up(Xc, ok)


def test_tilted_start():
    pitch = np.radians(12)                      # 相機往下看 12°：直立的人在相機座標裡的「上」
    up = np.array([0.0, -np.cos(pitch), np.sin(pitch)])
    lf = Lifter((1920, 1080))
    side = np.array([0.5, -0.85, 0.1])          # 一開始往旁邊歪
    bend = np.array([0.0, -0.3, 0.95])          # 再彎腰
    for _ in range(120):
        feed(lf, side / np.linalg.norm(side))
    for _ in range(60):
        feed(lf, bend / np.linalg.norm(bend))
    assert abs(lf.R[1][0]) < 1e-9               # 不採用側傾
    for _ in range(300):
        feed(lf, up)
    ang = np.degrees(np.arccos(np.clip(lf.R[1] @ up, -1, 1)))
    print(f'  一開始歪著／彎腰，之後直立：相機方向誤差 {ang:.2f}°')
    assert ang < 1.0


def test_stand_lock_and_fit_guard():
    pitch = np.radians(15)
    up = np.array([0.0, -np.cos(pitch), np.sin(pitch)])
    lf = Lifter((1920, 1080))
    lf._stand = [pitch + d for d in np.linspace(-0.01, 0.01, 10)]
    got = lf.stand_lock()
    assert got is not None and abs(got - 15) < 0.5
    bad = np.array([0.0, -np.cos(0.5), np.sin(0.5)])     # 之後地板擬合、軀幹都歪：仍固定
    lf.fit = (bad, 0.0)
    for _ in range(200):
        feed(lf, bad)
    ang = np.degrees(np.arccos(np.clip(lf.R[1] @ up, -1, 1)))
    print(f'  站直校正後被干擾：誤差 {ang:.2f}°')
    assert ang < 1.0
    lf2 = Lifter((1920, 1080))                            # 沒校正：地板擬合跟軀幹差 >10° 不採用
    lf2.fit = (bad, 0.0)
    for _ in range(300):
        feed(lf2, up)
    ang = np.degrees(np.arccos(np.clip(lf2.R[1] @ up, -1, 1)))
    print(f'  地板擬合偏 {np.degrees(0.5) - 15:.0f}°：誤差 {ang:.2f}°')
    assert ang < 1.0


if __name__ == '__main__':
    test_tilted_start()
    test_stand_lock_and_fit_guard()
    print('全部通過')
