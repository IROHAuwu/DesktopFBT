"""精度優先（桌面模式）：轉身時整個人左右／前後判反、腿標反、抖動，延遲後要修好。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import motion  # noqa: E402
import pose as P  # noqa: E402
import precise as PR  # noqa: E402


def simulate(consistent, face=True, seconds=24, seed=1, fps=30, decide=0.8):
    rng = np.random.default_rng(seed)
    cam = np.array([0.0, 1.2, 3.5])
    truth, raw, S = [], [], []
    bb = bl = 0
    for i in range(seconds * fps):
        t = i / fps
        J = np.array(motion.person(t), float)
        h = J[P.HIP].copy()
        J = (J - h) @ motion.rot_y(0.5 * t + 0.6 * np.sin(0.3 * t)).T + h      # 轉好幾圈
        truth.append(J)
        d = PR._unit((cam - h) * np.array([1, 0, 1]))
        toward = PR.facing(J) @ d
        s = np.full(P.NUM_JOINTS, 0.85)
        fs = (0.9 if toward > 0.2 else (0.25 if toward < -0.2 else 0.55)) if face else 0.5
        for j in (P.NOSE, P.L_EYE, P.R_EYE):
            s[j] = np.clip(fs + rng.normal(0, 0.08), 0, 1)
        X = J + rng.normal(0, 0.02, J.shape) + np.outer(rng.normal(0, 0.05, P.NUM_JOINTS), PR._unit(cam - h))
        if rng.random() < 0.01:
            X[rng.integers(26)] += rng.normal(0, 0.3, 3)
        k = 30.0 / fps
        if bb == 0 and rng.random() < k * (0.06 if abs(toward) < 0.6 else 0.01):
            bb = int(rng.integers(3, 30) / k) + 1
        if bl == 0 and rng.random() < k * 0.02:
            bl = int(rng.integers(2, 12) / k) + 1
        if bb:
            X, s = X[PR.PERM_BODY], s[PR.PERM_BODY]
            if consistent:                                   # 模型以為背對：深度也一起反過來
                X = PR.mirror(X, cam)
        if bl:
            X, s = X[PR.PERM_LEGS], s[PR.PERM_LEGS]
        bb, bl = max(0, bb - 1), max(0, bl - 1)
        raw.append(X)
        S.append(s)
    pr = PR.Precise(fps, decide_s=decide)
    out = {}
    for i in range(len(raw)):
        r = pr.push(i / fps, raw[i], S[i], np.ones(26, bool), None, cam)
        if r is not None:
            out[int(round(r[0] * fps))] = r[1]
    idx = sorted(out)
    err = lambda A, B: np.linalg.norm(A - B, axis=1).mean()  # noqa: E731
    big_raw = np.mean([err(raw[i], truth[i]) > 0.12 for i in idx])
    big_out = np.mean([err(out[i], truth[i]) > 0.12 for i in idx])
    jit = lambda seq: np.linalg.norm(np.diff(np.stack(seq), 2, axis=0), axis=2).mean()  # noqa: E731
    return big_raw, big_out, jit([raw[i] for i in idx]), jit([out[i] for i in idx]), \
        np.mean([err(out[i], truth[i]) for i in idx])


def test_label_swaps():
    br, bo, jr, jo, e = simulate(consistent=False)
    print(f'  標籤對調：判錯 {br:.1%} → {bo:.1%}，抖動 {jr:.4f} → {jo:.4f}，誤差 {e * 100:.1f} cm')
    assert br > 0.15 and bo < 0.03 and jo < jr / 4 and e < 0.045


def test_turn_confusion():
    br, bo, jr, jo, e = simulate(consistent=True)
    print(f'  轉身判反（深度也反）：判錯 {br:.1%} → {bo:.1%}，誤差 {e * 100:.1f} cm')
    assert br > 0.15 and bo < 0.05 and e < 0.045


def test_no_face_info():
    br, bo, *_ = simulate(consistent=False, face=False)
    print(f'  沒有臉的資訊：判錯 {br:.1%} → {bo:.1%}')
    assert bo < 0.03


def test_15fps():
    br, bo, jr, jo, e = simulate(consistent=True, fps=15)
    print(f'  15 FPS：判錯 {br:.1%} → {bo:.1%}，誤差 {e * 100:.1f} cm')
    assert bo < 0.06 and e < 0.05


def simulate_bend(seconds=30, seed=3, fps=60, flips=True):
    """扭腰、彎腰：模型把肩膀前後（扭腰方向）或上半身往前／往後（彎腰方向）判反。"""
    rng = np.random.default_rng(seed)
    cam = np.array([0.0, 1.2, 3.5])
    truth, raw, S = [], [], []
    ep, kind = 0, 0
    for i in range(seconds * fps):
        t = i / fps
        J, tw, bend = motion.bend_person(t)
        h = J[P.HIP].copy()
        J = (J - h) @ motion.rot_y(np.pi + 0.4 * np.sin(0.2 * t)).T + h       # 大致面向鏡頭
        truth.append(J)
        X = J + rng.normal(0, 0.015, J.shape) + np.outer(rng.normal(0, 0.04, P.NUM_JOINTS), PR._unit(cam - h))
        s = np.full(P.NUM_JOINTS, 0.85)
        hf = np.cross((0, 1.0, 0), J[P.R_EAR] - J[P.L_EAR])                # 頭的前方
        toward = PR._unit(hf * [1, 0, 1]) @ PR._unit((cam - h) * [1, 0, 1])
        s[[P.NOSE, P.L_EYE, P.R_EYE]] = 0.9 if toward > 0.2 else (0.25 if toward < -0.2 else 0.55)
        k = 30.0 / fps
        pt = abs(PR.facing(J) @ PR._unit((cam - h) * [1, 0, 1]))
        if flips and ep == 0 and pt > 0.6:                    # 肩線、上半身和視線垂直時才分不出前後
            if abs(tw) > 0.35 and rng.random() < k * 0.08:
                ep, kind = int(rng.integers(5, 40) / k) + 1, 1
            elif bend > 0.7 and rng.random() < k * 0.08:
                ep, kind = int(rng.integers(5, 40) / k) + 1, 2
        if ep:
            X = PR.upper_variant(X, kind, cam)
            ep -= 1
        raw.append(X)
        S.append(s)
    pr = PR.Precise(fps)
    out = {}
    for i in range(len(raw)):
        r = pr.push(i / fps, raw[i], S[i], np.ones(26, bool), None, cam)
        if r is not None:
            out[int(round(r[0] * fps))] = r[1]
    idx = sorted(out)
    U = PR.UPPER
    err = lambda A, B: np.linalg.norm(A[U] - B[U], axis=1).mean()  # noqa: E731
    return (np.mean([err(raw[i], truth[i]) > 0.12 for i in idx]),
            np.mean([err(out[i], truth[i]) > 0.12 for i in idx]),
            np.mean([err(out[i], truth[i]) for i in idx]))


def test_twist_and_bend():
    br, bo, e = simulate_bend()
    print(f'  扭腰／彎腰判反：判錯 {br:.1%} → {bo:.1%}，上半身誤差 {e * 100:.1f} cm')
    assert br > 0.1 and bo < 0.03 and e < 0.05


def test_twist_and_bend_no_false_flips():
    br, bo, e = simulate_bend(flips=False, seed=5)
    print(f'  扭腰／彎腰（模型都對）：判錯 {bo:.1%}，誤差 {e * 100:.1f} cm')
    assert bo < 0.01 and e < 0.04


def test_viterbi():
    u = np.array([[0, 5], [0, 5], [5, 0], [0, 5]], float)
    assert list(PR.viterbi(u, lambda t: np.array([[0, 1], [1, 0]]))) == [0, 0, 1, 0]
    assert list(PR.viterbi(u, lambda t: np.array([[0, 99], [99, 0]]))) == [0, 0, 0, 0]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
