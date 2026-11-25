"""即時追蹤的穩定化（steady.py、skeleton._limit、時間序列模型剛開始的補值）。執行：python tests/test_steady.py

這些修正來自真實畫面（真的 RTMW3D 模型＋人站著舉手、抬腳的影片）：模型的相對深度會跟著手腳的動作飄 10–30 cm，
造成頭亂轉、身體上下晃、站直的腿忽彎忽直。這裡用合成的資料確認每一個零件的行為。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose as P  # noqa: E402
import steady as S  # noqa: E402
import temporal as TM  # noqa: E402
from skeleton import Skeleton  # noqa: E402


def rot(axis, a):
    return S._rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * a)


def angle(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def face_points(H, scale=300.0, at=(400.0, 300.0)):
    """頭的方向 H（欄 = 左、上、前；座標 = 畫面右、畫面上、朝相機）→ 畫面上的五個點。"""
    k, sc = np.zeros((P.NUM_JOINTS, 2)), np.zeros(P.NUM_JOINTS)
    for j, m in S._FACE.items():
        p = H @ np.array(m)
        k[j], sc[j] = (at[0] + scale * p[0], at[1] - scale * p[1]), 1.0
    k[P.NECK], k[P.HEAD] = (at[0], at[1] + 0.11 * scale), (at[0], at[1] - 0.12 * scale)
    sc[P.NECK] = sc[P.HEAD] = 1.0
    return k, sc


def test_head_rotation():
    """臉上五個點在畫面上的位置 → 頭的方向：沒有雜訊時完全正確，每 1 畫素的雜訊約 3 度（兩耳相距 45 畫素）；背對時不用。"""
    rng = np.random.default_rng(0)
    assert angle(S.head_rotation(*face_points(np.eye(3))), np.eye(3)) < 1e-6
    for noise, lim in ((0.0, 0.01), (1.0, 4.5)):
        errs = []
        for _ in range(300):
            yaw, pitch, roll = (np.radians(rng.uniform(-a, a)) for a in (60, 35, 30))
            H = rot([0, 0, 1], roll) @ rot([0, 1, 0], yaw) @ rot([1, 0, 0], -pitch)
            k, sc = face_points(H)
            k[[P.NOSE, P.L_EYE, P.R_EYE, P.L_EAR, P.R_EAR]] += rng.normal(0, noise, (5, 2))
            errs.append(angle(S.head_rotation(k, sc), H))
        assert np.mean(errs) < lim, (noise, np.mean(errs))
    turned = rot([0, 1, 0], np.radians(30))                       # 臉轉向自己的左邊：鼻子在畫面上偏右
    k, sc = face_points(turned)
    assert k[P.NOSE, 0] > 0.5 * (k[P.L_EAR, 0] + k[P.R_EAR, 0]) and S.head_rotation(k, sc)[0, 2] > 0.4
    back = rot([0, 1, 0], np.radians(170))                        # 背對鏡頭（模型還是會猜出臉上的點）：左右反過來 → 不用
    assert S.head_rotation(*face_points(back)) is None
    k, sc = face_points(np.eye(3))
    sc[P.NOSE] = 0.3                                              # 鼻子看不清楚：不用
    assert S.head_rotation(k, sc) is None
    print('  頭的方向：從臉上五個點在畫面上的位置算，不用深度')


def person(lean=0.0):
    """面向 +Z（相機）站直的人；lean：整個上半身往相機傾（公尺，脖子的位移）。"""
    J = np.zeros((P.NUM_JOINTS, 3))
    J[P.HIP], J[P.L_HIP], J[P.R_HIP] = (0, 0.9, 0), (0.09, 0.9, 0), (-0.09, 0.9, 0)
    J[P.L_KNEE], J[P.R_KNEE] = (0.1, 0.5, 0.0), (-0.1, 0.5, 0.0)
    J[P.L_ANK], J[P.R_ANK] = (0.1, 0.09, 0), (-0.1, 0.09, 0)
    J[P.L_HEEL], J[P.R_HEEL] = (0.1, 0.02, -0.05), (-0.1, 0.02, -0.05)
    J[P.L_BIGTOE], J[P.R_BIGTOE] = (0.09, 0.02, 0.15), (-0.09, 0.02, 0.15)
    J[P.L_SMALLTOE], J[P.R_SMALLTOE] = (0.14, 0.02, 0.12), (-0.14, 0.02, 0.12)
    J[P.NECK], J[P.L_SHO], J[P.R_SHO] = (0, 1.42, lean), (0.18, 1.42, lean), (-0.18, 1.42, lean)
    J[P.L_ELB], J[P.R_ELB] = (0.2, 1.14, lean), (-0.2, 1.14, lean)
    J[P.L_WRI], J[P.R_WRI] = (0.2, 0.9, lean), (-0.2, 0.9, lean)
    J[P.L_EAR], J[P.R_EAR], J[P.NOSE] = (0.075, 1.53, lean), (-0.075, 1.53, lean), (0, 1.5, lean + 0.115)
    J[P.L_EYE], J[P.R_EYE], J[P.HEAD] = (0.032, 1.535, lean + 0.085), (-0.032, 1.535, lean + 0.085), (0, 1.655, lean)
    return J


CAM = np.array([0.0, 1.2, 3.0])
RCAM = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])      # lift.Lifter.R（水平的相機）：相機的右、下、前在世界裡的方向


def project(J, f=700.0, c=(640.0, 360.0)):
    d = J - CAM
    return np.stack([c[0] + f * d[:, 0] / -d[:, 2], c[1] - f * d[:, 1] / -d[:, 2]], axis=1)


def test_head_steady():
    """頭照畫面的方向接在脖子上：模型把耳朵的深度弄錯（頭仰 30 度）也不受影響；長時間的平均慢慢對齊模型的點。"""
    J = person()
    k, sc, ok = project(J), np.ones(P.NUM_JOINTS), np.ones(P.NUM_JOINTS, bool)
    hs = S.HeadSteady(0.3, 1.70)
    rng = np.random.default_rng(0)
    ups = []
    for i in range(300):
        X = J.copy()
        back = rng.normal(0, 0.05)                                 # 模型：兩耳、頭頂的深度每幀亂跳 5 cm
        X[[P.L_EAR, P.R_EAR, P.HEAD], 2] += back
        Y = hs(X, ok, k, sc, CAM, RCAM, i / 30.0)
        up = 0.5 * (Y[P.L_EAR] + Y[P.R_EAR]) - Y[P.NECK]
        ups.append(np.degrees(np.arctan2(up[2], up[1])))
        raw = 0.5 * (X[P.L_EAR] + X[P.R_EAR]) - X[P.NECK]
        if i == 0:
            first = abs(np.degrees(np.arctan2(raw[2], raw[1])))
    ups = np.array(ups)
    assert np.abs(np.diff(ups[60:])).mean() < 0.3, np.abs(np.diff(ups[60:])).mean()      # 幾乎不抖（原本每幀差十幾度）
    assert abs(ups[150:].mean()) < 4.0, ups[150:].mean()            # 平均對齊模型的點（這裡 = 沒有仰頭）
    assert abs(np.linalg.norm(Y[P.L_EAR] - Y[P.R_EAR]) - S.EAR_W) < 1e-6 and Y[P.NOSE, 2] > Y[P.L_EAR, 2]
    del first
    X = J.copy()
    sc2 = sc.copy()
    sc2[P.NOSE] = 0.1                                             # 看不到臉：原樣
    assert np.allclose(S.HeadSteady()(X, ok, k, sc2, CAM, RCAM, 0.0), X)
    print('  頭：照畫面的方向接在脖子上，不跟著模型的深度抖')


def test_depth_steady():
    """相對深度：畫面上不動的關節慢慢變（雜訊濾掉），畫面上在動的關節跟得上。"""
    ds = S.DepthSteady(0.3)
    J = person()
    rng = np.random.default_rng(1)
    s = np.ones(P.NUM_JOINTS)
    outs, ins = [], []
    for i in range(240):
        t = i / 30.0
        X = J.copy()
        swing = 0.3 * np.sin(2 * np.pi * 0.8 * t)                 # 左手前後擺（畫面上也在動）
        X[P.L_WRI] = (0.2 + 0.25 * np.cos(2 * np.pi * 0.8 * t), 1.0, swing)
        k = project(X)
        z = -(X[:, 2] - X[P.HIP, 2])                              # 相對深度（越靠近相機越小）
        z_noisy = z + rng.normal(0, 0.06, P.NUM_JOINTS)
        z_noisy[[P.L_HIP, P.R_HIP]] = z[[P.L_HIP, P.R_HIP]]
        outs.append(ds(k, z_noisy, s, t) - z)
        ins.append(z_noisy - z)
    outs, ins = np.array(outs)[90:], np.array(ins)[90:]
    assert outs[:, P.NECK].std() < 0.4 * ins[:, P.NECK].std(), (outs[:, P.NECK].std(), ins[:, P.NECK].std())   # 軀幹：雜訊少一半以上
    assert outs[:, P.R_ANK].std() < 0.6 * ins[:, P.R_ANK].std()
    assert np.abs(outs[:, P.L_WRI]).mean() < 0.09, np.abs(outs[:, P.L_WRI]).mean()        # 在動的手：跟得上（誤差沒有比雜訊大多少）
    z2 = z.copy()
    z2[P.NECK] += 5.0
    assert abs(ds(k, z2, s, 240 / 30.0 + 10.0)[P.NECK] - z2[P.NECK]) < 1e-9             # 中斷很久之後：重新開始，不拖著舊的值
    print('  相對深度：畫面上不動的關節不跟著雜訊晃，在動的關節跟得上')


def test_feet_and_floor():
    """腳尖、腳跟的深度錯得離譜時（腳尖跑到腳踝正下方 20 cm）：腳掌放回放平，最低的點不會把整個人頂高。"""
    J = person()
    ok = np.ones(P.NUM_JOINTS, bool)
    bad = J.copy()
    for toe, small, ank in ((P.L_BIGTOE, P.L_SMALLTOE, P.L_ANK), (P.R_BIGTOE, P.R_SMALLTOE, P.R_ANK)):
        for j in (toe, small):                                    # 沿視線移到腳踝下方很深的地方（畫面位置不變）
            u = (J[j] - CAM) / np.linalg.norm(J[j] - CAM)
            bad[j] = J[j] + 0.5 * u
    assert bad[P.L_BIGTOE, 1] < J[P.L_ANK, 1] - 0.2
    F = S.flat_feet(bad, ok, CAM, 1.70)
    for toe, heel, ank in ((P.L_BIGTOE, P.L_HEEL, P.L_ANK), (P.R_BIGTOE, P.R_HEEL, P.R_ANK)):
        v = F[toe] - F[ank]
        assert abs(v[1] + 0.07) < 0.02 and v[2] > 0.1, v            # 放平、朝相機（+Z）
        assert abs(F[heel, 1] - F[toe, 1]) < 0.02
    assert abs(S.floor_level(F, ok, 1.70) - 0.02) < 0.02
    assert S.floor_level(bad, ok, 1.70) > J[P.L_ANK, 1] - 0.13      # 沒修的點也不會把地板拉到腳踝下方 13 cm 以外
    good = S.flat_feet(J, ok, CAM, 1.70)                            # 本來就放平的腳：幾乎不動（小趾在外側）
    assert np.abs(good - J).max() < 0.03, np.abs(good - J).max()
    assert good[P.L_SMALLTOE, 0] > good[P.L_BIGTOE, 0] and good[P.R_SMALLTOE, 0] < good[P.R_BIGTOE, 0]
    turned = J.copy()                                               # 腳尖朝外 40 度：方向照畫面
    c, sn = np.cos(np.radians(40)), np.sin(np.radians(40))
    for j in (P.L_BIGTOE, P.L_SMALLTOE, P.L_HEEL):
        v = J[j] - J[P.L_ANK]
        turned[j] = J[P.L_ANK] + [c * v[0] + sn * v[2], v[1], -sn * v[0] + c * v[2]]
    T = S.flat_feet(turned, ok, CAM, 1.70)
    v = T[P.L_BIGTOE] - T[P.L_ANK]
    assert abs(np.degrees(np.arctan2(v[0], v[2])) - 40.0) < 8.0, np.degrees(np.arctan2(v[0], v[2]))
    lifted = J.copy()                                               # 抬起來的腳（小腿往後勾）：腳掌跟著小腿，不是放平
    lifted[P.R_KNEE], lifted[P.R_ANK] = (-0.1, 0.5, 0.0), (-0.1, 0.5, -0.4)
    for j in (P.R_HEEL, P.R_BIGTOE, P.R_SMALLTOE):
        lifted[j] = lifted[P.R_ANK] + [0, 0, -0.5]                  # 腳尖的點是壞的
    L = S.flat_feet(lifted, ok, CAM, 1.70)
    shin = (lifted[P.R_ANK] - lifted[P.R_KNEE]) / 0.4
    foot = L[P.R_BIGTOE] - L[P.R_ANK]
    assert abs(foot @ shin - 0.068 * 1.70 / 1.65) < 0.01 and np.linalg.norm(foot) < 0.2
    ok_toe = lifted.copy()                                          # 抬起來的腳、腳尖量得準（距離腳踝一個腳掌長）：照量到的
    ok_toe[P.R_BIGTOE] = lifted[P.R_ANK] + [0.0, -0.14, -0.09]
    ok_toe[P.R_HEEL] = lifted[P.R_ANK] + [0.0, 0.02, -0.07]
    L2 = S.flat_feet(ok_toe, ok, CAM, 1.70)
    assert np.linalg.norm(L2[P.R_BIGTOE] - ok_toe[P.R_BIGTOE]) < 0.02
    print('  腳掌：深度錯的時候放回放平（抬起來的腳跟著小腿），量得準的保留；地板不會被壞掉的腳尖拉低')


def test_limit():
    """站直的腿：模型把膝蓋的深度弄錯 25 cm（變成彎的），超過骨長容許的部分收回來，腿回到幾乎伸直。"""
    J = person()
    ok = np.ones(P.NUM_JOINTS, bool)
    sk = Skeleton(1.70, learn=False)
    sk.L[sk._bone(P.L_HIP, P.L_KNEE)] = sk.L[sk._bone(P.R_HIP, P.R_KNEE)] = 0.40
    sk.L[sk._bone(P.L_KNEE, P.L_ANK)] = sk.L[sk._bone(P.R_KNEE, P.R_ANK)] = 0.41
    sk.L[sk._bone(P.L_SHO, P.L_HIP)] = sk.L[sk._bone(P.R_SHO, P.R_HIP)] = 0.53
    sk.L[sk._bone(P.L_SHO, P.L_ELB)] = sk.L[sk._bone(P.R_SHO, P.R_ELB)] = 0.28
    sk.L[sk._bone(P.L_ELB, P.L_WRI)] = sk.L[sk._bone(P.R_ELB, P.R_WRI)] = 0.24
    sk.L[sk._bone(P.L_HIP, P.R_HIP)], sk.L[sk._bone(P.L_SHO, P.R_SHO)] = 0.18, 0.36

    def along(X, j, d):
        u = (X[j] - CAM) / np.linalg.norm(X[j] - CAM)
        X[j] = X[j] + d * u

    def knee(X):
        a, b = X[P.L_KNEE] - X[P.L_HIP], X[P.L_ANK] - X[P.L_KNEE]
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    bad = J.copy()
    along(bad, P.L_KNEE, -0.25)                                     # 膝蓋往相機 25 cm
    along(bad, P.L_ANK, 0.15)                                       # 腳踝往後 15 cm
    assert knee(bad) > 50
    out = sk._limit(bad, ok, CAM)
    assert knee(out) < 25, knee(out)
    for j in (P.L_KNEE, P.L_ANK):                                   # 只沿視線移動：畫面上的位置不變
        a, b = (bad[j] - CAM), (out[j] - CAM)
        assert a @ b / np.linalg.norm(a) / np.linalg.norm(b) > 1 - 1e-9
    assert np.allclose(out[[P.R_KNEE, P.R_ANK, P.L_HIP, P.NECK]], bad[[P.R_KNEE, P.R_ANK, P.L_HIP, P.NECK]])   # 沒問題的不動
    same = sk._limit(J, ok, CAM)
    assert np.abs(same - J).max() < 1e-9                            # 本來就合理的姿勢：完全不動
    bent = J.copy()                                                 # 真的彎著膝蓋（畫面上腿變短）：不會被拉直
    bent[P.L_KNEE] = (0.1, 0.58, 0.25)
    bent[P.L_ANK] = (0.1, 0.25, 0.02)
    assert abs(knee(sk._limit(bent, ok, CAM)) - knee(bent)) < 3.0
    print('  站直的腿：模型的深度超過骨長容許的部分收回來（只沿視線），真的彎著的不動')


def test_temporal_warmup():
    """時間序列模型剛開始資料還不到視窗長度：前面當成維持第一筆的姿勢（以前填 0，剛開始一兩秒深度會亂掉）。"""
    ts = np.arange(30) / 30.0 + 10.0
    seq = np.zeros((30, 17, 3))
    seq[..., 0] = np.arange(30)[:, None]
    seq[..., 2] = 1.0
    grid = ts[-1] - np.arange(242, -1, -1) / 30.0
    X = TM.resample(ts, seq, grid)
    assert X.shape == (243, 17, 3) and np.allclose(X[0], seq[0]) and np.allclose(X[-1], seq[-1])
    assert (X[..., 2] == 1.0).all()
    gap = np.concatenate([ts[:10], ts[10:] + 2.0])                  # 中間斷了 2 秒：那一段還是當成沒有資料
    X = TM.resample(gap, seq, np.array([ts[9] + 1.0]))
    assert (X == 0).all()
    print('  時間序列模型：剛開始的空白用第一筆的姿勢補')


if __name__ == '__main__':
    test_head_rotation()
    test_head_steady()
    test_depth_steady()
    test_feet_and_floor()
    test_limit()
    test_temporal_warmup()
    print('全部通過')
