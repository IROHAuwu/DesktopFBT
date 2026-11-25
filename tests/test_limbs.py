"""單相機的手腳前後修正：手和頭重疊時的前後（看臉部的點）、頭與軀幹不會被穿過、膝蓋往後折。執行：python tests/test_limbs.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import limbs as LB  # noqa: E402
import pose as P  # noqa: E402


def hands_behind_head():
    """面向 +Z（相機），雙手抱頭：手腕在頭後面，手肘在兩側。"""
    J = np.zeros((P.NUM_JOINTS, 3))
    J[P.HIP], J[P.L_HIP], J[P.R_HIP] = (0, 0.9, 0), (0.09, 0.9, 0), (-0.09, 0.9, 0)
    J[P.L_KNEE], J[P.R_KNEE] = (0.1, 0.5, 0.01), (-0.1, 0.5, 0.01)
    J[P.L_ANK], J[P.R_ANK] = (0.1, 0.1, 0), (-0.1, 0.1, 0)
    J[P.NECK], J[P.L_SHO], J[P.R_SHO] = (0, 1.42, 0), (0.18, 1.42, 0), (-0.18, 1.42, 0)
    J[P.L_EAR], J[P.R_EAR], J[P.NOSE] = (0.07, 1.58, -0.01), (-0.07, 1.58, -0.01), (0, 1.57, 0.1)
    J[P.L_EYE], J[P.R_EYE], J[P.HEAD] = (0.03, 1.6, 0.08), (-0.03, 1.6, 0.08), (0, 1.68, 0)
    J[P.L_ELB], J[P.R_ELB] = (0.3, 1.58, -0.04), (-0.3, 1.58, -0.04)
    J[P.L_WRI], J[P.R_WRI] = (0.06, 1.6, -0.12), (-0.06, 1.6, -0.12)
    return J


ARM = {P.L_WRI: P.L_ELB, P.R_WRI: P.R_ELB}


def palm(Y, w, n=25):
    """手腕到指尖（手掌是直的：沿著前臂延伸）上的點。"""
    v = Y[w] - Y[ARM[w]]
    return Y[w] + np.linspace(0, 1, n)[:, None] * LB.HAND_LEN * v


def gap(Y, w):
    """手掌離頭中心最近的距離。"""
    head = 0.5 * (Y[P.L_EAR] + Y[P.R_EAR])
    return float(np.linalg.norm(palm(Y, w) - head, axis=1).min())


def where(Y, w, cam):
    """手掌和頭在畫面上重疊的部分在頭的前面 (+1) 還是後面 (-1)；沒有重疊 → 0。"""
    head = 0.5 * (Y[P.L_EAR] + Y[P.R_EAR])
    d = (cam - head) / np.linalg.norm(cam - head)
    rel = palm(Y, w) - head
    t = rel @ d
    over = np.linalg.norm(rel - t[:, None] * d, axis=1) < LB.HEAD_R
    if not over.any():
        return 0
    assert (t[over] > 0).all() or (t[over] < 0).all(), '手掌穿過頭'
    return 1 if t[over][0] > 0 else -1


def test_hands_behind_head():
    J = hands_behind_head()
    X = J.copy()
    X[[P.L_WRI, P.R_WRI], 2] = 0.08                     # 模型把手估到臉前面
    X[[P.L_ELB, P.R_ELB], 2] = 0.12
    X[P.L_KNEE, 2] = -0.1                               # 膝蓋往後折
    ok = np.ones(P.NUM_JOINTS, bool)
    sc = np.full(P.NUM_JOINTS, 0.9)
    sc[[P.L_WRI, P.R_WRI]] = 0.7                        # 手被頭擋住
    cam = np.array([0.0, 1.2, 3.0])
    assert gap(X, P.L_WRI) < LB.HEAD_R                  # 原本：手掌在頭裡面
    Y, _ = LB.fix(X, ok, sc, {}, cam)
    for w in (P.L_WRI, P.R_WRI):
        assert where(Y, w, cam) == -1 and gap(Y, w) >= LB.R_BACK - 0.005, (Y[w], gap(Y, w))     # 整個手掌回到頭後面
        assert np.allclose(Y[ARM[w]], X[ARM[w]])        # 手肘不動、前臂長度不變：只轉前臂
        assert abs(np.linalg.norm(Y[w] - Y[ARM[w]]) - np.linalg.norm(X[w] - X[ARM[w]])) < 1e-9
    assert abs(Y[P.L_KNEE, 2]) < 0.03
    sc2 = sc.copy()
    sc2[[P.L_WRI, P.R_WRI]] = 0.9
    sc2[[P.NOSE, P.L_EYE, P.R_EYE]] = 0.6               # 臉被手擋住：手在前面
    Y2, _ = LB.fix(J, ok, sc2, {}, cam)
    assert where(Y2, P.L_WRI, cam) == 1 and gap(Y2, P.L_WRI) >= LB.R_FRONT - 0.005
    sc3 = np.full(P.NUM_JOINTS, 0.9)                    # 分數差不多（不知道前後）：留在模型的那一側，但手掌不能在頭裡面
    Y3, _ = LB.fix(X, ok, sc3, {}, cam)
    for w in (P.L_WRI, P.R_WRI):
        assert gap(Y3, w) >= LB.R_BACK - 0.005 and where(Y3, w, cam) != 0
    Y4, _ = LB.fix(J, ok, sc3, {}, cam)                 # 本來就在頭後面、沒有碰到：幾乎不動
    assert np.abs(Y4 - J).max() < 0.02
    print('  手抱頭：整個手掌回到頭後面（只轉前臂），膝蓋拉直')



def wholebody(hand_at, hand_score, covered_score=0.9, face_score=0.9, side='Left'):
    """133 點（畫素）：臉在 (300,200)、寬約 80px；一隻手放在 hand_at，另一隻放得很遠。covered_score：被手蓋住的臉部點的分數。"""
    k = np.zeros((133, 2))
    s = np.full(133, 0.9)
    k[:17] = [300, 400]
    k[0], k[1], k[2], k[3], k[4] = (300, 205), (315, 190), (285, 190), (342, 200), (258, 200)
    a = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    k[23:91] = np.stack([300 + 40 * np.cos(a) * np.linspace(1, 0.3, 68), 200 + 48 * np.sin(a) * np.linspace(1, 0.3, 68)], -1)
    s[23:91] = face_score
    s[0:3] = face_score
    wi, sl, other_w, other = (9, slice(91, 112), 10, slice(112, 133)) if side == 'Left' else (10, slice(112, 133), 9, slice(91, 112))
    hand_at = np.asarray(hand_at, float)
    k[wi] = hand_at + [0, 22]
    k[sl] = hand_at + np.stack([np.linspace(-12, 12, 21), np.linspace(-10, 10, 21) % 7 - 3], -1)
    s[sl] = hand_score
    k[other_w], k[other] = (300, 520), (300, 540)
    under = np.linalg.norm(k[23:91][:, None] - np.vstack([k[wi][None], k[sl]])[None], axis=2).min(1) < 0.35 * 48
    s[23:91][under] = covered_score
    return k, s, int(under.sum())


def test_head_cue():
    """畫面上的判斷：手蓋住的地方五官看得到 → 頭後面；被蓋住 → 臉前面；背對（沒有臉）而手看得到 → 鏡頭這一側。"""
    k, s, n = wholebody((330, 205), 0.2)                  # 手在臉的左半邊，但五官都看得到、手看不到：抱頭
    assert n >= 5
    d = {}
    c = LB.head_cue(k, s, d)
    assert c == {'Left': -1, 'Right': 0} and d['Left']['under'] >= 5 and 'Right' not in d, (c, d)
    k, s, _ = wholebody((330, 205), 0.9)                  # 手的分數也高（模型對被擋住的點常常還是很有把握）：五官看得到就算後面
    assert LB.head_cue(k, s)['Left'] == -1
    k, s, _ = wholebody((300, 205), 0.9, covered_score=0.3)     # 手蓋住的五官分數低：手在臉前面
    assert LB.head_cue(k, s)['Left'] == 1
    k, s, _ = wholebody((300, 205), 0.9, covered_score=0.3, side='Right')
    assert LB.head_cue(k, s) == {'Left': 0, 'Right': 1}
    k, s, _ = wholebody((300, 205), 0.9, covered_score=0.15, face_score=0.15)   # 雙手把整張臉遮住
    assert LB.head_cue(k, s)['Left'] == 1
    k, s, _ = wholebody((300, 205), 0.85, covered_score=0.62)   # 五官分數降了一些、手很清楚：前面
    assert LB.head_cue(k, s)['Left'] == 1
    k, s, _ = wholebody((300, 205), 0.55, covered_score=0.62)   # 五官和手差不多清楚：不判斷
    assert LB.head_cue(k, s)['Left'] == 0
    k, s, _ = wholebody((300, 205), 0.4, covered_score=0.62)    # 五官比手清楚：後面
    assert LB.head_cue(k, s)['Left'] == -1
    k, s, _ = wholebody((300, 420), 0.9)                  # 手在肚子前面：和頭無關
    assert LB.head_cue(k, s) == {'Left': 0, 'Right': 0}
    k, s, n = wholebody((300, 140), 0.2)                  # 手在頭頂（頭髮）上、手看不到：被頭擋住
    d = {}
    assert LB.head_cue(k, s, d)['Left'] == -1 and d['Left']['under'] < 5, d
    k, s, _ = wholebody((300, 140), 0.9)                  # 手在頭頂上、看得到：不知道前後（交給模型和「不穿過頭」）
    assert LB.head_cue(k, s)['Left'] == 0
    k, s, _ = wholebody((300, 140), 0.9, face_score=0.1)  # 背對鏡頭（沒有臉）、手看得到：在鏡頭這一側（後腦）
    k[23:91] = k[23:91] * 0 + [300, 200]                  # 背對時臉部點擠成一團
    assert LB.head_cue(k, s)['Left'] == 1
    assert LB.head_cue(k[:26], s[:26]) == {'Left': 0, 'Right': 0}    # 只有 26 點的模型：沒有這份資訊
    print('  頭與手（畫面）：五官看得到 → 後面，被蓋住 → 前面')


def test_head_cue_in_3d():
    J = hands_behind_head()
    ok = np.ones(P.NUM_JOINTS, bool)
    sc = np.full(P.NUM_JOINTS, 0.9)                     # 臉和手腕的分數一樣高：舊的做法判斷不出來
    cam = np.array([0.0, 1.2, 3.0])
    head = 0.5 * (J[P.L_EAR] + J[P.R_EAR])
    X = J.copy()
    X[[P.L_WRI, P.R_WRI], 2] = 0.08                     # 模型把手估到臉前面
    X[[P.L_ELB, P.R_ELB], 2] = 0.12
    hand = {'Left': None}
    Y, hp = LB.fix(X, ok, sc, hand, cam, cue={'Left': -1, 'Right': -1})
    for w in (P.L_WRI, P.R_WRI):
        assert where(Y, w, cam) == -1 and gap(Y, w) >= LB.R_BACK - 0.005     # 五官看得到 → 整個手掌在頭後面、不碰到頭
    Y2, _ = LB.fix(J, ok, sc, {}, cam, cue={'Left': 1, 'Right': 0})      # 左手：五官被蓋住 → 臉前面；右手不知道：留在後面
    assert where(Y2, P.L_WRI, cam) == 1 and gap(Y2, P.L_WRI) >= LB.R_FRONT - 0.005
    assert where(Y2, P.R_WRI, cam) == -1 and np.abs(Y2[P.R_WRI] - J[P.R_WRI]).max() < 0.02
    far = J.copy()
    far[P.L_WRI] = head + [0.02, 0.0, 0.4]              # 本來就在臉前面 40cm：不拉回來
    far[P.L_ELB] = head + [0.2, -0.2, 0.3]
    Y3, _ = LB.fix(far, ok, sc, {}, cam, cue={'Left': 1})
    assert np.allclose(Y3[P.L_WRI], far[P.L_WRI])
    # 手伸向鏡頭、只是畫面上蓋到臉（模型把手腕放在頭前面很遠）：就算臉部的點「看得到」也不拉到頭後面
    reach = J.copy()
    reach[P.L_WRI] = head + [0.02, 0.0, 0.45]
    reach[P.L_ELB] = head + [0.15, -0.15, 0.25]
    lb = LB.Limbs()
    Y5, _ = lb(reach, ok, sc, None, cam, {'Left': -1})
    assert np.allclose(Y5[P.L_WRI], reach[P.L_WRI]) and 'Left' not in lb.head_side
    # 手腕在耳朵旁邊、手肘在外側、前臂指向頭（使用者回報的姿勢）：手腕沒有碰到頭，但手掌伸進頭裡 → 整個手掌要繞到頭後面
    ear = J.copy()
    ear[P.L_ELB], ear[P.L_WRI] = head + [0.35, -0.15, 0.09], head + [0.13, -0.07, -0.03]
    ear[P.R_ELB], ear[P.R_WRI] = head + [-0.35, -0.15, 0.09], head + [-0.13, -0.07, -0.03]
    assert np.linalg.norm(ear[P.L_WRI] - head) > LB.HEAD_R and gap(ear, P.L_WRI) < LB.HEAD_R
    for cue in ({'Left': -1, 'Right': -1}, None):       # 判斷成後面，或判斷不出來（模型本來就偏後面）：都繞到後面
        Y6, _ = LB.fix(ear, ok, sc, {}, cam, cue=cue)
        for w in (P.L_WRI, P.R_WRI):
            assert where(Y6, w, cam) == -1 and gap(Y6, w) >= LB.R_BACK - 0.005, (cue, gap(Y6, w))
            assert np.linalg.norm(Y6[w] - ear[w]) < 0.12                  # 轉得不多
    Y7, _ = LB.fix(ear, ok, sc, {}, cam, cue={'Left': 1, 'Right': 1})     # 判斷成前面：手掌擋在臉前面
    assert where(Y7, P.L_WRI, cam) == 1 and gap(Y7, P.L_WRI) >= LB.R_FRONT - 0.005
    # 手慢慢滑過頭的邊緣：修正量連續，不會跳
    prev, steps = None, []
    for x in np.linspace(0.30, 0.10, 41):
        Z = J.copy()
        Z[P.L_ELB], Z[P.L_WRI] = head + [x + 0.22, -0.15, 0.09], head + [x, -0.07, -0.03]
        y = LB.fix(Z, ok, sc, {}, cam, cue={'Left': -1})[0][P.L_WRI] - Z[P.L_WRI]
        if prev is not None:
            steps.append(float(np.linalg.norm(y - prev)))
        prev = y
    assert max(steps) < 0.035, max(steps)
    # 有手的點（即時）：照實際的手掌方向檢查，手的點跟著手腕移動
    H = ear[P.L_WRI] + np.linspace(0, 1, 21)[:, None] * np.array([-0.15, 0.0, -0.02])
    Y8, hp = LB.fix(ear, ok, sc, {'Left': H}, cam, cue={'Left': -1})
    assert np.allclose(hp['Left'] - H, Y8[P.L_WRI] - ear[P.L_WRI]) and np.linalg.norm(Y8[P.L_WRI] - ear[P.L_WRI]) > 0.02
    tip = hp['Left'][12]
    seg = Y8[P.L_WRI] + np.linspace(0, 1, 25)[:, None] * (tip - Y8[P.L_WRI])
    assert np.linalg.norm(seg - head, axis=1).min() >= LB.R_BACK - 0.005
    print('  頭與手（3D）：整個手掌照畫面的判斷放到頭後面／臉前面，不會穿過頭，滑過頭的邊緣時連續')


def test_steady_and_head_size():
    """離線（影片）：前後的判斷用前後幾幀的多數決；頭的大小可以調（Avatar 的頭比較大時）。"""
    seq = [{'Left': -1, 'Right': 0}] * 10 + [{'Left': 1, 'Right': 0}] + [None] * 3 + [{'Left': -1, 'Right': 0}] * 10
    out = LB.steady_cues(seq, 30.0)
    assert len(out) == len(seq) and all(o['Left'] == -1 and o['Right'] == 0 for o in out), out     # 零星的相反判斷、空白都補成「後面」
    lone = LB.steady_cues([None] * 20 + [{'Left': 1}] + [None] * 20, 30.0)
    assert lone[20]['Left'] == 1 and lone[19]['Left'] == 0          # 只有一幀的判斷：照用，但不擴散
    flip = LB.steady_cues([{'Left': -1}] * 30 + [{'Left': 1}] * 30, 30.0)
    assert [f['Left'] for f in flip][:20] == [-1] * 20 and [f['Left'] for f in flip][-20:] == [1] * 20   # 真的換邊：跟著換
    assert LB.steady_cues([], 30.0) == []
    J = hands_behind_head()
    ok, sc, cam = np.ones(P.NUM_JOINTS, bool), np.full(P.NUM_JOINTS, 0.9), np.array([0.0, 1.2, 3.0])
    head = 0.5 * (J[P.L_EAR] + J[P.R_EAR])
    ear = J.copy()
    ear[P.L_ELB], ear[P.L_WRI] = head + [0.35, -0.15, 0.09], head + [0.13, -0.07, -0.03]
    try:
        gaps = []
        for r in (0.10, 0.13):
            LB.set_head_size(r)
            Y = LB.fix(ear, ok, sc, {}, cam, cue={'Left': -1})[0]
            gaps.append(gap(Y, P.L_WRI))
            assert gaps[-1] >= r + 0.03 - 0.005
        assert gaps[1] > gaps[0] + 0.02
    finally:
        LB.set_head_size(LB.HEAD_R)
    assert LB.GROW == 0.0
    print('  影片：前後的判斷前後幾幀一起看；頭的大小可以調')


def test_head_memory():
    """即時、影片：有幾幀判斷不出來時沿用上一次的判斷，不會前後跳；離開頭附近就忘掉。"""
    J = hands_behind_head()
    ok = np.ones(P.NUM_JOINTS, bool)
    sc = np.full(P.NUM_JOINTS, 0.9)
    cam = np.array([0.0, 1.2, 3.0])
    X = J.copy()
    X[[P.L_WRI, P.R_WRI], 2] = 0.2                      # 模型一直說手在臉前面
    lb = LB.Limbs()
    behind = lambda Y: Y[P.L_WRI, 2] < J[P.L_EAR, 2] - 0.04     # noqa: E731
    assert behind(lb(X, ok, sc, None, cam, {'Left': -1, 'Right': -1})[0])
    for _ in range(LB.HOLD):
        assert behind(lb(X, ok, sc, None, cam, {'Left': 0, 'Right': 0})[0])
    assert not behind(lb(X, ok, sc, None, cam, {'Left': 0, 'Right': 0})[0])       # 太久沒有新的判斷：回到模型的結果
    assert behind(lb(X, ok, sc, None, cam, {'Left': -1, 'Right': -1})[0])
    away = X.copy()
    away[P.L_WRI] = (0.5, 1.2, 0.2)                     # 手離開頭附近
    lb(away, ok, sc, None, cam, None)
    assert not behind(lb(X, ok, sc, None, cam, None)[0])
    knee = X.copy()
    knee[P.L_KNEE, 2] = -0.1
    assert lb.head(knee, ok, sc, None, cam, {'Left': -1})[0][P.L_KNEE, 2] == -0.1   # head()：只管頭，不動膝蓋
    print('  頭與手：判斷不出來時沿用上一次的判斷')


def test_estimator_carries_cue():
    """模型的 133 點 → 每一幀的判斷跟著手的點一起傳到後面的流程。"""
    import lift
    k, s, _ = wholebody((330, 205), 0.2)
    est = object.__new__(lift.Pose3DEstimator)
    k26, z26, s26 = est._halpe((k, np.zeros(133), s))
    assert k26.shape == (P.NUM_JOINTS, 2) and est.cue == {'Left': -1, 'Right': 0}
    hk, hz, hs = est.hands                              # 仍然可以當成三個陣列的 tuple 用
    assert hk.shape == (42, 2) and est.hands.cue is est.cue and est.cue_detail['Left']['cue'] == -1
    est._halpe((k[:26], np.zeros(26), s[:26]))          # 26 點的模型：沒有
    assert est.cue is None and est.hands is None


def test_hands_not_through_torso():
    """手從右邊經過肚子移到左邊：在中線前面的走前面的弧線，後面的走後面，不會穿過去。"""
    base = hands_behind_head()
    ok = np.ones(P.NUM_JOINTS, bool)
    sc = np.full(P.NUM_JOINTS, 0.9)
    for z0, front in ((0.03, True), (-0.03, False)):
        lm = LB.Limbs()
        zs = []
        for i in range(41):
            X = base.copy()
            xw = -0.25 + 0.5 * i / 40
            X[P.R_WRI] = (xw, 1.1, z0)
            X[P.R_ELB] = (-0.2, 1.2, z0 * 2)
            X[P.L_WRI], X[P.L_ELB] = (0.4, 1.0, 0), (0.3, 1.2, 0)
            Y, _ = lm(X, ok, sc)
            zs.append(Y[P.R_WRI, 2])
            inside = (Y[P.R_WRI, 0] / 0.15) ** 2 + (Y[P.R_WRI, 2] / 0.105) ** 2
            assert inside >= 1.0 or abs(Y[P.R_WRI, 0]) > 0.15, (i, Y[P.R_WRI])
        zs = np.array(zs)
        assert (zs.min() > 0) if front else (zs.max() < 0), zs
        mid = zs[20]
        assert abs(mid) > 0.1                                # 經過中間時繞到身體表面外
    print('  手經過肚子：前面走前面、後面走後面，不會穿過軀幹')


if __name__ == '__main__':
    test_hands_behind_head()
    test_head_cue()
    test_head_cue_in_3d()
    test_steady_and_head_size()
    test_head_memory()
    test_estimator_carries_cue()
    test_hands_not_through_torso()
    print('全部通過')
