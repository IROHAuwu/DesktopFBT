"""GEM-X 的前後處理（不需要模型）：裁切、熱圖解碼、時序模型的輸入、解碼、骨架 → 26 點、整段求解（用假的時序模型）。
和官方 PyTorch 實作的比對在 tools/export_gemx.py（CI，需要真的權重）。執行：python tests/test_gemx.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gemx as GX  # noqa: E402
import pose as P  # noqa: E402

SKELETON = os.path.join(HERE, 'data', 'gemx_skeleton.npz')       # 匯出流程算出來的 SOMA 零姿勢骨架（和發佈的那份相同）


def rotation(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * Kx @ Kx


def test_rot6d_mask():
    R = np.stack([rotation([1, 2, 3], 40), rotation([0, 1, 0], -100)])
    assert np.allclose(GX.rot6d(R[:, :2, :].reshape(2, 6)), R, atol=1e-9)
    for L in (50, 120, 121, 300):                      # 和官方的迴圈寫法相同
        ref = np.zeros((L, L), bool)
        if L > GX.MAX_LEN:
            ref[:] = True
            for i in range(L):
                lo = min(L - GX.MAX_LEN, max(0, i - GX.MAX_LEN // 2))
                hi = max(GX.MAX_LEN, min(L, i + GX.MAX_LEN // 2))
                ref[i, lo:hi] = False
        assert (GX.attn_mask(L) == ref).all(), L


def test_crops_and_boxes():
    img = np.zeros((480, 640, 3), np.uint8)
    img[200:280, 300:340] = 255
    box = np.float32([320, 240, 300])
    assert GX.crop_body(img, box).shape == (3, 256, 256) and GX.crop_kp(img, box).shape == (3, 256, 192)
    k = np.array([[300.0, 100], [340, 400]] + [[320, 250]] * 24)
    b = GX.person_box(k, np.ones(26))
    assert np.allclose([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], [320, 250]) and b[3] - b[1] > 300
    assert GX.person_box(k, np.zeros(26)) is None
    sq = GX.square_boxes(np.tile(b, (20, 1)))
    assert sq.shape == (20, 3) and np.allclose(sq[:, 2], (b[3] - b[1]) * 1.2, rtol=1e-5)      # 直立的人：邊長 = 高 × 1.2
    wide = GX.square_boxes(np.array([[0.0, 0, 300, 100]]))
    assert abs(wide[0, 2] - 300 / 0.75 * 1.2) < 1e-3


def test_heatmaps():
    """在已知位置放一個峰 → 解碼回全畫面座標；翻轉的那份轉回來後左右關節對調。"""
    box = np.float32([[400, 300, 256]])
    heat = np.zeros((1, 77, 64, 48), np.float32)
    for j in range(77):
        heat[0, j, 10 + j % 40, 5 + j % 30] = 0.9
    kp = GX.decode_heatmaps(heat, box)
    j = 12
    x = (5 + j % 30) / 47 * (256 * 0.75) + 400 - 256 * 0.375
    y = (10 + j % 40) / 63 * 256 + 300 - 128
    assert np.allclose(kp[0, j], [x, y, 0.9], atol=1e-3), kp[0, j]
    f = GX.flip_heatmaps(heat[..., ::-1])               # 把同一張圖水平翻轉後「當成翻轉輸入的結果」
    assert np.allclose(f[0, GX.FLIP77], heat[0])
    assert sorted(GX.FLIP77.tolist()) == list(range(77)) and (GX.FLIP77[GX.FLIP77] == np.arange(77)).all()
    assert GX.FLIP77[GX.L_ARM] == GX.R_ARM and GX.FLIP77[GX.L_FOOT] == GX.R_FOOT and GX.FLIP77[GX.L_EYE_J] == GX.R_EYE_J


def test_observations_decode():
    boxes = np.float32([[320, 240, 200]])
    kp = np.zeros((1, 77, 3), np.float32)
    kp[0, 0] = [320, 240, 0.9]                          # 框的中心
    kp[0, 1] = [420, 340, 0.8]                          # 框的右下角
    kp[0, 2] = [500, 240, 0.8]                          # 框外
    o = GX.observations(kp, boxes)
    assert np.allclose(o[0, 0], [0, 0, 0.9]) and np.allclose(o[0, 1], [1, 1, 0.8]) and o[0, 2, 2] == 0
    K = GX.intrinsics(640, 480)
    assert abs(K[0, 0] - 800) < 1e-3 and np.allclose(GX.cliff(boxes, K), [[0, 0, 0.25]])
    x = np.zeros((3, 655))
    x[:, GX.S_POSE] = np.tile([1, 0, 0, 0, 1, 0], 76)
    x[:, GX.S_RC] = x[:, GX.S_RGV] = [1, 0, 0, 0, 1, 0]
    x[:, GX.S_VEL] = [0, 0, 0.1]
    d = GX.decode(x)
    assert np.allclose(d['R_body'], np.eye(3)) and np.allclose(d['scale'], 1)
    R, t = GX.world_static(d['R_gv'], d['vel'])
    assert np.allclose(R, GX.FLIP_Z) and np.allclose(t[:, 2], [0, 0.1, 0.2]) and np.allclose(t[:, :2], 0)
    _, t2 = GX.world_static(d['R_gv'], d['vel'], np.full(3, 0.5))          # 60fps 的影片：每一幀走一半
    assert np.allclose(t2[:, 2], [0, 0.05, 0.1])
    cam = GX.camera_transl(np.array([[1.0, 0, 0]]), boxes, K)
    assert np.allclose(cam, [[0, 0, 2 * 800 / 200]])


def test_fk_chain():
    rest = np.zeros((77, 3))
    parents = np.arange(-1, 76)
    rest[:, 1] = np.arange(77) * 0.1                    # 一條往上的鏈
    I = np.tile(np.eye(3), (1, 76, 1, 1))
    J = GX.fk(rest, parents, np.eye(3)[None], I, np.array([[1.0, 2, 3]]))
    assert np.allclose(J[0, 0], [1, 2, 3]) and np.allclose(J[0, 76], [1, 2 + 7.6, 3])
    Rb = I.copy()
    Rb[0, 0] = rotation([0, 0, 1], 90)                  # 第二個關節轉 90°：之後的骨頭都倒向 -x
    J, Gr = GX.fk(rest, parents, np.eye(3)[None], Rb, rotations=True)
    assert np.allclose(J[0, 1], [0, 0.1, 0]) and np.allclose(J[0, 3], [-0.2, 0.1, 0], atol=1e-9)
    assert np.allclose(Gr[0, 5], rotation([0, 0, 1], 90))


def body():
    if not os.path.exists(SKELETON):
        print('  skip：沒有 tests/data/gemx_skeleton.npz')
        return None
    return GX.Body(SKELETON)


def test_body_points():
    b = body()
    if b is None:
        return
    assert b.rest.shape == (77, 3) and b.parents[0] == -1 and (b.parents[1:] < np.arange(1, 77)).all()
    assert 1.4 < b.height < 2.0, b.height
    I = np.tile(np.eye(3), (1, 76, 1, 1))
    X = b.points(np.eye(3)[None], I, np.zeros((1, 3)))[0]
    up, left, fwd = (X - X[P.HIP]) @ b.up, (X - X[P.HIP]) @ b.left, (X - X[P.HIP]) @ b.fwd
    # 零姿勢：站直、雙手打開
    assert up[P.HEAD] > up[P.NOSE] > up[P.NECK] > up[P.L_SHO] - 0.1 and up[P.NECK] > 0.3
    assert up[P.L_KNEE] < -0.25 and up[P.L_ANK] < up[P.L_KNEE] - 0.25
    for l, r in ((P.L_SHO, P.R_SHO), (P.L_ELB, P.R_ELB), (P.L_WRI, P.R_WRI), (P.L_HIP, P.R_HIP), (P.L_ANK, P.R_ANK),
                 (P.L_EAR, P.R_EAR), (P.L_EYE, P.R_EYE), (P.L_HEEL, P.R_HEEL), (P.L_SMALLTOE, P.R_SMALLTOE)):
        assert left[l] > 0 > left[r] and abs(left[l] + left[r]) < 0.03, (l, left[l], left[r])
        assert abs(up[l] - up[r]) < 0.02
    assert left[P.L_WRI] > left[P.L_ELB] > left[P.L_SHO] > 0.1
    assert fwd[P.NOSE] > fwd[P.L_EYE] > fwd[P.L_EAR] and fwd[P.NOSE] - fwd[P.L_EAR] > 0.08
    assert abs(left[P.L_EAR] - left[P.R_EAR]) > 0.12
    for big, small, heel, ank in ((P.L_BIGTOE, P.L_SMALLTOE, P.L_HEEL, P.L_ANK), (P.R_BIGTOE, P.R_SMALLTOE, P.R_HEEL, P.R_ANK)):
        assert fwd[big] > fwd[small] > fwd[ank] > fwd[heel], (fwd[big], fwd[small], fwd[ank], fwd[heel])
        assert up[heel] < up[ank] - 0.03 and abs(up[heel] - up[big]) < 0.03       # 腳跟和腳尖在同一個平面（腳底）
        assert abs(left[small]) > abs(left[big])                                   # 小趾在外側
    # 整個人轉 90°、放大 1.1 倍：每一點都跟著轉、距離放大
    Ry = rotation(b.up, 90)
    X2 = b.points(Ry[None], I, np.array([[1.0, 0, 0]]), 1.1)[0]
    assert np.allclose(X2 - X2[P.HIP], 1.1 * (X - X[P.HIP]) @ Ry.T, atol=1e-9)
    # 低頭 40°：鼻子往下往前，耳朵跟著頭轉；腳不動
    Rb = I.copy()
    Rb[0, GX.HEAD_J - 1] = rotation(b.left, 40)
    X3 = b.points(np.eye(3)[None], Rb, np.zeros((1, 3)))[0]
    assert (X3[P.NOSE] - X[P.NOSE]) @ b.up < -0.03 and np.allclose(X3[P.L_HEEL], X[P.L_HEEL])
    assert abs(np.linalg.norm(X3[P.NOSE] - X3[P.L_EAR]) - np.linalg.norm(X[P.NOSE] - X[P.L_EAR])) < 1e-9


REAL_DOWNLOAD = GX.download
REAL_BROKEN = GX.broken


def unavailable(progress=None):
    raise GX.Unavailable('test')


class FakeNet:
    """回傳「零姿勢、等速往前走」的時序模型輸出。"""

    def __init__(self, R_c, R_gv, vel):
        self.R_c, self.R_gv, self.vel = R_c, R_gv, vel

    def run(self, obs, cliffs, feats, pos):
        L = len(obs)
        self.pos = pos
        x = np.zeros((L, 655), np.float32)
        x[:, GX.S_POSE] = np.tile([1, 0, 0, 0, 1, 0], 76)
        x[:, GX.S_RC] = self.R_c[:2].reshape(6)
        x[:, GX.S_RGV] = self.R_gv[:2].reshape(6)
        x[:, GX.S_VEL] = self.vel
        return x, np.tile(np.float32([0.9, 0, 0]), (L, 1))


def test_solve_and_scale():
    b = body()
    if b is None:
        return
    L, size = 60, (1280, 720)
    K = GX.intrinsics(*size)
    boxes = np.tile(np.float32([640, 360, 600]), (L, 1))
    # 相機座標（x 右、y 下、z 前）裡站直、面向鏡頭：人的左邊在畫面右邊 (+x)、上 → -y、前 → -z
    A = np.stack([b.left, b.up, b.fwd])                  # 零姿勢座標 → (左, 上, 前) 的分量
    R_c = np.diag([1.0, -1.0, -1.0]) @ A
    R_gv = GX.FLIP_Z @ GX.yaw(np.radians(70)) @ A        # 模型的世界座標和相機差一個任意的水平角：求解時要對齊回來
    net = FakeNet(R_c, R_gv, np.array([0, 0, 0.01]) @ A)  # 每 1/30 秒往「前」走 1cm
    d = GX.decode(net.run(np.zeros((L, 77, 3)), None, None, None)[0])
    cam = np.tile(np.float32([0.9, 0, 0]), (L, 1))
    true = 1.15                                          # 這個人比平均體型大 15%
    J = GX.fk(b.rest * true, b.parents, d['R_c'], d['R_body'], GX.camera_transl(cam, boxes, K))
    kp = np.concatenate([K[0, 0] * J[..., :2] / J[..., 2:3] + K[:2, 2], np.ones((L, 77, 1))], -1)
    assert abs(GX.body_scale(b, d, cam, kp, boxes, K) - true) < 0.01
    assert GX.body_scale(b, d, cam, kp * [1, 1, 0], boxes, K) == 1.0       # 沒有可信的關鍵點
    X, scale = GX.solve(net, b, kp, boxes, np.zeros((L, 1024), np.float32), size, fps=30.0)
    assert abs(scale - true) < 0.01 and X.shape == (L, 26, 3)
    x = X[0]
    assert x[P.HEAD, 1] > x[P.NECK, 1] > x[P.HIP, 1] > x[P.L_KNEE, 1] > x[P.L_ANK, 1]        # y 朝上
    assert x[P.NOSE, 2] > x[P.L_EAR, 2]                 # 面向鏡頭（+z 朝相機）
    assert x[P.L_SHO, 0] > 0.1 > -0.1 > x[P.R_SHO, 0]     # 面向鏡頭：人的左邊在畫面右邊（+x）
    assert abs((x[P.HEAD, 1] - x[P.L_HEEL, 1]) - b.height * true) < 0.08
    step = X[1:, P.HIP] - X[:-1, P.HIP]
    assert np.allclose(np.linalg.norm(step, axis=1), 0.01, atol=1e-6) and abs(step[:, 1]).max() < 1e-9
    assert step[0, 2] > 0.009                           # 朝著臉的方向（鏡頭）走
    # 60fps 的影片每兩幀取一幀之外，其他幀率：時間位置與每幀位移要照比例
    X25, _ = GX.solve(net, b, kp, boxes, np.zeros((L, 1024), np.float32), size, fps=25.0)
    assert np.allclose(net.pos, np.arange(L) * 1.2)
    assert np.allclose(np.linalg.norm(X25[1:, P.HIP] - X25[:-1, P.HIP], axis=1), 0.012, atol=1e-6)


def test_engine_export():
    """整個匯出流程（假的模型）：60fps 的影片每兩幀取一幀、沒有人的幀、四幀一批、寫出 .anim。"""
    b = body()
    if b is None:
        return
    import tempfile
    import time
    import cv2
    tmp = tempfile.mkdtemp()
    import paths
    old_base = paths.base_dir
    paths.base_dir = lambda: tmp
    try:
        import engine as E
        W, H, N = 640, 360, 150                          # 60fps × 2.5 秒 → 75 幀
        path = os.path.join(tmp, 'clip.mp4')
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (W, H))
        for i in range(N):
            img = np.full((H, W, 3), 60, np.uint8)
            if not 40 <= i < 52:                         # 這幾幀沒有人（畫面全暗）
                img[100:300, 280:360] = 200
            vw.write(img)
        vw.release()

        class Est:
            def __init__(self, *a, **k):
                self.bbox, self.freq, self.tta, self.precision, self.device, self.ms = None, 8, False, 'fp32', 'cpu', 1.0

            def __call__(self, img):
                if img[200, 320].mean() < 120:
                    return None, None, None
                k = np.tile([320.0, 200], (P.NUM_JOINTS, 1))
                k[P.HEAD], k[P.L_HEEL], k[P.L_WRI], k[P.R_WRI] = [320, 100], [320, 300], [360, 200], [280, 200]
                return k, np.zeros(P.NUM_JOINTS), np.full(P.NUM_JOINTS, 0.9)
        calls = dict(feat=[], kp=[])

        def feat(crops, cliffs):
            calls['feat'].append(len(crops))
            assert crops[0].shape == (3, 256, 256) and len(cliffs) == len(crops)
            return np.ones((len(crops), 1024), np.float32)

        def kp2d(crops, boxes):
            calls['kp'].append(len(crops))
            assert crops[0].shape == (3, 256, 192) and abs(boxes[0, 0] - 320) < 2 and abs(boxes[0, 1] - 200) < 2
            return np.zeros((len(crops), 77, 3), np.float32)
        A = np.stack([b.left, b.up, b.fwd])
        net = FakeNet(np.diag([1.0, -1.0, -1.0]) @ A, GX.FLIP_Z @ A, np.array([0, 0, 0.005]) @ A)
        saved = E.Pose3DEstimator, E.models.pose3d_ready, E.models.ensure_pose3d, E.P.resolve_device, E.Engine._gemx
        E.Pose3DEstimator = Est
        E.models.pose3d_ready = lambda: True
        E.models.ensure_pose3d = lambda progress=None: {}
        E.P.resolve_device = lambda pref: 'cpu'
        try:
            eng = E.Engine(preload=False)
            eng.update_config({'video': {'method': 'gemx'}, 'pose': {'device': 'cpu'}})
            GX.download = unavailable                    # 測試不連網路：下載來源當成還不能公開下載
            eng.start_export(path)                       # 沒有模型、也下載不到：要說清楚缺哪些檔、放哪裡
            t = time.time()
            while eng.get_state(False)['mode'] != 'idle' and time.time() - t < 20:
                time.sleep(0.05)
            err = eng.get_state(False)['error']
            assert err and 'ERR_GEMX_MISSING' in str(err) and 'gemx_motion.onnx' in str(err), err
            E.Engine._gemx = lambda self: (feat, kp2d, net, b)
            eng.start_export(path)
            t = time.time()
            while eng.get_state(False)['mode'] != 'idle' and time.time() - t < 120:
                time.sleep(0.1)
            st = eng.get_state(False)
            assert st['error'] is None, st['error']
            job = st['job']
            assert job['frames'] == N // 2, job             # 60fps → 30fps
            assert os.path.exists(job['export']) and os.path.exists(job['anim']) and os.path.getsize(job['anim']) > 1000
            n_person = N // 2 - 6                            # 12 幀（60fps）沒有人 = 6 幀（30fps）
            assert sum(calls['feat']) == sum(calls['kp']) == n_person, calls
            assert max(calls['feat']) <= 4
            z = np.load(job['export'][:-4] + '_joints.npz')
            assert abs(float(z['fps']) - 30.0) < 0.5 and z['joints'].shape == (N // 2, 26, 3)
            assert int(z['valid'][:, P.HIP].sum()) == n_person
            assert z['raw'].shape == z['joints'].shape and z['cue'].shape == z['cue_used'].shape == (N // 2, 2)     # 回報問題用的記錄
            assert z['to_cam'].shape == (N // 2, 3) and z['cue_local'].shape == (N // 2, 2)
            J = z['joints']
            assert J[:, :, 1].min() > -1e-6 and 1.3 < J[0, P.HEAD, 1] < 2.1      # 站在地上、身高合理
            assert J[-1, P.HIP, 2] - J[0, P.HIP, 2] > 0.2                        # 往鏡頭走了 0.005 × 74 ≈ 0.37 m
            # 第二次讀影片時後面讀不出來（損壞的檔案）：沒算到的幀不能當成有人
            Real, opened = E.V.VideoFile, []

            class Short(Real):
                def __init__(self, *a, **k):
                    super().__init__(*a, **k)
                    opened.append(self)

                def read_next(self):
                    r = super().read_next()
                    return None if r is not None and len(opened) >= 2 and r[2] >= 100 else r
            E.V.VideoFile = Short
            try:
                calls['feat'].clear()
                eng.start_export(path)
                t = time.time()
                while eng.get_state(False)['mode'] != 'idle' and time.time() - t < 120:
                    time.sleep(0.1)
            finally:
                E.V.VideoFile = Real
            st = eng.get_state(False)
            assert st['error'] is None and len(opened) == 2, st['error']
            z = np.load(st['job']['export'][:-4] + '_joints.npz')
            assert sum(calls['feat']) == 50 - 6 and int(z['valid'][:, P.HIP].sum()) == 50 - 6, (calls['feat'], z['valid'][:, P.HIP].sum())
            assert not z['valid'][50:, P.HIP].any() and z['joints'].shape == (N // 2, 26, 3)
            eng.stop()
        finally:
            E.Pose3DEstimator, E.models.pose3d_ready, E.models.ensure_pose3d, E.P.resolve_device, E.Engine._gemx = saved
            GX.download = REAL_DOWNLOAD
    finally:
        paths.base_dir = old_base


def test_engine_images():
    """圖片模式選 GEM-X：每張圖一幀、站在地板上；沒有人的圖略過；沒放模型時說明缺哪些檔。"""
    b = body()
    if b is None:
        return
    import tempfile
    import time
    import cv2
    tmp = tempfile.mkdtemp()
    import paths
    old_base = paths.base_dir
    paths.base_dir = lambda: tmp
    try:
        import engine as E
        person = np.full((360, 640, 3), 60, np.uint8)
        person[100:300, 280:360] = 200
        files = [os.path.join(tmp, 'a.png'), os.path.join(tmp, 'empty.png')]
        cv2.imwrite(files[0], person)
        cv2.imwrite(files[1], np.full((360, 640, 3), 60, np.uint8))

        class Est:
            def __init__(self, *a, **k):
                self.bbox, self.freq, self.tta, self.precision, self.device, self.ms, self.n = None, 8, False, 'fp32', 'cpu', 1.0, 0
                self.cue = {'Left': 0, 'Right': 0}

            def __call__(self, img):
                if img[200, 320].mean() < 120:
                    return None, None, None
                k = np.tile([320.0, 200], (P.NUM_JOINTS, 1))
                k[P.HEAD], k[P.L_HEEL], k[P.L_WRI], k[P.R_WRI] = [320, 100], [320, 300], [360, 200], [280, 200]
                return k, np.zeros(P.NUM_JOINTS), np.full(P.NUM_JOINTS, 0.9)
        A = np.stack([b.left, b.up, b.fwd])
        net = FakeNet(np.diag([1.0, -1.0, -1.0]) @ A, GX.FLIP_Z @ GX.yaw(1.0) @ A, np.zeros(3))
        seen = []

        def feat(crops, cliffs):
            seen.append(len(crops))
            return np.ones((len(crops), 1024), np.float32)
        gx = (feat, lambda crops, boxes: np.zeros((len(crops), 77, 3), np.float32), net, b)
        X = E.Engine._gemx_image(gx, person, *Est()(person)[::2])
        assert X.shape == (26, 3) and abs(X[:, 1].min()) < 1e-9 and np.allclose(X[P.HIP, [0, 2]], 0)
        assert 1.5 < X[P.HEAD, 1] < 1.8 and X[P.NOSE, 2] > X[P.L_EAR, 2] and X[P.L_SHO, 0] > 0 > X[P.R_SHO, 0]
        saved = E.Pose3DEstimator, E.models.pose3d_ready, E.models.ensure_pose3d, E.P.resolve_device, E.Engine._gemx
        E.Pose3DEstimator = Est
        E.models.pose3d_ready = lambda: True
        E.models.ensure_pose3d = lambda progress=None: {}
        E.P.resolve_device = lambda pref: 'cpu'
        try:
            eng = E.Engine(preload=False)
            eng.update_config({'video': {'method': 'gemx'}, 'pose': {'device': 'cpu'}})

            def run():
                eng.start_image_export(files)
                t = time.time()
                while eng.get_state(False)['mode'] != 'idle' and time.time() - t < 60:
                    time.sleep(0.05)
                return eng.get_state(False)
            GX.download = unavailable
            st = run()
            assert st['error'] and 'ERR_GEMX_MISSING' in str(st['error']), st['error']
            E.Engine._gemx = lambda self: gx
            seen.clear()
            st = run()
            assert st['error'] is None, st['error']
            assert st['job']['saved'] == 1 and st['job']['skipped'] == ['empty.png'] and seen == [1], (st['job'], seen)
            assert os.path.exists(st['job']['export']) and os.path.exists(st['job']['unity'])
            eng.update_config({'video': {'method': 'standard'}})
            seen.clear()
            st = run()                                  # 標準方式不會用到 GEM-X
            assert st['error'] is None and seen == [], (st['error'], seen)
        finally:
            E.Pose3DEstimator, E.models.pose3d_ready, E.models.ensure_pose3d, E.P.resolve_device, E.Engine._gemx = saved
            GX.download = REAL_DOWNLOAD
    finally:
        paths.base_dir = old_base


def test_view_direction():
    """相機俯視時「指向相機」的方向不是水平的：頭和手的前後要沿著真正的視線移動。"""
    b = body()
    if b is None:
        return
    L, size = 4, (1280, 720)
    boxes = np.tile(np.float32([640, 360, 600]), (L, 1))
    kp = np.zeros((L, 77, 3))
    A = np.stack([b.left, b.up, b.fwd])
    D = np.diag([1.0, -1.0, -1.0])
    level = FakeNet(D @ A, GX.FLIP_Z @ GX.yaw(0.7) @ A, np.zeros(3))
    X, _, v = GX.solve(level, b, kp, boxes, np.zeros((L, 1024), np.float32), size, view=True)
    assert np.allclose(v, [0, 0, 1], atol=1e-6)
    th = np.radians(25)                                 # 相機在上面往下看 25 度：畫面裡人往後仰
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    for sign in (1, -1):
        tilt = FakeNet(D @ (Rx if sign > 0 else Rx.T) @ A, GX.FLIP_Z @ GX.yaw(0.7) @ A, np.zeros(3))
        X, _, v = GX.solve(tilt, b, kp, boxes, np.zeros((L, 1024), np.float32), size, view=True)
        up_in_cam = (D @ GX.decode(tilt.run(kp, None, None, None)[0])['R_c'][0]) @ b.up
        assert np.allclose(np.linalg.norm(v, axis=1), 1) and abs(v[0, 0]) < 1e-6 and v[0, 2] > 0.8
        assert abs(v[0, 1] - up_in_cam[2]) < 1e-6 and abs(abs(v[0, 1]) - np.sin(th)) < 1e-6, v[0]
        assert X[0, P.HEAD, 1] > X[0, P.HIP, 1] + 0.4   # 世界座標裡人還是站直的
        import engine as E
        far = E._far(X[0], v[0])
        head = 0.5 * (X[0, P.L_EAR] + X[0, P.R_EAR])
        assert np.allclose((far - head) / np.linalg.norm(far - head), v[0])
    # 坐在地上往後躺、身體斜一邊（身體的「前方」幾乎朝上），相機俯視 15 度：只看「前方」的水平方向會轉錯，要用整個旋轉對齊
    ph = np.radians(15)
    Rp = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    lean = np.radians(-80)
    Rl = np.array([[1, 0, 0], [0, np.cos(lean), -np.sin(lean)], [0, np.sin(lean), np.cos(lean)]])
    pose = GX.yaw(np.radians(40)) @ Rl @ rotation([0, 0, 1], 25)          # 世界座標（y 上）裡的姿勢
    lying = FakeNet(D @ Rp @ pose @ A, GX.FLIP_Z @ GX.yaw(2.1) @ pose @ A, np.zeros(3))
    X, _, v = GX.solve(lying, b, kp, boxes, np.zeros((L, 1024), np.float32), size, view=True)
    assert abs(v[0, 0]) < 1e-6 and abs(abs(v[0, 1]) - np.sin(ph)) < 1e-6 and v[0, 2] > 0.9, v[0]
    want = (pose @ A) @ (b.rest[GX.L_ARM] - b.rest[GX.R_ARM])            # 兩肩的方向：和世界座標裡擺的姿勢一樣
    got = X[0, P.L_SHO] - X[0, P.R_SHO]
    assert np.allclose(got / np.linalg.norm(got), want / np.linalg.norm(want), atol=1e-6)
    d0 = GX.decode(lying.run(kp, None, None, None)[0])
    fc, fw = (D @ d0['R_c'][0]) @ b.fwd, GX.world_static(d0['R_gv'], d0['vel'], np.ones(L))[0][0] @ b.fwd
    old = GX.yaw(np.arctan2(fc[0], fc[2]) - np.arctan2(fw[0], fw[2]))    # 舊的對齊方式：這個姿勢會差很多
    new = GX.yaw(-2.1)
    assert np.degrees(np.arccos(np.clip((np.trace(old.T @ new) - 1) / 2, -1, 1))) > 20
    print('  視線方向：相機俯視／仰視時沿著真正的視線；坐著、躺著時朝向也對得上')


def test_release():
    """GEM-X 的模型：匯出後過一陣子才釋放（接著匯出不用重新載入）、開始即時追蹤馬上釋放、顯示卡出錯後重新載入。"""
    import tempfile
    import time
    tmp = tempfile.mkdtemp()
    import paths
    old_base = paths.base_dir
    paths.base_dir = lambda: tmp
    import engine as E
    loads = []

    class Fake:
        device = 'cpu'

        def __init__(self, *a):
            loads.append(type(self).__name__)

    class Est:
        device = 'cpu'
    saved = GX.missing, GX.Body, GX.Features, GX.Keypoints, GX.Net, E.GEMX_KEEP_S
    GX.missing = lambda: []
    GX.Body = GX.Features = GX.Keypoints = GX.Net = Fake
    try:
        eng = E.Engine(preload=False)
        eng._est = Est()
        a = eng._gemx()
        assert len(loads) == 4 and eng._gemx() is a and len(loads) == 4
        eng._gemx_free(0.3)                             # 排定釋放
        assert eng._gx is not None
        assert eng._gemx() is a                         # 又要用了：取消釋放
        time.sleep(0.6)
        assert eng._gx is not None and len(loads) == 4
        eng._gemx_free(0.1)
        time.sleep(0.4)
        assert eng._gx is None
        eng._gemx()
        assert len(loads) == 8
        E.GEMX_KEEP_S = 0.2

        def wait():
            t = time.time()
            while eng.get_state(False)['mode'] != 'idle' and time.time() - t < 10:
                time.sleep(0.02)
        eng._start('export', lambda: None)              # 匯出結束：先留著，過一陣子才釋放
        wait()
        assert eng._gx is not None
        time.sleep(0.5)
        assert eng._gx is None
        eng._gemx()

        def gpu():
            raise P.GPUError('ERR_GPU', 'DML device removed')
        eng._start('export', gpu)                       # 顯示卡出錯：馬上放掉
        wait()
        assert eng._gx is None and eng.get_state(False)['error'] == 'ERR_GPU'
        eng._gemx()
        eng._start('live', lambda: None)                # 即時追蹤：顯示記憶體留給遊戲
        wait()
        assert eng._gx is None
        eng.stop()
    finally:
        GX.missing, GX.Body, GX.Features, GX.Keypoints, GX.Net, E.GEMX_KEEP_S = saved
        paths.base_dir = old_base
    print('  模型釋放：連續匯出不重新載入，閒置／即時追蹤／顯示卡出錯時釋放')


class _Release:
    """本機的假 Release：支援續傳（Range），可以讓某個檔案傳到一半就斷線、或回傳 404。"""

    def __init__(self, files):
        import hashlib
        import http.server
        import threading
        self.files = dict(files)
        self.files['SHA256SUMS'] = ''.join(f'{hashlib.sha256(v).hexdigest()}  {k}\n' for k, v in files.items()).encode()
        self.cut = {}                                   # 檔名 → 傳這麼多位元組就斷線（一次）
        self.missing = False
        self.ranges = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.rsplit('/', 1)[-1]
                if outer.missing or name not in outer.files:
                    self.send_error(404)
                    return
                body = outer.files[name]
                start = 0
                rng = self.headers.get('Range')
                if rng:
                    start = int(rng.split('=')[1].split('-')[0])
                    outer.ranges.append((name, start))
                    if start >= len(body):
                        self.send_error(416)
                        return
                self.send_response(206 if rng else 200)
                self.send_header('Content-Length', str(len(body) - start))
                self.end_headers()
                stop = outer.cut.pop(name, None)
                self.wfile.write(body[start:] if stop is None else body[start:start + stop])
                if stop is not None:
                    self.wfile.flush()
                    self.connection.close()
        self.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.srv.server_address[1]}/gemx-models/'
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def test_download():
    """自動下載：核對檢查碼、斷線後接著下載、按停止會留下 .part、來源還不能下載時丟 Unavailable、內容不對不會留下。"""
    import tempfile
    tmp = tempfile.mkdtemp()
    import paths
    old_base, old_url, old_sizes, old_retry = paths.base_dir, GX.BASE_URL, GX.SIZES, (GX.RETRIES, GX.RETRY_WAIT)
    paths.base_dir = lambda: tmp
    rng = np.random.default_rng(0)
    files = {f: rng.integers(0, 255, 200_000 + 50_000 * i, dtype=np.uint8).tobytes() for i, f in enumerate(GX.FILES)}
    rel = _Release(files)
    GX.BASE_URL, GX.SIZES = rel.url, {f: len(v) for f, v in files.items()}
    GX.RETRIES, GX.RETRY_WAIT = 0, 0.0                  # 先看「不重試」時的行為
    d = GX.folder()
    try:
        assert set(GX.missing()) == set(GX.FILES)
        rel.missing = True                              # repo 還沒公開：404
        try:
            GX.download()
            raise AssertionError('should be unavailable')
        except GX.Unavailable:
            pass
        rel.missing = False
        big = 'gemx_kp2d_fp16.onnx.data'
        rel.cut[big] = 60_000                           # 這個檔傳到一半斷線
        seen = []
        try:
            GX.download(lambda done, total: seen.append((done, total)))
            raise AssertionError('should fail')
        except GX.DownloadError as e:
            assert e.file == big and e.why == 'incomplete' and str(e) == f'ERR_MODEL_DOWNLOAD:{big}|incomplete', (e.file, e.why)
        later = {f for f in GX.FILES if GX.SIZES[f] >= GX.SIZES[big]}       # 小的先下載：比它小的都已經好了
        assert set(GX.missing()) == later, GX.missing()
        assert os.path.getsize(os.path.join(d, big + '.part')) == 60_000
        assert seen and all(a <= b for a, b in seen) and seen[-1][1] == sum(GX.SIZES.values())
        rel.ranges.clear()
        GX.download()                                   # 再按一次：從斷掉的地方接著下載
        assert (big, 60_000) in rel.ranges and GX.missing() == []
        for f, v in files.items():
            assert open(os.path.join(d, f), 'rb').read() == v, f
        assert not [x for x in os.listdir(d) if x.endswith('.part')]
        # 連線中途斷掉：自己接著下載（不用使用者再按一次）；一直斷就放棄
        os.remove(os.path.join(d, big))
        GX.RETRIES = 3
        rel.cut[big] = 80_000
        rel.ranges.clear()
        seen = []
        GX.download(lambda done, total: seen.append(done))
        assert (big, 80_000) in rel.ranges and GX.missing() == []
        assert open(os.path.join(d, big), 'rb').read() == files[big]
        assert seen == sorted(seen), '重試時進度不能倒退'
        os.remove(os.path.join(d, big))
        GX.RETRIES = 1

        class Flaky(dict):                              # 每次都傳一點就斷
            def pop(self, k, default=None):
                return 10_000 if k == big else default
        rel.cut = Flaky()
        try:
            GX.download()
            raise AssertionError('should give up')
        except GX.DownloadError as e:
            assert e.why == 'incomplete'
        assert os.path.getsize(os.path.join(d, big + '.part')) == 20_000      # 兩次各 10000：都留著
        rel.cut = {}
        GX.download()
        assert GX.missing() == []
        GX.RETRIES = 0
        # 按停止：例外原樣傳出，.part 留著
        os.remove(os.path.join(d, big))

        class Stop(Exception):
            pass

        def stop(done, total):
            raise Stop()
        try:
            GX.download(stop)
            raise AssertionError('should stop')
        except Stop:
            pass
        assert os.path.exists(os.path.join(d, big + '.part')) and GX.missing() == [big]
        # .part 的內容不對（例如檔案換過版本）：重新下載一次就好
        open(os.path.join(d, big + '.part'), 'wb').write(b'x' * 1000)
        GX.download()
        assert open(os.path.join(d, big), 'rb').read() == files[big]
        # 伺服器上的檔案和檢查碼對不上：不會留下錯的檔案
        os.remove(os.path.join(d, 'gemx_skeleton.npz'))
        rel.files['gemx_skeleton.npz'] = b'not the right file'
        try:
            GX.download()
            raise AssertionError('should fail')
        except GX.DownloadError as e:
            assert e.why == 'invalid'
        assert GX.missing() == ['gemx_skeleton.npz'] and not os.path.exists(os.path.join(d, 'gemx_skeleton.npz.part'))
        GX.BASE_URL = 'http://127.0.0.1:9/'             # 連不上
        try:
            GX.download()
            raise AssertionError('should fail')
        except GX.DownloadError as e:
            assert e.why == 'network'
    finally:
        rel.close()
        paths.base_dir, GX.BASE_URL, GX.SIZES = old_base, old_url, old_sizes
        GX.RETRIES, GX.RETRY_WAIT = old_retry


def test_engine_downloads_models():
    """按匯出時缺模型 → 自動下載（顯示進度）→ 接著載入；下載失敗時顯示原因。"""
    import tempfile
    import time
    tmp = tempfile.mkdtemp()
    import paths
    old_base = paths.base_dir
    paths.base_dir = lambda: tmp
    try:
        import engine as E
        eng = E.Engine(preload=False)
        phases = []

        def fake(progress):
            progress(50, 100)
            phases.append((eng.state['phase'], eng.state['model']['status']))
            for f in GX.FILES:
                os.makedirs(GX.folder(), exist_ok=True)
                open(os.path.join(GX.folder(), f), 'wb').write(b'0')
        GX.download = fake
        GX.broken = lambda: []                          # 這一段不核對檔案（不連網路）
        try:
            eng._gemx()
            raise AssertionError('fake files cannot load')
        except RuntimeError as e:
            assert str(e) == 'ERR_GEMX_MODEL', e         # 下載完了才會走到載入（假的檔案載入失敗）
        assert phases == [('downloading_gemx', 'downloading')] and eng.state['model']['status'] == 'ready'
        for f in GX.FILES:
            os.remove(os.path.join(GX.folder(), f))

        def broken(progress):
            raise GX.DownloadError('gemx_motion.onnx', 'network', 'boom')
        GX.download = broken
        try:
            eng._gemx()
            raise AssertionError('should fail')
        except GX.DownloadError as e:
            assert str(e) == 'ERR_MODEL_DOWNLOAD:gemx_motion.onnx|network'
        del time
        # 檔案壞掉（複製到一半、磁碟錯誤）：載入失敗時核對檢查碼，壞的刪掉重新下載，再載入一次
        GX.download, GX.broken = REAL_DOWNLOAD, REAL_BROKEN
        rng = np.random.default_rng(1)
        files = {f: rng.integers(0, 255, 5000, dtype=np.uint8).tobytes() for f in GX.FILES}
        files['gemx_skeleton.npz'] = open(os.path.join(HERE, 'data', 'gemx_skeleton.npz'), 'rb').read()
        rel = _Release(files)
        old = GX.BASE_URL, GX.SIZES
        GX.BASE_URL, GX.SIZES = rel.url, {f: len(v) for f, v in files.items()}
        try:
            os.makedirs(GX.folder(), exist_ok=True)
            for f, v in files.items():
                open(os.path.join(GX.folder(), f), 'wb').write(b'damaged' if f == 'gemx_skeleton.npz' else v)
            try:
                eng._gemx()
                raise AssertionError('random bytes are not a model')
            except RuntimeError as e:
                assert str(e) == 'ERR_GEMX_MODEL', e     # 骨架修好了，但其他（假的）模型還是載入不了：不會一直重試
            assert open(os.path.join(GX.folder(), 'gemx_skeleton.npz'), 'rb').read() == files['gemx_skeleton.npz']
            assert GX.missing() == [] and GX.broken() == []
            GX.BASE_URL = 'http://127.0.0.1:9/'         # 連不上網路：不刪任何東西
            open(os.path.join(GX.folder(), 'gemx_skeleton.npz'), 'wb').write(b'damaged')
            assert GX.broken() == [] and GX.missing() == []
        finally:
            rel.close()
            GX.BASE_URL, GX.SIZES = old
    finally:
        GX.download, GX.broken = REAL_DOWNLOAD, REAL_BROKEN
        paths.base_dir = old_base


def test_old_setting_migrates():
    """設定檔裡舊的 GVHMR 選項 → GEM-X。"""
    import json
    import tempfile
    tmp = tempfile.mkdtemp()
    import paths
    old_base = paths.base_dir
    paths.base_dir = lambda: tmp
    try:
        import engine as E
        with open(os.path.join(tmp, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({'video': {'method': 'gvhmr'}}, f)
        eng = E.Engine(preload=False)
        assert eng.cfg['video']['method'] == 'gemx'
    finally:
        paths.base_dir = old_base


def test_missing_files():
    assert set(GX.missing()) <= set(GX.FILES)
    assert GX.paths()['gemx_motion.onnx'].endswith(os.path.join('models', 'gemx', 'gemx_motion.onnx'))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
