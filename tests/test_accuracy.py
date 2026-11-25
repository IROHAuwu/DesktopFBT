"""精度：模擬的姿勢模型（tests/sim.py）跑和引擎同一份即時計算（track.Tracker），和已知的 3D 動作比較。

數字是模擬出來的（畫面誤差小、深度誤差大而且前後連續），不是真實影片的精度；用來確認每一步都有幫助、之後改動不退步。
執行：python tests/test_accuracy.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import bvh  # noqa: E402
import desktop as DK  # noqa: E402
import pose as P  # noqa: E402
import precise as PR  # noqa: E402
import sim  # noqa: E402
import track  # noqa: E402
from lift import SOLE, Lifter, Pose3DEstimator  # noqa: E402
from skeleton import Skeleton  # noqa: E402

LIMBS = ('LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm', 'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg')
HEIGHT, THR = 1.70, 0.4
HEAD_MUSCLES = {f'm{i}' for i in range(9, 15)}


def rot_angle(A, B):
    R = A @ B.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def run(motion, seconds=30.0, fps=30.0, seed=1, rays=True, smoothness=0.85, cam=None, **noise):
    """回傳 dict：dir 骨頭方向誤差（度）、rot 手腳旋轉誤差（度，含扭轉）、mus 肌肉值誤差、jit 肌肉值的抖動、sig 量到的誤差大小。"""
    cam, seq = sim.sequence(motion, seconds=seconds, seed=seed, fps=fps, cam=cam, **noise)
    trk = track.Tracker(cam.size, cam.hfov, HEIGHT, THR, smoothness=smoothness)
    if not rays:
        full = trk.sk.apply
        trk.sk.apply = lambda X, ok, conf=None, **kw: full(X, ok, conf, **{k: v for k, v in kw.items() if k != 'cam'})
    desk, ref = DK.DesktopPose(), DK.DesktopPose()
    ok1 = np.ones(P.NUM_JOINTS, bool)
    truth = {round(t * fps): cam.world(J) for t, J, _ in seq}
    D, R, Mu, Hd, prm_seq = [], [], [], [], []
    for t, _, (k, z, s) in seq:
        r = trk(t, k, z, s)
        if r is None or r[0] < 6.0:
            continue
        T = truth[round(r[0] * fps)]
        D.append(sim.bone_angles(r[1], T).mean())
        G, GT = bvh.global_rotations(r[1]), bvh.global_rotations(T)
        R.append(np.mean([rot_angle(G[b], GT[b]) for b in LIMBS]))
        prm, tp = desk(r[1], r[2], r[3], r[0]), ref(T, ok1, {}, r[0])
        mus = [n for n in prm if n[0] == 'm']
        # 頭（脖子、頭的肌肉 9–14）另外算：模擬資料裡「脖子到兩耳中點」跟著軀幹傾斜、臉卻一直是正的，
        # 所以用肌肉值比不公平；改比「臉朝哪裡」（兩耳中點 → 鼻子的方向）
        body = [n for n in mus if n not in HEAD_MUSCLES]
        Mu.append(np.mean([abs(prm[n] - tp[n]) for n in body]))
        face = [J[P.NOSE] - 0.5 * (J[P.L_EAR] + J[P.R_EAR]) for J in (r[1], T)]
        Hd.append(np.degrees(np.arccos(np.clip(face[0] @ face[1] / np.linalg.norm(face[0]) / np.linalg.norm(face[1]), -1, 1))))
        prm_seq.append([prm[n] for n in mus])
    jit = float(np.mean(np.abs(np.diff(np.array(prm_seq), 2, axis=0))) * fps * fps)
    return dict(dir=float(np.mean(D)), rot=float(np.mean(R)), mus=float(np.mean(Mu)), jit=jit, head=float(np.mean(Hd)),
                sig=(trk.sk.sig_L, trk.sk.sig_z))


# v1.6.6（改之前）在同一組模擬資料上的數字（三個亂數種子的平均）：方向、旋轉（度），肌肉值誤差，抖動
BEFORE = {'walk': (9.2, 22.4, 0.084, 15.8), 'turn': (9.7, 22.4, 0.090, 16.5), 'bend': (9.5, 21.5, 0.091, 16.7)}


def test_live_accuracy():
    for name, mo in (('walk', sim.motion_walk), ('turn', sim.motion_turn), ('bend', sim.motion_bend)):
        r = run(mo)
        b = BEFORE[name]
        print(f'  {name}：骨頭方向 {b[0]:.1f}° → {r["dir"]:.1f}°，手腳旋轉 {b[1]:.1f}° → {r["rot"]:.1f}°，'
              f'肌肉值 {b[2]:.3f} → {r["mus"]:.3f}（不含頭），臉的方向誤差 {r["head"]:.1f}°，抖動 {b[3]:.1f} → {r["jit"]:.1f}')
        assert r['dir'] < 8.6 and r['rot'] < 17.0 and r['mus'] < 0.080 and r['jit'] < 8.0 and r['head'] < 19.0, (name, r)


def test_rays_never_hurt():
    """沿視線修深度：誤差的大小是自己量的，所以不管模型是哪一種誤差比較大，都不能比不修還差。"""
    cases = (('一般', {}), ('深度誤差大', dict(zs=0.09, zw=0.03)), ('深度被壓扁', dict(comp=0.7)),
             ('深度很準', dict(zs=0.012, zw=0.005, comp=0.97)), ('畫面誤差大', dict(px=0.016, px_w=0.008)))
    sig = {}
    for label, kw in cases:
        a, b = run(sim.motion_turn, seconds=22.0, rays=True, **kw), run(sim.motion_turn, seconds=22.0, rays=False, **kw)
        sig[label] = a['sig'][1]
        print(f'  {label}：骨頭方向 不修 {b["dir"]:.2f}° → 修 {a["dir"]:.2f}°（量到：畫面 {100 * a["sig"][0]:.1f} cm、'
              f'深度 {100 * a["sig"][1]:.1f} cm）')
        # 0.35°：深度的幾何上限（skeleton._limit）在「模型深度很準」的模擬資料上會多 0.2° 左右；
        # 真實畫面上模型的深度會錯 20–30 cm（模擬資料沒有這種誤差），那時候它讓站直的腿不會被算成彎的
        assert a['dir'] < b['dir'] + 0.35, (label, a, b)
    assert sig['深度誤差大'] > 2.0 * sig['深度很準']                # 量到的深度誤差真的跟著模型的誤差變


def test_floor_and_pitch_consistent():
    """相機的俯仰角和離地高度要一致：以前兩者會各用各的估計，人轉身時腳一度掉到地板下 0.9 公尺、腿被壓扁。"""
    cam, seq = sim.sequence(sim.motion_turn, seconds=30.0, seed=1)
    lf = Lifter(cam.size, cam.hfov, HEIGHT, THR)
    up_true = cam.R @ np.array([0.0, 1.0, 0.0])
    low, up_err, steps, last = [], [], [], None
    for t, _, (k, z, s) in seq:
        X, ok = lf(k, z, s, t)
        if X is None:
            continue
        p = np.degrees(lf.pitch)
        if last is not None and t > 4.0:
            steps.append(abs(p - last[1]) / (t - last[0]))
        last = (t, p)
        if t > 2.0:
            low.append(min(X[j, 1] for j in SOLE))
            up_err.append(np.degrees(np.arccos(np.clip(lf.R[1] @ up_true, -1, 1))))
    print(f'  最低的腳底點離地 {100 * min(low):+.0f} ～ {100 * max(low):+.0f} cm，「上」的誤差平均 {np.mean(up_err):.1f}°、'
          f'最大 {max(up_err):.1f}°，俯仰角最快每秒變 {max(steps):.1f}°')
    assert min(low) > -0.25 and max(low) < 0.25, (min(low), max(low))
    assert np.mean(up_err) < 2.0 and max(up_err) < 6.0
    assert max(steps) < 2.6                                   # 不會突然整個人傾斜


def test_skeleton_keeps_image_position():
    """沿視線修深度不能動到畫面上的位置（只有最後的收尾會微調）。"""
    cam, seq = sim.sequence(sim.motion_bend, seconds=6.0, seed=3)
    sk = Skeleton(HEIGHT)
    ok = np.ones(P.NUM_JOINTS, bool)
    c = np.array([0.0, cam.pos[1], 0.0])
    worst = 0.0
    for _, J, _ in seq:
        X = cam.world(J) + np.random.default_rng(0).normal(0, 0.03, (P.NUM_JOINTS, 3))
        Y = sk._rays(X, ok, c)
        u, v = X - c, Y - c
        cosang = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        worst = max(worst, float(np.degrees(np.arccos(np.clip(cosang[sim.BODY], -1, 1))).max()))
    assert worst < 0.02, worst                              # 脖子、骨盆中心跟著兩肩、兩髖的平均移動，差一點點


def _tpose():
    J = np.zeros((P.NUM_JOINTS, 3))
    J[P.HIP], J[P.NECK] = (0, 0.9, 0), (0, 1.4, 0)
    for sx, hip, kn, an, toe, small, heel, sho, elb, wri, ear in (
            (1, P.L_HIP, P.L_KNEE, P.L_ANK, P.L_BIGTOE, P.L_SMALLTOE, P.L_HEEL, P.L_SHO, P.L_ELB, P.L_WRI, P.L_EAR),
            (-1, P.R_HIP, P.R_KNEE, P.R_ANK, P.R_BIGTOE, P.R_SMALLTOE, P.R_HEEL, P.R_SHO, P.R_ELB, P.R_WRI, P.R_EAR)):
        J[hip], J[kn], J[an] = (0.09 * sx, 0.9, 0), (0.09 * sx, 0.48, 0), (0.09 * sx, 0.07, 0)
        J[toe], J[small], J[heel] = (0.09 * sx, 0.01, 0.137), (0.13 * sx, 0.01, 0.11), (0.09 * sx, 0.0, -0.05)
        J[sho], J[elb], J[wri] = (0.18 * sx, 1.4, 0), (0.48 * sx, 1.4, 0), (0.73 * sx, 1.4, 0)
        J[ear] = (0.075 * sx, 1.57, 0)
    J[P.NOSE] = (0, 1.55, 0.1)
    return J


def test_bend_axis():
    """手腳的扭轉：打直時量到的彎曲平面幾乎是亂的，要靠「不扭轉」的預設穩住。"""
    J = _tpose()
    G = bvh.global_rotations(J)
    assert all(G[n] is not None for n in bvh.NAMES)
    assert max(np.abs(G[n] - np.eye(3)).max() for n in bvh.NAMES if 'Foot' not in n and 'Toe' not in n) < 1e-9
    for sx, elb, wri in ((1, P.L_ELB, P.L_WRI), (-1, P.R_ELB, P.R_WRI)):       # 手臂垂在身體兩側、完全打直（最常見的姿勢）
        J[elb], J[wri] = (0.18 * sx, 1.10, 0), (0.18 * sx, 0.85, 0)
    base = bvh.global_rotations(J)
    assert base['LeftArm'] is not None and base['RightArm'] is not None       # 以前這裡算不出來，沿用上一幀的旋轉
    assert np.allclose(base['LeftArm'] @ np.array([0, -1.0, 0]), [-1, 0, 0], atol=1e-6)   # 手肘往前彎
    rng = np.random.default_rng(0)
    dev = []
    for _ in range(200):
        Gn = bvh.global_rotations(J + rng.normal(0, 0.005, J.shape))
        dev += [rot_angle(Gn[b], base[b]) for b in ('LeftArm', 'RightArm', 'LeftUpLeg', 'RightUpLeg')]
    print(f'  打直的手腳加 5 mm 雜訊：旋轉變動平均 {np.mean(dev):.1f}°、最大 {np.max(dev):.1f}°（以前手臂平均 94°）')
    assert np.mean(dev) < 5.0 and np.max(dev) < 18.0
    prev = None
    for t in np.linspace(0, 6, 120):                                              # 彎曲角度慢慢變：軸連續，不會在某個角度跳一下
        Ja = _tpose()
        b = 0.02 + 1.2 * abs(np.sin(t))
        Ja[P.L_WRI] = Ja[P.L_ELB] + 0.25 * np.array([np.cos(b), 0, np.sin(b)])
        Ja[P.L_ANK] = Ja[P.L_KNEE] + 0.41 * np.array([0, -np.cos(b), -np.sin(b)])
        Gt = bvh.global_rotations(Ja)
        if prev is not None:
            assert rot_angle(Gt['LeftArm'], prev['LeftArm']) < 4.0 and rot_angle(Gt['LeftUpLeg'], prev['LeftUpLeg']) < 4.0
        prev = Gt
    assert np.allclose(bvh.swing((1, 0, 0), (-1, 0, 0)) @ np.array([1.0, 0, 0]), [-1, 0, 0], atol=1e-9)


def test_smoother_no_lag():
    """去抖動以每一幀為中心（看前後）：抖動小很多，而且不會像一般的濾波一樣落後。"""
    rng = np.random.default_rng(2)
    fps, f = 30.0, 1.2
    ok, s = np.ones(P.NUM_JOINTS, bool), np.full(P.NUM_JOINTS, 0.85)
    sm = PR.Smoother(0.25)
    assert abs(sm.delay_s - 0.25) < 1e-9
    out = {}
    raw_err = []
    for i in range(int(12 * fps)):
        t = i / fps
        X = np.zeros((P.NUM_JOINTS, 3))
        X[:, 0] = 0.3 * np.sin(2 * np.pi * f * t)
        n = rng.normal(0, 0.02, X.shape)
        raw_err.append(np.abs(n[:, 0]).mean())
        r = sm.push(t, X + n, s, ok, {'Left': np.zeros((21, 3))})
        if r is not None:
            out[round(r[0] * fps)] = r[1][:, 0].mean()
            assert r[3]['Left'].shape == (21, 3)
    ts = np.array(sorted(out))[10:]
    y = np.array([out[i] for i in ts])
    ref = lambda lag: 0.3 * np.sin(2 * np.pi * f * (ts / fps - lag))  # noqa: E731
    lags = np.linspace(-0.1, 0.1, 81)
    best = lags[int(np.argmin([np.mean((y - ref(lg)) ** 2) for lg in lags]))]
    amp = float(np.sqrt(2) * y.std() / 0.3)
    print(f'  1.2 Hz、振幅 30 cm 的動作加 2 cm 雜訊：落後 {1000 * best:.0f} ms，振幅保留 {amp:.0%}')
    assert abs(best) < 0.012 and 0.93 < amp < 1.03
    assert sm.push(100.0, np.zeros((P.NUM_JOINTS, 3)), s, ok) is None            # 中斷很久：重新開始，不會把舊的混進來
    passthrough = PR.Smoother(0.0).push(1.0, np.ones((P.NUM_JOINTS, 3)), s, ok)
    assert passthrough is not None and passthrough[0] == 1.0


def test_live_tta():
    """即時的「高精度推論」：原圖和左右翻轉各算一次。模型對左右對稱時，結果要和只算一次一樣（左右標籤、深度都要換回來）。"""
    W, H = 640, 480
    rng = np.random.default_rng(5)
    base_k = np.column_stack([rng.uniform(150, 500, 133), rng.uniform(60, 420, 133)])
    base_z = rng.normal(0, 0.2, 133)
    PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)]

    def model_out(flipped):
        """假的模型：照實回答畫面上的點。翻轉的畫面裡，人的左手出現在右手的位置，所以標籤是對調的。"""
        k, z = base_k.copy(), base_z.copy()
        if flipped:
            k[:, 0] = W - 1 - k[:, 0]
            for a, b in PAIRS:
                k[[a, b]], z[[a, b]] = k[[b, a]], z[[b, a]]
        return k, z, np.full(133, 0.9)

    est = Pose3DEstimator.__new__(Pose3DEstimator)
    est.bbox, est.n, est.freq, est.live_tta, est.ms, est.raw, est.hands = np.array([100.0, 40, 540, 440]), 2, 8, False, 0.0, None, None
    est._crop = lambda img, bbox: (bool(img[0, 0, 0]), None, None)       # 假的裁切：只記這張是不是翻轉過的
    calls = []

    def fake_run(crops, frames=None):
        calls.append((len(crops), frames))
        return [model_out(c[0]) for c in crops]
    est._run = fake_run
    est._update_bbox = lambda img, k, s: True
    img = np.zeros((H, W, 3), np.uint8)
    img[:, W // 2:] = 255                                                 # 翻轉後左上角變成 255
    plain = est.batch([img, img])
    est.live_tta = True
    tta = est.batch([img, img])
    assert calls == [(2, 2), (4, 2)]
    for (k1, z1, s1, h1), (k2, z2, s2, h2) in zip(plain, tta):
        assert np.allclose(k1, k2, atol=1e-6) and np.allclose(z1, z2, atol=1e-9) and np.allclose(s1, s2)
        assert np.allclose(h1[0], h2[0])                                  # 手：用原圖的


def _arms_forward(seconds=24.0, fps=30.0, yaw=0.0, aim=0.0, **mk):
    """手指向鏡頭的那段時間：(手臂方向誤差的平均（度）, 往前伸出的程度（真實值的幾成，中位數）, 有標記的幀的比例)。"""
    cam = sim.Camera()
    model = sim.OverlapModel(cam, seed=1, **mk)
    trk = track.Tracker(cam.size, cam.hfov, HEIGHT, THR)
    truth, err, reach, marked = {}, [], [], []
    for i in range(int(seconds * fps)):
        t = i / fps
        J, th = sim.arms_forward(t, yaw, aim)
        truth[i] = (cam.world(J), th)
        r = trk(t, *model(J, 1 / fps))
        if th > np.radians(85) and t > 6:
            marked.append(any(j in trk.sk.fs_now for j in (P.L_WRI, P.R_WRI, P.L_ELB, P.R_ELB)))
        if r is None or r[0] < 6:
            continue
        T, th = truth[round(r[0] * fps)]
        if th < np.radians(75):
            continue
        for sh, el, wr in sim.ARMS:
            for a, b in ((sh, el), (el, wr)):
                u, v = r[1][b] - r[1][a], T[b] - T[a]
                err.append(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))
            reach.append((r[1][wr][2] - r[1][sh][2]) / (T[wr][2] - T[sh][2]))
    return float(np.mean(err)), float(np.median(reach)), float(np.mean(marked))


def test_arm_towards_camera():
    """手往鏡頭伸直（肩膀、手肘、手腕在畫面上疊成一點）：模型把深度縮回去、分不出前後、手肘被擋住，手都要伸得出去。
    v1.7.0：深度完全縮回去時手伸出 1%，手肘被擋住時 18%。"""
    cases = (('模型深度正常', {}), ('深度只剩 20%', dict(arm_comp=0.2)), ('完全分不出前後', dict(arm_comp=0.0)),
             ('手肘被擋住（分數 0.25）', dict(arm_comp=0.2, elbow_score=0.25)), ('稍微判反（-25%）', dict(arm_comp=-0.25)),
             ('背對鏡頭、完全分不出', dict(arm_comp=0.0, yaw=np.pi)), ('手往外偏 25°', dict(arm_comp=0.2, aim=np.radians(25))))
    for label, kw in cases:
        e, reach, marked = _arms_forward(**kw)
        print(f'  {label}：手臂方向誤差 {e:.1f}°，伸出去 {reach:.0%}')
        assert e < 12.0 and 0.8 < reach < 1.15, (label, e, reach)
        if kw.get('arm_comp', 1.0) <= 0.2 and not kw.get('aim'):
            assert marked > 0.8, (label, marked)                         # 預覽畫面上有「朝前／朝後」的記號
    sk = Skeleton(HEIGHT)                                                 # 手長時間朝鏡頭、深度被縮回去：學到的骨長不會跟著變短
    cam = sim.Camera()
    model = sim.OverlapModel(cam, seed=2, arm_comp=0.1)
    trk = track.Tracker(cam.size, cam.hfov, HEIGHT, THR)
    for i in range(600):
        J, _ = sim.arms_forward(2.0 + 2.0 * (i % 60) / 60.0)            # 幾乎一直舉在前面
        trk(i / 30.0, *model(J, 1 / 30))
    i_arm = sk._bone(P.L_SHO, P.L_ELB)
    assert trk.sk.L[i_arm] > 0.88 * motion_upper(), trk.sk.L[i_arm]


def motion_upper():
    import motion
    return motion.UPPER


def main():
    for f in (test_bend_axis, test_smoother_no_lag, test_live_tta, test_skeleton_keeps_image_position, test_arm_towards_camera,
              test_floor_and_pitch_consistent, test_live_accuracy, test_rays_never_hurt):
        f()
        print('ok', f.__name__)
    print('全部通過')


if __name__ == '__main__':
    main()
