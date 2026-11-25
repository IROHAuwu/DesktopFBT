"""用真的模型（RTMW3D ＋ GEM-X）在有正確答案的圖片上跑 App 的「圖片 → 姿勢」流程，和標準方式比較。
圖片與 3D 正確答案來自 mmpose 的測試資料（Human3.6M 的 4 張圖）。同時檢查模型的自動下載（約 3.5 GB）。
只在 GitHub Actions 手動執行：DESKTOPFBT_REAL_MODEL=1 python tests/test_gemx_real.py
"""
import io
import os
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
TMP = tempfile.mkdtemp()
import paths  # noqa: E402
paths.base_dir = lambda: TMP

import gemx as GX  # noqa: E402
import models  # noqa: E402
import pose as P  # noqa: E402
from lift import Pose3DEstimator, lift_image  # noqa: E402

RAW = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/h36m/'
# H36M 17 點 → Halpe26
H36 = {1: P.R_HIP, 2: P.R_KNEE, 3: P.R_ANK, 4: P.L_HIP, 5: P.L_KNEE, 6: P.L_ANK,
       11: P.L_SHO, 12: P.L_ELB, 13: P.L_WRI, 14: P.R_SHO, 15: P.R_ELB, 16: P.R_WRI}


def note(msg):
    print(msg, flush=True)
    print(f'::notice title=GEM-X::{msg}', flush=True)


def get(name):
    with urllib.request.urlopen(RAW + name, timeout=60) as r:
        return r.read()


def align(A, B, scale=True, reflect=False):
    """把 A 對到 B（旋轉＋平移，scale：加上縮放，reflect：允許鏡射）。回傳 (平均誤差, 旋轉角度, 縮放, det)。"""
    a, b = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    det = float(np.sign(np.linalg.det(U @ Vt)))
    d = np.ones(3)
    if not reflect and det < 0:
        d[2] = -1
    R = (U * d) @ Vt                                     # a @ R ≈ b
    k = float((S * d).sum() / (a ** 2).sum()) if scale else 1.0
    err = float(np.linalg.norm(k * a @ R - b, axis=1).mean())
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))) if np.linalg.det(R) > 0 else 180.0
    return err, ang, k, det


def main():
    if os.environ.get('DESKTOPFBT_REAL_MODEL') != '1':
        print('略過（需要網路與真的模型）')
        return
    from engine import Engine
    ann = np.load(io.BytesIO(get('test_h36m_body3d.npz')), allow_pickle=True)
    est = Pose3DEstimator(models.ensure_pose3d(), 'cpu', 1)
    # 自動下載（和 App 走同一段程式）
    t0, last = time.time(), [0.0]

    def prog(done, total):
        if time.time() - last[0] > 20:
            last[0] = time.time()
            print(f'  下載 {done >> 20} / {total >> 20} MB', flush=True)
    GX.download(prog)
    assert GX.missing() == [] and GX.broken() == []
    assert not [f for f in os.listdir(GX.folder()) if f.endswith('.part')]
    size = sum(os.path.getsize(p) for p in GX.paths().values())
    note(f'自動下載：{size / 1e9:.2f} GB，{time.time() - t0:.0f} 秒，檢查碼正確')
    p = GX.paths()
    t0 = time.time()
    gx = (GX.Features(p['gemx_body_fp16.onnx'], 'cpu'), GX.Keypoints(p['gemx_kp2d_fp16.onnx'], 'cpu'),
          GX.Net(p['gemx_motion.onnx'], 'cpu'), GX.Body(p['gemx_skeleton.npz']))
    note(f'載入：{time.time() - t0:.0f} 秒')
    D = np.diag([1.0, -1.0, -1.0])                       # 相機座標（y 下、z 往前）→ App 的座標（y 上、z 朝相機）
    res = {'標準': [], 'GEM-X': []}
    est.tta = True
    for i, name in enumerate(ann['imgname']):
        img = cv2.imdecode(np.frombuffer(get(name), np.uint8), cv2.IMREAD_COLOR)
        est.bbox, est.n = None, 0
        k, z, s = est(img)
        if k is None:
            note(f'{name}: 沒偵測到人')
            continue
        S = ann['S'][i][:, :3] @ D.T
        gt = np.array([S[j] for j in H36])
        height = 1.0 / 0.491 * float(np.mean([np.linalg.norm(S[1] - S[3]), np.linalg.norm(S[4] - S[6])]))
        t0 = time.time()
        Xg = Engine._gemx_image(gx, img, k, s, getattr(est, 'cue', None))
        ms = (time.time() - t0) * 1000
        Xs, _ = lift_image(k, z, s, (img.shape[1], img.shape[0]), 60, height, 0.3)
        line = [os.path.basename(name)]
        for key, X in (('標準', Xs), ('GEM-X', Xg)):
            if X is None:
                line.append(f'{key} 沒有結果')
                continue
            ours = np.array([X[h] for h in H36.values()])
            pa, _, sc_, det = align(ours, gt, reflect=True)
            rot, ang, _, _ = align(ours, gt)
            rigid, _, _, _ = align(ours, gt, scale=False)
            root = float(np.linalg.norm((ours - ours.mean(0)) - (gt - gt.mean(0)), axis=1).mean())
            res[key].append((pa, rot, rigid, root, ang, sc_, det))
            line.append(f'{key}：對齊後 {rot * 100:.1f}cm（允許鏡射 {pa * 100:.1f}、不縮放 {rigid * 100:.1f}、只平移 {root * 100:.1f}），'
                        f'和相機座標差 {ang:.0f} 度，大小 ×{sc_:.2f}')
        note('；'.join(line) + f'；GEM-X {ms:.0f} ms（CPU）')
        cue = getattr(est, 'cue_detail', None) or {}
        print('  頭與手：', {k_: {a: (round(float(b), 2) if isinstance(b, float) else b) for a, b in v.items()}
                         for k_, v in cue.items()}, flush=True)
    for key, r in res.items():
        r = np.array(r)
        note(f'{key}（{len(r)} 張）平均：對齊後 {r[:, 1].mean() * 100:.1f}cm、不縮放 {r[:, 2].mean() * 100:.1f}cm、'
             f'只平移 {r[:, 3].mean() * 100:.1f}cm、方向差 {r[:, 4].mean():.0f} 度')
    g = np.array(res['GEM-X'])
    assert len(g) >= 3
    assert g[:, 1].mean() < 0.12, 'GEM-X 的姿勢和正確答案差太多（左右或座標可能反了）'
    assert g[:, 1].mean() < g[:, 0].mean() + 0.03, '只有鏡射後才對得上：左右反了'
    assert g[:, 4].max() < 40, '朝向和相機看到的不一樣'
    assert 0.8 < g[:, 5].mean() < 1.25, '大小不對'
    print('全部通過')


if __name__ == '__main__':
    main()
